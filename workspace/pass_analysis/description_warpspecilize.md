# Warp specialization (matmul pass analysis)

Companion notes for `_p_matmul_…_64x256x128x1`.

| | |
|--|--|
| **Pipeline** | `AutomaticWarpSpecialization` (`AutomaticWarpSpecialization.cpp`) |
| **IR spine** | `input_ir.ttir` → … → `stage_cluster.ttir` → `after_partition_scheduling.ttir` → … |
| **This doc** | walk each AWS sub-pass; **§1 = PartitionScheduling** (filled in); later §§ TBD |

**AWS order (high level):**

```text
1 PartitionScheduling
2 NVWSHoistTmemStore
3 NVWSInsertAref / InsertTmemAref
4 SCCP + CSE
5 NVWSLowerAref
6 PartitionLoops          ← physical per-partition scf.for clones
7 NVWSLowerWarpGroup
8 ScheduleLoops
9 multiBufferTMADescriptors
```

---

# 1. PartitionScheduling

| | |
|--|--|
| **Pass** | `tritongpu-partition-scheduling` (`PartitionScheduling.cpp` + `PartitionSchedulingUtility.h`) |
| **Role** | first step of AWS |
| **IR in** | `stage_cluster.ttir` / `input_ir.ttir` — one `scf.for` with `tt.warp_specialize` |
| **IR out** | `after_partition_scheduling.ttir` — same for, ops tagged `ttg.partition` |
| **DOT / PNG** | `partition_scheduling_dots/graph-_p_matmul_…_0-*.png` |

**Goal.** Split the warp-specialize loop body into **warp partitions** (here: load vs compute/store) so later passes can insert arefs and `PartitionLoops` can clone one `scf.for` per partition region.

```text
buildGraph → data mark → duplicateViewOps → (manual) → initial 1-node partitions
    → mergePartitions → propagatePartitions → assign ids → … → serialize attrs
```

## 1.0 General idea and primitives

### General idea

1. **Mark the data path** from payload seeds (descriptor **load**, TMEM load, **MMAv5**, ops with `data` attr, …). On *this* matmul the heavy anchors are the two `descriptor_load`s and the compute/store chain (`tt.dot` is classic MMA, not MMAv5 — still treated as data/`NONE` until merged with store).
2. **Give each data op its own partition**, then **merge** along crossing edges using **heuristics** (what *may* share a partition) and **constraints** (what *must not*, e.g. TMEM↔MMA, two MANUALS). That decides the warp cut: here **LOAD** vs **STORE/compute**.
3. **Solve the rest** (index arith, `scf.if`/`scf.for`, reduce bodies, …) via **node–edge** dependency propagation (region union, backward/forward along edges) so every warp that needs control owns a copy of it — without merging load and MMA into one partition.

```text
data seeds → 1-node partitions → merge (heuristics ∩ constraints)
                                    ↓
                         fixed data partitions (anchors)
                                    ↓
              propagate along nesting + SSA edges → non-data / regions
```

### Primitives

Analysis overlay on the for (not a full-module rewrite).

| type | meaning |
|------|---------|
| **`Node`** | vertex wrapping an `Operation*` or `Value` (iter_arg / result); also parent/child tree for regions |
| **`Port`** | `{Node*, idx}` — result slot (`OutputPort`) or operand slot (`InputPort`) |
| **`Edge`** | `OutputPort → InputPort` (one SSA use); `getFromNode()` / `getToNode()` |
| **`Partition`** | set of nodes + flags (`LOAD`/`STORE`/`VIEW`/…) + cost |

Several graph `Node`s may temporarily share one MLIR `op*` (e.g. after `duplicateViewOps`). Real IR clones happen later (`cloneMultiPartitionDataOps`, `PartitionLoops`).

### In/out edges and from/to (IR example)

From `stage_cluster.ttir`:

```mlir
%43 = tt.descriptor_load ... -> tensor<64x128xf8E5M2, ...>
%45 = ttg.convert_layout %43 ...
%47 = tt.dot %45, %46, %arg51 ...
```

```text
  Node(load %43) ──edge──► Node(convert %45) ──edge──► Node(dot %47)
       result0                  in0 / out0                 in0 (A)
```

For one edge `%43 → %45`:

| API | value |
|-----|--------|
| `edge.getFromNode()` | Node(`descriptor_load` `%43`) — **producer** |
| `edge.getFromIdx()` | `0` — load’s result slot |
| `edge.getToNode()` | Node(`convert_layout` `%45`) — **consumer** |
| `edge.getToIdx()` | `0` — convert’s operand slot |

From a **node’s** point of view:

| API | meaning | on `%45` |
|-----|---------|----------|
| `node->getInEdges()` | edges **into** this node (one per operand) | from `%43` (and any other operands) |
| `node->getOutEdges()` | edges **out of** this node (all uses of each result) | to `%47`’s A operand |

So: **in-edge** = “who feeds me?” → `getFromNode()` is upstream; **out-edge** = “who uses me?” → `getToNode()` is downstream. Backward `propagatePartitions` walks `getInEdges()` then `edge.getFromNode()`.

## 1.1 `initialDataValues` + `propagateDataValues`

**DOT:** `…-0000-input` → `…-0001-input` (after data; blue = data).

**Seeds** (`initialDataValues`): result slots of payload producers — e.g. `tt.descriptor_load`, `TMEMLoad`, MMAv5, ops with a `data` attr. Represented as `(Node*, result_idx)` / `OutputPort`.

**Propagate:** BFS **forward** along uses (`getOutputsFromPort`). Mark consumer result slots (and thus nodes) as data until fixed point.

**On this IR:** data ≈ `%43/%44` loads → converts → `%47` dot → select/epilogue → store path.
**Not data:** pure tile-index arith before loads (`%55`… inside `%41`’s then, `%39/%40`, …).

## 1.2 `duplicateViewOps`

**DOT:** `…-0002-input`.

**View ops (syntactic):** `BroadcastOp`, `ExpandDimsOp`, `ConvertLayoutOp`, `MemDescViewTrait` — `Flags::VIEW` for partitioning (not a pedantic zero-copy alias test). Even `1→N` broadcast is `VIEW`; merge later uses **element counts** to prefer consumer-side expand and keep aref small.

**Why:** merge rules assume **one consumer per view** (`assert(out_edges.size()==1)` on `view_producer`).

**How:** if a **data** view has multiple out-edges, keep the first; for each extra user, add a **sibling** `Node` (same `op*`) under `getParent()`, rewire that edge, copy in-edges + `dataValue` flags. Load is **not** duplicated.

This matmul’s converts already have one user → often a no-op here.

## 1.3 `deserializeManualPartitions`

If ops already have `ttg.partition = array<i32: …>`, create `MANUAL` partitions and attach nodes.

**This IR:** no pre-existing `ttg.partition` → no-op. Partitions are assigned automatically below.

## 1.4 `initialPartitionAssignment`

**DOT:** `…-0004-initial.png`.

Every **data** node without a partition gets its **own** partition (22 one-node partitions here). Index/control stay unpartitioned (white).

## 1.5 `mergePartitions`

**DOT:** `…-0005`…`0026-merge-step` → **`…-0027-merge.png`**.

### Mechanism

1. Worklist = **crossing** data edges (producer/consumer in different partitions).
2. Try `heuristics` **in order**; if `apply(edge)` and all `constraints` pass → `Partition::merge(A,B)`.
3. Merge moves **all** nodes of one partition into the other (not only the two endpoint ops).
4. Repeat to fixpoint; then pair-wise `partition_heuristics`.

| item | role |
|--|------|
| **`heuristics`** | `(name, Edge→bool)` — “should we try merge along this edge?” e.g. `view_consumer`, `sequence`, `none_consumer`, `load_local_alloc`, … |
| **`constraints`** | hard vetoes — e.g. don’t merge two `MANUAL`s; don’t merge `TMEM`↔`MMA` |

`VIEW` on a **partition** is kept only while **all** nodes in it are view ops; after merge with `dot`/`store`, partition loses `VIEW`.

### Result on this matmul (`0027-merge`)

Collapsed to **2 data partitions**:

| role | contents (SSA from `stage_cluster.ttir`) |
|------|------------------------------------------|
| **LOAD** | `%43`, `%44` `tt.descriptor_load` |
| **STORE / compute** | `%45/%46` convert → `%47` `tt.dot` → `%50` select → epilogue (`%51` if: mulf/addf/reduce/… ) → `%99` convert → `descriptor_store` |

Cross-partition data edges (red in DOT) are the future **aref** cuts: load tiles → compute converts/dot.

After `assignPartitionIds`, serialized ids match `after_partition_scheduling.ttir` (typically compute/store → `0`, loads → `1`).

## 1.6 `propagatePartitions`

**DOT:** `…-0028`…`0033-propagate` (and again after assign-no-use).

Phases: region upward → backward onto non-data → forward onto leftovers → reduce body → tmem init-store patch → **re-propagate from patched stores**.



### 1.6.1 Region upward (leaves → parents)

A **leaf** = region node whose children are all non-region (e.g. `%41 = scf.if` under `%22 = scf.for`).

```text
leaf.partitions = ∪ child.partitions     # SetVector → dedup
then walk leaf->getParent() …            # for %41, parent = scf.for %22
  parent.partitions ∪= children
```

The `scf.if` region may get multiple partition labels `ttg.partition = array<i32: 0, 1>` while body ops stay split. That means “clone this `if` into both warp regions,” not “merge load with MMA.”

### 1.6.2 Backward onto non-data

Seeds = **region nodes ∪ data nodes**. For each seed, BFS **backward** (`getInEdges` → `getFromNode`):

- skip if target is **data**
- else `addPartitions(seed.partitions)` — **accumulate / union**, do not replace; `changed` only if the set grows

Partition labels move **consumer → producer** (upstream). A non-data op reached from load (part 1) and later from compute (part 0) ends as `{0,1}`. Address math feeding loads picks up the load partition(s); shared trip-count updates (`%52/%53/%54`) can become `{0,1}`.

### 1.6.3 Forward onto remaining unpartitioned nodes

After 1.6.2, collect nodes with **`!hasPartition()`**. Repeatedly:

```text
for each unpartitioned node N:
  for edge in N->getInEdges():          # look at producers
    if producer has partition:
      N->setPartition(producer's partition)   # write onto N (consumer)
drop N once it has a partition; stop if no progress
```

Same API as 1.6.2 (`getInEdges` / `getFromNode`), but labels move **producer → consumer** (downstream) — that is why the code comment says “forward.” Uses **`setPartition`** (replace with one partition), not `addPartitions` union. Covers leftovers that never got a partition from the backward sweep.

### 1.6.4 Push partitions into `tt.reduce` bodies

```cpp
for each ReduceOp node:
  child->addPartitions(reduce.partitions)  // for every nested child
```

`tt.reduce` owns a **region** (combiner lambda). `buildGraph` nests that body under the reduce node. Example from `stage_cluster.ttir`:

```mlir
%89 = "tt.reduce"(%88) <{axis = 0}> ({
^bb0(%arg55: f32, %arg56: f32):
  %100 = tt.elementwise_inline_asm "max.NaN..." %arg55, %arg56 -> f32
  tt.reduce.return %100 : f32
})
```

```text
Node(tt.reduce %89)     ← parent (epilogue / partition 0)
  ├── inline_asm %100   ← must same partition
  └── reduce.return
```

The combiner is not separate warp work; it must stay with the reduce. Different partition ids would break `PartitionLoops` (region cloned apart from its parent). After serialize: reduce, asm, and `reduce.return` share `ttg.partition = array<i32: 0>`.

### 1.6.5 Patch: `tmem_alloc` → init `tmem_store` (non-MMA partition)

**Not used on this FP8 `tt.dot` matmul** (no TMEM/MMAv5). Relevant when accumulator lives in TMEM.

Corner case (code comment): an **initial** `ttng.tmem_store` that fills a buffer right after `ttng.tmem_alloc`, before MMA, may have been forward-propagated into the **MMA** partition. It should stay with the **alloc** (non-MMA / e.g. 4-warp) partition instead.

`TMEMStore` operands: `0:dst`, `1:dep(token)`, `2:src`, `3:pred`.

```text
%buf, %tok0 = ttng.tmem_alloc ...
%tok1 = ttng.tmem_store ..., %buf, %tok0, ...   // dep operand idx 1 ← alloc token
// later MMA uses %buf / tokens
```

Logic:

- only **non-data** `TMEMStore` nodes
- find in-edge with `getToIdx() == 1` (token); `getFromNode()` must be **`TMEMAllocOp`** (not MMA)
- `setPartition` to alloc’s first partition whose flags are **not** `MMA`

If the token came from MMA instead (`MMA → store`), the `isa<TMEMAllocOp>` check fails → no-op. That is a different chain than alloc→init-store→MMA.

### 1.6.6 Re-propagate from patched `tmem_store`s

After 1.6.5 moves those stores off MMA, their **non-data producers** may still hold the old MMA partition from earlier propagate. Fix by BFS **backward** from each `patched_nodes` entry only:

```text
seed = patched tmem_store (new non-MMA partition)
for in-edges: skip if edge.isDataValue()
  fromNode->addPartitions(seed.partitions)   # union onto upstream non-data
  continue BFS on fromNode
```

Narrower than §1.6.3 (not “all unpartitioned nodes”) — only the fan-in of the patched stores. No-op on this matmul when `patched_nodes` is empty.

## 1.7 After scheduling (attrs vs real clones)

| step | effect |
|------|--------|
| `assignPartitionIds` | stable numeric ids (store/other → mma → load ordering) |
| `assignPartitionsForOpsWithNoUse` | e.g. `llvm.intr.assume` |
| `duplicateCheapOps` | optional cheap NONE path clones to avoid aref |
| `serialize` | write `ttg.partition` / `ttg.partition.outputs` / stages |
| `cloneMultiPartitionDataOps` | IR-clone data ops that still have multiple partition ids |

**For-loop control duplication** is **not** finished here: still one `scf.for` with multi-id attrs.
Physical per-partition `scf.for` copies happen later in **§6 PartitionLoops** (`cloneForOp`).

## 1.8 Quick map: dump index → step

| PNG | step |
|-----|------|
| `0000`–`0003` | input / after data / after duplicate views / final pre-assign |
| `0004` | initial 1-node data partitions |
| `0005`–`0026` | merge steps |
| `0027` | merged (2 partitions) |
| `0028`–`0033` | propagate |
| `0034`+ | assign ids / no-use / more propagate / duplicate / final |

---

# 2. NVWSHoistTmemStore

*(TBD)*

---

# 3. NVWSInsertAref / InsertTmemAref

*(TBD — cross-partition data edges from §1 become aref channels)*

---

# 4. SCCP + CSE

*(TBD)*

---

# 5. NVWSLowerAref

*(TBD)*

---

# 6. PartitionLoops

*(TBD — `cloneForOp`: one `scf.for` per partition region; shared control with multi-`ttg.partition` is cloned into each)*

---

# 7. NVWSLowerWarpGroup

*(TBD)*

---

# 8. ScheduleLoops

*(See `description_stage_cluster.md` for the coarse SWP schedule; ordering vs AWS may differ by dump recipe.)*

---

# 9. multiBufferTMADescriptors

*(TBD)*
