# Warp specialization (matmul pass analysis) — PartitionScheduling

Companion notes for `_p_matmul_…_64x256x128x1`.

| | |
|--|--|
| **Pipeline** | `AutomaticWarpSpecialization` (`AutomaticWarpSpecialization.cpp`) |
| **IR spine** | `input_ir.ttir` → … → `stage_cluster.ttir` → `after_partition_scheduling.ttir` → … |
| **This doc** | **§1 = PartitionScheduling** |
| **Sibling** | `WarpSpecialization_Rest.md` — AWS parts 2–10 (HoistTmemStore → …) |

**AWS order (high level):**

```text
1 PartitionScheduling          ← this file
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
2. **Seed partitions from payload data values**, then only consider merges across **crossing data edges** (i.e., edges whose endpoints are marked as payload). `heuristics` pick merges that keep co-location valid; `constraints` veto incompatible groups (e.g., TMEM↔MMA, multiple MANUALs). The objective is to **minimize how much marked payload has to cross partition boundaries** (aref communication), which yields the warp cut (here **LOAD** vs **STORE/compute**).
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

Every **data** node without a partition gets its **own** partition_idx (22 one-node partitions here). Index/control stay unpartitioned (white).

**partition flags are set here as well.** `node->setPartition(p)` calls `p->add(node)`, which ORs `getNodeFlags(node)` onto `partition->flags`. With one node per partition, the partition’s flags **are exactly that op’s kind**:

| op (via `getNodeFlags`) | partition flag |
|-------------------------|----------------|
| `descriptor_load` / `gather` | `LOAD` |
| `descriptor_store` / `scatter` (or attr `"store"`) | `STORE` |
| MMAv5 / attr `"mma"` | `MMA` |
| `TMEMLoad` / `TMEMStore` | `TMEM` |
| `math.exp2` | `SFU` |
| view (`convert_layout`, broadcast, …) | `VIEW` |
| else (e.g. `tt.dot`, `arith.*` on this path) | `NONE` |

So in `0004-initial`, each badge’s `[LOAD]` / `[VIEW]` / `[STORE]` / `[NONE]` is already the partition flag from that single op. Later **merge** ORs flags of merged nodes together (and drops `VIEW` unless the whole partition stays all-view). Numeric `Partition::id` values come only in §1.7 `assignPartitionIds`.

**DOT badge caveat.** Labels look like `21{0}[STORE]` = `vizId{cost}[flags]`. Only **`[flags]`** reliably match the analysis. The leading number is a **dump-only** id from `VisualizationInfo` (first time that `Partition*` was drawn; reused across all dumps in one `analyze()`). It is **not** `Partition::id` and will **not** match final `ttg.partition` (e.g. after merge/ids you may still see viz `3`/`21` while IR gets `0`/`1`).

## 1.5 `mergePartitions`

**DOT:** `…-0005`…`0026-merge-step` → **`…-0027-merge.png`**.

### Mechanism

1. Worklist = **crossing** data edges (producer/consumer in different partitions).
2. Try `heuristics` **in order**; if `apply(edge)` and all `constraints` pass → `Partition::merge(A,B)`.
3. Merge moves **all** nodes of one partition into the other (not only the two endpoint ops) by changing the partition flag.
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

**Background.** After merge (§1.5), only **data** ops have partitions (the warp cut is already decided: here LOAD vs STORE/compute). Index arith, `scf.if`/`scf.for`, reduce combiners, and similar **non-data** / region structure are still unlabeled, yet each warp region later needs its own copy of the control and addressing that feeds its work. Merging those into a single partition would undo the cut; leaving them empty would break `PartitionLoops` cloning.

**Work in §1.6.** Propagate existing partition membership along nesting and SSA edges **without merging partitions**: union labels onto regions and non-data producers/consumers (and a few TMEM/reduce corner cases) so shared control can be tagged `ttg.partition = array<i32: 0, 1>` while payload stays split. Phases below: region upward → backward onto non-data → forward onto leftovers → reduce body → tmem init-store patch → re-propagate from patched stores.

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

## 1.7 `assignPartitionIds`

**DOT:** `…-0034-assign-partition-ids`.

Until here, partitions are unnamed buckets with **flags** only (`LOAD` / `STORE` / `MMA` / …). This step assigns stable numeric `Partition::id` values that later become `ttg.partition = array<i32: …>`.

### Four ordered groups (kind heuristic)

Every partition is classified into **one** of four buckets (first match wins: `STORE` before `MMA` before `LOAD`):

| group | test | role |
|-------|------|------|
| **other** | no STORE/MMA/LOAD | default / leftover compute |
| **store** | `flags & STORE` | descriptor store (+ merged compute on this matmul) |
| **mma** | `flags & MMA` | MMAv5 warp (not classic `tt.dot`) |
| **load** | `flags & LOAD` | descriptor load |

Ids are then renumbered **in that group order**:

```text
other → store → [reserve default] → mma → load
```

So the intended warp-specialize shape is at most these **four kinds** of specialized warps (default/other, store, mma, load). Multiple partitions of the same kind still get consecutive ids within their group (not a hard global cap of four `Partition*` objects).

### Default / id `0`

```cpp
// after assigning other + store:
if (idx == 0)
  idx++;   // skip 0 so MMA/LOAD never become the default partition
```

If there were no `other` and no `store` partitions, `idx` would still be `0`; bumping ensures **MMA and LOAD never own partition 0**. Partition `0` is treated as the **default** group elsewhere (e.g. `assignPartitionsForOpsWithNoUse` looks up `id == 0`).

### On this matmul

After merge: one **STORE** (dot+epilogue+`descriptor_store`) and one **LOAD** (two loads). No separate MMA flag (`tt.dot` is `NONE` until merged with store). Typical serialize: store/compute → `0`, loads → `1` (matches `after_partition_scheduling.ttir`).

## 1.8 `assignPartitionsForOpsWithNoUse`

**DOT:** `…-0035-assign-no-use`.

Some nodes still have **empty** partition sets after propagate (e.g. `llvm.intr.assume`, other side-effect / no-use ops that never sat on a data edge).

For each such node:

1. Look at **siblings** under the same parent region (`parent->getNodes()`, skip self).
2. For every sibling that is an **op** and already `hasPartition()`, **`addPartitions`** that sibling’s set (union; not only the first sibling).
3. If **no** partitioned sibling was found (`!done`): assign the **default** partition with **`id == 0`** (`setPartition`). Create one with `id = 0` if it does not exist yet.

Cannot just copy the parent op’s partitions (comment in code: that can pull in extras like tmem tokens). Prefer “same region peers,” else default `0`.

Nuances: **all** partitioned siblings (union); fallback is “no usable sibling partition,” not merely “no siblings.”

## 1.9 `duplicateCheapOps`

**DOT:** `…-0042`…`duplicate` dumps.

Goal: if payload leaves partition **A**, wanders through a cheap chain in partition **B**, then returns to **A**, assign that cheap path to **both** partitions (`addPartition`) so later IR cloning can run the cheap ops in A too and **skip the A→B→A aref**. No new graph `Node` here — multi-membership only; real clones are in §1.11 `cloneMultiPartitionDataOps`.

1. **Cheap op + single-partition threshold**
   Candidate ops: `getNodeFlags == NONE` or `SFU`.
   Both ends of the starting crossing edge must have **exactly one** partition (`getPartitions().size() == 1`). Multi-partition nodes are skipped.

2. **Cross-partition situation**
   For each partition, take **out-crossing** data edges. Pattern: producer in **A** (`startPartition`) → consumer cheap node in **B** (`partition`). Search only while staying on cheap, single-partition nodes in **B**.

3. **DFS via `parentMap.emplace(child, node)`**
   From the first B node, DFS on out-edges among candidates still in **B**.
   `std::map<Node*, Node*> parentMap`: `parentMap[child] = node` records the predecessor so the path can be reconstructed. `stack.push_back(child)` continues the search.

4. **Backward update when path re-enters A**
   If a child is again in `startPartition` (**A**), walk the predecessor chain:

   ```cpp
   node->addPartition(startPartition);
   while (parentMap.find(node) != parentMap.end()) {
     node = parentMap[node];
     node->addPartition(startPartition);
   }
   ```

   Every node on the cheap B-path also gets **A** (multi-partition). Not recursive child mutation — path reconstruction only.

## 1.10 `serialize`

Transfers partition analysis from the temporary **graph** onto the live **MLIR module** (attrs on ops / the warp-specialize `scf.for`). After this, later AWS passes read IR attrs, not the `Graph`.

| attr | written on | meaning |
|------|------------|---------|
| `ttg.partition` | ops (and yields of `scf.for` / `scf.if`) | sorted list of partition **ids** owning that op |
| `ttg.partition.outputs` | `scf.for` / `scf.if` / `tt.reduce` (etc.) | **per-result** partition id lists (boundary values) |
| `ttg.partition.stages` | warp-specialize `scf.for` | array indexed by partition id: MMA → `1`, else `0` (aref buffering lag vs consumers; **not** `loop.stage`) |
| `ttg.warp_specialize.tag` | that `scf.for` | tags which WS loop this scheduling belongs to |

Also merges partition sets when several graph nodes map to the same MLIR op. Result matches `after_partition_scheduling.ttir` (e.g. `ttg.partition.stages = [0, 0]` for load+store on this matmul).

## 1.11 After serialize

| step | effect |
|------|--------|
| `cloneMultiPartitionDataOps` | IR-clone data ops that still have multiple partition ids (so aref insert can handle them) |

**For-loop control duplication** is **not** finished here: still one `scf.for` with multi-id attrs.
Physical per-partition `scf.for` copies happen later in **§6 PartitionLoops** (`cloneForOp`).

## 1.12 Quick map: dump index → step

| PNG | step |
|-----|------|
| `0000`–`0003` | input / after data / after duplicate views / final pre-assign |
| `0004` | initial 1-node data partitions |
| `0005`–`0026` | merge steps |
| `0027` | merged (2 partitions) |
| `0028`–`0033` | propagate |
| `0034` | `assignPartitionIds` |
| `0035` | `assignPartitionsForOpsWithNoUse` |
| `0036`–`0041` | propagate (again) |
| `0042`+ | `duplicateCheapOps` / final |

---
