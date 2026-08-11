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

# 2. NVWSHoistTmemStore

| | |
|--|--|
| **Pass** | `nvws-hoist-tmem-store` (`HoistTmemStore.cpp`) |
| **When** | right after PartitionScheduling in AWS |
| **This matmul** | no TMEM/MMAv5 → typically a no-op |

**Goal (alloc part).** Move `ttng.tmem_alloc` **out of** the `tt.warp_specialize` `scf.for` (and nested fors) so TMEM is allocated **once**, not every iteration. Then **thread the async token** through the loop nest like any other carried value.

Also folds some `tmem_store` into alloc (`FoldTmemStoreIntoAlloc`) and related cleanups; below focuses on `hoistTmemAlloc`.

## 2.1 Why must the token be an iter_arg / yield?

TMEM ops (`alloc` / `store` / `load` / MMAv5) form an SSA **token chain**: each op takes a dep token and produces a new one. That encodes “this buffer use happens after that one.”

If alloc stays **inside** the body, the token is local to that iteration. After hoist, alloc is **outside**, but MMA still runs **inside** every iter. The token that MMA depends on (and the token MMA produces for the next use) must therefore be:

1. passed **in** as a for **init / region iter_arg** (like `%acc` or any carried state), and  
2. passed **out** via **`scf.yield`** so the next iteration / outer loop sees the updated token.

Without yield, the final in-body token would die at the end of the iteration and outer SSA / next-iter deps would be broken — same reason other loop-carried args are yielded.

## 2.2 Safety check (when hoist is refused)

Do **not** hoist across an outer loop if an inner MMA loop’s trip count **depends on that outer IV** and you cannot prove the inner loop always executes (≥1). Otherwise some outer iters would skip MMA while a single outer alloc/token still exists (mismatched lifetime). See § comment in `hoistTmemAlloc`.

## 2.3 Fake IR walkthrough (multi-tier fors)

Illustrative only (types shortened). Partition ids: MMA warp = `0`.

### Before hoist

Alloc lives **inside** the WS for. Its token’s **only** use is as **init** of the inner MMA for (`getUniqueUserLoopAndMMA`).

```mlir
// tt.warp_specialize on outer for
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32
%n = ... : i32
%k = ... : i32   // independent of %i  (safe to hoist)

scf.for %i = %c0 to %n step %c1 iter_args(%arg_other = ...)
    -> (...) {
  // still INSIDE ws for:
  %buf, %tok0 = ttng.tmem_alloc : ... -> !ttg.memdesc<...>, !ttg.async.token
      {ttg.partition = array<i32: 0>}

  %inner:1 = scf.for %kk = %c0 to %k step %c1
      iter_args(%tok = %tok0) -> (!ttg.async.token) {
    // MMA consumes/produces token (partition 0)
    %tok1 = ttng.tc_gen5_mma ..., %tok {ttg.partition = array<i32: 0>}
    scf.yield {ttg.partition = array<i32: 0>} %tok1 : !ttg.async.token
  } {ttg.partition = array<i32: 0>,
     ttg.partition.outputs = [array<i32: 0>]}

  scf.yield {ttg.partition = array<i32: 0>} %arg_other : ...
} {tt.warp_specialize, ttg.partition = array<i32: 0>, ...}
```

### What `hoistTmemAlloc` does

1. **Validate** nest + trip-count independence (or `assume` proves execute-once).  
2. **`moveBefore`** outer WS for; **remove** `ttg.partition` on alloc.  
3. Read **`tokenPartition`** from the for that currently consumes the token (`partition.outputs` at that init-arg index).  
4. **Outer→inner:** `addIterArgsToLoop` with the hoisted token; extend `ttg.partition` / `ttg.partition.outputs` with `tokenPartition`; set inner init to the new iter_arg.  
5. Walk the in-body token chain to the **last unused** token (after MMA/load/store).  
6. **Inner→outer:** `appendToForOpYield` that token so each for **yields** the updated token; result becomes the next outer yield input.

### After hoist (shape)

```mlir
// OUTSIDE ws for — once:
%buf, %tok0 = ttng.tmem_alloc : ... -> !ttg.memdesc<...>, !ttg.async.token
// (no ttg.partition)

%ws:2 = scf.for %i = %c0 to %n step %c1
    iter_args(%arg_other = ..., %tok_o = %tok0)
    -> (..., !ttg.async.token) {
  %inner:1 = scf.for %kk = %c0 to %k step %c1
      iter_args(%tok = %tok_o) -> (!ttg.async.token) {
    %tok1 = ttng.tc_gen5_mma ..., %tok {ttg.partition = array<i32: 0>}
    scf.yield {ttg.partition = array<i32: 0>} %tok1
  } {ttg.partition = array<i32: 0>,
     ttg.partition.outputs = [array<i32: 0>]}

  scf.yield {ttg.partition = array<i32: 0>} %arg_other, %inner#0
} {tt.warp_specialize,
   ttg.partition = array<i32: 0>,
   ttg.partition.outputs = [..., array<i32: 0>], ...}
```

```text
before:  [ws for] { alloc;  inner for(tok=alloc_tok) { mma; yield tok } }
after:   alloc;
         [ws for](tok_o=alloc_tok) {
           inner for(tok=tok_o) { mma; yield tok };
           yield ..., inner_tok
         }
```

**One-liner:** hoist moves **storage creation** out; **token** stays a normal loop-carried SSA value (init + yield) so each iteration’s MMA still chains dependencies correctly.

---

# 3. NVWSInsertAref (`InsertAref.cpp`) / InsertTmemAref

| | |
|--|--|
| **Pass** | `nvws-insert-aref` (`NVWSArefInsertion`) |
| **When** | after HoistTmemStore in AWS |
| **Sibling** | `nvws-insert-tmem-aref` handles **TMEM / MMAv5** separately — this pass **skips** `MMAv5OpInterface`, `TMEMAllocOp`, `TMEMStoreOp` |
| **This matmul** | load partition `1` → compute/store partition `0` via aref on `tt.descriptor_load` results (and similar register values) |

## 3.1 General idea / mechanism

### Why aref?

After PartitionScheduling the IR still looks like **one** loop with shared SSA and `ttg.partition` attrs:

```mlir
%43 = tt.descriptor_load ... {ttg.partition = array<i32: 1>}
%45 = ttg.convert_layout %43 {ttg.partition = array<i32: 0>}
```

That `%43 → %45` edge is fine only while both ops share one region / register file. **PartitionLoops** later splits the WS for into **separate partition regions** (different warp groups). Then p1 keeps the load, p0 keeps convert/dot — they **do not share SSA**. A raw cross-partition SSA use cannot survive.

So the producer must leave data somewhere both sides can see: usually **shared memory** (TMEM for the sibling pass). **Aref** is the handle for that buffer; **put / get** are the protocol for using it safely.

| Piece | Role |
|--------|------|
| **`aref.create`** | Handle wrapping an allocated buffer. Created **outside** the WS loop so both partitions see the same channel. |
| **`aref.put.enter` / `put.exit`** | Producer critical section: enter → writable buf + token; write; exit = published. |
| **`aref.get.enter` / `get.exit`** | Consumer critical section: enter → readable buf + token; read; exit = done reading. |
| **token** | SSA stand-in for the handshake; **LowerAref** turns enter/exit into **mbarriers**. |

Aref does **not** compute — only names the shared buffer and brackets who may touch it when. Same-partition SSA stays as normal SSA (no aref).

### How data moves (put / get)

Same edge as this matmul (`p1` load → `p0` compute):

**1. Create channel once (outside the loop)**

```mlir
%buf  = ttg.local_alloc ...           // real smem
%aref = nvws.aref.create %buf         // handle both partitions use
```

**2. Producer (partition 1) — put = write into the channel**

```mlir
%tok_p, %buf_w = nvws.aref_put_enter %aref   // may write
nvws.descriptor_load ... into %buf_w         // TMA fills aref buffer
nvws.aref_put_exit %aref, %tok_p             // publish
```

No longer “produce register `%43` for someone else” — produce by **storing into the aref buffer**.

**3. Consumer (partition 0) — get = read from the channel**

```mlir
%tok_g, %buf_r = nvws.aref_get_enter %aref
%43' = ttg.local_load %buf_r                 // p0-local registers
%45  = ttg.convert_layout %43'
%47  = tt.dot %45, ...
nvws.aref_get_exit %aref, %tok_g
```

**4. After PartitionLoops (conceptually)**

```
warp group 1 (load):   put_enter → TMA → put_exit
warp group 0 (compute): get_enter → local_load → cvt/dot → get_exit
         \________________ aref / smem ________________/
```

No SSA edge between groups — only **shared buffer + sync**. Pipelining/multibuffering (later) can stage multiple slots on the same aref so load and compute overlap.

### What this pass processes

`runOnFunction` walks each `tt.warp_specialize` for that has partitions, then handles **three producer kinds independently** (same put/get mechanism; different where the produced value comes from). MMAv5/TMEM are skipped here (`InsertTmemAref`). Details in §3.2.

## 3.2 Three cases (where producers come from)

| # | Case | Producer value | Pass order |
|---|------|----------------|------------|
| 1 | **Loop-carried args** | Region iter_arg (tensor / float / int); producer partitions from `ttg.partition.outputs[argNumber − 1]` (IV = arg 0) | At **start of for body** |
| 2 | **Memory ops** | `local_alloc` or `local_alloc(desc_load)` | **First** among in-body ops — put can TMA straight into the aref buf and mark alloc/load stale; prefer before bare register uses of the same load |
| 3 | **Other non-TMEM ops** | Remaining partitioned ops (incl. leftover `desc_load` register uses) | **After** memory ops; insert after the producer. This FP8 matmul: `descriptor_load` (p1) → `convert_layout` / `dot` (p0) |

**MMAv5 / TMEM:** skipped (`MMAv5OpInterface`, `TMEMAllocOp`, `TMEMStoreOp`) → sibling `InsertTmemAref`.

### Case sketches

**Loop-carried** (fake; this matmul’s `%arg51` is often same-partition only):

```mlir
// partition.outputs for %acc includes p1; use in p0 → aref at body start
scf.for ... iter_args(%acc = %cst) -> (...) {
  %x = arith.addf %acc, %one {ttg.partition = array<i32: 0>}
  scf.yield ... %newAcc ...   // produced under p1
}
```

**Memory ops** — `local_alloc(desc_load)` processed before bare load uses:

```mlir
// Before
%ld = tt.descriptor_load %desc[...] {ttg.partition = array<i32: 1>}
%mem = ttg.local_alloc %ld {ttg.partition = array<i32: 1>}
%c  = ttg.local_load %mem {ttg.partition = array<i32: 0>}

// After: put TMA into aref buf (erase %ld/%mem); get retargets / loads for p0
```

If alloc and load have **different** partition ids, they are **not** fused (`isLoadAndAlloc`) — two separate producers.

**Non-tmem register** (real pattern from `after_partition_scheduling.ttir`):

```mlir
%43 = tt.descriptor_load ... {ttg.partition = array<i32: 1>}
%45 = ttg.convert_layout %43 {ttg.partition = array<i32: 0>}
// → put TMA/store into aref; get local_load; %45 uses loaded value
```

## 3.3 How we detect “producer ≠ consumer partitions”

Core gate is inside `insertArefs` → `processResultUses`:

```cpp
auto userPartitions = getPartitionIds(&use);
for (auto id : producedValue.partitions)
  userPartitions.remove(id);
for (auto id : userPartitions) {
  resultsPerPartition[id].insert(result);
  usesPerPartition[id].push_back(&use);
}
if (resultsPerPartition.empty())
  return false;  // no cross-partition consumer → no aref
```

- `getPartitionIds(&use)` = partitions of the **consumer** op (or per-operand attrs when present).
- Remove every **producer** partition id. Whatever remains is a **true cross-partition** consumer.
- Bucket uses by remaining consumer partition id → one get sequence per consumer partition.

Producer partitions for a normal op result: `getProducedValues` → `getPartitionIds(op)` (or `partition.outputs` for region ops). For loop iter_args: `getPartitionOutputs(for)[argNumber − 1]`.

### IR example (real) — from `after_partition_scheduling.ttir`

Load partition **1**, compute partition **0**:

```mlir
%43 = tt.descriptor_load %arg13[...] {ttg.partition = array<i32: 1>}
      : ... -> tensor<64x128xf8E5M2, #blocked2>
%45 = ttg.convert_layout %43 {ttg.partition = array<i32: 0>}
      : tensor<64x128xf8E5M2, #blocked2> -> tensor<64x128xf8E5M2, #ttg.dot_op<...>>
%47 = tt.dot %45, %46, %arg51 {ttg.partition = array<i32: 0>} ...
```

For produced value `%43` with `partitions = {1}`:

| Use | `userPartitions` | after `remove(1)` | action |
|-----|------------------|-------------------|--------|
| `%45` convert_layout | `{0}` | `{0}` | aref get on partition **0** |
| (if some same-p1 use existed) | `{1}` | `∅` | ignored |

Same idea for `%44` → `%46`. Loop-carried detection uses the same `remove` rule on iter_arg uses (see §3.2 sketch).

## 3.4 Create put / get, how they cowork, replace old SSA

Once `resultsPerPartition` is non-empty:

```cpp
aref = createAref(...);           // before outer WS loop: smem buffer(s)
staleOps = createArefPut(...);    // producer: enter → fill buf → exit
for each consumerPartition:
  createArefGet(...);             // consumer: enter → local_load / rewire → exit
for (op : staleOps) op->erase();  // old desc_load / local_alloc when absorbed
```

### Put (`createArefPut`) — producer side

1. `ArefPutEnterOp` → get writable buffer view + token (tagged with **producer** partition).
2. Fill buffer depending on producer kind:
   - `local_alloc(desc_load)` / bare `desc_load` → **TMA into aref buf** (`createNVWSDescriptorLoadOp`), mark load(/alloc) stale.
   - plain tensor / scalar → `LocalStoreOp` (scalar via splat).
3. `ArefPutExitOp` with async kind (`TMALoad` / `NONE`).

Producer SSA (`%43`) is no longer the channel; the aref buffer is.

### Get (`createArefGet`) — consumer side

1. Insert before earliest consumer use in the block.
2. `ArefGetEnterOp` → read-only buffer + token (**consumer** partition).
3. Replace consumer operands:
   - tensor → `LocalLoadOp` from buf, `use->set(localLoad)`;
   - `LocalAllocOp` result → retarget uses to aref memdesc (`replaceUsesAndPropagateType`);
   - scalar → load + unsplat.
4. `ArefGetExitOp` after last consumer / post-dominator.

Put publishes; get consumes (full IR sketch in §3.1). Later `LowerAref` turns enter/exit into barriers / multibuffers; PartitionLoops keeps put ops in p1’s clone and get ops in p0’s clone.

**One-liner:** same-partition SSA stays; cross-partition edges become aref put (producer fills smem) + get (consumer loads / retargets); three producer kinds are processed independently; MMAv5/TMEM use `InsertTmemAref`.

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
