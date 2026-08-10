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

## 1.0 Graph primitives

Analysis overlay on the for (not a full-module rewrite).

| type | meaning |
|------|---------|
| **`Node`** | vertex wrapping an `Operation*` or `Value` (iter_arg / result); also parent/child tree for regions |
| **`Port`** | `{Node*, idx}` — result slot (`OutputPort`) or operand slot (`InputPort`) |
| **`Edge`** | `OutputPort → InputPort` (one SSA use); `getFromNode()` / `getToNode()` |
| **`Partition`** | set of nodes + flags (`LOAD`/`STORE`/`VIEW`/…) + cost |

Several graph `Node`s may temporarily share one MLIR `op*` (e.g. after `duplicateViewOps`). Real IR clones happen later (`cloneMultiPartitionDataOps`, `PartitionLoops`).

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

Fills partitions onto **regions** and **non-data** producers so every warp that needs control/index logic owns it.

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
- else `addPartitions(seed.partitions)` (union)

So address math feeding loads picks up the load partition(s); shared trip-count updates (`%52/%53/%54`) can become `{0,1}`.

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
