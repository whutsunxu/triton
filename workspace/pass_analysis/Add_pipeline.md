# `add_pipeline` (`SoftwarePipeliner.cpp`)

| | |
|--|--|
| **Pass** | `tritongpu-pipeline` (`TritonGPUPipeline` / `add_pipeline`) |
| **When** | after `assign_latencies` + `schedule_loops`, and (on Blackwell) after `add_warp_specialize` / AWS |
| **Role** | turn coarse `loop.stage` / `loop.cluster` into a real **software-pipelined** for (overlap method ①) |
| **Code** | `SoftwarePipeliner.cpp` + `LowerLoops.cpp` + `PipelineExpander` |
| **IR** | Walkthrough below starts from `ir/stage_cluster_analysis/stage_cluster.ttir` (post `schedule_loops`, one for, attrs only). Dump after full pass: `ir/add_pipeline/` |
| **Also see** | `Assign_latency_And_Schedule_loop_stage_cluster.md` (how attrs are built); `Latency_Overlap_SWP_vs_WarpSpecialization.md` |

File comment: SWP is usually **schedule** + **expander**. This pass prepares async ops from the schedule, then expands the loop into prologue / steady state / (optional) epilogue.

```text
assign_latencies → schedule_loops     ← attrs only (stage/cluster)
        │
        ▼  [warp_specialize / AWS on Blackwell]
        │
        ▼  add_pipeline                 ← materialize SWP
   lowerLoops → expandLoops → cleanup → wgmma/waits → TMA stores
```

---

## 1. General idea

Before `add_pipeline`, the for body is still **one iteration’s worth of ops**, only *labeled* with stages. Overlap across iterations is not real yet.

`add_pipeline` **materializes** that schedule:

1. Rewrite loads/MMA/TMA into **async** forms + buffers keyed by the schedule (`lowerLoops`).
2. **Expand** the for so stage-k work from iter `i` runs in the same steady-state trip as stage-0 work from iter `i+k` (`expandLoops` / pipeline expander).
3. Fix waits, drop schedule attrs, optional WGMMA async + TMA-store pipelining.

```text
Before (attrs only):
  for i:
    load  {stage=0}     // conceptually “for future iter”
    dot   {stage=2}     // conceptually “for current iter”
    // still sequential in one body

After expand:
  prologue: issue early stages for first iters
  for i' in steady:
    // mix of stages from different logical iters
  epilogue: (optional) drain remaining stages
```

On this matmul after AWS, each WS partition’s for often has a **collapsed 1-stage** schedule (Rest2 §9). Then expansion may do little for TMA↔dot overlap (already paid by warps); the pass still runs for async lowering, waits, and any fors with `num_stages > 1`.

---

## 2. Workflow on `stage_cluster.ttir`

Input for this section: **one** `scf.for %22` with `tt.scheduled_max_stage = 2` and `tt.warp_specialize` (attr only — not yet split into warp groups). Core body schedule:

| Ops (sketch) | `loop.stage` | `loop.cluster` |
|--|--|--|
| tile-index `scf.if`, `%43`/`%44` `descriptor_load` | 0 | 1 / 5 |
| k-counter / some selects | 1 | 0 / 4 |
| `convert_layout`, `tt.dot`, acc select, store `scf.if` | 2 | 2 / 6 |

```mlir
// stage_cluster.ttir — still synchronous descriptor_load → convert → dot
%43 = tt.descriptor_load %arg13[...] {loop.cluster = 5, loop.stage = 0}
%44 = tt.descriptor_load %arg24[...] {loop.cluster = 5, loop.stage = 0}
%45 = ttg.convert_layout %43 {loop.cluster = 2, loop.stage = 2}
%46 = ttg.convert_layout %44 {loop.cluster = 2, loop.stage = 2}
%47 = tt.dot %45, %46, %arg51 {loop.cluster = 2, loop.stage = 2}
// ... store path in scf.if {loop.cluster = 6, loop.stage = 2}
} {tt.scheduled_max_stage = 2, tt.warp_specialize}
```

`runOnOperation` then runs the seven steps below.

### (1) `lowerLoops` — async producers + multibuffers

**Code:** `SoftwarePipeliner.cpp` → `lowerLoops` → `LowerLoops.cpp` `lowerLoop`:

```cpp
schedule.deSerialize(forOp);
newForOp = lowerMMAs(forOp, schedule);
newForOp = lowerLoads(newForOp, schedule, axisInfoAnalysis);
newForOp = lowerTMADescriptors(newForOp, schedule);
schedule.serialize(newForOp);
```

| Substep | On this IR |
|--|--|
| **`lowerMMAs`** | Walks `MMAv5OpInterface` only. Body has classic `tt.dot` (`#nvidia_mma` v2) → **no-op**. |
| **`lowerLoads`** | Pipelines `%43`/`%44`. |
| **`lowerTMADescriptors`** | `%0 = make_tensor_descriptor` is **outside** the for → **no-op**. |

**`lowerLoads` detail for `%43`/`%44`:**

1. `getDefUseStageDiff`: def stage 0, first use (`convert`/`dot`) stage 2 → `stageDiff = 2` (cluster ordering can bump further).
2. TMA loads → `asyncLoads` + `createAlloc(..., distance=stageDiff)` → smem ring `memdesc<2x…>` (or 3 if `loadRequiresAdditionalBuffer`).
3. Add loop-carried `insertIdx` / `extractIdx` / `phase`; `createTMABarrierAndWait`.
4. `createTMAAsyncLoad` → `createTMAAsyncCopy`:

```cpp
// at load's stage/cluster:
AsyncTMACopyGlobalToLocalOp(desc, indices, barrier, view[insertIdx], pred);
// after wait, at first-use stage/cluster:
replaceUsesWithLocalLoad(..., view[extractIdx]);  // feeds convert/dot
loadOp->erase();
```

**Core IR after (1) (conceptual):**

```mlir
%allocA = ttg.local_alloc : () -> !ttg.memdesc<2x64x128xf8E5M2, ...>   // before for
%barA   = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, ...>
%22 = scf.for ... iter_args(..., %ins, %ext, %phase, ...) {
  // stage 0: async fill slot %ins
  ttng.barrier_expect %barA[%ins], ...
  ttng.async_tma_copy_global_to_local %arg13[...] %viewA, %barA[%ins], %true
  // stage 2: wait + local_load slot %ext → same SSA users as old %43
  ttng.wait_barrier %barA[%ext], %phase
  %43' = ttg.local_load %viewA_ext
  %45 = ttg.convert_layout %43' {loop.stage = 2, ...}
  %47 = tt.dot %45, %46, %arg51 {loop.stage = 2, ...}
}
```

Same pattern for B (`%44`). Schedule attrs still on ops; body still **one** logical iteration — not expanded yet.

### (2) `expandLoops` — prologue / steady / epilogue

**Code:** `SoftwarePipeliner.cpp` `expandLoops`:

```cpp
schedule.deSerialize(forOp);
finalSchedule = schedule.createFinalSchedule(forOp);
pipelineForLoop(rewriter, forOp, options);  // PipelineExpander
// optional peelLoopEpilogue for MMAv5+wait-in-last-stage (not this IR)
resolveMaskOp(moduleOp);
```

With `tt.scheduled_max_stage = 2`, the expander treats the schedule as a **modulo-3** pipeline (stages 0..2). Steady-state body mixes:

- stage-0 TMA for logical iter `i+2`
- stage-1 bookkeeping for `i+1`
- stage-2 `local_load`/`dot`/store for iter `i`

**Core IR after (2) (conceptual):**

```mlir
// prologue: peel early stages for first iters (issue TMA without matching dots)
scf.for %i = ... {   // fewer trips; induction advanced by maxStage
  // predicated fragments of stages from different logical iters
  async_tma ...     // "future" tile
  wait + local_load + convert + dot   // "current" tile
  // store if if applicable
}
```

Dynamic bounds → `MaskOp` / `PredicateStageOp` then resolved. This IR is **not** inside `ttg.warp_specialize` yet, so the MMAv5 epilogue-peel heuristic does not apply (`tt.dot` anyway).

### (3) `removePipeliningAttributes`

**Code:** `PipeliningUtility.cpp` `removePipeliningAttributes`.

Strips `loop.stage` / `loop.cluster` / related from the expanded for — schedule has been consumed by the expander.

### (4) `pipelineWgmma`

**Code:** `pipelineWgmma` → `asyncLaunchDots`.

```cpp
if (getNumStagesOrDefault(forOp, numStages) >= 1)
  asyncLaunchDots(forOp);
```

Targets async **WGMMA** (`ttng.warp_group_dot` style). This matmul’s compute is `tt.dot` on `#nvidia_mma` v2 → typically **little/no change**.

### (5) `updateWaits`

**Code:** `WGMMAPipeline.cpp` `updateWaits`.

Recomputes async wait distances (`pendings`) after expansion / any WGMMA launch so waits sit at the right pipeline distance from commits/copies.

### (6) Arith canonicalize

Greedy `arith` dialect patterns — fold adds/selects left messy by peeling and idx/phase updates.

### (7) `pipelineTMAStores`

**Code:** `TMAStoresPipeline.cpp` `pipelineTMAStores`, only if `getNumStagesOrDefault(forOp, numStages) > 1`.

In `stage_cluster.ttir`, `tt.descriptor_store %0[...]` lives in the stage-2 epilogue `scf.if`. This step can rewrite it toward async store + wait (and desc multibuffer if needed), e.g. conceptually:

```mlir
ttng.async_tma_copy_local_to_global %0[...] %smem_view
ttng.async_tma_store_wait {pendings = 0}
```

(Exact shape depends on store pipelining heuristics; see `ir/add_pipeline/after_pipeline.ttir` for a post-AWS+pipeline example.)

---

### Step summary for this IR

| Step | Key code | Effect on `stage_cluster` for |
|--|--|--|
| (1) lowerLoops | `lowerLoads` / `createTMAAsyncLoad` | `%43`/`%44` → async TMA + smem ring + `local_load`; MMA/desc helpers no-op |
| (2) expandLoops | `pipelineForLoop` | Prologue + steady for mixing stages across iters |
| (3) remove attrs | `removePipeliningAttributes` | Drop stage/cluster |
| (4) wgmma | `asyncLaunchDots` | Usually no-op (`tt.dot`) |
| (5) waits | `updateWaits` | Fix async wait distances |
| (6) arith CSE-ish | arith canonicalization | Cleanup |
| (7) TMA stores | `pipelineTMAStores` | Async-ify in-loop `descriptor_store` when `num_stages > 1` |

**One-liner (this IR):** stage 0 loads become buffered async TMAs; expander overlaps those copies with stage-2 `dot` from earlier iters; attrs then disappear.

---

## 3. Relation to warp specialization

| | After AWS | What `add_pipeline` still does |
|--|--|--|
| TMA vs compute overlap | Concurrent partitions + aref ring | Not the main lever anymore if stages collapsed to 1 |
| Per-partition for | May still have multi-stage *within* a role | Expand that residual SWP |
| Async forms / waits / TMA stores | Barriers already from LowerAref | lowerLoads/MMA, `updateWaits`, `pipelineTMAStores` |

**One-liner:** `schedule_loops` decides *when* ops should overlap across iters; `add_pipeline` **rewrites the for** so that schedule actually executes (async ops + prologue/steady/epilogue).

---

## 4. Pointers

| Topic | Doc / path |
|--|--|
| Building stage/cluster | `Assign_latency_And_Schedule_loop_stage_cluster.md` |
| SWP vs WS overview | `Latency_Overlap_SWP_vs_WarpSpecialization.md` |
| AWS then schedule prune | `WarpSpecialization_Rest2.md` §9, §11 |
| Dump recipe | `ir/add_pipeline/dump_pipeline.sh` (post-AWS); walkthrough input: `ir/stage_cluster_analysis/stage_cluster.ttir` |
