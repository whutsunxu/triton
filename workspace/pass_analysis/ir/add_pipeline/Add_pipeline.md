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
| **`lowerLoads`** | Pipelines `%43`/`%44` (ground truth: `test/TritonGPU/pipeline-lower-loop.mlir` `@tma_load_lowering`). |
| **`lowerTMADescriptors`** | `%0 = make_tensor_descriptor` is **outside** the for → **no-op**. |

**`lowerLoads` detail for `%43`/`%44`:**

1. `getDefUseStageDiff`: def stage 0, first use (`convert`/`dot`) stage 2 → `stageDiff = 2`.
2. `createAlloc(..., distance=2)` → ring `memdesc<2x…>` + barrier alloc; `init_barrier` each slot.
3. Iter args init `ins=ext=-1`, `phase=0`; each trip:
   - `ins_n = (ins+1) % 2` (stage 0)
   - `ext_n = (ext+1) % 2` (stage 2); on wrap `phase_n = phase xor 1`
4. `createTMABarrierAndWait`: `barrier_expect` on **`bar[ins_n]`**; `wait_barrier` on **`bar[ext_n], phase_n`**.
5. `createTMAAsyncLoad`: TMA into **`buf[ins_n]`**; after wait, `local_load` from **`buf[ext_n]`**.

**Order after lowerLoop (confirmed CHECKs — not wait-before-expect):**  
Expect/copy (stage 0, **insert** slot) appear *above* wait/`local_load` (stage 2, **extract** slot). Wait is **not** before expect in this IR. Insert and extract are different ring indices: you arm/fill a *future* slot and wait a *past* slot. Safety is lag + phase, not “wait then overwrite” in program order.

```mlir
// after lowerLoops — ONE for; attrs remain; matches pipeline-lower-loop CHECKs
%bufA = ttg.local_alloc : () -> !ttg.memdesc<2x64x128xf8E5M2, ...>
%bar  = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, ...>  // A+B may share one wait group
// init_barrier bar[0], bar[1]

scf.for ... iter_args(%ins = -1, %ext = -1, %phase = 0, ...) {
  %ins_n = (%ins + 1) % 2                          // stage 0
  %ext_n = (%ext + 1) % 2                          // stage 2
  %phase_n = wrap(%ext) ? (%phase xor 1) : %phase  // stage 2

  // stage 0 — INSERT: fire TMA for a future consumer
  ttng.barrier_expect %bar[%ins_n], BYTES, %true
  ttng.async_tma_copy_global_to_local %arg13[...] %bufA[%ins_n], %bar[%ins_n], %true
  // (+ B similarly)

  // stage 2 — EXTRACT: wait until past TMA done, then use (old %43 users)
  ttng.wait_barrier %bar[%ext_n], %phase_n
  %43p = ttg.local_load %bufA[%ext_n]
  %45 = ttg.convert_layout %43p
  %47 = tt.dot %45, %46, %arg51

  scf.yield %ins_n, %ext_n, %phase_n, ...
}
```

### (2) `expandLoops` — prologue / steady; future vs current tile

**Code:** `SoftwarePipeliner.cpp` `expandLoops` → `pipelineForLoop`:

```cpp
schedule.deSerialize(forOp);
finalSchedule = schedule.createFinalSchedule(forOp);
pipelineForLoop(rewriter, forOp, options);
resolveMaskOp(moduleOp);
```

`tt.scheduled_max_stage = 2` → stages `{0,1,2}`. Prologue peels early stages (issue TMAs / advance `ins` without matching dots). Steady body mixes stage-2 wait/`dot` for tile `i` with stage-0 expect/TMA for tile `i+2`. Expander placement often puts **wait (current) before expect/TMA (future)** in the steady text — that is post-expand order, not lowerLoop order.

**Concise IR after expand (steady trip):**

```mlir
// prologue: expect+TMA (+ bump ins) peeled so the ring already holds tiles

scf.for %iv' = ... iter_args(%ins, %ext, %phase, %acc, ...) {
  // current tile (stage 2) — data already in buf[ext] from an earlier trip's TMA
  %ext_n = (%ext + 1) % 2
  %phase_n = wrap(%ext) ? (%phase xor 1) : %phase
  ttng.wait_barrier %bar[%ext_n], %phase_n
  %tile = ttg.local_load %buf[%ext_n]
  %acc_n = tt.dot (cvt %tile), ..., %acc

  // future tile (stage 0) — arm+fill INSERT for use ~stageDiff trips later
  %ins_n = (%ins + 1) % 2
  ttng.barrier_expect %bar[%ins_n], BYTES, %pred
  ttng.async_tma_copy_global_to_local %desc[...] %buf[%ins_n], %bar[%ins_n], %pred

  scf.yield %ins_n, %ext_n, %phase_n, %acc_n, ...
}
```

| Carried | Across iterations |
|--|--|
| `%ins` | Next TMA write index; `yield %ins_n` |
| `%ext` | Next wait/`local_load` index; lags `%ins` by pipeline depth |
| `%phase` | Wait parity for `%ext`; XOR when `%ext` wraps |

Epilogue peel is an MMAv5 heuristic and is skipped under `ttg.warp_specialize` / for this `tt.dot`. Dynamic bounds → masks then `resolveMaskOp`.

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
| (1) lowerLoops | `lowerLoads` / `createTMAAsyncLoad` | expect+TMA on `ins`, wait+`local_load` on `ext`; yield `ins/ext/phase` |
| (2) expandLoops | `pipelineForLoop` | Prologue fills ring; steady mixes wait(current)+TMA(future); carry idxs/phase |
| (3) remove attrs | `removePipeliningAttributes` | Drop stage/cluster |
| (4) wgmma | `asyncLaunchDots` | Usually no-op (`tt.dot`) |
| (5) waits | `updateWaits` | Fix async wait distances |
| (6) arith CSE-ish | arith canonicalization | Cleanup |
| (7) TMA stores | `pipelineTMAStores` | Async-ify in-loop `descriptor_store` when `num_stages > 1` |

**One-liner (this IR):** lowerLoop writes expect/TMA(`ins`) then wait/load(`ext`) with carried phase; expand overlaps wait/`dot` of tile `i` with TMA of tile `i+2`.

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
| Dump recipe | §5 below; also `ir/add_pipeline/dump_pipeline.sh` |
| lowerLoop CHECK golden | `test/TritonGPU/pipeline-lower-loop.mlir` `@tma_load_lowering` |

---

## 5. Server `triton-opt` commands (confirm IR)

Set paths to your Linux build and this repo:

```bash
export TRITON_OPT=/path/to/build/.../bin/triton-opt   # Linux binary
export IRDIR=/path/to/triton/workspace/pass_analysis/ir
export SRC="$IRDIR/stage_cluster_analysis/stage_cluster.ttir"
export OUT="$IRDIR/add_pipeline"
mkdir -p "$OUT"
NUM_STAGES=3
```

### (A) Only `lowerLoops` (step 1)

```bash
"$TRITON_OPT" "$SRC" \
  -tritongpu-test-pipeline-lower-loop \
  -o "$OUT/01-after-lower-loops_from_stage_cluster.ttir"
```

Inspect: `barrier_expect` / `async_tma_copy` on insert idx **before** `wait_barrier` / `local_load` on extract idx; `scf.yield` of `ins, ext, phase`.

### (B) Full `add_pipeline` with intermediate dumps (steps 1→2 visible in log)

```bash
"$TRITON_OPT" "$SRC" \
  -tritongpu-pipeline="num-stages=${NUM_STAGES} dump-intermediate-steps=true" \
  -o "$OUT/after_pipeline_from_stage_cluster.ttir" \
  >"$OUT/pipeline_from_stage_cluster.log" 2>&1
```

Log contains:

- `SoftwarePipeliner internal IR Dump After: LowerLoops`
- `SoftwarePipeliner internal IR Dump After: ExpandLoops`

Split dumps (optional):

```bash
csplit -f "$OUT/dump-" -b '%02d.ttir' -z \
  "$OUT/pipeline_from_stage_cluster.log" \
  '/SoftwarePipeliner internal IR Dump/' '{*}'
```

### (C) Full pipeline without dumps

```bash
"$TRITON_OPT" "$SRC" \
  -tritongpu-pipeline="num-stages=${NUM_STAGES}" \
  -o "$OUT/after_pipeline_from_stage_cluster.ttir"
```

### (D) Post-AWS pipeline (compiler order on Blackwell)

```bash
"$TRITON_OPT" \
  "$IRDIR/automatic-warp-specialization/after_automatic_warp_specialization.ttir" \
  -tritongpu-pipeline="num-stages=${NUM_STAGES} dump-intermediate-steps=true" \
  -o "$OUT/after_pipeline.ttir" \
  >"$OUT/pipeline_debug.log" 2>&1
```

Or run `ir/add_pipeline/dump_pipeline.sh` after fixing `TRITON_OPT` inside the script.
