# Warp specialization (matmul pass analysis) — Rest of AWS

Companion notes for `_p_matmul_…_64x256x128x1`.

| | |
|--|--|
| **Pipeline** | `AutomaticWarpSpecialization` (`AutomaticWarpSpecialization.cpp`) |
| **IR spine** | `input_ir.ttir` → … → `stage_cluster.ttir` → `after_partition_scheduling.ttir` → … |
| **This doc** | AWS parts **§2–§11** (after PartitionScheduling) |
| **Sibling** | `WarpSpecialization_PartitionScheduling.md` — §1 PartitionScheduling |

**AWS order (high level):**

```text
1 PartitionScheduling          ← see sibling doc
2 NVWSHoistTmemStore           ← this file starts here
3 NVWSInsertAref / InsertTmemAref
4 SCCP
5 CSE
6 NVWSLowerAref
7 PartitionLoops          ← physical per-partition scf.for clones
8 NVWSLowerWarpGroup
9 ScheduleLoops
10 multiBufferTMADescriptors
```

---

# 6. NVWSLowerAref (`LowerAref.cpp`)

| | |
|--|--|
| **Pass** | `nvws-lower-aref` (`NVWSLowerAref`) |
| **When** | after SCCP + CSE in AWS |
| **Role** | turn high-level aref put/get into multibuffers + barriers; first may **combine** arefs that share a consumer so arrive/wait coalesce |

`runOnOperation` roughly:

```text
1. collect WS fors (+ nested fors) → combineArefs(loop)   ← §6.1–6.2
2. multiBufferAref (producer-load arefs)                  ← §6.3
3. NVWSAssignStagePhase                                   ← §6.4
4. pattern-rewrite LowerArefCreate → mbarriers / waits / …  ← §6.5
```

## 6.1 General idea — same dominant consumer **and** same partition

InsertAref often creates **one aref per cross-partition value** (e.g. A-tile and B-tile each get their own put/get). That means **separate** empty/full barriers later.

`combineArefs` is an optimization: if several **get-enters** in the **same loop body**

1. share the same **dominant buffer consumer** (`liveBeforeOp`), and
2. sit in the **same get partition**, and
3. their **puts** also share one producer partition,

then merge those arefs into **one multi-buffer aref** and replace the old put/get enter/exit sequences with **combined** ones — so barrier arrive/wait can be coalesced (especially for TMA loads).

## 6.2 Workflow — group by dominant consumer, then combine (`combineArefs`)

```cpp
void combineArefs(scf::ForOp loop) {
  auto getEnterOps = loop.getOps<ArefGetEnterOp>();  // this body only
  DominanceInfo domInfo(loop);
  DenseMap<pair<Operation*, int>, SmallVector<ArefGetEnterOp>> liveBeforeGroups;

  for (auto getEnterOp : getEnterOps) {
    if (auto liveBeforeOp = getDominantConsumer(...))
      liveBeforeGroups[{liveBeforeOp, partitionIds.front()}].push_back(getEnterOp);
  }

  for (auto getEnterOps : make_second_range(liveBeforeGroups)) {
    if (getEnterOps.size() == 1) continue;
    // same put partition? → new aref.create + combined put/get; erase olds
  }
}
```

Flow: collect gets → group by `(liveBeforeOp, partitionId)` → skip size-1 buckets → combine (≥2) into one aref + coalesced put/get → erase olds.

`liveBeforeOp` from `getDominantConsumer`:

```cpp
buf = getEnter.getResult(0);              // buffer, not token
sinkOps = findSharedMemorySinkOps(buf);   // LocalLoad / MMAv5 / views
liveBeforeOp = findNearestCommonDominator(sinkOps, domInfo);
```

WS collection: outer walk finds `tt.warp_specialize` fors; inner `loop->walk` collects that for **and** nested fors; `combineArefs` on each.

### IR — **no** combine (typical two `local_load`s before `dot`)

```mlir
scf.for %i = ... {
  // partition 1 puts into %aref_a / %aref_b (omitted)

  %buf_a, %tok_a = nvws.aref_get_enter %aref_a {ttg.partition = array<i32: 0>}
  %ld_a = ttg.local_load %buf_a {ttg.partition = array<i32: 0>}

  %buf_b, %tok_b = nvws.aref_get_enter %aref_b {ttg.partition = array<i32: 0>}
  %ld_b = ttg.local_load %buf_b {ttg.partition = array<i32: 0>}

  %cvt_a = ttg.convert_layout %ld_a {ttg.partition = array<i32: 0>}
  %cvt_b = ttg.convert_layout %ld_b {ttg.partition = array<i32: 0>}
  %acc = tt.dot %cvt_a, %cvt_b, %acc0 {ttg.partition = array<i32: 0>}
  // get exits ...
}
```

| get | `liveBeforeOp` | key |
|-----|----------------|-----|
| get A | `%ld_a` | `(%ld_a, 0)` |
| get B | `%ld_b` | `(%ld_b, 0)` |

→ two size-1 groups → no combine. Same `dot` does not merge them (`tt.dot` is not a listed sink).

### IR — **yes** combine (same sink / same NCD)

```mlir
%buf_a, %tok_a = nvws.aref_get_enter %aref_a {ttg.partition = array<i32: 0>}
%buf_b, %tok_b = nvws.aref_get_enter %aref_b {ttg.partition = array<i32: 0>}
%sink = ttng.tc_gen5_mma %view_a, %view_b, ... {ttg.partition = array<i32: 0>}
```

```text
liveBeforeGroups[(%sink, 0)] = [getA, getB]   // size 2 → combine
```

Puts of those arefs must also share one producer partition.

### After replace (sketch)

```mlir
%aref = nvws.aref.create %buf_a_alloc, %buf_b_alloc

%tok_p, %w0, %w1 = nvws.aref_put_enter %aref ...   // partition 1
// TMA/fills into %w0 / %w1 ...
nvws.aref_put_exit %aref, %tok_p ...

%tok_g, %r0, %r1 = nvws.aref_get_enter %aref ...   // partition 0
// consumers use %r0 / %r1
nvws.aref_get_exit %aref, %tok_g ...
```

Old per-aref create / put enter-exit / get enter-exit are erased.

**One-liner:** same `(dominant buffer consumer, get partition)` + same put partition → merge arefs and coalesced put/get; separate `local_load` sinks before `tt.dot` usually stay uncombined.

## 6.3 `multiBufferAref` — deepen smem aref buffers to `num_stages`

After combine, LowerAref walks **producer-load** arefs (put side feeds `nvws.descriptor_load` / TMA) and expands each smem operand from InsertAref’s **depth-1** buffer to a **circular buffer of depth `numStages`** (pass option `num-stages`, default **3**; AWS forwards the user’s `num_stages`).

Skipped:

- arefs that are **not** producer-load (`isProducerLoad` false), and
- arefs whose operands lack a defining op or are **`ttng.tmem_alloc`** (`eligible = false` inside `multiBufferAref`).

Per eligible aref operand:

```cpp
arefBufType = getMultiBufferedType(getBufferViewType(arefBufType, true), numStages);
// e.g. <1x128x64> → drop leading 1 → <128x64> → prepend numStages → <3x128x64, mutable>
newAlloc = createAlloc(..., arefBufType, ...);
oldAlloc->replaceAllUsesWith(newAlloc);
// then new aref.create on the new allocs; erase old aref + old allocs
```

Put/get enter–exit stay as aref ops for now; later `LowerArefCreate` / stage-phase indexing picks `memdesc_index` into the leading dim.

### IR — before (`num_stages = 3`)

From InsertAref: one slot per aref (leading dim `1`). Sketch from `test/NVWS/lower_aref.mlir` (`warp_specialize_tma_matmul`):

```mlir
%buf_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
%aref_a = nvws.aref.create %buf_a : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
%buf_b = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
%aref_b = nvws.aref.create %buf_b : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>

scf.for ... {
  %wa, %tok_pa = nvws.aref.put.enter %aref_a ...
  nvws.descriptor_load ... %wa   // producer-load → isProducerLoad
  nvws.aref.put.exit %aref_a, %tok_pa [#nvws.async_op<tma_load>] ...
  // … same for B; gets / MMA …
}
```

### IR — after `multiBufferAref` (type change only at this step)

```mlir
%buf_a = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
%aref_a = nvws.aref.create %buf_a : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
%buf_b = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
%aref_b = nvws.aref.create %buf_b : <[!ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>]>
// put/get still reference %aref_*; stage index into the leading 3 comes in later lowering
```

TMEM arefs keep their allocs unchanged here. Full barrier / `memdesc_index` materialization is steps 3–4 (`AssignStagePhase` + `LowerArefCreate`).

**One-liner:** for TMA producer-load smem arefs, replace depth-1 allocs with depth-`numStages` multibuffers and rewire `aref.create`.

## 6.4 `NVWSAssignStagePhase` — fill `[stage, phase]` on put/get

Nested inside `NVWSLowerAref::runOnOperation` (`createNVWSAssignStagePhase()`), after combine + multiBuffer. Standalone pass `nvws-assign-stage-phase`; AWS never lists it on its own.

### 6.4.1, Mechanism — ring of slots, two barriers each

This pass set up a ring of buffer, the ring number is `depth`, besides the each buffer, it also set the empty/full flags to mark if the buffer is ready to read or write.

During the scf.for loop, it(both the producer and consumer) use the loop_idx to round-rubin the ring buffer, usually starting with buffer_0.

Before using the buffer(write the data into the buffer, or read the data from the buffer), it will compare the phase in the empty/full flags with its local loop-carried flag, if they match, then it can use the buffer, otherwise it spins until the buffer is ready.

Because there are producer and consumer, so need to carefully process the start values to avoid the deadlock.

Also because the buffer and its empty/full flags are accessed by round-robin, it need to properly process the buffer's empty/full flags and the loop-carried flag. Below are the details of the general idea.

### 6.4.2, Primitives — key components,

A multibuffered aref is a ring of `depth` slots (`depth = 3` here), two barriers each slot. Producer (put) and consumer (get) walk it independently so TMA can run ahead of `dot`.

```text
slot 0:  buf[0]   empty[0]   full[0]     ← three hardware objects
slot 1:  buf[1]   empty[1]   full[1]
slot 2:  buf[2]   empty[2]   full[2]
```

**Reused** means the **same warp** coming back around the ring: iter 0 and iter 3 both use `empty[0]` / `full[0]`. `stage` picks the slot; `phase` tells “first use of slot 0” from “second use of slot 0”.

| | **stage** | **phase** |
|--|-----------|-----------|
| **What** | ring index `0 .. depth-1` | 0/1 **parity** of that slot’s mbarrier |
| **Used for** | `memdesc_index buf[stage]`, `empty[stage]`, `full[stage]` | `wait_barrier bar, phase` |
| **Advances** | `(stage + 1) % depth` | flip **only when stage wraps** to 0 |

Two mbarriers per slot because the two events differ:

| barrier | who arrives (signal to update) | who waits | means |
|--|--|--|--|
| **full[stage]** | producer (TMA HW) | consumer | **write** of this slot finished |
| **empty[stage]** | consumer | producer | **read** of this slot finished; free to overwrite |

### 6.4.3, Workflow

#### initial status

All the `empty` and `full` barrier phases are set to zero.

```text
empty[0], empty[1], empty[2]   → hardware phase = 0
full[0],  full[1],  full[2]    → hardware phase = 0
```

That phase bit lives in the mbarrier object, **not** in SSA. It **toggles** when arrivals hit the expected count:

- **empty[i]** flips when the **consumer** arrives (get.exit)
- **full[i]** flips when the **producer** arrives (TMA / put.exit)

#### Software cursors — start values & rules

Put and get each have **one** walking pair `(stage, phase)`, loop-carried. Not one pair per slot.
Start values setting:

| cursor | init `stage` | init `phase` | after first wrap (first enter) |
|--|--|--|--|
| **put** | `depth-1` (=2) | **0** | `[0, 1]` |
| **get** | `depth-1` (=2) | **1** | `[0, 0]` |

the below codes show how the `(stage, phase)` are wrapped:
   ```text
   next = stage + 1
   wrap = (next == depth)
   stage = wrap ? 0 : next
   phase = wrap ? (phase xor 1) : phase
   enter %aref[stage, phase]
   ```

Rules:

which buffer to use?
```text
According to the value of `stage`, producer/consumer decide which buffer slot to use and which barrier(empty/full) slot to check.
```

How to check?
```text
Producer(TMA) check the empty.phase with the carried phase(the legal values can only be 0 or 1).
If they are same, keep the producer spin, otherwise just let the producer go ahead to do the task(usually to write the data into the buffer slot).
After finish the task, it toggles the full.phase to update the status.
then it can advance into next iteration.
```
Similarly,
```text
Consumer check the full.phase with the carried phase.
If they are same, keep the consumer spin, otherwise just let the consumer go ahead to do the task(usually to read the data into the buffer slot).
After finish the task, it toggles the empty.phase to update the status.
```

#### Loop example (`depth = 3`)

All six barriers start at hardware **0**. Cursors enter the loop as put `(2, 0)`, get `(2, 1)`.

**Iter 0 — first use of slot 0**

- Put initial values `(2, 0)`, wraps to `[0, 1]`: `wait(empty[0], 1)` → 0≠1 → proceed. TMA writes `buf[0]`, arrives on `full[0]` → flip **full[0]: 0→1**.
- Get initial values `(2, 1)`, wraps to `[0, 0]`: `wait(full[0], 0)` → spin until that is flipped, arrive on `empty[0]` → flip **empty[0]: 0→1**.

**Iters 1–2** — same on slots 1 and 2. Put SSA phase stays 1; get SSA phase stays 0. No wrap, no XOR.

**Iter 3 — reuse slot 0** (same `empty[0]` / `full[0]`)

- Put wraps to `[0, 0]`: `wait(empty[0], 0)` → hardware is 1 from the first read → wait until consumer arrives again (**empty[0]: 1→0**). TMA overwrite, arrive **full[0]: 1→0**.
- Get wraps to `[0, 1]`: `wait(full[0], 1)` → wait until that second write flips full off 1.

In Iter 0, if both cursors started at phase 0, the first put would `wait(empty[0], 0)` and spin forever: empty never completes until a consumer who cannot start.

During the first 3 iterations, because producer and consumer keep **separate** cursors so TMA can be several iterations ahead.

#### IR — before

From `02-after-insert-tmem-aref.ttir` (depth-1; multiBuffer has already turned these into `3x…` by the time AssignStagePhase runs):

```mlir
%buf, %tok = nvws.aref.put.enter %aref   // no [stage, phase]
nvws.aref.put.exit %aref, %tok [#nvws.async_op<tma_load>]
%r, %tg = nvws.aref.get.enter %aref
nvws.aref.get.exit %aref, %tg [#nvws.async_op<none>]
```

#### IR — after (`depth = 3`)

Sketch from `test/NVWS/assign_stage_phase.mlir`:

```mlir
%aref = nvws.aref.create %alloc : <[!ttg.memdesc<3x…>]>
// inits: stage=2, put phase=0, get phase=1
scf.for %i = ... iter_args(%s_put = %c2, %p_put = %c0,
                           %s_get = %c2, %p_get = %c1) {
  %sp1 = arith.addi %s_put, %c1
  %wp  = arith.cmpi eq, %sp1, %c3
  %sp  = arith.select %wp, %c0, %sp1
  %pp  = arith.select %wp, xor(%p_put, 1), %p_put
  %b, %tok = nvws.aref.put.enter %aref[%sp, %pp]
  nvws.aref.put.exit %aref[%sp], %tok [...]   // same stage, no phase

  // get: same wrap on its own iter_args → [%sg, %pg]
  %r, %tg = nvws.aref.get.enter %aref[%sg, %pg]
  nvws.aref.get.exit %aref[%sg], %tg [...]
  scf.yield %sp, %pp, %sg, %pg
}
```

This matmul (`03-after-lower-aref.ttir`) still shows that wrap after `LowerArefCreate`, now feeding `wait_barrier`. A and B were **not** combined (separate `local_load`s before `tt.dot`), so **four** cursors: put A `(2,0)`, get A `(2,1)`, put B `(2,0)`, get B `(2,1)`.

**One-liner:** `stage` is the circular-buffer index; SSA `phase` is the wait parity (put↔empty, get↔full); hardware empty/full all start at 0; first put P=1 is only so the first wait on an unused empty does not deadlock.

## 6.5 `LowerArefCreate` — enter/exit → wait / expect / TMA / arrive

Nested inside `NVWSLowerAref` **after** AssignStagePhase: greedy `OpRewritePattern<ArefCreateOp>`. AssignStagePhase only **fills** `[stage, phase]`; this pattern **erases** `aref.create` / put / get and materializes mbarriers.

### 6.5.1 Mechanism

Per `aref.create`, allocate two mbarrier arrays of length `depth` (**empty**, **full**), `init_barrier` every slot (HW phase = 0). Then rewrite each user using the stage/phase already on the op:

```text
put.enter [%s,%p]  →  wait empty[s], p ; memdesc_index buf[s]
put  + TMA         →  barrier_expect full[s], txBytes ; async_tma_copy …, full[s]
put.exit  [%s]     →  TMA HW arrive on full (no arrive_barrier for tma_load)
get.enter [%s,%p]  →  wait full[s], p ; memdesc_index buf[s]
get.exit  [%s]     →  arrive empty[s]  (+ fence if producer was TMA and consumer is NONE)
```

`descriptor_load` is on the **put** side. Combined arefs still share **one** empty/full pair; `getSubViews` indexes **all** buffers with the same `%s`.

`barrier_expect` is **not** a toggle. It is `mbarrier.arrive.expect_tx`: arm full with pending TMA bytes. The full parity flips when those bytes land. Expect **before** the copy is required; wait-empty **before** expect is what makes overwrite safe.

### 6.5.2 Primitives

| before (after §6.4) | after this pattern | HW effect |
|--|--|--|
| `put.enter %aref[%s,%p]` | `wait_barrier empty[%s], %p` | observe **empty** |
| `descriptor_load … %buf` | `memdesc_index buf[%s]` + `barrier_expect full[%s], N` + `async_tma_copy …, full[%s]` | arm then complete **full** |
| `put.exit %aref[%s], %tok [#tma_load]` | (nothing extra; TMA already arrives) | **full** toggles on copy done |
| `get.enter %aref[%s,%p]` | `wait_barrier full[%s], %p` | observe **full** |
| `local_load` / `dot` | same, on the indexed view | — |
| `get.exit %aref[%s], %tok [#none]` | `fence_async_shared` + `arrive_barrier empty[%s], 1` | **empty** toggles |

Software `%s` / `%p` stay SSA (`addi` / `xori` / `select`). Empty/full **parity** is not SSA — it lives in the `memdesc<1xi64>` object.

### 6.5.3 Workflow (`matchAndRewrite`)

```cpp
arefVal = createAndInitMbar(aref.create);   // empty[depth], full[depth], init each
for (user : aref.create.users()) {
  put.enter  → rewritePutEnterOp   // wait empty; TMA expect+copy on full
  get.enter  → rewriteGetEnterOp   // wait full; replace buffers with views
  put.exit   → rewritePutExitOp    // arrive full if not TMA
  get.exit   → rewriteGetExitOp    // arrive empty
  aref.buffer → memdesc_index only
}
// erase create + enter/exit; leftover tokens → ub.poison (hoisted later)
```

### IR — before (after AssignStagePhase)

```mlir
%b, %tok = nvws.aref.put.enter %aref[%sp, %pp] {ttg.partition = array<i32: 1>}
nvws.descriptor_load ... %b ... 8192
nvws.aref.put.exit %aref[%sp], %tok [#nvws.async_op<tma_load>] ...

%r, %tg = nvws.aref.get.enter %aref[%sg, %pg] {ttg.partition = array<i32: 0>}
%ld = ttg.local_load %r ...
nvws.aref.get.exit %aref[%sg], %tg [#nvws.async_op<none>] ...
```

### IR — after (`03-after-lower-aref.ttir`, A-tile put then get)

Put (partition 1) — wait **empty** `%21`, expect+TMA on **full** `%25`:

```mlir
ttng.wait_barrier %76, %75                          // empty[%73], put phase
%77 = ttg.memdesc_index %20[%73]                    // bufA[stage]
%78 = ttg.memdesc_index %25[%73]                    // full[stage]
ttng.barrier_expect %78, 8192, %true                // arm full (not a toggle)
ttng.async_tma_copy_global_to_local ... %77, %78    // HW arrive on full
```

Get (partition 0) — wait **full**, load, arrive **empty**:

```mlir
ttng.wait_barrier %92, %91                          // full[%89], get phase
%93 = ttg.memdesc_index %20[%89]
%ld = ttg.local_load %93
ttng.fence_async_shared {bCluster = false}
ttng.arrive_barrier %95, 1                          // empty[%89]
```

B-tile is the same pattern on its own empty/full (`%30` / `%34`) and wrap vs `%c3`.

**One-liner:** `LowerArefCreate` turns annotated put/get into wait-empty + expect/TMA-full (producer) and wait-full + arrive-empty (consumer); `barrier_expect` arms full, TMA toggles it.

---

# 7. PartitionLoops (`PartitionLoops.cpp`)

| | |
|--|--|
| **Pass** | `triton-gpu-partition-loops` (`TritonGPUPartitionLoops`) |
| **When** | after `NVWSLowerAref` in AWS (`createTritonGPUPartitionLoops`) |
| **Role** | turn one tagged `scf.for` into `nvws.warp_group` with **one physical `scf.for` per partition region** |
| **IR** | `03-after-lower-aref.ttir` → `04-after-partition-loops.ttir` |

Until here the WS loop is still **one** region. Ops are only *labeled* `ttg.partition = [0]`, `[1]`, or `[0,1]`. After this pass they live in **separate warp-group regions** and no longer share SSA.

## 7.1 Mechanism — clone the for, do not share registers across warps

Aref already replaced cross-partition **data** edges (TMA → `local_load`) with smem + mbarriers. What remains in one `scf.for` is:

- p1 ops: wait-empty, TMA, expect/full
- p0 ops: wait-full, `local_load`, `convert_layout`, `dot`, store
- both: index arith tagged `[0,1]`

To better assign the partitions onto warpgroups, We split the for-loop's SSAs into warp-specialized for-loop. In such situation, existing SSA from p1 to p0 would dangle. So `partitionLoop`:

1. **Refuse** leftover register edges between distinct partitions (`iterateUses`).
2. **Classify** the original iter_args from partition **0** (the only region that can `warp_group.yield` SSA).
3. **Clone** one `scf.for` into each `nvws.warp_group` region, keeping only that partition’s iter_args / ops.
4. **Rewire** uses of the old for that sit **after** the loop (no `ttg.partition` = “root”).

`nvws.warp_group` results **are** partition 0’s yielded values. Other regions only `warp_group.return`. A tensor computed only in p1 that is still used after the loop cannot come out as SSA — it would go through a smem `local_alloc` / store / load. This matmul does not need that path.

## 7.2 Workflow (`partitionLoop`)

`PartitionSet::fromLoop` reads `ttg.partition.stages` and each op’s `ttg.partition`. Then the four jobs from §7.1:

### (1) Refuse leftover register edges

For each partition, `iterateUses(loop, callback)` walks that partition’s SSA **outputs**: uses in another partition this trip, or uses in a **future** trip via `scf.yield` → iter_arg.

The callback is `(output, use, distance)`:

- `output` — the `OpResult` this partition produced
- `use` — the consuming operand (`use.get()` **is** `output` when `distance == 0`; after a yield it is the iter_arg)
- `distance` — how many future trips (`0` = same iter)

Skip if the user is also tagged with this partition (including `A(i) → A(i+1)`). Skip if every consumer partition id is already on the producer (`consumer ⊆ producer`): after clone, that region still has a copy of the def. Anything else is a dangling register edge — warn `"non-root partition #P has direct SSA consumer"` and fail.

That is the pre-aref `descriptor_load {p1}` → `dot {p0}` edge. After LowerAref, p1 and p0 only share smem/barriers, so this check passes.

### (2) Classify iter_args from partition 0

`nvws.warp_group` SSA results come only from p0’s `warp_group.yield`. So classification is vs **partition 0**, using `ttg.partition.outputs` on the **for** (not on post-loop ops like `%51`):

| category | when | what happens |
|--|--|--|
| `Used` | `0 ∈ outputs[i]` | keep this iter_arg on p0’s cloned for; it can be yielded out of `warp_group` |
| `TensorResultFromOtherPartition` | `0 ∉ outputs[i]`, yielded value is a **tensor** computed by another partition, **and** `%38#i` still has uses after the for | `local_alloc` a smem buffer in front of `warp_group` |
| `Unused` | otherwise | drop from p0 (p1-only i32s; tensors nobody uses after the loop) |

`getLoopVarIndicesToKeep` keeps only `Used`. Those types become `WarpGroupOp` result types. The smem alloc is empty on this matmul (see IR below).

### (3) Clone one `scf.for` per warp-group region

Create `nvws.warp_group` with `numPartitions` regions (`num_warps` each from the module). Walk the **containing block** (func body): skip unlabeled ops (allocs, `init_barrier`, post-loop bitcast); for the tagged for, `cloneForOp`.

`cloneForOp` builds one `scf.for` in each region whose id is on the original for (`[0, 1]` here). Each clone keeps only iter_args whose `partition.outputs` contain **that** id, then `cloneOpsInBlock`:

- `ttg.partition = [0]` → only p0 (wait-full, `local_load`, `dot`, store)
- `[1]` → only p1 (wait-empty, TMA)
- `[0,1]` → **both** (e.g. k-counter `%arg46`)

Close each region:

- **p0:** `warp_group.yield` the cloned for’s results. Those **are** `%38#k` after the op.
- **p1+:** if this partition computed a `TensorResultFromOtherPartition` tensor, `local_store` it into the buffer from (2); then `warp_group.return` (no SSA). This matmul: no store, just `return`. p1’s TMA cursors die in-region; A/B tiles already moved through barriers inside the body.

### (4) Rewire uses of the old for (root)

The original `%38:17` still exists. For each result that still has users:

- `TensorResultFromOtherPartition` → `local_load` from the buffer in (2), replace uses, `local_dealloc`. Not taken here.
- else if some user has **no** `ttg.partition` (code after the WS loop = “root”) or is another WS for → replace with `wgOp.getResult(newResultIndices[i])`.

Then erase the original for. `%51 = tt.bitcast %38#3` is root: index 3 maps to warp-group result 3, so the bitcast keeps using `%38#3` even though `%38` is now the warp-group.

### IR — before (`03-after-lower-aref.ttir`)

One for, 17 iter_args, still `ttg.partition` on every body op:

```mlir
%38:17 = scf.for %arg45 = %c0 to %ub step %c1
    iter_args(%arg46, %arg47, %arg48, %arg49, %arg50, %arg51, ... %arg62)
    -> (i32, i32, i32, tensor<128xi32>, i32, tensor<64x256xf32, #mma>, i32, ...)
{
  // p1: wait empty, TMA A/B
  // p0: wait full, local_load, cvt, tt.dot %arg51, store
  scf.yield %114, %69#3, %111#0, %111#1, %108, %110, ...
} { ttg.partition = array<i32: 0, 1>,
    ttg.partition.outputs = [
      [0,1], [0,1], [0], [0], [1], [0], [0,1], [0,1],
      [1], [1], [1], [0], [0], [1], [1], [0], [0]
    ] }

%51 = tt.bitcast %38#3 : tensor<128xi32, ...> -> tensor<128xf32, ...>   // root, no partition
```

Index **3** (`%arg49` / `%38#3`, amax tensor) is `outputs = [0]` — computed in p0, used after the loop. Index **5** (`%arg51` acc) is also `[0]` but **`%38#5` has no uses** after the for. p1-only slots are i32 (k-index, put stage/phase) → `Unused` for p0, not smem-export.

p0 keeps indices `0,1,2,3,5,6,7,11,12,15,16` (11). p1 keeps the 10 whose outputs contain `1`.

### IR — after (`04-after-partition-loops.ttir`)

```mlir
%38:11 = nvws.warp_group
partition0 num_warps(4) {
  %67:11 = scf.for ... iter_args(/* 11: counters, amax, acc, get stage/phase */)
  {
    // wait full, local_load, cvt, tt.dot, store  — no TMA
    scf.yield ...
  }
  nvws.warp_group.yield %67#0, ... %67#3, %67#4, ...   // #3 = amax tensor
}
partition1 num_warps(4) {
  %67:10 = scf.for ... iter_args(/* 10 i32s: counters, TMA coords, put stage/phase */)
  {
    // wait empty, expect, async_tma_copy A/B  — no dot
    scf.yield ...
  }
  nvws.warp_group.return
} : i32, i32, i32, tensor<128xi32, ...>, tensor<64x256xf32, #mma>, ...

%51 = tt.bitcast %38#3 : tensor<128xi32, ...> -> tensor<128xf32, ...>
```

`%38` is now the warp-group, not the for. `%51` still uses `%38#3` because `newResultIndices[3] = 3` (p0 kept the amax as its 4th result). No extra `local_alloc` in front of `warp_group`.

**One-liner:** `PartitionLoops` splits the tagged for into per-warp-group clones; p0 `yield`s SSA (here the amax `%38#3`); p1 `return`s; leftover cross-partition SSA is illegal because warp groups do not share registers.

---

# 8. NVWSLowerWarpGroup (`LowerWarpGroup.cpp`)

| | |
|--|--|
| **Pass** | `nvws-lower-warp-group` (`NVWSLowerWarpGroup`) |
| **When** | after `PartitionLoops` in AWS (`createNVWSLowerWarpGroup`) |
| **Role** | rewrite `nvws.warp_group` → `ttg.warp_specialize` (analysis container → executable form) |
| **IR** | `04-after-partition-loops.ttir` → `05-after-lower-warp-group.ttir` |

`partition0` / `partition1` on `nvws.warp_group` are **regions**, not SSA. Only the op results (`%38`) and in-region values (e.g. `%67`) are SSA. This pass keeps that result contract while switching dialect.

## 8.1 Mechanism — default vs specialized partitions

`ttg.warp_specialize` has a **default** region (may `ttg.warp_yield` SSA results) plus zero or more specialized partitions (only `ttg.warp_return`). Captures used by specialized partitions must be explicit operands or rematerialized inside the region; default may still use outer SSA freely.

So `matchAndRewrite(WarpGroupOp)`:

1. **Promote** region 0 to `default` when `numWarps[0] == module ttg.num-warps` (both 4 here). Remaining regions become specialized partitions (old p1 → new `partition0`).
2. **Capture** values defined above specialized regions (`getUsedValuesDefinedAbove`).
3. **Build** `ttg.warp_specialize`, move bodies, rename terminators, replace `%38` uses, erase `nvws.warp_group`.

If the op has results but region 0 is not the default-warp count, the pass fails — results must come from default.

## 8.2 Workflow (`createWarpSpecializeOp`)

### (1) Default region

Move old partition0’s ops into `default`; `nvws.warp_group.yield` → `ttg.warp_yield`. Same 11 results become `%38` of the warp_specialize. Outer uses like `%51 = tt.bitcast %38#3` are unchanged.

### (2) Capture handling for specialized partitions

For each capture:

| kind | action |
|--|--|
| pure constant / pure tensor op | clone into each specialized region (and chase operands) |
| other `RankedTensorType` | `local_alloc` outside + `local_load` per region; pass the memdesc as an operand |
| else (i32, i1, memdesc, tensordesc, …) | pass as `ttg.warp_specialize` operand → partition block arg |

This matmul’s TMA partition mostly takes the “else” path (scalars + memdescs + descriptors); constants like `%c0` are rematerialized inside.

### (3) Move specialized bodies

Create `WarpSpecializePartitionsOp` with one region per remaining warp-group region. `populateRegion` adds block args for the capture list, moves the for-loop body, ends with `ttg.warp_return` (was `nvws.warp_group.return`).

### IR — before (`04-after-partition-loops.ttir`)

```mlir
%38:11 = nvws.warp_group
partition0 num_warps(4) {          // compute; uses outer %5, %20, … freely
  %67:11 = scf.for ...
  nvws.warp_group.yield %67#0, ... %67#3, ...
}
partition1 num_warps(4) {          // TMA; still refers to outer SSA
  %67:10 = scf.for ...
  nvws.warp_group.return
}
%51 = tt.bitcast %38#3 ...
```

### IR — after (`05-after-lower-warp-group.ttir`)

```mlir
%38:11 = ttg.warp_specialize(%5, %6, %arg42, %10, %21, %20, %25, ...)
default {                          // was partition0
  %67:11 = scf.for ...             // still uses outer SSA
  ttg.warp_yield %67#0, ... %67#3, ...
}
partition0(%arg45: i32, ... %arg59: i32) num_warps(4) {  // was partition1
  %67:10 = scf.for ...             // uses %arg* for captures
  ttg.warp_return
} : (...) -> (i32, i32, i32, tensor<128xi32, ...>, ...)
%51 = tt.bitcast %38#3 ...
```

Root `%51` still reads `%38#3`; the hop is now `scf.for → warp_yield → %38`, same as before but on `ttg.warp_specialize`.

**One-liner:** `NVWSLowerWarpGroup` maps `nvws.warp_group` → `ttg.warp_specialize`: compute becomes `default` (yields SSA), TMA becomes a specialized partition with captures as operands / rematerialized constants.

---

# 9. ScheduleLoops (`ScheduleLoops.cpp`)

| | |
|--|--|
| **Pass** | `tritongpu-schedule-loops` (`TritonGPUScheduleLoops`) |
| **When** | after `NVWSLowerWarpGroup` in AWS (`createTritonGPUScheduleLoops`) |
| **Role** | rewrite stale `loop.stage` / `loop.cluster` on each **partitioned** `scf.for` |
| **IR** | `05-after-lower-warp-group.ttir` → `06-after-schedule-loops.ttir` |
| **Also see** | `description_stage_cluster.md` for the full non-WS coarse SWP recipe |

Ops and control flow are unchanged; only schedule attrs are rewritten. Overlap of TMA vs compute is already **warp concurrency**; each for no longer needs the old multi-stage SWP numbering from the combined loop.

## 9.1 Mechanism — prune stages after partition

Before PartitionLoops, one body used stages/clusters to software-pipeline TMA ahead of dot (e.g. TMA `cluster=5, stage=0`; load/dot `cluster=2, stage=2`). After the split, default only has compute latency ops and partition0 only has TMA latency ops — each for holds a **partial** view of the old schedule.

`getInitialSchedule` special-cases `tt.warp_specialize` fors:

1. **Deserialize** existing `loop.stage` / `loop.cluster` into a `CoarseSchedule`.
2. Collect **latency ops** still in this for (`wait_barrier`, `async_tma_copy`, `local_load`, …).
3. If they all sit in **≤1 stage**, build a **1-stage** schedule and put every body op in one cluster.
4. Else shrink/keep a multi-stage schedule (not this matmul).

Then the usual prologue/deps/dist-1/remaining path runs and `serialize` stamps attrs.

## 9.2 What changes on this matmul

| Region | 05 (stale half of old SWP) | 06 (after) |
|--|--|--|
| default wait / `local_load` / `dot` | `cluster=2, stage=2` | `cluster=1, stage=0` |
| partition0 wait / expect / TMA | `cluster=5, stage=0` | `cluster=1, stage=0` |
| leftover index / epilogue `scf.if` | mixed `0/4/5/6` | mostly `cluster=1, stage=0` |

Both fors still carry `tt.scheduled_max_stage = 2` on the for op from earlier serialization metadata; body ops are normalized to stage 0.

### IR sketch

```mlir
// 05 — default still tagged as late SWP stage
ttng.wait_barrier %75, %74 {loop.cluster = 2, loop.stage = 2}
tt.dot ...                   {loop.cluster = 2, loop.stage = 2}

// 05 — TMA partition still tagged as early SWP cluster
ttng.async_tma_copy ...      {loop.cluster = 5, loop.stage = 0}

// 06 — each partitioned for collapsed to one stage/cluster
ttng.wait_barrier %75, %74 {loop.cluster = 1, loop.stage = 0}
tt.dot ...                   {loop.cluster = 1, loop.stage = 0}
ttng.async_tma_copy ...      {loop.cluster = 1, loop.stage = 0}
```

**One-liner:** after WS split, `ScheduleLoops` re-stamps stage/cluster per physical for; when a partition’s latency ops share one stage, the old multi-stage attrs collapse to a single stage.

---

# 10. multiBufferTMADescriptors (`AutomaticWarpSpecialization.cpp` + `PipeliningUtility.cpp`)

| | |
|--|--|
| **Pass** | not a standalone pass — helper at end of AWS (`multiBufferTMADescriptors`) |
| **When** | after the AWS `OpPassManager` finishes (`ScheduleLoops` last in the PM) |
| **Role** | multi-buffer in-loop `tt.make_tensor_descriptor` so desc writes do not clobber an in-flight TMA |
| **IR** | no change on this matmul (`06-after-schedule-loops.ttir` is already final for AWS); only kernels that rebuild descriptors inside a WS for |

AWS comment: cannot rely on SWP alone for this, especially when desc updates live in **nested** loops. So it runs as an explicit post-PM step with `numStages` from the AWS option.

## 10.1 Mechanism — ring of descriptor slots

A tensordesc used by `async_tma_copy` is a small object in memory. If the loop **rebuilds** it each trip (`MakeTensorDesc`) into one slot while a prior TMA is still outstanding, the next write can corrupt that load. Fix: allocate `numDescs` slots and rotate a counter.

`numDescs = numStages + 1` (extra slot so “next update” and “oldest in-flight load” do not collide). The helper builds `CoarseSchedule(numDescs + 1)` because CoarseSchedule’s `numStages` means max stage index + 1; `lowerTMADescriptors` then uses `maxStage = schedule.getNumStages() - 1` (= `numDescs`).

## 10.2 Workflow (`multiBufferTMADescriptors` → `lowerTMADescriptors`)

### (1) Collect loops

Walk module fors with `tt.warp_specialize`. If a **nested** for contains `tt.make_tensor_descriptor`, insert that nested for into `descUpdateLoops`.

### (2) Allocate + rewrite (`lowerTMADescriptors`)

For each collected for:

1. `allocTMABuffers` — one global-scratch alloc of `maxStage * TMA_SIZE` per in-loop `MakeTensorDesc` (Hopper may bump `maxStage` if a `WarpGroupDot` is present).
2. Add a loop-carried **counter** per descriptor (init 0).
3. Replace each `MakeTensorDesc` with: subview slot `counter` → `createTMADesc` write → `tensormap_fenceproxy_acquire` → `reinterpret_tensor_desc`; uses of the old result follow the reinterpret; counter = `(counter+1) % numBuffers`; erase `MakeTensorDesc`.

If `tmaBufferMapping` is empty, return the for unchanged.

### IR — this matmul (no-op)

```mlir
// Outside WS fors — store descriptor only; not multi-buffered by this helper
%0 = tt.make_tensor_descriptor %arg7, ...

ttg.warp_specialize(...)
default { scf.for ... { ... descriptor_store %0[...] ... } }
partition0(... %arg52: !tt.tensordesc<...>, %arg56: !tt.tensordesc<...>, ...) {
  scf.for ... {
    // A/B loads use stable captured tensordescs — no MakeTensorDesc in the for
    ttng.async_tma_copy_global_to_local %arg52[...] ...
    ttng.async_tma_copy_global_to_local %arg56[...] ...
  }
}
```

`descUpdateLoops` stays empty → nothing rewritten.

**One-liner:** `multiBufferTMADescriptors` multi-buffers in-loop TMA descriptor rebuilds for WS kernels; this matmul’s A/B descriptors are stable captures, so the helper is a no-op.

---

# 11. “Simulate then materialize later” passes in MLIR

What you noticed in SCCP—**simulate in analysis**, then **materialize later** in a rewrite—is a fairly common MLIR pattern.

## 11.1 Is this style popular in MLIR?

Yes. MLIR frequently separates:

1. **Analysis / decision-making**: compute facts (constants, ranges, shapes, legality) *without changing IR semantics*.
2. **Rewrite / materialization**: apply the chosen simplifications using `OpBuilder` / `PatternRewriter`.

This shows up in multiple places:

- Dataflow-based optimizations: SCCP, constant propagation-like passes.
- Range/value analyses used to drive canonicalization.
- Shape/rank/stride analyses feeding targeted rewrites.

The motivation is usually the same:

- Analyses can be iterative and conservative (prove what’s safe).
- Rewrites should be localized and deterministic (apply only proven-safe changes).

## 11.2 Mechanism: simulate first, then materialize

Think of it as “two-phase optimization”:

### (A) Simulation phase (analysis)

The pass speculatively tries to evaluate what would happen **if** some operand values were constant.

In SCCP, this is:

- Build lattice facts per SSA value.
- For each op whose operands are known constants, call its dialect folding hook:
  - `op->fold(constantOperands, foldResults)`
- Update the lattice with either:
  - a new constant (if fold succeeded out-of-place), or
  - “unknown/overdefined” (if fold fails / is unsafe).

Importantly, the simulation phase **does not permanently rewrite** the IR—otherwise the analysis would be order-dependent or incorrect.

That’s why `ConstantPropagationAnalysis.cpp` has a guard for **in-place folds**:
- if folding mutates the op and produces no explicit `foldResults`, the analysis restores operands/attrs and conservatively marks results as not-constant.

### (B) Materialization phase (rewrite)

After the solver reaches a stable answer (fixpoint), SCCP then:

- replaces uses with materialized constants
- deletes ops that are now trivially dead

In SCCP’s `SCCP.cpp`, the rewrite happens in `rewrite(solver, ...)`, after `initializeAndRun(...)`.

## 11.3 Example: constant propagation with SCCP (conceptual IR)

### Before

```mlir
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32
%x  = arith.addi %c0, %c1 : i32
%y  = arith.muli %x, %c1 : i32
```

### Simulation phase

SCCP sees `%c0` and `%c1` are constants, calls folding:

- fold(`addi`, [0, 1]) → constant `1`
- fold(`muli`, [1, 1]) → constant `1`

### Fixpoint answer

Lattices say `%x` and `%y` are constants.

### Materialization phase

SCCP replaces `%x`/`%y` uses with constants and removes trivially dead ops.

## 11.4 How to “support this style” when writing/using passes

If you want to implement your own SCCP-like or simulate→materialize pass pattern in MLIR, the key support points are:

1. **Separate “facts” from “changes”.**
   - Represent results in analysis data structures (lattices/ranges/booleans).
   - Only rewrite in the final step.
2. **Use out-of-place folding, or restore on in-place folding.**
   - Provide correct `Op::fold(...)` implementations in the dialect so analyses can ask:
     “given constant operands, can you compute a constant result?”
   - If a fold can be in-place, analyses like SCCP must defensively guard and restore.
3. **Materialize using a builder/folder.**
   - In the rewrite phase, use the dialect folder / constant materialization to create the real IR constants.
4. **Keep rewrites small and local.**
   - Replace specific uses; then rely on canonicalization/CSE for broader cleanup.

For SCCP specifically, the “support” is already built into MLIR’s dataflow + fold interface:
- `SCCP.cpp` runs analysis to fixpoint
- `ConstantPropagationAnalysis.cpp` uses `op->fold(...)` for simulation
- `rewrite(...)` applies replacements and deletes dead ops

If the user wants, the next step is to point out another MLIR pass in Triton that follows the same pattern (analysis → rewrite), but SCCP is the clearest example.
