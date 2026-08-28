# Producer–consumer pipeline lowering (mbarrier + proxy fence)

| | |
|--|--|
| **Example IR** | `…/irs/19_after_ttnvgpuir_add_nvgpu_to_llvm.mlir___p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1` |
| **Kernel** | `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1` (FP8 batched matmul, warp-specialized) |
| **Related passes** | `add_nvgpu_to_llvm`, `add_nvvm_to_llvm`, `add_proxy_fence_insertion` |
| **Related docs** | [proxy_fence_insert](../proxy_fence_insert/proxy_fence_insert.md), [nvgpu_to_llvm](../NVGPUTOLLVM/nvgpu_to_llvm.md) |

At LLVM-dialect stage (IR 19), producer–consumer synchronization appears as:

- **`llvm.inline_asm`** — `mbarrier.*`, `cp.async.bulk.tensor.*`, tensormap proxy fences
- **`nvvm.fence.proxy`** — `fence.proxy.async.shared::cta` (from proxy-fence insertion)
- **`nvvm.barrier`** — CTA-wide barriers around handoffs

There are **two** double-buffered mbarrier pipelines in this kernel, plus tensormap init fences and an output-store fence.

---

## Warp-specialization context

```text
^bb0 entry
  tid.x → warp id via shfl
  warp < 4  → ^bb11  consumer warpgroup (matmul MMA, 256 threads → 128 active)
  warp ≥ 4  → ^bb1   producer dispatcher loop (switch on smem[99392+…])

^bb1 dispatcher
  case 0 → ^bb4 → ^bb5/^bb8   scale/TMA producer (local_absmax)
  case 1 → ^bb10              stub (sync, back to ^bb1)
  case 2 → ^bb3               return
```

Consumer matmul loop: `^bb13` → `^bb16` → `^bb18` → `^bb13`.  
Scale producer loop: `^bb5` → `^bb8` → `^bb5`.

---

## Shared-memory mbarrier layout

Eight `mbarrier.init …, 1` at IR lines 551–583. Two slots each (double buffer), four roles:

| Shared offset | SSA base | Role | Consumer action |
|---------------|----------|------|-----------------|
| `99328` | `%455` | A **empty** | `mbarrier.arrive` after consuming A (release to producer) |
| `99344` | `%467` | A **full** | `mbarrier.try_wait.parity` before `ldmatrix` A |
| `99360` | `%476` | B **empty** | `mbarrier.arrive` after consuming B |
| `99376` | `%485` | B **full** | `mbarrier.try_wait.parity` before loading B |

Scale pipeline reuses overlapping regions earlier in the kernel:

| Offset | SSA | Use in scale pipeline (`^bb8`) |
|--------|-----|--------------------------------|
| `99328` | `%143` | wait barrier before reading Y scale |
| `99344` | `%141` | TMA mbarrier for Y (`arg13` / X desc) |
| `99360` | `%140` | wait barrier before reading X/W scale |
| `99376` | `%139` | TMA mbarrier for W (`arg24`) |

Data buffers: A tile at `smem+65536` (`%454`), B tile at `smem+32768` (`%107`).

---

## Pipeline 1 — Scale / TMA producer (`^bb5` → `^bb8`)

Producer warpgroup `^bb4`. Software-pipelined TMA loads for scale tensors (X desc `%arg13`, W desc `%arg24`).

### Phase flip (buffer index wraps at 2)

```mlir
// lines 255–256
%183 = llvm.xor %158, %115 : i32          // phase ^ 1  (%115 = 1)
%184 = llvm.select %181, %183, %158 : i1, i32   // flip when buffer index wraps
```

- `%158` — loop-carried phase (starts `0` in `^bb5`)
- `%182` / `%157+1` — buffer index `0/1`, wrap when `== 2`
- Phase carried on back-edge: `llvm.br ^bb5(..., %182, %184)` (line 315)

### Wait — consumer waits for TMA completion

PTX spin loop (lines 259, 280):

```ptx
{
  .reg .pred complete;
waitLoop:
  mbarrier.try_wait.parity.shared::cta.b64 complete, [barrier], phase;
  @!complete bra.uni waitLoop;
}
```

| Line | Barrier ptr | Phase | Waits for |
|------|-------------|-------|-----------|
| 259 | `%186` = `%143[%185]` (Y wait @ 99328) | `%184` | Y-scale TMA done |
| 280 | `%206` = `%140[%185]` (X wait @ 99360) | `%184` | X/W-scale TMA done |

### Producer — declare bytes + issue TMA

```mlir
// Y scale (lines 268, 278)
%196 = llvm.inline_asm ... "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], 8192;" ... %190
%205 = llvm.inline_asm ... "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes ..." ... %arg13 ... %190

// X/W scale (lines 284, 309)
%211 = llvm.inline_asm ... "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], 32768;" ... %210
%234 = llvm.inline_asm ... "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes ..." ... %arg24 ... %210
```

TMA `complete_tx` automatically completes the mbarrier when the async copy finishes.

### Fence in scale pipeline (line 285)

```mlir
nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
```

**When:** after `expect_tx` for W/X (line 284), before second TMA issue and `nvvm.barrier`.  
**PTX:** `fence.proxy.async.shared::cta;`  
**Why:** order prior async shared traffic before the next TMA + mbarrier sequence.

---

## Pipeline 2 — Matmul K-loop (`^bb13` → `^bb16`)

Consumer warpgroup `^bb11`. Double-buffered A/B tiles with mbarrier handoff.

### Phase flip

```mlir
// lines 627–628
%530 = llvm.xor %510, %115 : i32
%531 = llvm.select %528, %530, %510 : i1, i32
```

- `%509` — buffer index, `%510` — phase
- Carried on loop back-edge: `llvm.br ^bb13(..., %529, %531)` (line 5525)

### Wait — consumer waits for full buffer

| Line | Barrier | Phase | Then |
|------|---------|-------|------|
| 632 | `%533` = `%467[%532]` (A full) | `%531` | `ldmatrix` from `%454` (A @ 65536) |
| 1088 | `%985` = `%485[%532]` (B full) | `%531` | `llvm.load` from B @ 32768 |

Preceded by `nvvm.barrier` (lines 631, 1087) to align warps before `try_wait`.

### Release — consumer signals empty buffer

| Line | Barrier | Predicate | Meaning |
|------|---------|-----------|---------|
| 1085 | `%983` = `%455[%532]` (A empty) | `%387` (elect leader) | done with A; producer may refill |
| 2391 | `%2285` = `%476[%532]` (B empty) | `%387` | done with B; producer may refill |

```mlir
llvm.inline_asm ... "mbarrier.arrive.shared::cta.b64 _, [$1];" ...
```

### Fences at consumer handoff

| Line | When | Why |
|------|------|-----|
| 1081 | after all `ldmatrix` A, before `mbarrier.arrive` A empty | ensure MMA/ldmatrix visible w.r.t. async proxy before release |
| 2387 | after B `llvm.load`s, before `mbarrier.arrive` B empty | same for B consumption |

```mlir
nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
```

**Note:** no `fence.proxy` between `try_wait` (632) and `ldmatrix`. Ordering there comes from the mbarrier wait itself.

### Teardown (lines 5535–5556)

At kernel exit (`^bb20`): `mbarrier.inval.shared::cta.b64` on all eight barrier objects.

---

## Tensormap proxy fences (prologue, lines 494–495)

One-time setup after `tensormap.replace.tile.*` ops (lines 477–493):

```mlir
%405 = llvm.inline_asm ... "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned ..."
%406 = llvm.inline_asm ... "fence.proxy.tensormap::generic.acquire.gpu ...;
                            cp.async.bulk.commit_group ;
                            cp.async.bulk.wait_group.read 0 ;"
```

| PTX | Role |
|-----|------|
| `tensormap.cp_fenceproxy...release` | publish tensormap writes in shared → visible to TMA unit |
| `fence.proxy.tensormap::generic.acquire` | TMA engine may safely read tensormap |
| `cp.async.bulk.commit_group` / `wait_group.read 0` | drain pending bulk async after tensormap init |

Also: `nvvm.bar.warp.sync` (line 475) before tensormap updates; `nvvm.barrier` (line 496) after.

---

## Output store fence (`^bb17`, line 5493)

After threads pack FP8 results into shared staging (`llvm.store` into `%500` @ 81920):

```mlir
nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}
nvvm.barrier ...
// then cp.async.bulk.tensor.3d.global.shared::cta.bulk_group (TMA store to global, line 5518)
```

Ensures shared stores are visible to the async copy engine before it reads shared for the global write.

---

## End-to-end flow

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ Scale pipeline (^bb8), per iteration                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  phase flip (%158 ^ 1 when buffer wraps)                                │
│  try_wait(Y) @ %143 ──► expect_tx + TMA(Y) ──► mbarrier complete_tx     │
│  try_wait(X) @ %140 ──► fence.proxy.async.shared ──► expect_tx + TMA(X) │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Matmul K-loop (^bb16), per iteration                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  phase flip (%510 ^ 1)                                                  │
│  barrier → try_wait(A full @ %467) → ldmatrix A                         │
│         → fence.proxy.async.shared → arrive(A empty @ %455)             │
│  barrier → try_wait(B full @ %485) → load B                             │
│         → fence.proxy.async.shared → arrive(B empty @ %476)             │
│  ... mma / accumulate ...                                               │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Tensormap init (once, ^bb11 prologue)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  tensormap.replace.* → cp_fenceproxy.release → fence.proxy.tensormap    │
│                      → acquire → commit_group → wait_group              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Output store (^bb17)                                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  shared stores → fence.proxy.async.shared → TMA store to global         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## IR line index (quick reference)

| Topic | Lines |
|-------|-------|
| Scale phase flip | 255–256, 315 |
| Scale try_wait | 259, 280 |
| Scale expect_tx + TMA | 268, 278, 284, 309 |
| Scale fence.proxy | 285 |
| mbarrier.init (×8) | 551–583 |
| Matmul phase flip | 627–628, 5525 |
| Matmul try_wait A/B | 632, 1088 |
| Matmul arrive A/B | 1085, 2391 |
| Matmul fence.proxy | 1081, 2387 |
| Tensormap fences | 494–495 |
| Output fence.proxy | 5493 |
| mbarrier.inval | 5535–5556 |

---

## Lowering notes

- **`mbarrier.try_wait.parity` / `arrive` / `expect_tx`** — lowered via `BasicPtxBuilderInterface` in `convert-nvvm-to-llvm` (`PtxLowering` in `NVVMToLLVM.cpp`) to `llvm.inline_asm`.
- **`nvvm.fence.proxy`** — stays as `nvvm.fence.proxy` through IR 20; translated to NVVM intrinsics / PTX at `llvm.to_module`.
- **Producer TMA for matmul A/B tiles** — consumer-side `try_wait` + `arrive` is visible in IR 19; the warpgroup that refills `%467`/`%485` and signals `complete_tx` is coordinated via warp-specialization (`^bb1` / `^bb4`) and may not appear as separate TMA ops in the consumer path.
