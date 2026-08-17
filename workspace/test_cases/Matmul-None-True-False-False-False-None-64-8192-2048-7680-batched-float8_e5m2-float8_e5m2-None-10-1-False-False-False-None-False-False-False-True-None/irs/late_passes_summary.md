# Late-pass IR dumps (`block_m=64` persistent FP8)

Kernel: `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1`  
Test: `test_matmul.py::test_op` (`8192×2048×7680` batched FP8, `is_persistent=True`, `block_m=64`)  
Dump dir: this `irs/` folder  
Compile: `TRITON_PERF_IR_DUMP=<irs> TRITON_ALWAYS_COMPILE=1` (pytest **PASSED**)

Each file is the module **after** the named pass. Unlisted passes between dump points are included in the later snapshot. Dumps restart the pass manager only when `TRITON_PERF_IR_DUMP` is set.

---

## Pass-by-pass

### 1. `add_optimize_partition_warps` → `10_after_…ttgir`

Shrinks extra warps on `ttg.warp_specialize` partitions and trims their register ask.

**This kernel:** still one load partition with `num_warps(2)` and `requestedRegisters = [24]`. Default group stays `ttg.num-warps = 4`. Depth-2 A/B SMEM and `tt.dot` (MMAv2 `16×8`) are already in place from pipeline.

### 2. `add_remove_tmem_tokens` → `11_after_…ttgir`

Drops Tensor-Memory (TMEM) dependence tokens after they are no longer needed (Hopper/Blackwell tcgen05 path).

**This kernel (sm120):** no `ttng.tmem_alloc`. Snapshot still has a leftover `ub.poison : !ttg.async.token` from intervening hoist/cleanup; the next canonicalizer removes it. Not a functional issue.

### 3. `add_optimize_dot_operands` → `12_after_…ttgir`

Rewrites MMA operand layouts so loads can feed Tensor Cores without extra transposes / converts.

**This kernel:** `ttg.local_load` of A/B now yields `#ttg.dot_op<{opIdx = 0/1, parent = #mma, kWidth = 4}>` instead of a blocked layout plus `convert_layout`. Poison token is gone.

### 4. `add_fence_insertion` → `13_after_…ttgir`

Inserts generic vs async-proxy memory fences at optimized points. A later pass covers remaining functional fences.

**This kernel:** `ttng.fence_async_shared` was already present around TMA wait/arrive (count stays 3). Intervening TMA lowering adds `ttng.tensormap_create` + `ttng.tensormap_fenceproxy_acquire` for the Y descriptor.

### 5. `add_allocate_warp_groups` → `14_after_…ttgir`

Counts warps for warp specialization, assigns warp-group IDs, and sets module `ttg.total-num-warps`.

**This kernel:**

| Attr | Value |
|------|------:|
| `ttg.num-warps` (default / MMA) | **4** |
| extra partitions | **2** × `num_warps(2)` |
| `warpGroupStartIds` | `[4, 6]` |
| `ttg.total-num-warps` | **8** → launch **256** threads |
| `actualRegisters` | `[488, 24, 24]` (budget; later capped) |
| `ttg.maxnreg` | **256** |

Matches the baseline nsys launch `(36, 256)`.

### 6. `add_allocate_shared_memory_nv` → `15_after_…ttgir`

Assigns SMEM offsets and the module `ttg.shared` size (NVIDIA layout).

**This kernel:** `ttg.shared = 99396` (~97.1 KiB), same as CUPTI. Offsets: B tile `0`, A tile `65536`, mbarriers `99328…99376`, epilogue tile `81920`.

### 7. `add_allocate_tensor_memory` → `16_after_…ttgir`

Assigns Blackwell tensor-memory (TMEM) slots for tcgen05 MMA.

**This kernel:** `ttg.tensor_memory_size = 0`. Expected: consumer sm120 stays on **MMAv2** (`mma.sync.m16n8k32`), not TMEM.

### 8. `add_proxy_fence_insertion` → `17_after_…ttgir`

Inserts remaining generic/async-proxy fences that the earlier fence pass did not place.

**This kernel:** one extra `ttng.fence_async_shared` (3 → 4). `check_matmul_two_cta` (just before this dump) sets `ttng.two-ctas = false`.

### 9. `add_warp_specialize_to_llvm` → `18_after_…mlir`

Lowers `ttg.warp_specialize` to LLVM: default vs worker warps communicate through SMEM + barriers.

**This kernel:** `ttg.warp_specialize` is gone; one `llvm.func` with the same kernel name; `nvvm.barrier` appears (26). File is large because debug `loc`s are still attached.

### 10. `llvm.optimize_module(..., O3)` → `03_after_llvm_optimize_module_O3.llir`

LLVM-IR (LLVM) after O3. Already dumped as `03_…`.

**This kernel:** `nvvm.reqntid = 256`, `nvvm.maxnreg = 256`, `setmaxnreg` inc **256** / dec **24** on WS partitions, **128** `mma.sync` in the K-body. Same launch story as the experiment report.

---

## Verification

No compile/IR failure. Pytest **PASSED**.

| Check | Result |
|-------|--------|
| Kernel symbol through TTGIR | `@_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1` |
| WS present until LLVM lower | yes (`10`–`17`); gone in `18` |
| SMEM size | **99396 B** (matches nsys) |
| TMEM | **0** (MMAv2 / sm120) |
| CTA threads | **8 warps × 32 = 256** |
| MMA | `#nvidia_mma` 16×8, `kWidth=4` FP8 |

**Not bugs, just this path:**

1. `remove_tmem_tokens` / TMEM alloc are no-ops here (no TMEM).
2. `actualRegisters = 488` is a partition budget, not 488 physical registers; PTX/LLVM later uses 256 / 24.
3. Dump `18` originally named `___kernel` (`llvm.func` not matched); renamed to the real kernel.

---

## Reproduce

```bash
export PYTHONPATH=python/triton_kernels
export TRITON_ALWAYS_COMPILE=1
export TRITON_PERF_IR_DUMP=$PWD/workspace/test_cases/.../irs
/Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
  python/triton_kernels/tests/test_matmul.py::test_op -v
```
