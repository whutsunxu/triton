# Batched FP8 Matmul Experiment Analysis

Target kernel: `_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`
Profile dir: `work_space` → `/Volumes/case_sensitive_workspace/triton/workspace/profile_01/`
Frame adapted from [`perf_ana` MLP-MoE report](https://github.com/whutsunxu/triton/blob/perf_ana/work_space/MLP-MOE-B64-2048-1440/experiment_report.md).

---

## 1. Platform

### Software


**Table: Software stack versions**

| Component              | Version                             | Source                                                        |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------- |
| NVIDIA Driver          | 580.95.05                           | `nvidia-smi`                                                  |
| CUDA (driver-reported) | 13.0                                | `nvidia-smi`                                                  |
| CUDA Toolkit (`nvcc`)  | 13.0, V13.0.48                      | container `nvcc --version`                                    |
| PyTorch                | 2.12.0+cu130 (built with CUDA 13.0) | workspace `venv`                                              |
| Triton                 | 3.7.0 (`+git`, editable)            | `/Volumes/case_sensitive_workspace/triton` via `PYTHONPATH` / editable install |
| Nsight Systems         | 2025.3.2                            | container (`nsys`)                                            |
| Nsight Compute         | 2025.3.0                            | container (`ncu`)                                             |




### Hardware


**Table: GPU hardware overview**

| Item               | Value                                     |
| ------------------ | ----------------------------------------- |
| GPU                | NVIDIA GeForce RTX 5060 Ti                |
| SKU / VRAM         | **16 GB** GDDR7 (`nvidia-smi`: 16311 MiB) |
| Architecture       | Blackwell (consumer GB206-class), CC 12.0 |
| SMs                | 36 (`torch.cuda.get_device_properties`)  |
| L2 cache           | 32 MB                                     |
| CUDA Cores         | 4608                                      |
| Tensor Cores       | **144** (4 per SM × 36 SMs)               |
| Boost clock (ref.) | 2.57 GHz ([NVIDIA 5060 family](https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5060-family/)); **759 AI TOPS** |
| SM frequency (peak model) | **2.40 GHz** (NCU measured during this kernel) |
| Memory interface   | 128-bit GDDR7                             |

#### SM on-chip resources (CC 12.0)

Limits from the [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/) (consumer / CC 12.0):

**Table: SM on-chip resource limits (CC 12.0)**

| Resource | Per SM | Per CTA (thread block) | Notes |
| -------- | ------ | ---------------------- | ----- |
| Register file | **65536** × 32-bit regs (**64K**) | ≤ **255** regs/thread | Occupancy: `floor(65536 / (regs/thread × threads/CTA))` CTAs |
| Max resident warps | **48** | — | 4 schedulers × 12 warps |
| Max resident CTAs | **32** | — | Usually hit regs/SMEM first |
| L1 + SMEM pool | **128 KB** | — | Carveout split L1 vs SMEM at runtime |
| Shared memory | ≤ pool above | **≤ 99 KB** (`sharedMemPerBlockOptin`) | ~1 KB driver reserve/CTA; static default **48 KB** without opt-in |

This kernel (NCU): **255 regs/thread**, **256 threads**, **~49 KB** dynamic SMEM → both **regs** and **SMEM** limit to **1 CTA/SM** (`Block Limit Registers = 1`, `Block Limit Shared Mem = 1`).


### Peak performance (RTX 5060 Ti 16GB)

Peaks use the profiled SM clock `f_SM = 2.40 GHz` (not 2.57 GHz boost), unless noted.

**2D (Tensor Cores, dense)** — Blackwell dense MAC rates; 1 MAC = 2 FLOPs; 4 TC/SM:

```text
Arch (dense):  BF16/FP16 = 512 MACs/TC/cycle  → 1024 FLOPs/TC/cycle
               FP8       = 1024 MACs/TC/cycle → 2048 FLOPs/TC/cycle

Peak_2D = (FLOPs/TC/cycle) * (TC/SM) * (#SM) * f
        = (FLOPs/TC/cycle) * 4 * 36 * f
```

```text
Peak_BF16_2D @ 2.40 GHz = 1024 * 4 * 36 * 2.40e9 = 354 TFLOPS
Peak_FP8_2D  @ 2.40 GHz = 2048 * 4 * 36 * 2.40e9 = 708 TFLOPS
```

Cross-check at boost 2.57 GHz: `2048 * 4 * 36 * 2.57e9 ≈ 759 TFLOPS`, matching NVIDIA’s **759 AI TOPS** on the 5060 Ti product page.

**Table: Peak performance summary**

| Metric | Peak | Meaning |
| ------ | ---- | ------- |
| Memory bandwidth | **448 GB/s** | GDDR7 peak |
| **2D** BF16 (Tensor Core, dense) | **≈354 TFLOPS** | `1024 * 4 * 36 * 2.40e9` |
| **2D** FP8 (Tensor Core, dense) | **≈708 TFLOPS** | `2048 * 4 * 36 * 2.40e9` |
| **1D** FP32 (CUDA core) | **≈22.1 TFLOPS** | Epilogue / elementwise roof |
| **Phys. 2D dens.** (FP8) | **≈1580 ops/B** | `708 TFLOPS / 448 GB/s` |
| **Phys. 1D dens.** (CUDA) | **≈49.4 ops/B** | `22.1 TFLOPS / 448 GB/s` |

Do **not** derive Tensor peaks from `k * Peak_FP32_1D`; TC throughput is independent of the CUDA-core FP32 roof.

---

## 2. Test Case

### Script and launch


**Table: Test script and launch**

| Item | Value |
| ---- | ----- |
| Test | `python/triton_kernels/tests/test_matmul.py::test_op` |
| Active case (branch `matmul_perf_analysis`) | `Case(8192, 2048, 7680, "batched", "float8_e5m2", "float8_e5m2")` |
| Pytest node id (abbrev.) | `…-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-…` |
| Launch (smoke) | `PYTHONPATH=python/triton_kernels python -m pytest …::test_op -v` |
| Launch (nsys) | see §3 |
| Container | `ir_dev` (`jasonsun11/ir_dev:cuda13_v2`), `--gpus all --privileged` |




### Workload parameters


**Table: Workload parameters**

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| `m` | **8192** | rows of A / C |
| `n` | **2048** | cols of B / C |
| `k` | **7680** | reduction dim |
| `mode` | **batched** | batch dim = `n_slices` |
| `n_slices` | **10** | default for non-`plain` `Case` |
| Activation dtype | **FP8 E5M2** (`float8_e5m2`) | A and C |
| Weight dtype | **FP8 E5M2** | B |
| Bias | FP32 `[10, 2048]` | enabled (`do_bias=True` in this config) |
| `block_m` | **128** | pytest param |
| Tile (from kernel name) | **128×256×128×1** | `BM×BN×BK×splitK` |
| Layout tag | **NNN** | A/B/C non-transposed path in this naming |
| `is_persistent` | False | |
| `split_k` | 1 | |
| Fused SwiGLU | None | pure matmul + bias |


### Tensor shapes (batched) and theoretical perf feature analysis

Shapes for `mode=batched`, `n_slices=B=10`, `M=8192`, `N=2048`, `K=7680`. Sizes use 1 byte/elem (FP8) and 4 bytes/elem (FP32 bias); MB = bytes / 1e6.

**Table: Batched tensor shapes and sizes**

| Tensor | Shape | Dtype | Bytes | Formula |
| ------ | ----- | ----- | ----- | ------- |
| A (activation) | `[10, 8192, 7680]` | FP8 | **629.146 MB** | `B*M*K` |
| B (weight) | `[10, 7680, 2048]` | FP8 | **157.286 MB** | `B*K*N` |
| C (output) | `[10, 8192, 2048]` | FP8 | **167.772 MB** | `B*M*N` |
| bias | `[10, 2048]` | FP32 | **0.082 MB** | `B*N*4` |
| **Traffic A+B+C** | — | — | **954.204 MB** | bias excluded from DRAM traffic model |

**Launch geometry (nsys / CUPTI):** grid `(5120,1,1)`, block `(256,1,1)` (= 8 warps), 255 regs/thread, 48 KiB dynamic SMEM.
Grid check: `ceil(M/128) * ceil(N/256) * n_slices = 64 * 8 * 10 = 5120`.

Peaks used below (@ `f_SM = 2.40 GHz`, see §1): **Peak_FP8_2D = 708 TFLOPS**, **Peak_FP32_1D ≈ 22.1 TFLOPS**, **Peak_BW = 448 GB/s**.

Formulas:

```text
FLOPs_2D       = 2 * B * M * N * K                         # GEMM MACs on Tensor Cores
FLOPs_1D       = B * M * N                                 # bias add (scale mul is same order)
Traffic        = size(A) + size(B) + size(C)               # FP8; bias excluded
Op_2D_dens     = FLOPs_2D / Traffic
Op_1D_dens     = FLOPs_1D / Traffic
t_2D_theo      = FLOPs_2D / Peak_FP8_2D
t_1D_theo      = FLOPs_1D / Peak_FP32_1D
t_BW_theo      = Traffic / Peak_BW
t_lower        = max(t_2D_theo, t_1D_theo, t_BW_theo)      # ideal lower bound
Effective_BW   = Traffic / t_meas
FLOPS_util     = (FLOPs_2D / t_meas) / Peak_FP8_2D
```

**Table: Roofline quantities for target kernel**

| Quantity | Task amount | Theoretical time |
| -------- | ----------- | ---------------- |
| 2D FLOPs (FP8 TC) | `2*10*8192*2048*7680 = 2.577e12` | **`3.640e-3` s (3.64 ms)**  |
| 1D FLOPs (FP32 CUDA, bias and scale introduced by convert) | `3*10*8192*2048 = 5.033e8` | **`2.28e-5` s (22.8 µs)** |
| Traffic A+B+C | **954.204 MB** | **`2.130e-3` s (2.13 ms)**  |
| Phys. 2D dens. (FP8 knee) | **≈1580 ops/B** (`708e12 / 448e9`) | — |
| Phys. 1D dens. (FP32 knee) | **≈49.3 ops/B** (`22.1e12 / 448e9`) | — |
| Ideal lower bound | `max(t_2D, t_1D, t_BW)` | **`3.640e-3` s (3.64 ms)** |

**Summary:** from the above theoretical time of 2D task(on tensor core), 1D tasks(on cuda core), data traffic tasks(on
LDGSTC), we can find the case is bounded by tensor core.

---

## 3. Parallelism and Tiling Decomposition

Sources: TTGIR `irs/02_after_ttnvgpuir_add_lower_mma.ttgir___matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`, PTX `irs/04_after_llvm_translate_to_asm.ptx___…`, launch path in `triton_kernels/matmul.py` + `matmul_details/_common.py::compute_pids`.

### 3.1 Parallel split at CTA level (output / batch)

Problem shape (batched): `Y[B, M, N]` with `B=10`, `M=8192`, `N=2048`, reduction `K=7680`.

Launch (1D grid flattened to `(5120,1,1)`):

```text
grid_m = ceil(M / BLOCK_M) = ceil(8192 / 128) = 64
grid_n = ceil(N / BLOCK_N) = ceil(2048 / 256) = 8
grid   = B * grid_m * grid_n * SPLIT_K
       = 10 * 64 * 8 * 1 = 5120
```

Each `pid = tl.program_id(0)` ∈ `[0, 5120)` is **one CTA**. For `mode=batched` (`RAGGED_DIMENSION is None`), `compute_pids` decodes:

**Table: CTA program-id decoding**

| Logical id | Range | Role |
| ---------- | ----- | ---- |
| `pid_s` (`pid_z`) | `0..9` | batch / slice index |
| `pid_m` | `0..63` | M-tile index → rows `[pid_m*128, (pid_m+1)*128)` |
| `pid_n` | `0..7` | N-tile index → cols `[pid_n*256, (pid_n+1)*256)` |
| `pid_k` | `0` | `SPLIT_K=1` (no K-split across CTAs) |

So the **5120 CTAs tile the output volume** `(B, M, N)`; K is reduced **inside** each CTA by looping over `BLOCK_K=128` tiles (`k_tiles = 7680/128 = 60`).

Thread layout per CTA: **8 warps × 32 threads = 256 threads** (matches NCU `Block Size = 256`).

### 3.2 Parallel split at warp level (within one CTA tile)

MMA layout from TTGIR:

```text
#mma = #ttg.nvidia_mma<{…, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
```

Warps form a **2×4 grid** over the CTA output tile `128×256`:

**Table: Warp ownership of CTA accumulator tile**

| Warp coord `(w_m, w_n)` | Owns accumulator subtile |
| ---------------------- | ------------------------ |
| `w_m in {0,1}`, `w_n in {0,1,2,3}` | `64×64` fp32 (`128/2` × `256/4`) |

All 8 warps **share** the CTA SMEM buffers for A/B; they do **not** each own a private `128×128` / `128×256` allocation. Parallelism is over the **output (accumulator) plane**: each warp `local_load`s the A rows / B cols needed for its `64×64` from the common SMEM tiles.

### 3.3 CTA-level tiling (SMEM vs registers)

IR: `irs/02_after_ttnvgpuir_add_lower_mma.ttgir___matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`. Launch opts: `num_stages=2` (`opt_flags.py` default for this path).

#### SMEM (CTA shared)

Only the K-pipelined **activation** and **weight** tiles are allocated in shared memory. Bias is **not** in SMEM.

**Table: CTA SMEM buffer allocations**

| Buffer | Role | IR `local_alloc` | Dtype | Logical tile | Bytes |
| ------ | ---- | ---------------- | ----- | ------------ | ----- |
| `%x` | activation (A / X) | `!ttg.memdesc<1×128×128×f8E5M2, #shared, #smem>` | FP8 E5M2 | `BLOCK_M × BLOCK_K = 128×128` | `128 * 128 * 1 = 16384` |
| `%w` | weight (B / W) | `!ttg.memdesc<1×128×256×f8E5M2, #shared1, #smem>` | FP8 E5M2 | `BLOCK_K × BLOCK_N = 128×256` | `128 * 256 * 1 = 32768` |
| — | bias | *(none)* | FP32 | `256` (N-vector) | **0 in SMEM** — `tt.load` from global → registers in epilogue |

```text
SMEM_tiles = 16384 + 32768 = 49152 B = 48 KiB
```

48 KB < 98 KB, less than the SM's SHMEM limiter. with num_stages=2, in total 96 KB, less than the SM's SHMEM limiter. so each SM can hold one CTA from the SHEME perspective(for better CTA scheduling, 2-4 CTA on one SM might be better.).

**Note on `num_stages=2` vs alloc shape `1×…`:** launch uses `num_stages=2`, and Triton keeps `num_stages − 1` SMEM slots → IR `local_alloc` is `1×128×128` / `1×128×256` (**48 KiB** actual, matches NCU). The K-loop still does generic stage arithmetic (`idx = stage+1; if idx >= 1: idx = 0`) before `memdesc_index`, so SSA can momentarily hold `1`, but the wrap threshold equals alloc depth **1**, so only slot **0** is ever used — single-buffer, not true 0/1 ping-pong. A real double-buffer would be `2×128×128` + `2×128×256` ≈ **96 KiB** (still &lt; ~98 KB CTA SHMEM), which is the “with num_stages=2, in total 96 KB” packing view above.

#### Registers

No `local_alloc` for C or bias — they live in the register file. Pressure is set by **peak live set**, not the sum of every value that appears in the kernel: A/B fragment regs and epilogue (bias / scale) **reuse** the same physical registers across phases.

**Phases (TTGIR + LLVM O3):**

```text
K-loop:     acc + A/B MMA fragments (+ addrs/masks)
              bias / XScale / WScale NOT loaded yet

after loop: local_dealloc %x / %w (SMEM)
            A/B fragment regs die → free for epilogue

epilogue:   acc still live
            + load bias, XScale, WScale
            + scale splat, bias broadcast, addf, fp_to_fp, store
```

**Per-thread footprint (from `03_after_llvm_optimize_module_O3.llir`, one `BLOCK_K` body):**

**Table: Per-thread register footprint (A/B/out/bias/scale)**

| Value | Source | Per-thread regs | Per-warp regs (`×32`) | Live when | Notes |
| ----- | ------ | --------------- | --------------------- | --------- | ----- |
| Out / acc | `128×256` fp32 `#mma` → warp `64×64` | **128** | **4096** | K-loop + epilogue | `4096/32`; dominates and spans both phases |
| A fragments | 16× `ldmatrix.x4` | **64** | **2048** | K-loop only | warp A `64×128` fp8 → `256 B/thread` |
| B fragments | 256× `load i8` → 64× `bitcast <4xi8>→i32` | **64** | **2048** | K-loop only | warp B `128×64` fp8 → `256 B/thread` |
| A+B together | — | **128** | **4096** | K-loop only | this LLIR loads **all** A/B for the K-tile before 128× `mma.sync` (no micro-tile reuse) |
| Bias broadcast | TTGIR `broadcast` → `128×256` (matches out tile) | **128** | **4096** | epilogue only | same shape as Out/acc; **reuses A/B-freed regs** after K-loop (not stacked on K-loop A/B peak) |
| Scale broadcast | TTGIR `splat`/`mulf` → `128×256` (matches out tile) | **128** | **4096** | epilogue only | same shape as Out/acc; **reuses A/B-freed regs**; `fmul` can also overwrite acc in place |
| **Peak K-loop** | `Out + A/B` | **256** | **8192** | K-loop | `128 + 128`; bias/scale not live |
| **Peak epilogue** | `Out + Bias` (Scale via in-place / same A/B pool) | **256** | **8192** | epilogue | Bias takes freed A/B (`128`); Scale reuses Out or same freed pool — **not** `128+128+128` |
| **Peak overall (w/ reuse)** | `max(K-loop, epilogue)` | **256** | **8192** | — | naive sum `128*4=512` is wrong; NCU **255**/thread ≈ this peak (+ addrs/masks) |

```text
naive (no reuse):     Out+A+B+Bias+Scale = 128+64+64+128+128 = 512 / thread
with phase reuse:     max(Out+A/B, Out+Bias) = max(256, 256) = 256 / thread
                      (= 8192 / warp; CTA ≈ 256*256 = 65536 ≈ full 64K → 1 CTA/SM)
NCU compiled:         255 regs/thread
```

**Bias broadcast (TTGIR epilogue):**

```text
%bias   = tt.load … : tensor<256xf32, #blocked>
%acc_82 = ttg.convert_layout %bias → #ttg.slice<{dim=0, parent=#mma}>
%acc_83 = tt.expand_dims %acc_82 {axis=0} → tensor<1x256xf32, #mma>
%acc_84 = tt.broadcast %acc_83 → tensor<128x256xf32, #mma>
%acc_85 = arith.addf %acc_81, %acc_84   // scaled acc + bias
```

### 3.4 Warp MMA grain (instruction-level loops)

Hardware instruction (PTX): `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` — grain **16×8×32**.

Per warp, for **one** K-tile (`BLOCK_K=128`), covering its `64×64` accumulator:

```text
#M = 64 / 16 = 4
#N = 64 / 8  = 8
#K = 128 / 32 = 4
mma_per_warp_per_K_tile = 4 * 8 * 4 = 128
```

PTX check: the inner K-loop body (`$L__BB0_3` … branch back) contains **exactly 128** `mma.sync…m16n8k32` instructions (fully unrolled M/N/K micro-tiles for one `tt.dot`). Outer loop runs **60** times over K → `128 * 60` MMA issues per warp for the full reduction.

**Note on K:** warps do **not** partition K among themselves in parallel; every warp walks the same K tiles for its own `(M,N)` subtile. The “×4 on K” is sequential micro-tiling inside the warp’s MMA expansion of one `128`-wide K block.

---

## 4. Profiler Analysis

### Nsight Systems

```bash
export PATH="/Volumes/case_sensitive_workspace/venv/bin:$PATH"
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --cudabacktrace=none \
  --stats=true \
  -o workspace/profile_01/test_matmul_nsys \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
    python/triton_kernels/tests/test_matmul.py::test_op -v \
  > workspace/profile_01/test_matmul_nsys.log 2>&1
```

**Target kernel (nsys `cuda_gpu_kern_sum`):** `_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`

| Item | Value |
| ---- | ----- |
| Launches | **1** |
| GPU time | **40.781553 ms** (`40,781,553` ns) |
| Stability (10×) | mean **40.822 ms**, CV **0.14%** |

**Tensor-Core utilization** (vs Peak_FP8_2D @ 2.40 GHz = **708 TFLOPS**, FLOPs_2D = `2.577e12`):

```text
Achieved_FP8 = FLOPs_2D / t = 2.577e12 / 40.781553e-3 = 63.19 TFLOPS
TC_util      = Achieved_FP8 / Peak_FP8_2D = 63.19 / 708 ≈ 8.9%

# equivalent: t_2D_theo / t_meas = 3.64 ms / 40.782 ms ≈ 8.9%
```



### Nsight Compute

Key options (stall-capable; not `--set basic` alone): `--kernel-name _matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`, `--launch-count 1`, sections `LaunchStats` / `Occupancy` / `SpeedOfLight` / `WorkloadDistribution` / `SchedulerStats` / `WarpStateStats` / `ComputeWorkloadAnalysis` (or `--set full`). Needs privileged / counter access. Optional PM Sampling: `--section PmSampling` + tensor/`l1tex` metrics (this kernel uses `cp.async`/L1TEX, not TMA).

**SOL / occupancy:** NCU duration **~46.8 ms** (replay; prefer nsys for wall time). DRAM only **~6.3%**; SM/Memory SOL **~37%**. Theoretical occupancy **16.7%** (regs + SMEM → **1 CTA/SM**). Schedulers idle hard: **No Eligible ~89.6%**, eligible warps/scheduler **~0.16** with only **~2** active warps vs max **12** — primary concurrency problem.

**Warp stalls:** Dominated by **`math_pipe_throttle` (~47%)**, then **`mio_throttle` (~20%)** and **`wait` (~14%)**. Memory-side stalls (`long_scoreboard` ~5%, `barrier` ~3%, `lg_throttle` ~0.1%) are small — not DRAM-bound; issue latency is pipe / MIO under low warp concurrency.

### IR dumps (compiler pass snapshots)

Dumped under `workspace/profile_01/irs/` after the NVIDIA backend points aligned with [`compiler.py` on `perf_ana`](https://github.com/whutsunxu/triton/blob/perf_ana/third_party/nvidia/backend/compiler.py):

**Table: IR dump files by compiler pass**

| File prefix | After pass |
| ----------- | ---------- |
| `01_after_ttir_add_loop_unroll.ttir__…` | `passes.ttir.add_loop_unroll` |
| `02_after_ttnvgpuir_add_lower_mma.ttgir__…` | `nvidia.passes.ttnvgpuir.add_lower_mma` |
| `03_after_llvm_optimize_module_O3.llir__…` | `llvm.optimize_module(..., OPTIMIZE_O3)` |
| `04_after_llvm_translate_to_asm.ptx__…` | `llvm.translate_to_asm(...)` |

See `irs/MANIFEST.txt`. Env: `TRITON_PERF_IR_DUMP=…/irs TRITON_ALWAYS_COMPILE=1`.

---

## 5. Summary

**Perf problems**

1. **No ping-pong SMEM buffer** — `num_stages=2` but IR alloc is `1×…` (only stage index 0); need to enable true double-buffering (`2×…` / effective ping-pong).
2. **Too few live warps per scheduler** — only **~2** eligible/active warps per warp scheduler vs HW max **12** (`<<< 12`); need to tune (tile / regs / occupancy / multi-CTA) to raise concurrency.
