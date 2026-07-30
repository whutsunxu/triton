# Batched FP8 Matmul Experiment Analysis

Target kernel: `_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`  
Profile dir: `work_space` → `/Volumes/case_sensitive_workspace/triton/workspace/profile_01/`  
Frame adapted from [`perf_ana` MLP-MoE report](https://github.com/whutsunxu/triton/blob/perf_ana/work_space/MLP-MOE-B64-2048-1440/experiment_report.md).

---

## 1. Platform

### Software


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


| Item               | Value                                     |
| ------------------ | ----------------------------------------- |
| GPU                | NVIDIA GeForce RTX 5060 Ti                |
| SKU / VRAM         | **16 GB** GDDR7 (`nvidia-smi`: 16311 MiB) |
| Architecture       | Blackwell (consumer GB206-class), CC 12.0 |
| SMs                | 36 (`torch.cuda.get_device_properties`)  |
| L2 cache           | 32 MB                                     |
| CUDA Cores         | 4608                                      |
| Boost clock (ref.) | 2.57 GHz (NVIDIA product page)            |
| SM frequency (peak model) | **2.40 GHz** (NCU measured during this kernel) |
| Memory interface   | 128-bit GDDR7                             |




### Peak performance (RTX 5060 Ti 16GB)

Same roof model as the [MLP-MoE EP=1 report](https://github.com/whutsunxu/triton/blob/perf_ana/work_space/MLP-MOE-B64-2048-1440/experiment_report.md), but **FLOPS peaks use the profiled SM clock $f_{\mathrm{SM}}{=}2.40\,\mathrm{GHz}$** (not the 2.57 GHz marketing boost):

$$
\mathrm{FP32}_{1\mathrm{D}} \approx N_{\mathrm{cores}} \times f_{\mathrm{SM}} \times 2
= 4608 \times 2.40\,\mathrm{GHz} \times 2 \approx 22.1\,\mathrm{TFLOPS}
$$

Dense Tensor peaks follow the same scaling as before ($\mathrm{BF16}_{2\mathrm{D}}{\approx}2\times\mathrm{FP32}_{1\mathrm{D}}$, $\mathrm{FP8}_{2\mathrm{D}}{\approx}4\times\mathrm{FP32}_{1\mathrm{D}}$):

| Metric | Peak | Meaning |
| ------ | ---- | ------- |
| Memory bandwidth | **448 GB/s** | GDDR7 peak |
| **2D** BF16 (Tensor Core, dense) | **≈44.2 TFLOPS** | Matrix / Tensor Core MMA peak (BF16) @ 2.40 GHz |
| **2D** FP8 (Tensor Core, dense) | **≈88.5 TFLOPS** | Matrix / Tensor Core MMA peak (FP8) @ 2.40 GHz |
| **1D** FP32 / BF16 (CUDA core) | **≈22.1 TFLOPS** | Elementwise / epilogue roof @ 2.40 GHz |
| **Phys. 2D dens.** (FP8) | **197.5 ops/B** | $88.5\,\mathrm{TFLOPS} / 448\,\mathrm{GB/s}$ |
| **Phys. 1D dens.** (CUDA core) | **49.4 ops/B** | $22.1\,\mathrm{TFLOPS} / 448\,\mathrm{GB/s}$ |

(At the 2.57 GHz boost clock the same model would give ≈23.7 / 47.4 / 94.8 TFLOPS; those marketing-boost numbers are **not** used for utilization below.)

---

## 2. Test Case

### Script and launch


| Item | Value |
| ---- | ----- |
| Test | `python/triton_kernels/tests/test_matmul.py::test_op` |
| Active case (branch `matmul_perf_analysis`) | `Case(8192, 2048, 7680, "batched", "float8_e5m2", "float8_e5m2")` |
| Pytest node id (abbrev.) | `…-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-…` |
| Launch (smoke) | `PYTHONPATH=python/triton_kernels python -m pytest …::test_op -v` |
| Launch (nsys) | see §3 |
| Container | `ir_dev` (`jasonsun11/ir_dev:cuda13_v2`), `--gpus all --privileged` |




### Workload parameters


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


### Tensor shapes (batched)


| Tensor | Shape | Dtype | Bytes |
| ------ | ----- | ----- | ----- |
| A | `[10, 8192, 7680]` | FP8 | 629.15 MB |
| B | `[10, 7680, 2048]` | FP8 | 157.29 MB |
| C | `[10, 8192, 2048]` | FP8 | 167.77 MB |
| bias | `[10, 2048]` | FP32 | 0.082 MB |

**Launch geometry (nsys / CUPTI):** grid `(5120,1,1)`, block `(256,1,1)` (= 8 warps), 255 regs/thread, 48 KiB dynamic SMEM.  
Grid check: $\lceil M/128\rceil\cdot\lceil N/256\rceil\cdot n\_slices = 64\cdot 8\cdot 10 = 5120$.

---

## 3. Parallelism and Tiling Decomposition

Sources: TTGIR `irs/02_after_ttnvgpuir_add_lower_mma.ttgir___matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`, PTX `irs/04_after_llvm_translate_to_asm.ptx___…`, launch path in `triton_kernels/matmul.py` + `matmul_details/_common.py::compute_pids`.

### 3.1 Parallel split at CTA level (output / batch)

Problem shape (batched): $Y \in \mathbb{R}^{B \times M \times N}$ with $B{=}10$, $M{=}8192$, $N{=}2048$, reduction $K{=}7680$.

Launch (1D grid flattened to `(5120,1,1)`):

$$
\begin{aligned}
\mathrm{grid\_m} &= \lceil M / \mathrm{BLOCK\_M}\rceil = \lceil 8192/128\rceil = 64 \\
\mathrm{grid\_n} &= \lceil N / \mathrm{BLOCK\_N}\rceil = \lceil 2048/256\rceil = 8 \\
\mathrm{grid} &= B \cdot \mathrm{grid\_m} \cdot \mathrm{grid\_n} \cdot \mathrm{SPLIT\_K}
          = 10 \cdot 64 \cdot 8 \cdot 1 = \mathbf{5120}
\end{aligned}
$$

Each `pid = tl.program_id(0)` ∈ `[0, 5120)` is **one CTA**. For `mode=batched` (`RAGGED_DIMENSION is None`), `compute_pids` decodes:

| Logical id | Range | Role |
| ---------- | ----- | ---- |
| `pid_s` (`pid_z`) | $0..9$ | batch / slice index |
| `pid_m` | $0..63$ | M-tile index → rows `[pid_m·128, (pid_m+1)·128)` |
| `pid_n` | $0..7$ | N-tile index → cols `[pid_n·256, (pid_n+1)·256)` |
| `pid_k` | $0$ | `SPLIT_K=1` (no K-split across CTAs) |

So the **5120 CTAs tile the output volume** $(B,M,N)$; K is reduced **inside** each CTA by looping over `BLOCK_K=128` tiles (`k_tiles = 7680/128 = 60`).

Thread layout per CTA: **8 warps × 32 threads = 256 threads** (matches NCU `Block Size = 256`).

### 3.2 Parallel split at warp level (within one CTA tile)

MMA layout from TTGIR:

```text
#mma = #ttg.nvidia_mma<{…, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
```

Warps form a **2×4 grid** over the CTA output tile `128×256`:

| Warp coord `(w_m, w_n)` | Owns accumulator subtile |
| ---------------------- | ------------------------ |
| $w_m \in \{0,1\}$, $w_n \in \{0,1,2,3\}$ | $64 \times 64$ fp32 (`128/2` × `256/4`) |

All 8 warps **share** the CTA SMEM buffers for A/B; they do **not** each own a private `128×128` / `128×256` allocation. Parallelism is over the **output (accumulator) plane**: each warp `local_load`s the A rows / B cols needed for its `64×64` from the common SMEM tiles.

### 3.3 CTA-level tiling (SMEM vs registers)

| Buffer | Shape / dtype | Where | Evidence |
| ------ | ------------- | ----- | -------- |
| Input A (`%x`) | `1×128×128` f8E5M2 (stage depth 1) | **SMEM** | `ttg.local_alloc` → `#smem` |
| Weight B (`%w`) | `1×128×256` f8E5M2 | **SMEM** | `ttg.local_alloc` → `#smem` |
| Accumulator | `128×256` **fp32**, layout `#mma` | **Registers** | `scf.for` iter_arg `tensor<128x256xf32, #mma>`; **no** `local_alloc` for C |
| Epilogue store | `128×256` f8E5M2 | registers → **global** | `tt.fp_to_fp` then `tt.store` |

Confirmed: the CTA output tile lives in **registers** (MMA accumulator), not SRAM. SMEM holds only the current K-slice of A (`BLOCK_M×BLOCK_K`) and B (`BLOCK_K×BLOCK_N`).

### 3.4 Warp MMA grain (instruction-level loops)

Hardware instruction (PTX): `mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32` — grain **$16\times8\times32$**.

Per warp, for **one** K-tile (`BLOCK_K=128`), covering its `64×64` accumulator:

$$
\begin{aligned}
\#M &= 64/16 = 4 \\
\#N &= 64/8 = 8 \\
\#K &= 128/32 = 4 \\
\#\mathrm{mma/warp/K\text{-}tile} &= 4 \cdot 8 \cdot 4 = \mathbf{128}
\end{aligned}
$$

PTX check: the inner K-loop body (`$L__BB0_3` … branch back) contains **exactly 128** `mma.sync…m16n8k32` instructions (fully unrolled M/N/K micro-tiles for one `tt.dot`). Outer loop runs **60** times over K → $128 \times 60$ MMA issues per warp for the full reduction.

**Note on K:** warps do **not** partition K among themselves in parallel; every warp walks the same K tiles for its own `(M,N)` subtile. The “×4 on K” is sequential micro-tiling inside the warp’s MMA expansion of one `128`-wide K block.

---

## 4. Profiler Analysis

### Nsight Systems (worked)

From repo root inside `ir_dev`:

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

**Artifacts** (under `workspace/profile_01/`):

| File | Role |
| ---- | ---- |
| `test_matmul_nsys.nsys-rep` | Nsight Systems report |
| `test_matmul_nsys.sqlite` | SQLite backing store |
| `test_matmul_nsys.log` | Captured nsys console + `--stats` tables |

#### CUDA GPU kernel summary (full capture)

From `cuda_gpu_kern_sum` on `test_matmul_nsys.nsys-rep`:

| Time % | Total (ns) | Inst. | Name |
| ------ | ---------- | ----- | ---- |
| **54.8** | **40,781,553** | **1** | **`_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`** |
| 19.1 | 14,204,855 | 2 | `at::native::elementwise_kernel` (PyTorch helpers) |
| 10.9 | 8,093,059 | 2 | `neg_kernel_cuda` |
| 7.9 | 5,867,092 | 2 | `float8_copy_kernel_cuda` |
| 7.3 | 5,460,056 | 2 | `distribution_elementwise_grid_stride_kernel` |
| ~0 | ≪1 µs | — | tiny fill / RNG helpers |

The Triton matmul is a **single launch** and dominates GPU kernel time in this smoke profile (~55% of summed kernel time; absolute duration **40.782 ms**).

#### Stability (10× nsys, earlier `workspace/nsys_fluctuation/`)

| Stat | Value |
| ---- | ----- |
| mean | 40.822 ms |
| stdev | 0.058 ms (**CV 0.14%**) |
| range (max−min)/mean | **0.40%** |

Duration is highly stable across repeated nsys captures.

#### Roofline for the target kernel (nsys duration)

Formulas (same as MLP-MoE report):

- **2D ops** $= 2 \cdot B \cdot M \cdot N \cdot K$
- **Traffic** $= \mathrm{size}(A)+\mathrm{size}(B)+\mathrm{size}(C)$ (FP8; bias excluded from 2D traffic)
- **Op 2D dens.** $= (2BMNK) / \mathrm{traffic}$
- **Effective BW** $= \mathrm{traffic} / t$
- **FLOPS util.** $= (2BMNK/t) / \mathrm{Peak}_{2\mathrm{D}}(\mathrm{FP8})$

| Quantity | Value |
| -------- | ----- |
| Kernel time $t$ (nsys) | **40.781553 ms** |
| 2D FLOPs | $2\cdot10\cdot8192\cdot2048\cdot7680 = 2.577\times10^{12}$ |
| Traffic $A{+}B{+}C$ | **954.204 MB** |
| **Op 2D dens.** | **2700.7 ops/B** |
| Phys. 2D dens. (FP8) @ 2.40 GHz | 197.5 ops/B |
| **2D bound** | **Compute** (2700.7 ≫ 197.5) |
| **1D dens.** (bias adds $B\cdot M\cdot N$) | **0.176 ops/B** (≪ 49.4 → epilogue BW-cheap vs GEMM) |
| Achieved 2D FP8 | **63.19 TFLOPS** |
| **FLOPS utilization** | **71.4%** of **88.5 TFLOPS** (peak @ $f_{\mathrm{SM}}{=}2.40\,\mathrm{GHz}$) |
| Effective bandwidth | **23.4 GB/s** (**5.2%** of 448 GB/s) |

**Reading:** arithmetic intensity is far above the FP8 knee, so the kernel is **compute-bound on paper**. Measured effective DRAM traffic rate is low (~5% of peak), consistent with a Tensor-Core–heavy GEMM that reuses tiles in SMEM/L1/L2. Achieved **~71%** of dense FP8 Tensor peak at the measured SM clock (vs ~67% if the 2.57 GHz / 94.8 TFLOPS boost roof were used).

---

### Nsight Compute (worked; privileged container)

`--set basic` only includes LaunchStats / Occupancy / SpeedOfLight / WorkloadDistribution — **no** Warp State Statistics, Scheduler Statistics, or Compute Workload Analysis — so it **cannot** name dominant stalls (`lg_throttle`, `long_scoreboard`, `mio_throttle`, `barrier`, `math_pipe_throttle`, …).

**Fix:** re-profile with stall-capable sections (preferred over bare `basic`):

```bash
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels

ncu \
  --force-overwrite \
  --kernel-name _matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1 \
  --launch-count 1 \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section WorkloadDistribution \
  --section SchedulerStats \
  --section WarpStateStats \
  --section ComputeWorkloadAnalysis \
  -o workspace/.../test_matmul_ncu_stalls \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
    python/triton_kernels/tests/test_matmul.py::test_op -q \
  > workspace/.../test_matmul_ncu_stalls.log 2>&1
```

(Equivalent heavier option: `ncu --set full …`.)

**Artifacts:** `test_matmul_ncu_stalls.ncu-rep` / `.log`, `test_matmul_ncu_details.log`, `ncu_stall_breakdown.md` (also synced to `test_matmul_ncu.*`).

#### PM Sampling (intra-kernel Tensor / L1TEX timeline)

This kernel loads via **`cp.async` (SM async-copy / L1TEX)**, not TMA (`cp.async.bulk`). NCU does **not** expose a per-instruction event Gantt of `cp.async` vs `mma.sync`; the closest timeline is **PM Sampling** (metric samples over the kernel duration).

Discover valid PM metrics on the GPU first:

```bash
ncu --query-metrics-collection pmsampling | rg -i 'tensor|l1tex'
```

Capture (Tensor pipe + L1TEX; open **Details → PM Sampling** in the NCU GUI):

```bash
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels
OUT=workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None

ncu \
  --force-overwrite \
  --kernel-name _matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1 \
  --launch-count 1 \
  --section PmSampling \
  --metrics pmsampling:sm__pipe_tensor_cycles_active_realtime_v2.avg.pct_of_peak_sustained_elapsed,\
pmsampling:l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
  -o ${OUT}/test_matmul_ncu_pmsampling \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
    python/triton_kernels/tests/test_matmul.py::test_op -q \
  > ${OUT}/test_matmul_ncu_pmsampling.log 2>&1
```

Notes:

- Exact `pmsampling:…` metric names vary by arch (CC 12.0 / Blackwell); if a metric is rejected, pick a matching `sm__pipe_tensor_*` / `l1tex__*` name from `--query-metrics-collection pmsampling`.
- Prefer the GUI timeline; CLI raw timestamps may use a different origin than the GUI (GUI normalizes to the first sample / kernel window).
- Requires the same privileged / `RmProfilingAdminOnly=0` setup as other NCU counter collects.
- **Expected artifacts:** `test_matmul_ncu_pmsampling.ncu-rep`, `test_matmul_ncu_pmsampling.log` (not collected yet in this profile dir).

#### SOL / occupancy (stall-capable capture)

| Metric | Value |
| ------ | ----- |
| Duration (NCU SOL) | **46.82 ms** |
| DRAM Throughput | **~6.3%** |
| Memory / SM Throughput | **~37.4%** |
| Grid / Block | 5120 / 256 |
| Theoretical occupancy | **16.7%** (limited by registers + shared memory) |
| No Eligible (scheduler) | **89.6%** of cycles |
| Eligible warps / scheduler | **0.16** (of 2.00 active) |
| Warp cycles per issued inst | **19.28** |

#### Warp stall breakdown (verified present)

From `WarpStateStats` metrics `smsp__average_warps_issue_stalled_*_per_issue_active`:

| Stall reason | Avg warps (per issue-active) | Share of warp latency |
| ------------ | ---------------------------- | --------------------- |
| `math_pipe_throttle` | 9.1472 | **47.4%** |
| `mio_throttle` | 3.8479 | **20.0%** |
| `wait` | 2.7528 | **14.3%** |
| `long_scoreboard` | 0.9221 | **4.8%** |
| `barrier` | 0.5792 | **3.0%** |
| `not_selected` | 0.5377 | **2.8%** |
| `short_scoreboard` | 0.4492 | **2.3%** |
| `lg_throttle` | 0.0227 | **0.1%** |
| `drain` / `dispatch_stall` / `tex_throttle` | ≈0 | ≈0% |

**Dominant stall:** `math_pipe_throttle` (~47% of issue latency) — execution pipe / oversubscribed math pipeline; Tensor pipe is highest-utilized in Compute Workload Analysis (~31% active cycles). Secondary: `mio_throttle` (~20%), then `wait` (~14%). Memory-side `long_scoreboard` / `lg_throttle` / `barrier` are present but small.

#### Earlier `--set basic` stability (3 runs)

| Run | Duration | DRAM % | SM % |
| --- | -------- | ------ | ---- |
| 1 | 46.880 ms | 6.44 | 37.40 |
| 2 | 46.900 ms | 6.42 | 37.39 |
| 3 | 46.910 ms | 6.43 | 37.39 |

- **CV (duration):** 0.033%. NCU duration remains ~15% above nsys due to replay; use nsys for wall-clock GPU time.

### IR dumps (compiler pass snapshots)

Dumped under `workspace/profile_01/irs/` after the NVIDIA backend points aligned with [`compiler.py` on `perf_ana`](https://github.com/whutsunxu/triton/blob/perf_ana/third_party/nvidia/backend/compiler.py):

| File prefix | After pass |
| ----------- | ---------- |
| `01_after_ttir_add_loop_unroll.ttir__…` | `passes.ttir.add_loop_unroll` |
| `02_after_ttnvgpuir_add_lower_mma.ttgir__…` | `nvidia.passes.ttnvgpuir.add_lower_mma` |
| `03_after_llvm_optimize_module_O3.llir__…` | `llvm.optimize_module(..., OPTIMIZE_O3)` |
| `04_after_llvm_translate_to_asm.ptx__…` | `llvm.translate_to_asm(...)` |

See `irs/MANIFEST.txt`. Env: `TRITON_PERF_IR_DUMP=…/irs TRITON_ALWAYS_COMPILE=1`.

---

## 5. Summary

- **Workload:** batched FP8×FP8 matmul $B{=}10$, $M{=}8192$, $N{=}2048$, $K{=}7680$, tile **128×256×128**, kernel `_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`.
- **nsys:** single-launch GPU time **40.782 ms**; ~**71.4%** of dense **FP8 2D** peak at **2.40 GHz** (**63.2 / 88.5 TFLOPS**); op intensity **~2701 ops/B** → **compute-bound** vs FP8 knee **197.5**; effective DRAM BW only **~5%** of 448 GB/s (cache/SMEM reuse).
- **Repeatability:** nsys 10× CV **0.14%**; ncu 3× CV **0.033%**.
- **ncu (stall sections):** DRAM ~**6.3%**, SM/Memory SOL ~**37.4%**, occupancy ~**16.7%**. Dominant warp stall **`math_pipe_throttle` (47.4%)**, then **`mio_throttle` (20.0%)**; `long_scoreboard` 4.8%, `barrier` 3.0%, `lg_throttle` 0.1%. Requires `--privileged` for counters; do **not** rely on `--set basic` alone for stall naming. For an intra-kernel Tensor/L1TEX timeline use **`--section PmSampling`** (see §4); this kernel uses `cp.async`/L1TEX, not TMA.
- **Artifacts:** `test_matmul_nsys.*`, `test_matmul_ncu.*`, `irs/` under this profile dir; PM Sampling command prepared as `test_matmul_ncu_pmsampling.*` (run when collecting the timeline).
