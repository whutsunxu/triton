# Batched FP8 Matmul Experiment Analysis (persistent / TMA, `block_m=64`)

Target kernel: `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1`  
Profile dir: `workspace/Matmul-None-True-False-False-False-None-64-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/`  
Frame adapted from the persistent `block_m=128` sibling (`…-None-128-…/experiment_report.md`).  
**Nsight Compute was not run** for this capture (per request).

---

## 1. Platform

### Software

**Table: Software stack versions**

| Component              | Version                             | Source                                                        |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------- |
| NVIDIA Driver          | 580.126.09                          | `nvidia-smi`                                                  |
| CUDA (driver-reported) | 13.0                                | `nvidia-smi`                                                  |
| CUDA Toolkit (`nvcc`)  | 13.0, V13.0.48                      | `nvcc --version`                                              |
| PyTorch                | 2.12.0+cu130 (built with CUDA 13.0) | workspace `venv`                                              |
| Triton                 | 3.7.0 (`+git`, editable)            | `/Volumes/case_sensitive_workspace/triton`                    |
| Nsight Systems         | 2025.5.2                            | `nsys --version`                                              |

### Hardware

**Table: GPU hardware overview**

| Item               | Value                                     |
| ------------------ | ----------------------------------------- |
| GPU                | NVIDIA GeForce RTX 5060 Ti                |
| SKU / VRAM         | **16 GB** GDDR7 (`nvidia-smi`: 16311 MiB) |
| Architecture       | Blackwell (consumer), CC **12.0**         |
| SMs                | **36**                                    |
| L2 cache           | 32 MB                                     |
| Peak model clock   | **2.40 GHz** (same roof as sibling report) |
| Memory bandwidth   | **448 GB/s**                              |

**Table: Peak performance summary (@ 2.40 GHz)**

| Metric | Peak |
| ------ | ---- |
| Memory bandwidth | **448 GB/s** |
| **2D** FP8 (Tensor Core, dense) | **≈708 TFLOPS** |
| **2D** BF16 (Tensor Core, dense) | **≈354 TFLOPS** |
| Phys. 2D dens. (FP8) | **≈1580 ops/B** |

---

## 2. Test Case

### Script and launch

**Table: Test script and launch**

| Item | Value |
| ---- | ----- |
| Test | `python/triton_kernels/tests/test_matmul.py::test_op` |
| Active case | `Case(8192, 2048, 7680, "batched", "float8_e5m2", "float8_e5m2")` |
| Pytest node id | `…[None-True-False-False-False-None-64-8192-2048-7680-batched-float8_e5m2-float8_e5m2-…]` |
| Constraint change vs `block_m=128` persistent | **`block_m=64`** |
| Kernel selected | **`kernels._p_matmul`** (persistent TMA path) |
| Launch (smoke) | `PYTHONPATH=python/triton_kernels python -m pytest …::test_op -v` |
| Launch (nsys) | see §4 |
| Correctness | **PASSED** (`ir_dump.log` / `test_matmul_nsys.log`) |

### Workload parameters

**Table: Workload parameters**

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| `m` / `n` / `k` | **8192 / 2048 / 7680** | same problem as siblings |
| `mode` | **batched** | `n_slices=10` |
| Activation / weight | **FP8 E5M2** | |
| Tile (from kernel name) | **64×256×128×1** | `BM×BN×BK×splitK` |
| Layout tag | **NNN** | |
| **`is_persistent`** | **True** | TMA descriptors + persistent CTA loop |
| Outer loop attrs (TTIR) | `tt.flatten`, **`tt.warp_specialize`** | `_p_matmul.py` persistent tile loop |
| Fused SwiGLU | None | pure matmul + bias / flex scales |

### Tensor shapes and roofline

Same traffic / FLOPs model as the sibling reports:

| Quantity | Task amount | Theoretical time (@ peaks above) |
| -------- | ----------- | -------------------------------- |
| 2D FLOPs (FP8 TC) | `2*10*8192*2048*7680 = 2.577e12` | **3.64 ms** |
| Traffic A+B+C | **954.204 MB** | **2.13 ms** |
| Ideal lower bound | `max(t_2D, t_BW)` | **3.64 ms** (TC-bound) |

---

## 3. Parallelism and Tiling Decomposition

Sources: TTIR / TTGIR under `irs/`, PTX `04_after_llvm_translate_to_asm.ptx___p_matmul_…`, CUPTI launch row in `test_matmul_nsys.sqlite`.

### 3.1 CTA-level parallelism (persistent grid)

Launch (CUPTI):

```text
grid  = (36, 1, 1)     # = NUM_SMS
block = (256, 1, 1)    # = 8 warps  (WS: default + load partition)
regs/thread = 255
dynamic SMEM = 99396 B (~97.1 KiB); sharedMemoryExecuted = 102400 B
```

Unlike the non-persistent kernel (grid = all output tiles), each CTA walks a strided slice of the flattened tile space. With `BLOCK_M=64`:

```text
grid_m = ceil(8192/64) = 128
grid_n = ceil(2048/256) = 8
num_blocks = B * grid_m * grid_n = 10 * 128 * 8 = 10240
for block_id in range(program_id, num_blocks, NUM_SMS=36):  # flatten + WS
    decode (pid_z, pid_m, pid_n); K-loop over BLOCK_K; epilogue (+ TMA store)
```

So **36 persistent CTAs** cover all **10240** tiles (≈284 tiles/CTA on average) — **2×** the tile count of the `block_m=128` persistent run (5120 tiles).

### 3.2 Warp specialization (this path succeeds)

Outer persistent loop is marked `tt.warp_specialize`. After `add_warp_specialize`, IR contains a real **`ttg.warp_specialize`**:

- **default** partition: MMA / epilogue (`tt.dot`, bias/scale, TMA store)
- **partition0** `num_warps(2)`, `requestedRegisters = [24]`: TMA producers  
  `ttng.async_tma_copy_global_to_local` for A (`1×64×128` f8) and B (`1×128×256` f8)

PTX shows `.reqntid 256`, `.maxnreg 256`, `setmaxnreg` around **256** (default) / **24** (producer), and `cp.async.bulk.tensor.3d…mbarrier`.

Loads are **`tt.descriptor_load` / async TMA**, not `tt.load` — that is why WS partitioning fires here (unlike the non-persistent `_matmul` path).

### 3.3 MMA / SMEM tiling

Still consumer-Blackwell **MMAv2** (sm120):

```text
#mma = #ttg.nvidia_mma<{versionMajor = 2, …, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
PTX: mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
```

With `BLOCK_M=64`, the MMA layout shrinks to **1×4** warps (vs **2×4** for `BLOCK_M=128`).

**Table: CTA SMEM tile buffers (post-pipeline)**

| Buffer | IR alloc | Bytes (data, depth) |
| ------ | -------- | ------------------- |
| A tile | `!ttg.memdesc<2×64×128×f8E5M2, …>` | 2 × 8192 = **16384** |
| B tile | `!ttg.memdesc<2×128×256×f8E5M2, …>` | 2 × 32768 = **65536** |
| Y epilogue | `!ttg.memdesc<1×64×256×f8E5M2, …>` | 16384 |
| mbarriers / TMA side state | extra `2×1×i64` allocs + runtime | → total dyn SMEM **~97 KiB** (CUPTI) |

**Difference vs `block_m=128`:** here `num_stages=2` actually materializes a **2-deep** FP8 ping-pong (`2×64×128` / `2×128×256`). The 128-M sibling stayed at depth-**1** tile allocs under the same stage count.

### 3.4 K micro-tiling

Per K-tile (`BLOCK_K=128`), PTX contains **128** `mma.sync…m16n8k32` issues per warp body. Outer K extent ≈ `ceil(7680/128)=60` per output tile; each persistent CTA repeats that for every tile it owns.

---

## 4. Profiler Analysis

### Nsight Systems

```bash
export PYTHONPATH=python/triton_kernels
CASE=workspace/Matmul-None-True-False-False-False-None-64-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --cudabacktrace=none \
  --stats=true \
  -o "$CASE/test_matmul_nsys" \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
    python/triton_kernels/tests/test_matmul.py::test_op -v \
  >"$CASE/test_matmul_nsys.log" 2>&1
```

**Target kernel:** `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1`

| Item | Value |
| ---- | ----- |
| Launches | **1** |
| GPU time | **42.671134 ms** (`42,671,134` ns) |
| Grid / block | **(36,1,1)** / **(256,1,1)** |
| Regs / dyn SMEM | **255** / **99396 B** |

**Tensor-Core utilization** (Peak_FP8_2D = 708 TFLOPS, FLOPs_2D = `2.577e12`):

```text
Achieved_FP8 = 2.577e12 / 42.671134e-3 ≈ 60.39 TFLOPS
TC_util      = 60.39 / 708 ≈ 8.5%
# t_2D_theo / t_meas = 3.64 / 42.671 ≈ 8.5%
```

**Vs siblings (same problem):**

| Build | GPU time | TC util | Notes |
| ----- | -------- | ------- | ----- |
| Non-persistent `_matmul` | **~40.78 ms** | **~8.9%** | grid 5120, block 256 |
| Persistent `block_m=128` | **44.838 ms** | **8.1%** | grid 36, block 384, depth-1 SMEM |
| Persistent `block_m=64` (this) | **42.671 ms** | **8.5%** | grid 36, block 256, **depth-2** SMEM |

So `block_m=64` persistent is **~4.8% faster** than persistent-128, but still **~4.6% slower** than non-persistent on this GPU/problem.

### Nsight Compute

Collected 2026-08-16 under `ncu_20260816_142622/` (Nsight Compute **2025.3.1**, container `--privileged`).

See **`ncu_20260816_142622/ncu_bottleneck_summary.md`** for the full TC / TMA / warp-stall write-up.

| Artifact | Role |
| -------- | ---- |
| `test_matmul_ncu_main.ncu-rep` | SOL, TC roofline, Memory, Scheduler, WarpState, Compute, Instruction, Source |
| `test_matmul_ncu_pmsampling.ncu-rep` | `--set pmsampling` |
| `test_matmul_ncu_tma_tensor.ncu-rep` | TC/TMA pipe + stall % metrics |
| `test_matmul_ncu_tma_l1tex.ncu-rep` | TMA L1TEX load/store bytes |

**Headline stalls** (`*_per_warp_active.pct`): barrier **25.9%**, long_scoreboard **20.1%**, math_pipe_throttle **18.5%**, wait **16.9%**.  
Occupancy **16.7%** (regs **255**). Tensor pipe **28.5%**, TMA pipe **0.17%** (but **~25 GB** TMA loads — TMA not the limiter). SM/Mem SOL ~**34%** → latency-bound.

### IR dumps

Under `irs/` (see `irs/MANIFEST.txt`). Env: `TRITON_PERF_IR_DUMP=<dir> TRITON_ALWAYS_COMPILE=1` with dump hooks in `third_party/nvidia/backend/compiler.py`.

| File prefix | After / before pass |
| ----------- | ------------------- |
| `01_…ttir` | `add_loop_unroll` |
| `04a_…ttgir` | before `assign_latencies` |
| `04b_…ttgir` | before `add_pipeline` (after schedule + WS) |
| `05_…ttgir` | after `add_pipeline` |
| `02_…ttgir` | after `add_lower_mma` |
| `03_…llir` | LLVM O3 |
| `04_…ptx` | `translate_to_asm` |
| `mem_feature.log` | `--test-print-alignment` on 04a |

---

## 5. Summary

**What changed vs persistent `block_m=128`**

1. Tile is **64×256×128** → `grid_m=128`, **10240** output tiles (2× tile count).
2. Launch becomes **36×256** (8 warps) instead of **36×384** (12 warps); MMA layout **1×4** instead of **2×4**.
3. Pipeline materializes **2-deep** A/B SMEM (`2×64×128` / `2×128×256`); dyn SMEM rises to **~97 KiB**, regs/thread to **255**.
4. Measured **42.67 ms** (~8.5% TC util) — better than persistent-128, still behind non-persistent ~40.78 ms.

**Perf problems / open items**

1. **Still slower than non-persistent** despite successful WS + real 2-deep ping-pong; high register pressure (255) may limit occupancy.
2. **2× more tiles** per CTA (284 vs 142) increases persistent scheduling / epilogue overhead relative to compute.
3. **sm120 stays on MMAv2** (`mma.sync.m16n8k32`) — no tcgen05/MMAv5 path on this consumer GPU.
