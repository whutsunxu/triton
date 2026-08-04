# Batched FP8 Matmul Experiment Analysis (persistent / TMA)

Target kernel: `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`  
Profile dir: `workspace/Matmul-None-True-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/`  
Frame adapted from the non-persistent sibling report (`…-None-False-…-True-None/experiment_report.md`).  
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
| Pytest node id | `…[None-True-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-…]` |
| Constraint change vs sibling | **`is_persistent=True`** (was `False`) |
| Kernel selected | **`kernels._p_matmul`** (persistent TMA path) |
| Launch (smoke) | `PYTHONPATH=python/triton_kernels python -m pytest …::test_op -v` |
| Launch (nsys) | see §4 |
| Correctness | **PASSED** (`ir_dump.log` / `test_matmul_nsys.log`) |

### Workload parameters

**Table: Workload parameters**

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| `m` / `n` / `k` | **8192 / 2048 / 7680** | same problem as non-persistent |
| `mode` | **batched** | `n_slices=10` |
| Activation / weight | **FP8 E5M2** | |
| Tile (from kernel name) | **128×256×128×1** | `BM×BN×BK×splitK` |
| Layout tag | **NNN** | |
| **`is_persistent`** | **True** | TMA descriptors + persistent CTA loop |
| Outer loop attrs (TTIR) | `tt.flatten`, **`tt.warp_specialize`** | `_p_matmul.py` persistent tile loop |
| Fused SwiGLU | None | pure matmul + bias / flex scales |

### Tensor shapes and roofline

Same traffic / FLOPs model as the non-persistent report:

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
block = (384, 1, 1)    # = 12 warps  (WS: default + load partition)
regs/thread = 168
dynamic SMEM = 81980 B (~80.1 KiB); sharedMemoryExecuted = 102400 B
```

Unlike the non-persistent kernel (grid **5120** = all output tiles), each CTA walks a strided slice of the flattened tile space:

```text
num_blocks = B * grid_m * grid_n = 10 * 64 * 8 = 5120
for block_id in range(program_id, num_blocks, NUM_SMS=36):  # flatten + WS
    decode (pid_z, pid_m, pid_n); K-loop over BLOCK_K; epilogue (+ TMA store)
```

So **36 persistent CTAs** cover all **5120** tiles (≈142 tiles/CTA on average).

### 3.2 Warp specialization (this path succeeds)

Outer persistent loop is marked `tt.warp_specialize`. After `add_warp_specialize`, IR contains a real **`ttg.warp_specialize`**:

- **default** partition: MMA / epilogue (`tt.dot`, bias/scale, TMA store)
- **partition0** `num_warps(2)`, `requestedRegisters = [24]`: TMA producers  
  `ttng.async_tma_copy_global_to_local` for A (`1×128×128` f8) and B (`1×128×256` f8)

PTX shows `setmaxnreg` around **240** (default) / **24** (producer) and `cp.async.bulk.tensor.3d…mbarrier`.

Loads are **`tt.descriptor_load` / async TMA**, not `tt.load` — that is why WS partitioning fires here (unlike the non-persistent `_matmul` path).

### 3.3 MMA / SMEM tiling

Still consumer-Blackwell **MMAv2** (sm120):

```text
#mma = #ttg.nvidia_mma<{versionMajor = 2, …, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
PTX: mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32
```

**Table: CTA SMEM tile buffers (post-pipeline)**

| Buffer | IR alloc | Bytes (data) |
| ------ | -------- | ------------ |
| A tile | `!ttg.memdesc<1×128×128×f8E5M2, …>` | 16384 |
| B tile | `!ttg.memdesc<1×128×256×f8E5M2, …>` | 32768 |
| mbarriers / TMA side state | extra `1×1×i64` allocs + runtime | → total dyn SMEM **~80 KiB** (CUPTI) |

Same `num_stages=2` → depth-**1** data alloc pattern as the non-persistent case (no true 2-deep ping-pong of the FP8 tiles).

Warps in the default partition still own a **2×4** MMA layout over the `128×256` accumulator (each warp `64×64`).

### 3.4 K micro-tiling

Per K-tile (`BLOCK_K=128`), PTX contains **128** `mma.sync…m16n8k32` issues per warp body (same grain math as sibling: `4×8×4`). Outer K extent ≈ `ceil(7680/128)=60` per output tile; each persistent CTA repeats that for every tile it owns.

---

## 4. Profiler Analysis

### Nsight Systems

```bash
export PYTHONPATH=python/triton_kernels
CASE=workspace/Matmul-None-True-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None

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

**Target kernel:** `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`

| Item | Value |
| ---- | ----- |
| Launches | **1** |
| GPU time | **44.838187 ms** (`44,838,187` ns) |
| Grid / block | **(36,1,1)** / **(384,1,1)** |
| Regs / dyn SMEM | **168** / **81980 B** |

**Tensor-Core utilization** (Peak_FP8_2D = 708 TFLOPS, FLOPs_2D = `2.577e12`):

```text
Achieved_FP8 = 2.577e12 / 44.838187e-3 ≈ 57.47 TFLOPS
TC_util      = 57.47 / 708 ≈ 8.1%
# t_2D_theo / t_meas = 3.64 / 44.838 ≈ 8.1%
```

**Vs non-persistent sibling** (`_matmul_…`, grid 5120, block 256, ~40.78 ms, ~8.9% TC util): this persistent+WS+TMA build is **~10% slower** on this GPU/problem despite successful warp specialization — worth separate follow-up (SMEM depth-1, WS register split, persistent tile scheduling on 36 SMs).

### Nsight Compute

**Not collected** for this experiment.

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

**What changed vs non-persistent**

1. **`is_persistent=True`** selects `_p_matmul` with **TMA** loads/stores and a **persistent** CTA loop over tiles (`step=NUM_SMS`).
2. Outer loop **`tt.warp_specialize` + TMA** → real **`ttg.warp_specialize`** (producer warps for async TMA, compute default partition).
3. Launch becomes **36×384** (12 warps) instead of **5120×256** (8 warps).

**Perf problems / open items**

1. **Still no 2-deep FP8 ping-pong** — tile `local_alloc` remains `1×128×128` / `1×128×256` under `num_stages=2`.
2. **Measured slower than non-persistent** on this case (**44.84 ms** vs **~40.78 ms**); WS/TMA overhead and occupancy/reg split need NCU (or further nsys) to explain.
3. **sm120 stays on MMAv2** (`mma.sync.m16n8k32`) — no tcgen05/MMAv5 path on this consumer GPU.
