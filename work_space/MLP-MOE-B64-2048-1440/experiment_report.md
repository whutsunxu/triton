# MLP-MoE EP=1 Experiment Analysis

## 1. Platform

### Software

| Component | Version | Source |
|-----------|---------|--------|
| NVIDIA Driver | 595.71.05 | `nvidia-smi` |
| CUDA (driver-reported) | 13.2 | `nvidia-smi` |
| CUDA Toolkit (`nvcc`) | 13.0, V13.0.48 | `nvcc --version` |
| PyTorch | 2.12.0+cu130 (built with CUDA 13.0) | `venv` (`torch.__version__`, `torch.version.cuda`) |
| Triton | 3.7.0 | workspace tree `python/triton/__init__.py` (via `PYTHONPATH`) |

### Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| SKU / VRAM | **16 GB** GDDR7 (`nvidia-smi`: 16311 MiB) |
| Architecture | Blackwell (consumer GB206-class) |
| CUDA Cores | 4608 |
| Boost clock (ref.) | 2.57 GHz (NVIDIA product page) |
| Memory interface | 128-bit GDDR7 |

### Peak performance (RTX 5060 Ti 16GB)

Figures below are **theoretical peaks** for the 16GB SKU. Memory bandwidth is from NVIDIA’s published 5060-family specs. FLOPS use the common Blackwell consumer model: CUDA-core FMA rate for “1D”, and 5th-gen Tensor Core dense matrix rates for “2D”. Sparse Tensor peaks (≈2× dense) are noted but not used as the primary roof.

**Calculation basis (CUDA FP32):**

\[
\text{FP32}_{\text{1D}} \approx N_{\text{cores}} \times f_{\text{boost}} \times 2
= 4608 \times 2.57\,\text{GHz} \times 2 \approx 23.7\,\text{TFLOPS}
\]

Tensor-core dense rates on this SKU are typically quoted as **≈2×** that for BF16/FP16 and **≈4×** for FP8 (dense). Sources: [NVIDIA RTX 5060 Family](https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5060-family/) (bandwidth, clocks, core count); aggregated peak tables (e.g. VideoCardz / TechPowerUp-style listings, WareDB) for Tensor TFLOPS.

| Metric | Peak | Meaning |
|--------|------|---------|
| Memory bandwidth | **448 GB/s** | GDDR7 peak (`128-bit × 28 Gbps` class); NVIDIA published for 5060 Ti |
| **2D** BF16 (Tensor Core, dense) | **≈47.4 TFLOPS** | Matrix / Tensor Core MMA peak (BF16) |
| **2D** FP8 (Tensor Core, dense) | **≈94.8 TFLOPS** | Matrix / Tensor Core MMA peak (FP8, e.g. E4M3) |
| **2D** FP32 (Tensor Core) | **N/A** (no dense FP32 Tensor MMA on GeForce); TF32 Tensor ≈ **23.7 TFLOPS** if counted separately | True FP32 matrix Tensor path not applicable; TF32 is the closest Tensor float path |
| **1D** BF16 (CUDA core) | **≈23.7 TFLOPS** | Elementwise / non-Tensor path; same FMA rate as FP32 on modern GeForce |
| **1D** FP8 (CUDA core) | **N/A** | No meaningful CUDA-core FP8 ALU peak; FP8 is Tensor-Core–oriented |
| **1D** FP32 (CUDA core) | **≈23.7 TFLOPS** | Standard CUDA-core FMA peak |

**Notes**

- 8GB vs 16GB 5060 Ti share the same compute/bandwidth class in NVIDIA’s published table; this machine is the **16GB** SKU.
- “2D FLOPS” ≈ Tensor Core / dense GEMM-style throughput; “1D FLOPS” ≈ CUDA-core / elementwise throughput.
- With structured sparsity, Tensor BF16/FP8 peaks are often advertised at roughly **2×** the dense numbers above (≈94.8 / ≈189.6 TFLOPS). Roofline work for dense MoE matmuls should use the **dense** column unless sparsity is explicitly enabled.

---

## 2. Test Case

### Script and launch

| Item | Value |
|------|-------|
| Benchmark script | `/Volumes/case_sensitive_workspace/triton/python/triton_kernels/bench/bench_mlp.py` |
| Launch (from `work_space/run_command.sh`) | `torchrun --nproc-per-node=1 …/bench_mlp.py` |
| Expert parallelism | **EP = 1** (`EP = world_size`, `nproc=1`) |
| Scenario exercised | FP8×FP8 dense baseline (`roofline_mlp` with `x_dtype=fp8`, `w_dtype=fp8`; other MX4 scenarios commented out) |
| Run name | `gpt-oss-x2` |

### Model / workload defaults (`bench_mlp.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `batch_per_expt` | 64 | intensity proxy / roofline sweep value |
| Tokens (`batch`) | **2048** | `64 × 128 / 4` |
| `dim1` (d_model / hidden) | **1440** | |
| `dim2` (FC1 width, pre-SwiGLU) | **2880** | SwiGLU `reduction_n=2` → intermediate 1440 |
| `n_expts_tot` | **128** | |
| `n_expts_act` (topk) | **4** | |
| Activation dtype (expert path) | **FP8 E4M3** (`torch.float8_e4m3fn`) | gate path uses BF16 copy of activations |
| Weight dtype (FC1/FC2) | **FP8 E4M3** | gate weights remain BF16 |
| EP / ranks | **1 / 1** | single GPU, no cross-rank EP traffic |

### Operator pipeline (EP=1)

Operators executed inside `run_mlp()` (warmup + 1 profiled iter). Shapes match `work_space/op-triton_kernel.list` from the EP=1 run.

| # | Operator | Role | Key tensors / parameters |
|---|----------|------|---------------------------|
| 1 | **Gate matmul** (`matmul`) | Router logits | Act `[2048, 1440]` BF16 × `wg` `[1440, 128]` BF16 + `bg` `[128]` FP32 → logits `[2048, 128]` BF16 |
| 2 | **Top-k** (`topk`) | Select experts + softmax; builds dispatch/combine indices | Input `[2048, 128]` BF16; `n_expts_act=4`; `active_indx` `[2048, 4]` int16; `dispatch_indx` / `combine_indx` `[8192]` int32 |
| 3 | **`convert_dp_to_ep`** | Token→expert-sorted EP-local layout | `x` `[2048, 1440]` FP8; EP=1 → `y_ep` `[8192, 1440]` FP8 (`2048×4` active slots); `expt_map` / `boolmask` `[1, 128]` |
| 4 | **FC1 matmul + fused SwiGLU** (`matmul` + `swiglu_fn`) | First expert GEMM | Act `[8192, 1440]` FP8 × `w1` `[128, 1440, 2880]` FP8 + `b1` `[128, 2880]` FP32; ragged metadata; fused SwiGLU → `[8192, 1440]` FP8 |
| 5 | **FC2 matmul** (`matmul`) | Second expert GEMM | Act `[8192, 1440]` FP8 × `w2` `[128, 1440, 1440]` FP8 + `b2` `[128, 1440]` FP32; ragged metadata → `[8192, 1440]` FP8 |
| 6 | **`convert_ep_to_dp`** | Expert-sorted → token-sorted DP-local | `y_ep` `[8192, 1440]` FP8 → DP-local tokens with top-k expert outputs |
| 7 | **Reduce** (`reduce`) | Weighted combine over top-k | View `[-1, 4, 1440]` then `reduce(dim=1)` → `[2048, 1440]` |

**Precision / layout notes (from run log):** gate uses `PrecisionConfig` with `allow_tf32=True`; expert FP8 weights use flex scale metadata (`b_microblock_size=32`). With EP=1, all 128 experts reside on the single rank (`n_expts_tot // EP = 128`).

---

## 3. Profiler Analysis

Commands live in `work_space/run_command.sh` (nsys + ncu) and `work_space/MLP-MOE-B64-2048-1440/run_ncu.sh` (ncu helper). All runs use the workspace venv on `PATH` and `TRITON_PROTON_DISABLE=1`.

### Nsight Systems (worked)

From repo root (`/Volumes/case_sensitive_workspace/triton`):

```bash
export PATH="/Volumes/case_sensitive_workspace/venv/bin:$PATH"

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas,mpi \
  --cuda-memory-usage=true \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --python-sampling=true \
  --python-sampling-frequency=1000 \
  --stats=true \
  --output=work_space/MLP-MOE-B64-2048-1440/nsys_bench_mlp \
  env PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py
```

**Artifacts** (under `work_space/MLP-MOE-B64-2048-1440/`):

| File | Role |
|------|------|
| `nsys_bench_mlp.nsys-rep` | Nsight Systems report |
| `nsys_bench_mlp.sqlite` | SQLite backing store for the report |
| `nsight_perf.log` | Captured nsys console output + `--stats` tables |

Bench line from the profiled run: `batch_per_expt: 64 | MS: 3.06 | TFLOPS: 33.55 | TBPS: 0.28`. Full `cuda_gpu_kern_sum` is in `nsight_perf.log` (includes init/warmup as well as Triton MoE kernels such as `_p_matmul_NNT_fp8…` / `_convert_*` / `_topk_forward`).

### Nsight Compute (blocked)

Light set attempted (`--set basic`):

```bash
# via run_ncu.sh (defaults NCU_SET=basic), or:
ncu \
  --set basic \
  --force-overwrite \
  --target-processes all \
  --kernel-name-base demangled \
  --export work_space/MLP-MOE-B64-2048-1440/ncu_bench_mlp_basic \
  env PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py
```

**Result:** exit code 1, **`ERR_NVGPUCTRPERM`** — no `.ncu-rep` written. Host has `RmProfilingAdminOnly=1` and the container lacks `CAP_SYS_ADMIN`. Details and host fixes: `ncu_ERR_NVGPUCTRPERM.md`. Logs: `ncu_bench_mlp.log`, `ncu_bench_mlp_basic.log`.

---

## 4. Summary

Nsight Systems profiling completed and artifacts are under `MLP-MOE-B64-2048-1440/` (`nsys_bench_mlp.*`, `nsight_perf.log`). Nsight Compute remains blocked by GPU performance-counter permissions (`ERR_NVGPUCTRPERM`); re-run `run_ncu.sh` after the host/container fix documented in `ncu_ERR_NVGPUCTRPERM.md`. Deeper kernel-level conclusions TBD pending a focused pass over the nsys report (and ncu once unblocked).
