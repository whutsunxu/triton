# MLP-MoE EP=1 Experiment Analysis

## 1. Platform

### Software


| Component              | Version                             | Source                                                        |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------- |
| NVIDIA Driver          | 595.71.05                           | `nvidia-smi`                                                  |
| CUDA (driver-reported) | 13.2                                | `nvidia-smi`                                                  |
| CUDA Toolkit (`nvcc`)  | 13.0, V13.0.48                      | `nvcc --version`                                              |
| PyTorch                | 2.12.0+cu130 (built with CUDA 13.0) | `venv` (`torch.__version__`, `torch.version.cuda`)            |
| Triton                 | 3.7.0                               | workspace tree `python/triton/__init__.py` (via `PYTHONPATH`) |




### Hardware


| Item               | Value                                     |
| ------------------ | ----------------------------------------- |
| GPU                | NVIDIA GeForce RTX 5060 Ti                |
| SKU / VRAM         | **16 GB** GDDR7 (`nvidia-smi`: 16311 MiB) |
| Architecture       | Blackwell (consumer GB206-class)          |
| CUDA Cores         | 4608                                      |
| Boost clock (ref.) | 2.57 GHz (NVIDIA product page)            |
| Memory interface   | 128-bit GDDR7                             |




### Peak performance (RTX 5060 Ti 16GB)

Figures below are **theoretical peaks** for the 16GB SKU. Memory bandwidth is from NVIDIA’s published 5060-family specs. FLOPS use the common Blackwell consumer model: CUDA-core FMA rate for “1D”, and 5th-gen Tensor Core dense matrix rates for “2D”. Sparse Tensor peaks (≈2× dense) are noted but not used as the primary roof.

**Calculation basis (CUDA FP32):**

$$
\mathrm{FP32}_{1\mathrm{D}} \approx N_{\mathrm{cores}} \times f_{\mathrm{boost}} \times 2
= 4608 \times 2.57\,\mathrm{GHz} \times 2 \approx 23.7\,\mathrm{TFLOPS}
$$

Tensor-core dense rates on this SKU are typically quoted as **≈2×** that for BF16/FP16 and **≈4×** for FP8 (dense). Sources: [NVIDIA RTX 5060 Family](https://www.nvidia.com/en-eu/geforce/graphics-cards/50-series/rtx-5060-family/) (bandwidth, clocks, core count); aggregated peak tables (e.g. VideoCardz / TechPowerUp-style listings, WareDB) for Tensor TFLOPS.

| Metric | Peak | Meaning |
| ------ | ---- | ------- |
| Memory bandwidth | **448 GB/s** | GDDR7 peak (`128-bit × 28 Gbps` class); NVIDIA published for 5060 Ti |
| **2D** BF16 (Tensor Core, dense) | **≈47.4 TFLOPS** | Matrix / Tensor Core MMA peak (BF16) |
| **2D** FP8 (Tensor Core, dense) | **≈94.8 TFLOPS** | Matrix / Tensor Core MMA peak (FP8, e.g. E4M3) |
| **2D** FP32 (Tensor Core) | **N/A** (no dense FP32 Tensor MMA on GeForce); TF32 Tensor ≈ **23.7 TFLOPS** if counted separately | True FP32 matrix Tensor path not applicable; TF32 is the closest Tensor float path |
| **1D** BF16 (CUDA core) | **≈23.7 TFLOPS** | Elementwise / non-Tensor path; same FMA rate as FP32 on modern GeForce |
| **1D** FP8 (CUDA core) | **N/A** | No meaningful CUDA-core FP8 ALU peak; FP8 is Tensor-Core–oriented |
| **1D** FP32 (CUDA core) | **≈23.7 TFLOPS** | Standard CUDA-core FMA peak |
| **Phys. 2D dens.** (BF16) | **105.8 ops/B** | $47.4\,\mathrm{TFLOPS} / 448\,\mathrm{GB/s}$; roofline knee |
| **Phys. 2D dens.** (FP8) | **211.6 ops/B** | $94.8\,\mathrm{TFLOPS} / 448\,\mathrm{GB/s}$; roofline knee |
| **Phys. 1D dens.** (CUDA core) | **52.9 ops/B** | $23.7\,\mathrm{TFLOPS} / 448\,\mathrm{GB/s}$; roofline knee |

**Notes**

- 8GB vs 16GB 5060 Ti share the same compute/bandwidth class in NVIDIA’s published table; this machine is the **16GB** SKU.
- “2D FLOPS” ≈ Tensor Core / dense GEMM-style throughput; “1D FLOPS” ≈ CUDA-core / elementwise throughput.
- Physical calculation density $\kappa = \mathrm{Peak\,FLOPS} / \mathrm{Peak\,BW}$ [ops/byte] is the roofline knee intensity.
- With structured sparsity, Tensor BF16/FP8 peaks are often advertised at roughly **2×** the dense numbers above (≈94.8 / ≈189.6 TFLOPS). Roofline work for dense MoE matmuls should use the **dense** column unless sparsity is explicitly enabled.

---



## 2. Test Case



### Script and launch


| Item                                      | Value                                                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Benchmark script                          | `/Volumes/case_sensitive_workspace/triton/python/triton_kernels/bench/bench_mlp.py`                          |
| Launch (from `work_space/run_command.sh`) | `torchrun --nproc-per-node=1 …/bench_mlp.py`                                                                 |
| Expert parallelism                        | **EP = 1** (`EP = world_size`, `nproc=1`)                                                                    |
| Scenario exercised                        | FP8×FP8 dense baseline (`roofline_mlp` with `x_dtype=fp8`, `w_dtype=fp8`; other MX4 scenarios commented out) |
| Run name                                  | `gpt-oss-x2`                                                                                                 |




### Model / workload defaults (`bench_mlp.py`)


| Parameter                      | Value                                | Notes                                      |
| ------------------------------ | ------------------------------------ | ------------------------------------------ |
| `batch_per_expt`               | 64                                   | intensity proxy / roofline sweep value     |
| Tokens (`batch`)               | **2048**                             | `64 × 128 / 4`                             |
| `dim1` (d_model / hidden)      | **1440**                             |                                            |
| `dim2` (FC1 width, pre-SwiGLU) | **2880**                             | SwiGLU `reduction_n=2` → intermediate 1440 |
| `n_expts_tot`                  | **128**                              |                                            |
| `n_expts_act` (topk)           | **4**                                |                                            |
| Activation dtype (expert path) | **FP8 E4M3** (`torch.float8_e4m3fn`) | gate path uses BF16 copy of activations    |
| Weight dtype (FC1/FC2)         | **FP8 E4M3**                         | gate weights remain BF16                   |
| EP / ranks                     | **1 / 1**                            | single GPU, no cross-rank EP traffic       |




### Operator pipeline (EP=1)

Operators executed inside `run_mlp()` (warmup + 1 profiled iter). Shapes match `work_space/op-triton_kernel.list` from the EP=1 run.


| #   | Operator                                               | Role                                                      | Key tensors / parameters                                                                                                            |
| --- | ------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Gate matmul** (`matmul`)                             | Router logits                                             | Act `[2048, 1440]` BF16 × `wg` `[1440, 128]` BF16 + `bg` `[128]` FP32 → logits `[2048, 128]` BF16                                   |
| 2   | **Top-k** (`topk`)                                     | Select experts + softmax; builds dispatch/combine indices | Input `[2048, 128]` BF16; `n_expts_act=4`; `active_indx` `[2048, 4]` int16; `dispatch_indx` / `combine_indx` `[8192]` int32         |
| 3   | `convert_dp_to_ep`                                     | Token→expert-sorted EP-local layout                       | `x` `[2048, 1440]` FP8; EP=1 → `y_ep` `[8192, 1440]` FP8 (`2048×4` active slots); `expt_map` / `boolmask` `[1, 128]`                |
| 4   | **FC1 matmul + fused SwiGLU** (`matmul` + `swiglu_fn`) | First expert GEMM                                         | Act `[8192, 1440]` FP8 × `w1` `[128, 1440, 2880]` FP8 + `b1` `[128, 2880]` FP32; ragged metadata; fused SwiGLU → `[8192, 1440]` FP8 |
| 5   | **FC2 matmul** (`matmul`)                              | Second expert GEMM                                        | Act `[8192, 1440]` FP8 × `w2` `[128, 1440, 1440]` FP8 + `b2` `[128, 1440]` FP32; ragged metadata → `[8192, 1440]` FP8               |
| 6   | `convert_ep_to_dp`                                     | Expert-sorted → token-sorted DP-local                     | `y_ep` `[8192, 1440]` FP8 → DP-local tokens with top-k expert outputs                                                               |
| 7   | **Reduce** (`reduce`)                                  | Weighted combine over top-k                               | View `[-1, 4, 1440]` then `reduce(dim=1)` → `[2048, 1440]`                                                                          |


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

| File                      | Role                                            |
| ------------------------- | ----------------------------------------------- |
| `nsys_bench_mlp.nsys-rep` | Nsight Systems report                           |
| `nsys_bench_mlp.sqlite`   | SQLite backing store for the report             |
| `nsight_perf.log`         | Captured nsys console output + `--stats` tables |

Bench line from the profiled run: `batch_per_expt: 64 | MS: 3.06 | TFLOPS: 33.55 | TBPS: 0.28`. Full `cuda_gpu_kern_sum` is in `nsight_perf.log` (includes init/warmup as well as Triton MoE kernels such as `_p_matmul_NNT_fp8…` / `_convert_*` / `_topk_forward`).

### Operator timing window (2nd `run_mlp`)

nsys absolute time **4.812189 s – 4.816590 s** (stream 7). Kernel times below are GPU durations from `nsys_bench_mlp.sqlite` for that window.

#### Platform peaks (from §1)

| Peak | Value | Used for |
|------|-------|----------|
| Memory bandwidth | **448 GB/s** | Effective BW % |
| **2D** BF16 Tensor | **≈47.4 TFLOPS** | Gate matmul FLOPS util |
| **2D** FP8 Tensor | **≈94.8 TFLOPS** | FC1 / FC2 FLOPS util |
| **1D** BF16/FP32 CUDA-core | **≈23.7 TFLOPS** | Elementwise / epilogue roof |
| **Phys. 2D dens.** (BF16) | **105.8 ops/B** | $47.4\,\mathrm{TF}/448\,\mathrm{GB/s}$ |
| **Phys. 2D dens.** (FP8) | **211.6 ops/B** | $94.8\,\mathrm{TF}/448\,\mathrm{GB/s}$ |
| **Phys. 1D dens.** | **52.9 ops/B** | $23.7\,\mathrm{TF}/448\,\mathrm{GB/s}$ |

#### FLOPs / bandwidth formulas

- **Operator theoretical calculation density:** $\mathrm{ops} / (\mathrm{input}+\mathrm{out})$ traffic [ops/byte]. For matmul, traffic $= \mathrm{size}(A)+\mathrm{size}(B)+\mathrm{size}(C)$ (activations + weights + output). **2D ops** $= 2MNK$; **1D ops** = CUDA-core epilogue work (bias, SwiGLU, reduce adds, …).
- **2D bound:** compare op 2D dens. to phys. 2D dens. (knee). Op dens. **>** knee → **compute-bound**; op dens. **<** knee → **bandwidth-bound**.
- **Effective bandwidth:** $\mathrm{BW}_{\mathrm{eff}} = (\mathrm{input}+\mathrm{out}) / t_{\mathrm{kernel}}$. Values **above DRAM peak** imply L2/L1 reuse.
- **FLOPS utilization (matmul only):** $\mathrm{Util} = (2MNK/t) / \mathrm{Peak}_{2\mathrm{D}}(\mathrm{dtype})$.

#### Per-operator table (aligned with Operator pipeline)

| Op NO | Operator | Key tensors / parameters | Kernel time | Op 2D theor. dens. [ops/B] | 2D bound | Op 1D theor. dens. [ops/B] | Effective bandwidth | FLOPS utilization (matmul) |
| ----- | -------- | ------------------------ | ----------- | -------------------------- | -------- | -------------------------- | ------------------- | -------------------------- |
| 1 | **Gate matmul** (`matmul`) | Act `[2048, 1440]` BF16 × `wg` `[1440, 128]` BF16 + `bg` `[128]` FP32 → logits `[2048, 128]` BF16 | **24.448 µs** (`_matmul_NNT_fp32xbf16xbf16_128x128x64x2`) | **111.2** (= $2MNK$ / size(A+B+C)) | **Compute** (111.2 > 105.8 BF16 knee) | **0.039** (bias adds $MN$) | **277.8 GB/s** (62.0% of 448) | **65.1%** of 2D BF16 (30.88 / 47.4 TFLOPS) |
| 2 | **Top-k** (`topk`) | Input `[2048, 128]` BF16; `n_expts_act=4`; indices as in pipeline | **1.792 µs** (`_topk_forward`) | — | — | — (compare/select; not modeled) | **338.3 GB/s** (75.5% of 448) | — |
| 3 | `convert_dp_to_ep` | `x` `[2048, 1440]` FP8 → `y_ep` `[8192, 1440]` FP8 | **19.520 µs** (`_convert_dp_to_ep`) | **0** (move only) | **Bandwidth** | **0** | **755.4 GB/s** (>DRAM; L2 reorder) | — |
| 4 | **FC1 matmul + fused SwiGLU** | Act `[8192, 1440]` FP8 × `w1` `[128, 1440, 2880]` FP8 + `b1`; SwiGLU → `[8192, 1440]` FP8 | **1.917 ms** (`_p_matmul_…_swiglu`) | **122.6** (= $2 \cdot 8192 \cdot 2880 \cdot 1440$ / size(A+W+C)) | **Bandwidth** (122.6 < 211.6 FP8 knee) | **0.170** (bias + SwiGLU; see below) | **289.3 GB/s** (64.6% of 448) | **37.4%** of 2D FP8 (35.45 / 94.8 TFLOPS) |
| 5 | **FC2 matmul** (`matmul`) | Act `[8192, 1440]` FP8 × `w2` `[128, 1440, 1440]` FP8 → `[8192, 1440]` FP8 | **0.954 ms** (`_p_matmul_…_64x256x128x1`) | **117.6** (= $2 \cdot 8192 \cdot 1440 \cdot 1440$ / size(A+W+C)) | **Bandwidth** (117.6 < 211.6 FP8 knee) | **0.041** (bias adds $MN$) | **303.0 GB/s** (67.6% of 448) | **37.6%** of 2D FP8 (35.62 / 94.8 TFLOPS) |
| 6 | `convert_ep_to_dp` | `y_ep` `[8192, 1440]` FP8 → DP-local top-k outputs | **39.776 µs** (`_convert_ep_to_dp`) | **0** | **Bandwidth** | **0** | **593.1 GB/s** (>DRAM; cache) | — |
| 7 | **Reduce** (`reduce`) | `[-1, 4, 1440]` → `[2048, 1440]` | **7.776 µs** (`_reduce_forward`) | **0** | **Bandwidth** | **0.600** ($3 \cdot 2048 \cdot 1440$ adds / traffic) | **1896 GB/s** (>DRAM; tiny set) | — |

##### Gate matmul densities (worked example)

- Traffic $\mathrm{size}(A)+\mathrm{size}(B)+\mathrm{size}(C) = 2048\cdot1440\cdot2 + 1440\cdot128\cdot2 + 2048\cdot128\cdot2 = 6{,}791{,}168\,\mathrm{B}$.
- **2D ops** $= 2MNK = 2\cdot2048\cdot128\cdot1440 = 7.550\times10^8$.
- **Op 2D dens.** $= 7.550\times10^8 / 6.791\times10^6 =$ **111.2 ops/B** (slightly above phys. BF16 knee **105.8** → mildly compute-leaning on paper).
- **1D ops** ≈ bias broadcast-add $MN = 2048\cdot128$; dens. $= 2.62\times10^5 / 6.791\times10^6 =$ **0.039 ops/B** (≪ phys. 1D knee **52.9**).

##### FC1 matmul + fused SwiGLU — theoretical density from semantics

Fused kernel does **one Tensor-Core GEMM** then a **CUDA-core SwiGLU epilogue** on the GEMM result; only the reduced FP8 tensor is written.

1. **GEMM (2D)**
   - Shape: $M=8192$, $K=1440$, $N=2880$ (pre-SwiGLU width).
   - $\mathrm{FLOPs}_{2\mathrm{D}} = 2MNK = 2\cdot8192\cdot2880\cdot1440 = 6.795\times10^{10}$.
   - Traffic (fused, no full-$N$ store):
     $\mathrm{size}(A)=8192\cdot1440\cdot1$, $\mathrm{size}(W)=128\cdot1440\cdot2880\cdot1$, $\mathrm{size}(C)=8192\cdot1440\cdot1$ (post-SwiGLU)
     $\Rightarrow \mathrm{size}(A)+\mathrm{size}(W)+\mathrm{size}(C) = 554{,}434{,}560\,\mathrm{B}$.
   - **Op 2D dens.** $= 6.795\times10^{10} / 554.4\times10^6 =$ **122.6 ops/B** (below phys. FP8 knee **211.6** → BW-leaning vs FP8 roof).

2. **SwiGLU epilogue (1D)** — matches `compute_swiglu` / `swiglu_torch` with `reduction_n=2`, `alpha=1`, `limit=1`:
   Split GEMM row $[g, \ell]$ of length $N=2880$ into pairs $(g_i,\ell_i)$, $H=N/2=1440$ outputs:

$$
s = \frac{g}{1 + e^{-\alpha g}},\qquad y = s\cdot(\ell+1)\quad(\equiv \mathrm{fma}(s,\ell,s)).
$$

   Per output element, arithmetic counted here (clamps omitted as non-FLOP compares; flex `scale=1` muls omitted):
   - $1$ mul ($\alpha\cdot g$), $1$ exp, $1$ add, $1$ div, $1$ FMA ($=2$ FLOPs) → **6 FLOPs / out**.
   - Plus **bias add** on the full GEMM width before split: $M\cdot N$ adds.
   - $\mathrm{FLOPs}_{1\mathrm{D}} = M\cdot N + 6\cdot M\cdot H = 8192\cdot2880 + 6\cdot8192\cdot1440 = 9.437\times10^7$.
   - **Op 1D dens.** $= 9.437\times10^7 / 554.4\times10^6 =$ **0.170 ops/B** (≪ phys. 1D knee **52.9** → epilogue is cheap vs DRAM traffic of $W$).

3. **Roofline reading:** compare **op 2D dens. (122.6)** to **phys. 2D FP8 dens. (211.6)** for Tensor-Core bound vs BW-bound; compare **op 1D dens. (0.170)** to **phys. 1D dens. (52.9)** for the SwiGLU tail (always BW-bound if run alone).

**Notes**

- Window also contains small helper kernels not listed as separate pipeline ops: bitmatrix / ragged-metadata (`~6.4 µs` total) between top-k and `convert_dp_to_ep`, plus `_remap_ragged_tensor_metadata` (**3.840 µs**) after `convert_dp_to_ep`, and a short early `_reduce_forward` (**2.144 µs**) between gate and top-k.
- FC1/FC2 dominate: **1.917 + 0.954 = 2.871 ms** of the **~4.40 ms** graph (gate→final reduce).
- After **4.816590 s**, remaining GPU time is mostly cuBLAS roof calibration / teardown (`nvjet_…_mma_…`), not more `run_mlp` iterations.

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

**Result:** exit code 1, `ERR_NVGPUCTRPERM` — no `.ncu-rep` written. Host has `RmProfilingAdminOnly=1` and the container lacks `CAP_SYS_ADMIN`. Details and host fixes: `ncu_ERR_NVGPUCTRPERM.md`. Logs: `ncu_bench_mlp.log`, `ncu_bench_mlp_basic.log`.

---



## 4. Summary

Nsight Systems profiling completed and artifacts are under `MLP-MOE-B64-2048-1440/` (`nsys_bench_mlp.*`, `nsight_perf.log`). On the 2nd `run_mlp` window (**4.812–4.816 s**), FC1/FC2 FP8 matmuls dominate (~2.87 ms) at ~**37%** of 2D FP8 peak and ~**65–68%** of DRAM BW; gate BF16 matmul reaches ~**65%** of 2D BF16 peak. Nsight Compute remains blocked by GPU performance-counter permissions (`ERR_NVGPUCTRPERM`); re-run `run_ncu.sh` after the host/container fix documented in `ncu_ERR_NVGPUCTRPERM.md`.