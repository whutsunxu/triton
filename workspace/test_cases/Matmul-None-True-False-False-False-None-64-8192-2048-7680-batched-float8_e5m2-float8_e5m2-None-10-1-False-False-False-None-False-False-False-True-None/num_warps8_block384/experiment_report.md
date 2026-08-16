# Experiment: `num_warps=8` → blockDim **384** (persistent `block_m=64`)

**Date:** 2026-08-16  
**Kernel:** `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1`  
**Folder:** `num_warps8_block384/` (this directory)  
**Parent case:** batched FP8 `8192×2048×7680`, `is_persistent=True`, tile `64×256×128`

Compared to sibling baseline in parent `experiment_report.md` (auto warps → **block 256**, **42.671 ms**).

---

## 1. How to reproduce

```bash
# Inside ir_dev container, repo root:
# temporarily force num_warps=8 in test_matmul.py:
#   @pytest.mark.parametrize("num_warps", [8])

export PYTHONPATH=python/triton_kernels
export PATH=/opt/nvidia/nsight-systems/2025.3.2/bin:/opt/nvidia/nsight-compute/2025.3.1:$PATH
VENV=/Volumes/case_sensitive_workspace/venv/bin
CASE=workspace/test_cases/Matmul-None-True-False-False-False-None-64-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None
OUT=$CASE/num_warps8_block384
KERN=_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1

# IR dump
TRITON_ALWAYS_COMPILE=1 TRITON_PERF_IR_DUMP=$PWD/$OUT/irs \
  $VENV/python -m pytest python/triton_kernels/tests/test_matmul.py::test_op -v

# Nsight Systems
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none --cudabacktrace=none --stats=true \
  -o $OUT/test_matmul_nsys \
  $VENV/python -m pytest python/triton_kernels/tests/test_matmul.py::test_op -v \
  >$OUT/test_matmul_nsys.log 2>&1

# Nsight Compute (stall-capable sections; needs --privileged)
ncu --force-overwrite --target-processes all --kernel-name $KERN --launch-count 1 \
  --section LaunchStats --section Occupancy --section SpeedOfLight \
  --section SpeedOfLight_HierarchicalTensorRooflineChart \
  --section WorkloadDistribution --section SchedulerStats --section WarpStateStats \
  --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Chart --section InstructionStats \
  -o $OUT/test_matmul_ncu \
  $VENV/python -m pytest python/triton_kernels/tests/test_matmul.py::test_op -v
```

**Pytest node id:** `…[8-True-…-None-64-8192-2048-7680-batched-float8_e5m2-…]` — **PASSED**.

Restore tip after: `@pytest.mark.parametrize("num_warps", [4, 8] if is_hopper() else [None])`.

---

## 2. Launch & WS warp split

| Item | Value |
|------|------:|
| Grid | **(36, 1, 1)** |
| Block | **(384, 1, 1)** = **12 warps** |
| Regs / thread | **168** |
| Dyn SMEM | **99396 B** (~97 KiB) |
| PTX | `.reqntid 384` |

**Warp specialization (why 384 ≠ 8×32):**

| Partition | Role | Warps |
|-----------|------|------:|
| default | MMA `[1,8]` + epilogue + TMA store | **8** (`num_warps=8`) |
| partition0 | TMA loads A/B | **~4** (schedule-time); PTX `setmaxnreg` 240 vs 24 |
| **CTA total** | | **12** → **384 threads** |

Rule of thumb for this kernel: `block ≈ (num_warps + 4) × 32`.

**IR:** `#mma … warpsPerCTA = [1, 8]`, `#blocked1 … warpsPerCTA = [8]`.

---

## 3. Nsight Systems

| Item | Value |
|------|------:|
| Launches | 1 |
| GPU time | **33.346506 ms** (`33,346,506` ns) |

### vs siblings

| Build | Block | nsys GPU time | vs baseline 256 |
|-------|------:|--------------:|----------------:|
| Baseline (auto) | 256 | **42.671 ms** | — |
| **This (`num_warps=8`)** | **384** | **33.347 ms** | **−21.9%** |
| `num_warps=16` | 640 | 38.657 ms | −9.4% |

**Verdict:** best among {256, 384, 640} in the warp sweep.

Roofline (same FLOPs `2.577e12`, Peak FP8 2D ≈708 TFLOPS):

```text
Achieved ≈ 2.577e12 / 33.347e-3 ≈ 77.3 TFLOPS
TC_util  ≈ 77.3 / 708 ≈ 10.9%
```

---

## 4. Nsight Compute (summary)

| Metric | This (384) | Baseline ncu (256) |
|--------|----------:|-------------------:|
| Occupancy | **25.0%** (12/48 warps) | 16.7% (8/48) |
| Active warps / scheduler | **3.00** | 2.00 |
| Eligible / scheduler | **0.20** | 0.12 |
| No eligible | 84.3% | 87.9% |
| Compute (SM) SOL | **43.4%** | ~34% |
| Memory SOL | **48.1%** | ~34% |
| DRAM SOL | 5.6% | ~4.3% |
| Tensor pipe (elapsed) | **37.2%** | ~28.5% |
| Issue slots busy | 15.6% | ~12% |

### Stall mix (Warp State, inst units)

| Stall | 384 | 256 baseline |
|-------|----:|-------------:|
| **Math pipe throttle** | **6.52** | 3.06 |
| Barrier | 3.97 | **4.28** |
| Long scoreboard | 2.77 | 3.32 |
| Wait | 2.30 | 2.79 |
| MIO throttle | 1.65 | 0.89 |

More MMA warps improve eligibility/SOL and cut barrier/scoreboard share; **math_pipe** becomes the largest stall (tensor pipe fed harder).

---

## 5. Artifacts in this folder

| File | Contents |
|------|----------|
| `irs/` | IR dumps (MMA `[1,8]`, PTX `.reqntid 384`) |
| `test_matmul_nsys.{nsys-rep,sqlite,log}` | Nsight Systems |
| `test_matmul_ncu.ncu-rep` + `_details/_summary/_raw.txt` | Main NCU |
| `test_matmul_ncu_tma_tensor.ncu-rep` + `_raw.txt` | Focused TC/TMA/stall metrics |
| `cupti.txt` | Launch row (block 384, 33.35 ms) |
| `pytest_compile.log` | Correctness **PASSED** |

Parent sweep context: `../warp_block_dim_sweep_summary.md`.
