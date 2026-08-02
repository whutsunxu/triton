# Smoke test + Nsight perf check (`matmul_perf_analysis`)

Branch: `matmul_perf_analysis`  
Repo: `/Volumes/case_sensitive_workspace/triton`  
Date: 2026-08-02

## 0. Git identity

Already configured for this environment:

- `user.name`: `whutsunxu`
- `user.email`: `sunxu-cfd@outlook.com`

## 1. Fetch / checkout latest branch

```bash
cd /Volumes/case_sensitive_workspace/triton
git fetch origin matmul_perf_analysis
git checkout matmul_perf_analysis
git pull --ff-only origin matmul_perf_analysis
```

Result: up to date at `092dd2441` (tracks `origin/matmul_perf_analysis`).

## 2. Dependencies + smoke test

```bash
# deps
/Volumes/case_sensitive_workspace/venv/bin/python -m pip install -r \
  python/test-requirements.txt

# smoke
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels
/Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
  python/triton_kernels/tests/test_matmul.py::test_op -v \
  | tee workspace/smoke_test_matmul.log
```

Active case: `Case(8192, 2048, 7680, "batched", "float8_e5m2", "float8_e5m2")`, `block_m=128`.

**Result:** 1 collected, **PASSED** (~4.8s). Log: `workspace/smoke_test_matmul.log`.

## 3. Nsight install + profiler vs `experiment_report.md`

### Install

```bash
# Nsight Systems (nsys); Nsight Compute (ncu) already present from CUDA toolkit
sudo apt-get install -y cuda-nsight-systems-13-1
```

Versions used: Nsight Systems **2025.5.2**, GPU RTX 5060 Ti (driver 590.48.01).

### Profile

```bash
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels
OUTDIR=workspace/rerun_nsys_YYYYMMDD_HHMMSS

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --cudabacktrace=none \
  --stats=true \
  -o "$OUTDIR/test_matmul_nsys" \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
    python/triton_kernels/tests/test_matmul.py::test_op -v \
  >"$OUTDIR/test_matmul_nsys.log" 2>&1
```

This run: `workspace/rerun_nsys_20260802_101250/`.

### Kernel time comparison

Target kernel: `_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`

| Source | GPU time |
| --- | --- |
| `experiment_report.md` baseline | **40.781553 ms** (`40,781,553` ns) |
| Report 10× mean | 40.822 ms (CV 0.14%) |
| This nsys rerun | **40.538372 ms** (`40,538,372` ns) |

**Gap:** −0.243 ms (**−0.60%** vs baseline).

**Verdict:** similar / within normal run-to-run noise (well under 5%).

Baseline path:  
`workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/experiment_report.md`

## 4. Artifacts

| Path | Contents |
| --- | --- |
| `workspace/smoke_test_matmul.log` | pytest smoke log |
| `workspace/rerun_nsys_20260802_101250/` | nsys `.nsys-rep`, `.sqlite`, `.log`, compare txt |
| `workspace/smoke_test.md` | this summary |
