#!/usr/bin/env bash
# Nsight Compute profile for MLP MoE bench (B64-2048-1440)
# Default: light --set basic (override with NCU_SET=detailed|full if needed).
set -euo pipefail
cd /Volumes/case_sensitive_workspace/triton
export PATH="/Volumes/case_sensitive_workspace/venv/bin:/usr/local/cuda/bin:$PATH"
OUT_DIR="work_space/MLP-MOE-B64-2048-1440"
mkdir -p "$OUT_DIR"

NCU_SET="${NCU_SET:-basic}"
EXPORT_BASE="${NCU_EXPORT:-${OUT_DIR}/ncu_bench_mlp_basic}"

ncu \
  --set "$NCU_SET" \
  --force-overwrite \
  --target-processes all \
  --kernel-name-base demangled \
  --export "${EXPORT_BASE}" \
  env PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py

ls -lh "${EXPORT_BASE}"* 2>/dev/null || true
