#!/usr/bin/env bash
# Run tritongpu-optimize-partition-warps on post-pipeline IR (matches compiler.py:
#   add_pipeline → add_optimize_partition_warps).
set -euo pipefail

TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
IRDIR=/Volumes/case_sensitive_workspace/triton/workspace/pass_analysis/ir
OUTDIR="$IRDIR/add_optimize_partition_warps"
SourceFile="$IRDIR/add_pipeline/after_pipeline.ttir"
OutIR="$OUTDIR/after_optimize_partition_warps.ttir"
OutLog="$OUTDIR/optimize_partition_warps_debug.log"

mkdir -p "$OUTDIR"

"$TRITON_OPT" "$SourceFile" \
  -tritongpu-optimize-partition-warps \
  -o "$OutIR" \
  >"$OutLog" 2>&1

echo "wrote $OutIR"
echo "log:   $OutLog"
