#!/usr/bin/env bash
# Run tritongpu-pipeline on post-AWS IR (matches compiler.py:
#   add_warp_specialize → add_pipeline).
set -euo pipefail

TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
IRDIR=/Volumes/case_sensitive_workspace/triton/workspace/pass_analysis/ir
OUTDIR="$IRDIR/add_pipeline"
SourceFile="$IRDIR/automatic-warp-specialization/after_automatic_warp_specialization.ttir"
OutIR="$OUTDIR/after_pipeline.ttir"
OutLog="$OUTDIR/pipeline_debug.log"
NUM_STAGES=3

mkdir -p "$OUTDIR"

"$TRITON_OPT" "$SourceFile" \
  -tritongpu-pipeline=num-stages="${NUM_STAGES}" \
  -o "$OutIR" \
  >"$OutLog" 2>&1

echo "wrote $OutIR"
echo "log:   $OutLog"
