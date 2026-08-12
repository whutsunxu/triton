#!/usr/bin/env bash
# Run assign-latencies → schedule-loops → automatic-warp-specialization.
# Input is pre-latency IR; AWS nests partition-scheduling and related passes.
set -euo pipefail

TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
IRDIR=/Volumes/case_sensitive_workspace/triton/workspace/pass_analysis/ir
OUTDIR="$IRDIR/automatic-warp-specialization"
SourceFile="$IRDIR/input_ir.ttir"
OutIR="$OUTDIR/after_automatic_warp_specialization.ttir"
OutLog="$OUTDIR/automatic_warp_specialization_debug.log"

mkdir -p "$OUTDIR"

"$TRITON_OPT" "$SourceFile" \
  -tritongpu-assign-latencies=num-stages=3 \
  -tritongpu-schedule-loops \
  -tritongpu-automatic-warp-specialization=num-stages=3 \
  -o "$OutIR" \
  >"$OutLog" 2>&1

echo "wrote $OutIR"
echo "log:   $OutLog"
