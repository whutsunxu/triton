#!/usr/bin/env bash
# Dump loop.stage / loop.cluster after schedule-loops.
# Key: do NOT run automatic-warp-specialization (or pipeline) before this —
# AWS consumes the schedule and the attrs disappear from the IR.
# Also write -o ...; redirecting only stderr/stdout mixes debug text and hides attrs.
set -euo pipefail

TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
IRDIR=/Volumes/case_sensitive_workspace/triton/workspace/pass_analysis/ir
SourceFile="$IRDIR/input_ir.ttir"
OutIR="$IRDIR/stage_cluster.ttir"
OutLog="$IRDIR/schedule_loops_debug.log"

"$TRITON_OPT" "$SourceFile" \
  --debug-only=triton-loop-pipeline \
  -tritongpu-assign-latencies=num-stages=3 \
  -tritongpu-schedule-loops \
  -o "$OutIR" \
  >"$OutLog" 2>&1

echo "wrote $OutIR"
echo "log:   $OutLog"
