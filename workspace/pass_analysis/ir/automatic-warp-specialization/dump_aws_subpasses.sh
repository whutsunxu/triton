#!/usr/bin/env bash
# Dump IR after cumulative AutomaticWarpSpecialization sub-passes.
#
# AWS nest (AutomaticWarpSpecialization.cpp):
#   partition-scheduling
#   nvws-hoist-tmem-store          (required between checkpoints; included)
#   nvws-insert-aref              (required between checkpoints; included)
#   nvws-insert-tmem-aref         ← checkpoint 02
#   sccp + cse                    (required before lower-aref; included)
#   nvws-lower-aref               ← checkpoint 03
#   tritongpu-partition-loops     ← checkpoint 04
#   nvws-lower-warp-group         ← checkpoint 05
#   tritongpu-schedule-loops      ← checkpoint 06
#
# Note: full AWS also runs multiBufferTMADescriptors() after the nest;
# these dumps stop at schedule-loops (no multi-buffer post-step).
set -euo pipefail

TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
IRDIR=/Volumes/case_sensitive_workspace/triton/workspace/pass_analysis/ir
OUTDIR="$IRDIR/automatic-warp-specialization/subpasses"
SourceFile="$IRDIR/input_ir.ttir"
OutLog="$OUTDIR/dump_subpasses.log"
NUM_STAGES=3

mkdir -p "$OUTDIR"
: >"$OutLog"

# Shared prefix: assign latencies + schedule (feeds AWS nest).
PREFIX=(
  -tritongpu-assign-latencies=num-stages="${NUM_STAGES}"
  -tritongpu-schedule-loops
)

run_checkpoint() {
  local name="$1"
  shift
  local out="$OUTDIR/${name}.ttir"
  echo "==> ${name}" | tee -a "$OutLog"
  set +e
  "$TRITON_OPT" "$SourceFile" "${PREFIX[@]}" "$@" -o "$out" >>"$OutLog" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAILED ${name} (exit ${rc}); see $OutLog" >&2
    exit "$rc"
  fi
  echo "    wrote $out ($(wc -c <"$out") bytes)"
}

# 01 — partition-scheduling
run_checkpoint "01-after-partition-scheduling" \
  -tritongpu-partition-scheduling

# 02 — … + insert-tmem-aref  (includes hoist + insert-aref as in AWS)
run_checkpoint "02-after-insert-tmem-aref" \
  -tritongpu-partition-scheduling \
  -nvws-hoist-tmem-store \
  -nvws-insert-aref \
  -nvws-insert-tmem-aref

# 03 — … + lower-aref  (includes sccp + cse as in AWS)
run_checkpoint "03-after-lower-aref" \
  -tritongpu-partition-scheduling \
  -nvws-hoist-tmem-store \
  -nvws-insert-aref \
  -nvws-insert-tmem-aref \
  -sccp \
  -cse \
  -nvws-lower-aref=num-stages="${NUM_STAGES}"

# 04 — … + partition-loops
# Pretty form is readable but nvws.warp_group does not round-trip (printer omits
# result types). Also emit generic assembly that re-parses cleanly.
run_checkpoint "04-after-partition-loops" \
  -tritongpu-partition-scheduling \
  -nvws-hoist-tmem-store \
  -nvws-insert-aref \
  -nvws-insert-tmem-aref \
  -sccp \
  -cse \
  -nvws-lower-aref=num-stages="${NUM_STAGES}" \
  -tritongpu-partition-loops

echo "==> 04-after-partition-loops.generic (round-trip safe)" | tee -a "$OutLog"
"$TRITON_OPT" "$SourceFile" "${PREFIX[@]}" \
  -tritongpu-partition-scheduling \
  -nvws-hoist-tmem-store \
  -nvws-insert-aref \
  -nvws-insert-tmem-aref \
  -sccp \
  -cse \
  -nvws-lower-aref=num-stages="${NUM_STAGES}" \
  -tritongpu-partition-loops \
  --mlir-print-op-generic \
  -o "$OUTDIR/04-after-partition-loops.generic.ttir" >>"$OutLog" 2>&1
echo "    wrote $OUTDIR/04-after-partition-loops.generic.ttir"

# 05 — … + lower-warp-group
run_checkpoint "05-after-lower-warp-group" \
  -tritongpu-partition-scheduling \
  -nvws-hoist-tmem-store \
  -nvws-insert-aref \
  -nvws-insert-tmem-aref \
  -sccp \
  -cse \
  -nvws-lower-aref=num-stages="${NUM_STAGES}" \
  -tritongpu-partition-loops \
  -nvws-lower-warp-group

# 06 — … + schedule-loops
run_checkpoint "06-after-schedule-loops" \
  -tritongpu-partition-scheduling \
  -nvws-hoist-tmem-store \
  -nvws-insert-aref \
  -nvws-insert-tmem-aref \
  -sccp \
  -cse \
  -nvws-lower-aref=num-stages="${NUM_STAGES}" \
  -tritongpu-partition-loops \
  -nvws-lower-warp-group \
  -tritongpu-schedule-loops

echo "done. dumps in $OUTDIR"
ls -la "$OUTDIR"/*.ttir
