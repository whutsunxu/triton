#!/usr/bin/env bash
# Dump add_pipeline steps from stage_cluster.ttir (schedule attrs only).
# Usage:
#   TRITON_OPT=/path/to/triton-opt ./dump_pipeline_from_stage_cluster.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
IRDIR=$(cd "$SCRIPT_DIR/.." && pwd)
OUTDIR="$SCRIPT_DIR"
SRC="$IRDIR/stage_cluster_analysis/stage_cluster.ttir"
NUM_STAGES="${NUM_STAGES:-3}"

if [[ -z "${TRITON_OPT:-}" ]]; then
  echo "Set TRITON_OPT to your Linux triton-opt binary" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

echo "== lowerLoops only =="
"$TRITON_OPT" "$SRC" \
  -tritongpu-test-pipeline-lower-loop \
  -o "$OUTDIR/01-after-lower-loops_from_stage_cluster.ttir"

echo "== full pipeline + intermediate dumps =="
"$TRITON_OPT" "$SRC" \
  -tritongpu-pipeline="num-stages=${NUM_STAGES} dump-intermediate-steps=true" \
  -o "$OUTDIR/after_pipeline_from_stage_cluster.ttir" \
  >"$OUTDIR/pipeline_from_stage_cluster.log" 2>&1

echo "wrote:"
echo "  $OUTDIR/01-after-lower-loops_from_stage_cluster.ttir"
echo "  $OUTDIR/after_pipeline_from_stage_cluster.ttir"
echo "  $OUTDIR/pipeline_from_stage_cluster.log"
echo "In the log, search for: After: LowerLoops / After: ExpandLoops"
