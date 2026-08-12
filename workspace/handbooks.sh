pip install -r /Volumes/case_sensitive_workspace/triton/python/test-requirements.txt
## run test_matmul.py with pytest
cd /Volumes/case_sensitive_workspace/triton && PYTHONPATH=python/triton_kernels python -m pytest python/triton_kernels/tests/test_matmul.py::test_op -s -v > log 2>&1


## triton-opt build and run
TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
FileCheck=/Volumes/case_sensitive_workspace/triton/python/triton/FileCheck
#SourceFile=/Volumes/case_sensitive_workspace/triton/test/TritonGPU/loop-pipeline.mlir # /Volumes/case_sensitive_workspace/triton/workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/irs/04b_before_ttgpuir_add_pipeline.ttgir____matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1
SourceFile=/Volumes/case_sensitive_workspace/triton/workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/irs/04a_before_ttgpuir_add_assign_latencies.ttgir____matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1

"$TRITON_OPT" "$SourceFile" \
  -split-input-file \
  -tritongpu-assign-latencies=num-stages=2 \
  -tritongpu-schedule-loops \
  -tritongpu-pipeline=num-stages=2 \
  -canonicalize  >& log 2>&1 #| "$FileCheck" "$SourceFile" --check-prefixes=COMMON,CHECK

## print the contiguity, disvisibility and constancy
## triton-opt build and run
TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
FileCheck=/Volumes/case_sensitive_workspace/triton/python/triton/FileCheck

SourceFile=/Volumes/case_sensitive_workspace/triton/workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/irs/04a_before_ttgpuir_add_assign_latencies.ttgir____matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1

"$TRITON_OPT" "$SourceFile" --test-print-alignment >& mem.log 2>&1

## Dump loop.stage / loop.cluster after schedule-loops
## Key: do NOT run automatic-warp-specialization (or pipeline) before this —
## AWS consumes the schedule and the attrs disappear from the IR.
## Also write -o ...; redirecting only stderr/stdout mixes debug text and hides attrs.
TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
FileCheck=/Volumes/case_sensitive_workspace/triton/python/triton/FileCheck
SourceFile=/Volumes/case_sensitive_workspace/triton/workspace/Matmul-None-True-False-False-False-None-64-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/irs/04a_before_ttgpuir_add_assign_latencies.ttgir___p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1
OutIR=/Volumes/case_sensitive_workspace/triton/workspace/after_schedule_loops.ttgir
OutLog=/Volumes/case_sensitive_workspace/triton/workspace/schedule_loops_debug.log

"$TRITON_OPT" "$SourceFile" \
  -tritongpu-assign-latencies=num-stages=3 \
  -tritongpu-schedule-loops \
  -tritongpu-automatic-warp-specialization=num-stages=3 \
  -o "$OutIR" \
  >"$OutLog" 2>&1


## Dump partition-scheduling GraphViz (.dot) via visualize()
## Env: TRITON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT=1
## Optional filters:
##   TRITON_PARTITION_SCHEDULING_DUMP_DATA_ONLY=1
##   TRITON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY=1
## Dots are written to CWD as:
##   graph-<func>_<idx>-<NNNN>-<step>.dot
## Need tt.warp_specialize on the scf.for (use stage_cluster.ttir or IR that already has it).
TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
CASE_DIR=/Volumes/case_sensitive_workspace/triton/workspace/pass_analysis
SourceFile="$CASE_DIR/stage_cluster.ttir"
DOT_DIR="$CASE_DIR/partition_scheduling_dots"
OutIR="$CASE_DIR/after_partition_scheduling.ttir"
OutLog="$CASE_DIR/partition_scheduling_debug.log"

mkdir -p "$DOT_DIR"
cd "$DOT_DIR"
TRITON_PARTITION_SCHEDULING_ENABLE_DUMP_DOT=1 \
TRITON_PARTITION_SCHEDULING_DUMP_LOOP_ONLY=1 \
"$TRITON_OPT" "$SourceFile" \
  --debug-only=tritongpu-partition-scheduling \
  -tritongpu-partition-scheduling \
  -o "$OutIR" \
  >"$OutLog" 2>&1

# Render (needs graphviz `dot`):
# for f in graph-*.dot; do dot -Tpng "$f" -o "${f%.dot}.png"; done
# ls -1 graph-*.dot graph-*.png


## run test_matmul.py with nsys profile
cd /Volumes/case_sensitive_workspace/triton

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --cudabacktrace=none \
  --stats=true \
  -o /Volumes/case_sensitive_workspace/triton/workspace/test_matmul_nsys \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest python/triton_kernels/tests/test_matmul.py::test_op -v \
  >/Volumes/case_sensitive_workspace/triton/workspace/test_matmul_nsys.log 2>&1

## run test_matmul.py with ncu profile (stall-capable sections; needs GPU perf-counter access, e.g. --privileged)
## Note: --set basic lacks WarpStateStats / SchedulerStats / ComputeWorkloadAnalysis.
## Use explicit sections (or --set full) to capture lg_throttle / long_scoreboard / mio / barrier / math_pipe_throttle.
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels

ncu \
  --force-overwrite \
  --kernel-name _matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1 \
  --launch-count 1 \
  --section LaunchStats \
  --section Occupancy \
  --section SpeedOfLight \
  --section WorkloadDistribution \
  --section SchedulerStats \
  --section WarpStateStats \
  --section ComputeWorkloadAnalysis \
  -o /Volumes/case_sensitive_workspace/triton/workspace/test_matmul_ncu \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest python/triton_kernels/tests/test_matmul.py::test_op -v \
  >/Volumes/case_sensitive_workspace/triton/workspace/test_matmul_ncu.log 2>&1


  ## git
1 , push the local branch to the repo which doesn't have this branch,
git push -u origin matmul_perf_analysis
2, push the local branch to the repo which doesn have this branch,
git push origin matmul_perf_analysis
3, pull the remote branch to local which doesn't have this branch,
git fetch origin
git checkout -b matmul_perf_analysis origin/matmul_perf_analysis
4,  pull the remote branch to local which does have this branch,
git pull origin matmul_perf_analysis
