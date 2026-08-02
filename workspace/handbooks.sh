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
#SourceFile=/Volumes/case_sensitive_workspace/triton/test/TritonGPU/loop-pipeline.mlir # /Volumes/case_sensitive_workspace/triton/workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/irs/04b_before_ttgpuir_add_pipeline.ttgir____matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1
SourceFile=/Volumes/case_sensitive_workspace/triton/workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/irs/04a_before_ttgpuir_add_assign_latencies.ttgir____matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1

"$TRITON_OPT" "$SourceFile" --test-print-alignment >& mem.log 2>&1

## turn on pass's debug log
## print the contiguity, disvisibility and constancy
## triton-opt build and run
TRITON_OPT=/Volumes/case_sensitive_workspace/triton/build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
FileCheck=/Volumes/case_sensitive_workspace/triton/python/triton/FileCheck
#SourceFile=/Volumes/case_sensitive_workspace/triton/test/TritonGPU/loop-pipeline.mlir # /Volumes/case_sensitive_workspace/triton/workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/irs/04b_before_ttgpuir_add_pipeline.ttgir____matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1
SourceFile=/Volumes/case_sensitive_workspace/triton/workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/irs/04a_before_ttgpuir_add_assign_latencies.ttgir____matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1

"$TRITON_OPT" "$SourceFile" \
  -split-input-file \
  --debug-only=triton-loop-pipeline \
  -tritongpu-assign-latencies=num-stages=2 \
  -tritongpu-schedule-loops \
  -tritongpu-pipeline=num-stages=2 \
  -canonicalize  >& log 2>&1 #| "$FileCheck" "$SourceFile" --check-prefixes=COMMON,CHECK



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
