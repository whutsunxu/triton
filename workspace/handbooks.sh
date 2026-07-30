pip install -r /Volumes/case_sensitive_workspace/triton/python/test-requirements.txt

cd /Volumes/case_sensitive_workspace/triton && PYTHONPATH=python/triton_kernels python -m pytest python/triton_kernels/tests/test_matmul.py::test_set_idle_sms -v

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

## run test_matmul.py with ncu profile (target matmul kernel; needs GPU perf-counter access, e.g. --privileged)
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels

ncu \
  --force-overwrite \
  --set basic \
  --kernel-name _matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1 \
  --launch-count 1 \
  -o /Volumes/case_sensitive_workspace/triton/workspace/test_matmul_ncu \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest python/triton_kernels/tests/test_matmul.py::test_op -v \
  >/Volumes/case_sensitive_workspace/triton/workspace/test_matmul_ncu.log 2>&1
