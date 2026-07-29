cd /Volumes/case_sensitive_workspace/triton
export PATH="/Volumes/case_sensitive_workspace/venv/bin:$PATH"

# Plain bench (no profiler)
PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py

# Nsight Systems — outputs under case folder
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas,mpi \
  --cuda-memory-usage=true \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --python-sampling=true \
  --python-sampling-frequency=1000 \
  --stats=true \
  --output=work_space/MLP-MOE-B64-2048-1440/nsys_bench_mlp \
  env PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py

# Nsight Compute — outputs under case folder
# Requires GPU performance counter access (RmProfilingAdminOnly=0 on host,
# or container CAP_SYS_ADMIN when RmProfilingAdminOnly=1).
ncu \
  --set basic \
  --force-overwrite \
  --target-processes all \
  --kernel-name-base demangled \
  --export work_space/MLP-MOE-B64-2048-1440/ncu_bench_mlp_basic \
  env PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py
