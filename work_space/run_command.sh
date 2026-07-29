PYTHONPATH=:./python/triton_kernels/ torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py
