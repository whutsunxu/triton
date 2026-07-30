# VM Profiling Instructions (MLP-MoE B64-2048-1440)

## Server

| Item | Value |
|------|-------|
| SSH | `ssh -p 59727 root@77.104.167.149` |
| GPU | NVIDIA GeForce RTX 5060 Ti 16GB |
| Driver | 580.95.05 |
| Docker image | `jasonsun11/ir_dev:cuda13_v2` |
| Triton path (in container) | `/Volumes/case_sensitive_workspace/triton` |
| Branch | `perf_ana` |

## One-time setup (already done)

1. Install NVIDIA Container Toolkit on host (`nvidia-ctk runtime configure --runtime=docker`).
2. Generate SSH deploy key in container (mounted `/root/.ssh`):
   ```bash
   ssh-keygen -t ed25519 -C "sunxu-cfd@outlook.com" -f /root/.ssh/id_ed25519 -N ""
   ```
   Add public key to GitHub repo `whutsunxu/triton`.
3. Create host output dir:
   ```bash
   mkdir -p /root/triton/work_space/MLP-MOE-B64-2048-1440/VM-profiler
   ```

## Run profiling (all inside container)

**Important:** Do **not** mount over `/Volumes/case_sensitive_workspace/triton`. Use built-in image tree; only mount SSH keys and `VM-profiler` output.

Each `docker run` is ephemeral — run `git fetch && git reset --hard origin/perf_ana` at the start of every session.

```bash
docker run --rm \
  -v /root/.ssh:/root/.ssh:ro \
  -v /root/triton/work_space/MLP-MOE-B64-2048-1440/VM-profiler:/Volumes/case_sensitive_workspace/triton/work_space/MLP-MOE-B64-2048-1440/VM-profiler \
  --gpus all --cap-add=SYS_ADMIN \
  -w /Volumes/case_sensitive_workspace/triton \
  jasonsun11/ir_dev:cuda13_v2 bash -lc '
set -e
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes"
export PATH=/opt/nvidia/nsight-compute/2025.3.0/host/target-linux-x64:/usr/local/cuda/bin:/Volumes/case_sensitive_workspace/venv/bin:$PATH
export PYTHONPATH=./python/triton_kernels
export TRITON_PROTON_DISABLE=1
OUT=work_space/MLP-MOE-B64-2048-1440/VM-profiler

git fetch origin perf_ana && git reset --hard origin/perf_ana

# Bench (warmup + 1 timed run_mlp; no roofline/cuBLAS noise)
torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py --profile-only \
  2>&1 | tee $OUT/bench_smoke.log

# Nsight Compute — basic (all MoE kernels)
ncu --set basic --force-overwrite --target-processes all --kernel-name-base demangled \
  --kernel-name "regex:_p_matmul|_matmul_NNT|_topk_forward|_convert_|_reduce_forward" \
  --export $OUT/ncu_bench_mlp_basic \
  env PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py --profile-only \
  2>&1 | tee $OUT/ncu_basic_run.log

# Nsight Compute — detailed (FP8 matmul kernels only)
ncu --set detailed --force-overwrite --target-processes all --kernel-name-base demangled \
  --kernel-name regex:_p_matmul \
  --export $OUT/ncu_bench_mlp_detailed \
  env PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py --profile-only \
  2>&1 | tee $OUT/ncu_detailed_run.log
'
```

# Nsight Systems (same `--profile-only` script)

Use **standalone Nsight Systems** (not the `nsys` bundled inside Nsight Compute — that one only writes broken `.qdstrm`).

On the VM host, install once: `apt install nsight-systems-2025.3.2`

Inside the container, mount host Nsight Systems and profile:

```bash
docker run --rm \
  -v /root/.ssh:/root/.ssh:ro \
  -v /root/triton/work_space/MLP-MOE-B64-2048-1440/VM-profiler:/out \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  --gpus all --cap-add=SYS_ADMIN \
  -w /Volumes/case_sensitive_workspace/triton \
  jasonsun11/ir_dev:cuda13_v2 bash -lc '
export GIT_SSH_COMMAND="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes"
export PATH=/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64:/usr/local/cuda/bin:/Volumes/case_sensitive_workspace/venv/bin:$PATH
export PYTHONPATH=./python/triton_kernels
export TRITON_PROTON_DISABLE=1
git fetch origin perf_ana && git reset --hard origin/perf_ana

nsys profile --force-overwrite=true --stats=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --cuda-memory-usage=true --sample=none --cpuctxsw=none \
  --output=/out/nsys_bench_mlp \
  env PYTHONPATH=./python/triton_kernels/ TRITON_PROTON_DISABLE=1 \
  torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py --profile-only \
  2>&1 | tee /out/nsight_perf.log
'
```

Produces `nsys_bench_mlp.nsys-rep`, `nsys_bench_mlp.sqlite`, and `nsight_perf.log` (with `--stats` tables).

## NCU requirements

- Container needs `--cap-add=SYS_ADMIN` (host `RmProfilingAdminOnly=1`).
- Without it: `ERR_NVGPUCTRPERM` (see parent folder `ncu_ERR_NVGPUCTRPERM.md`).

## Commit artifacts (on host)

```bash
cd /root/triton
git pull origin perf_ana
git add work_space/MLP-MOE-B64-2048-1440/VM-profiler/
git commit -m "Add VM profiler artifacts for MLP-MoE EP=1"
git push origin perf_ana
```

## Artifacts in this folder

| File | Description |
|------|-------------|
| `bench_smoke.log` | `--profile-only` bench stdout |
| `nsys_bench_mlp.nsys-rep` | Nsight Systems report (`--profile-only`) |
| `nsys_bench_mlp.sqlite` | SQLite backing store for nsys report |
| `nsight_perf.log` | nsys console output + `--stats` tables |
| `ncu_bench_mlp_basic.ncu-rep` | NCU basic metrics, all MoE kernels |
| `ncu_bench_mlp_detailed.ncu-rep` | NCU detailed metrics, `_p_matmul*` only |
| `ncu_basic_run.log` / `ncu_detailed_run.log` | NCU console output |
| `vm-instructions.md` | This file |

## VM run results (2026-07-30)

- Bench `--profile-only`: **OK**
- nsys (`bench_mlp.py --profile-only`): **OK** — `nsys_bench_mlp.nsys-rep` (1.3 MB), `nsys_bench_mlp.sqlite` (4.4 MB); MoE kernels visible (`_p_matmul*`, `_matmul_NNT*`, `_topk_forward`, `_convert_*`, `_reduce_forward`)
- NCU basic: **OK** (~1.4 MB `.ncu-rep`, 9 passes per kernel)
- NCU detailed: **OK** (~4.2 MB `.ncu-rep`, 21 passes, `_p_matmul*` kernels)
