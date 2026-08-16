# VM bootstrap → Docker `ir_dev` → smoke test

Companion to [`workspace/smoke_test.md`](./smoke_test.md).  
Use this on a **new virtual server** to stand up the GPU env, pull the image, wire GitHub SSH, update `matmul_perf_analysis`, and run the matmul smoke test.

**Validated:** 2026-08-16 on RTX 5060 Ti (driver 580.95.05), image [`jasonsun11/ir_dev:cuda13_v2`](https://hub.docker.com/r/jasonsun11/ir_dev/).

---

## Checklist (order)

1. Host GPU + CUDA driver OK  
2. Host Nsight Systems (`nsys`) + Nsight Compute (`ncu`) + `nvcc` (optional but useful)  
3. Docker + NVIDIA Container Toolkit  
4. Pull `jasonsun11/ir_dev:cuda13_v2`  
5. GitHub SSH key (host) → mount into container  
6. Start container with `--gpus all --privileged`  
7. Fetch / checkout `matmul_perf_analysis` **via SSH only**  
8. Smoke pytest (`test_matmul.py::test_op`)  
9. (Optional) nsys / ncu profiles — see `smoke_test.md` §3 and experiment reports under `workspace/test_cases/`

---

## 1. Host GPU

```bash
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
# Expect: NVIDIA GeForce RTX 5060 Ti (or similar), CUDA Version ≥ 13.0 in nvidia-smi header
```

Driver API smoke (no toolkit required):

```bash
python3 - <<'PY'
import ctypes
lib = ctypes.CDLL("libcuda.so.1")
assert lib.cuInit(0) == 0
n = ctypes.c_int(); lib.cuDeviceGetCount(ctypes.byref(n))
print("devices:", n.value)
PY
```

---

## 2. Host Nsight / Nsight Compute / nvcc

```bash
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  nsight-compute-2025.3.1 \
  nsight-systems-2025.3.2 \
  cuda-nvcc-13-0 \
  cuda-cudart-dev-13-0

sudo ln -sf /opt/nvidia/nsight-compute/2025.3.1/ncu /usr/local/bin/ncu
echo 'export PATH=/usr/local/cuda/bin:/opt/nvidia/nsight-compute/2025.3.1:$PATH' \
  | sudo tee /etc/profile.d/cuda-nsight.sh
export PATH=/usr/local/cuda/bin:/opt/nvidia/nsight-compute/2025.3.1:/usr/local/bin:$PATH

nvcc --version    # 13.0.x
nsys --version    # 2025.3.2+
ncu --version     # 2025.3.1+
```

---

## 3. Docker + NVIDIA runtime

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker info | grep -i nvidia   # Runtimes should list nvidia
```

---

## 4. Pull the IR / Triton image

```bash
docker pull jasonsun11/ir_dev:cuda13_v2
docker images jasonsun11/ir_dev
# Expect ~46 GB local image (Hub lists ~21.8 GB compressed)
```

Image contents (high level): Triton + LLVM build tree under `/Volumes/case_sensitive_workspace/`, CUDA 13, workspace `venv`.

---

## 5. GitHub SSH key on the **host** (then reuse in container)

> **MUST use SSH remotes only** (`git@github.com:…`). Never `https://github.com/…` for fetch/pull/push.

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
KEY=~/.ssh/id_ed25519
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -C "sunxu-cfd@outlook.com" -f "$KEY" -N ""
fi
cat "$KEY.pub"
# → GitHub → Settings → SSH and GPG keys → New SSH key (Authentication)
```

Verify on host:

```bash
ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts 2>/dev/null
ssh -T git@github.com
# Hi whutsunxu! You've successfully authenticated...
```

---

## 6. Start container (`ir_dev`)

Mount the host SSH dir read-only; enable GPU + privileged (needed for `ncu` perf counters):

```bash
docker rm -f ir_dev 2>/dev/null || true
docker run -d --name ir_dev --gpus all --privileged \
  -v "$HOME/.ssh:/host_ssh:ro" \
  jasonsun11/ir_dev:cuda13_v2 \
  sleep infinity

docker exec ir_dev nvidia-smi -L
```

Wire SSH **inside** the container (matches `smoke_test.md` §0a layout):

```bash
docker exec ir_dev bash -lc '
set -e
mkdir -p /Volumes/case_sensitive_workspace/.ssh_github ~/.ssh
KEY=/Volumes/case_sensitive_workspace/.ssh_github/id_ed25519_github_whutsunxu
cp /host_ssh/id_ed25519 "$KEY"
cp /host_ssh/id_ed25519.pub "$KEY.pub"
chmod 600 "$KEY" && chmod 644 "$KEY.pub"
cat > ~/.ssh/config <<EOF
Host github.com
  HostName github.com
  User git
  IdentityFile $KEY
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts 2>/dev/null
ssh -T git@github.com
git config --global user.name "whutsunxu"
git config --global user.email "sunxu-cfd@outlook.com"
'
```

---

## 7. Update `matmul_perf_analysis`

```bash
docker exec ir_dev bash -lc '
set -e
cd /Volumes/case_sensitive_workspace/triton
git remote set-url origin git@github.com:whutsunxu/triton.git
git remote -v   # must be git@github.com:… (not https)
git fetch origin matmul_perf_analysis
git checkout matmul_perf_analysis
git pull --ff-only origin matmul_perf_analysis
git log -1 --oneline
'
```

Repo path in image: `/Volumes/case_sensitive_workspace/triton`  
Python venv: `/Volumes/case_sensitive_workspace/venv`

---

## 8. Smoke test

```bash
docker exec --privileged ir_dev bash -lc '
set -e
cd /Volumes/case_sensitive_workspace/triton
VENV=/Volumes/case_sensitive_workspace/venv/bin
"$VENV/python" -m pip install -q -r python/test-requirements.txt
export PYTHONPATH=python/triton_kernels
mkdir -p workspace
"$VENV/python" -m pytest \
  python/triton_kernels/tests/test_matmul.py::test_op -v \
  | tee workspace/smoke_test_matmul.log
'
```

**Expect:** `1 passed` (active case is batched FP8 `8192×2048×7680`; tip params may be `block_m=64`, `is_persistent=True` — see current `test_matmul.py`).

Interactive shell:

```bash
docker exec -it ir_dev bash
```

---

## 9. Next steps (profiling)

After smoke is green, follow [`smoke_test.md`](./smoke_test.md) §3 for **nsys** gap checks vs `experiment_report.md`, and existing recipes under:

- `workspace/handbooks.sh` — nsys / stall-capable ncu commands  
- `workspace/test_cases/…/experiment_report.md` — full analysis frame (nsys, ncu stalls, PM Sampling)

Notes for NCU:

- Use `--privileged` (or host `RmProfilingAdminOnly=0`) for counters.  
- `--set basic` lacks warp stall sections; use `WarpStateStats` / `SchedulerStats` / `ComputeWorkloadAnalysis` (or `--set full`).  
- PM Sampling: prefer `ncu --set pmsampling` or **one** realtime metric at a time on Blackwell (multi-metric CLI may fail with “only single-pass PM sampling”).

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `could not select device driver "" with capabilities: [[gpu]]` | Install/configure `nvidia-container-toolkit`, restart Docker |
| `Permission denied (publickey)` to GitHub | Add host `id_ed25519.pub` on GitHub; remount `/host_ssh`; check `IdentityFile` |
| HTTPS git remote | `git remote set-url origin git@github.com:whutsunxu/triton.git` |
| `ncu` `ERR_NVGPUCTRPERM` | Run container with `--privileged` or relax host profiling restriction |
| Image pull stalls on last layer | Retry `docker pull`; ensure disk space (~50+ GB free) |
| `python: command not found` in container | Use venv: `/Volumes/case_sensitive_workspace/venv/bin/python` |

---

## This run (2026-08-16) snapshot

| Item | Value |
|------|--------|
| Image | `jasonsun11/ir_dev:cuda13_v2` (~46.6 GB local) |
| Container | `ir_dev` (`--gpus all --privileged`, SSH via `/host_ssh`) |
| Branch | `matmul_perf_analysis` (SSH origin) |
| Host tools | nvcc 13.0.88, nsys 2025.3.2, ncu 2025.3.1 |
| Smoke | `test_op` **PASSED** (~7.5s); log `workspace/smoke_test_matmul.log` |
