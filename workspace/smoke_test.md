# Smoke test + Nsight perf check (`matmul_perf_analysis`)

Branch: `matmul_perf_analysis`  
Repo: `/Volumes/case_sensitive_workspace/triton`  
Date: 2026-08-04

## 0. Host bootstrap (SSH + C++ jump-to-def)

### 0a. Generate SSH key + configure GitHub access (**do this first on a new host**)

> **Required before `git fetch` / `git push` over SSH.**  
> Key lives on the durable volume so it survives most instance rebuilds.

**Step 1 — Generate the key** (skip if the files already exist):

```bash
mkdir -p /Volumes/case_sensitive_workspace/.ssh_github
KEY=/Volumes/case_sensitive_workspace/.ssh_github/id_ed25519_github_whutsunxu

# Generate only if missing
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -C "sunxu-cfd@outlook.com" -f "$KEY" -N ""
fi
chmod 600 "$KEY"
chmod 644 "$KEY.pub"
```

**Step 2 — Copy the public key and add it on GitHub:**

```bash
# Print the key (copy the whole line)
cat "$KEY.pub"
```

1. Open **[GitHub → Settings → SSH and GPG keys → New SSH key](https://github.com/settings/keys)**
2. Title: e.g. `vastai-matmul` / hostname
3. Key type: **Authentication Key**
4. Paste the `ssh-ed25519 AAAA… sunxu-cfd@outlook.com` line → **Add SSH key**

**Step 3 — Point SSH at that key and verify:**

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
# Ensure github.com uses this IdentityFile (rewrite if needed)
cat > ~/.ssh/config <<EOF
Host github.com
  HostName github.com
  User git
  IdentityFile /Volumes/case_sensitive_workspace/.ssh_github/id_ed25519_github_whutsunxu
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config

ssh -T git@github.com
# expect: Hi whutsunxu! You've successfully authenticated...
```

Expected local git identity (set if committing from this host):

- `user.name`: `whutsunxu`
- `user.email`: `sunxu-cfd@outlook.com`

This run: GitHub SSH authenticated as `whutsunxu`.

### 0b. C++ go-to-definition (clangd) — **per SSH host**

Cursor Remote-SSH needs clangd + extension on **each** server:

```bash
cd /Volumes/case_sensitive_workspace/triton
bash workspace/setup_clangd_cursor.sh
# same script also lives at:
#   /Volumes/case_sensitive_workspace/setup_clangd_cursor.sh
```

Then: Cursor **Developer: Reload Window**. Jump with **F12** / Ctrl+click.

**Verify (this run):**

```bash
clangd --version
# Ubuntu clangd version 18.1.3

# Extension installed: llvm-vs-code-extensions.vscode-clangd
# compile_commands.json present at repo root
```

Docs: `workspace/clangd_setup.md`. Repo config: `.clangd` (uses root `compile_commands.json` symlink).

## 1. Fetch / checkout latest branch

```bash
cd /Volumes/case_sensitive_workspace/triton
git fetch origin matmul_perf_analysis
git checkout matmul_perf_analysis
git pull --ff-only origin matmul_perf_analysis
```

Result: up to date at `bf45e6bf2` (tracks `origin/matmul_perf_analysis`).

## 2. Dependencies + smoke test

```bash
# deps
/Volumes/case_sensitive_workspace/venv/bin/python -m pip install -r \
  python/test-requirements.txt

# smoke
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels
/Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
  python/triton_kernels/tests/test_matmul.py::test_op -v \
  | tee workspace/smoke_test_matmul.log
```

Active case: `Case(8192, 2048, 7680, "batched", "float8_e5m2", "float8_e5m2")`, `block_m=128`.

**Result:** 1 collected, **PASSED** (~37.1s). Log: `workspace/smoke_test_matmul.log`.

## 3. Nsight install + profiler vs `experiment_report.md`

### Install

```bash
# Nsight Systems (nsys); Nsight Compute (ncu) already present from CUDA toolkit
sudo apt-get install -y cuda-nsight-systems-13-1
```

Versions used: Nsight Systems **2025.5.2**, GPU RTX 5060 Ti (driver 580.126.09).

### Profile

```bash
cd /Volumes/case_sensitive_workspace/triton
export PYTHONPATH=python/triton_kernels
OUTDIR=workspace/rerun_nsys_YYYYMMDD_HHMMSS

nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --sample=none \
  --cudabacktrace=none \
  --stats=true \
  -o "$OUTDIR/test_matmul_nsys" \
  /Volumes/case_sensitive_workspace/venv/bin/python -m pytest \
    python/triton_kernels/tests/test_matmul.py::test_op -v \
  >"$OUTDIR/test_matmul_nsys.log" 2>&1
```

This run: `workspace/rerun_nsys_20260804_145102/`.

### Kernel time comparison

Target kernel: `_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`

| Source | GPU time |
| --- | --- |
| `experiment_report.md` baseline | **40.781553 ms** (`40,781,553` ns) |
| Report 10× mean | 40.822 ms (CV 0.14%) |
| This nsys rerun | **39.265499 ms** (`39,265,499` ns) |

**Gap:** −1.516 ms (**−3.72%** vs baseline).

**Verdict:** similar / within a few percent (under 5%).

Baseline path:  
`workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/experiment_report.md`

## 4. Artifacts

| Path | Contents |
| --- | --- |
| `workspace/smoke_test_matmul.log` | pytest smoke log |
| `workspace/rerun_nsys_20260804_145102/` | nsys `.nsys-rep`, `.sqlite`, `.log`, compare txt |
| `workspace/setup_clangd_cursor.sh` | portable clangd bootstrap |
| `workspace/clangd_setup.md` | clangd docs |
| `workspace/smoke_test.md` | this summary |
