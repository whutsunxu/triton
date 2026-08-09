# Smoke test + Nsight perf check (`matmul_perf_analysis`)

Branch: `matmul_perf_analysis`  
Repo: `/Volumes/case_sensitive_workspace/triton`  
Date: 2026-08-05

## 0. Host bootstrap (SSH + C++ jump-to-def)

### 0a. Generate SSH key + configure GitHub access (**do this first on a new host**)

> **MUST use SSH for all git remote traffic — do not use HTTPS.**  
> Use `git@github.com:…` remotes only. Do **not** `git fetch/pull/push` via `https://github.com/…`.  
> Required before `git fetch` / `git push`.  
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

**Step 4 — Force the repo remote to SSH (not HTTPS):**

```bash
cd /Volumes/case_sensitive_workspace/triton
git remote -v
# must show: git@github.com:whutsunxu/triton.git

# If it shows https://github.com/..., fix it:
git remote set-url origin git@github.com:whutsunxu/triton.git
git remote -v
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

### 0c. tmux easy scroll (mouse + large history) — **per SSH host**

Durable config lives on the volume and is symlinked into `$HOME`:

```bash
# Already created at:
#   /Volumes/case_sensitive_workspace/.tmux.conf
ln -sfn /Volumes/case_sensitive_workspace/.tmux.conf ~/.tmux.conf

# Apply to a running session (or just start a new `tmux`):
tmux source-file ~/.tmux.conf
```

What it enables:

| Action | How |
|--------|-----|
| Scroll with mouse wheel | works in the active pane (`set -g mouse on`) |
| Enter scroll / copy mode | `Ctrl-b` then `[` |
| Exit copy mode | `q` or `Enter` |
| Half-page up / down (in copy mode) | `Ctrl-u` / `Ctrl-d` |
| Reload config | `Ctrl-b` then `r` |

Also sets `history-limit 100000` and vi keys in copy mode.

**Verify:**

```bash
tmux show -g mouse
# mouse on
tmux show -g history-limit
# history-limit 100000
```

### 0d. Graphviz (view partition-scheduling `.dot` dumps) — **per SSH host**

Needed for the Cursor Graphviz plugin and for rendering DOTs to PNG.

**Install:**

```bash
sudo apt-get install -y graphviz
which dot && dot -V
# expect: /usr/bin/dot
#         dot - graphviz version 2.43.0 …
```

**This run:** `graphviz` 2.42.2 / `dot` 2.43.0 installed.

**View in Cursor:**

1. Open a `.dot` under `…/partition_scheduling_dots/`
2. Command Palette → **Graphviz: Open Preview to the Side** (or the preview icon)

**Or render PNGs:**

```bash
DOT_DIR=workspace/Matmul-None-True-False-False-False-None-64-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/partition_scheduling_dots
cd /Volumes/case_sensitive_workspace/triton/$DOT_DIR

# key milestones
dot -Tpng graph-*_0-0000-input.dot -o graph-_p_matmul_…_0-0000-input.png
dot -Tpng graph-*_0-0004-initial.dot -o graph-_p_matmul_…_0-0004-initial.png
dot -Tpng graph-*_0-0044-final.dot -o graph-_p_matmul_…_0-0044-final.png

# or all:
# for f in graph-*.dot; do dot -Tpng "$f" -o "${f%.dot}.png"; done
```

Then open the `.png` in Cursor.

## 1. Fetch / checkout latest branch

> Use **SSH only** (`origin` = `git@github.com:whutsunxu/triton.git`). Never fetch via HTTPS.

```bash
cd /Volumes/case_sensitive_workspace/triton
git remote -v   # confirm git@github.com:… (not https://)
git fetch origin matmul_perf_analysis
git checkout matmul_perf_analysis
git pull --ff-only origin matmul_perf_analysis
```

Result: up to date at `d65651603` (tracks `origin/matmul_perf_analysis`).

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

For this env-verify nsys gap check, temporarily set `is_persistent=False`, `block_m=128` (branch tip is currently `True` / `64`; restored after profiling).

Active case: `Case(8192, 2048, 7680, "batched", "float8_e5m2", "float8_e5m2")`, `block_m=128`, `is_persistent=False`.

**Result:** 1 collected, **PASSED** (~3.5s, cached compile). Log: `workspace/smoke_test_matmul.log`.

## 3. Nsight install + profiler vs `experiment_report.md`

### Install

```bash
# Nsight Systems (nsys); Nsight Compute (ncu) already present from CUDA toolkit
sudo apt-get install -y cuda-nsight-systems-13-1
```

Versions used: Nsight Systems **2025.5.2**, GPU RTX 5060 Ti (driver **595.71.05**).

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

This run: `workspace/rerun_nsys_20260805_095409/`.

### Kernel time comparison

Target kernel: `_matmul_NNN_fp8e5xfp8e5xfp8e5_128x256x128x1`

| Source | GPU time |
| --- | --- |
| `experiment_report.md` baseline | **40.781553 ms** (`40,781,553` ns) |
| This nsys rerun | **41.048802 ms** (`41,048,802` ns) |

**Gap:** +0.267 ms (**+0.66%** vs baseline).

**Verdict:** similar / within a few percent (under 5%).

Baseline path:  
`workspace/Matmul-None-False-False-False-False-None-128-8192-2048-7680-batched-float8_e5m2-float8_e5m2-None-10-1-False-False-False-None-False-False-False-True-None/experiment_report.md`

## 4. Artifacts

| Path | Contents |
| --- | --- |
| `workspace/smoke_test_matmul.log` | pytest smoke log |
| `workspace/rerun_nsys_20260805_095409/` | nsys `.nsys-rep`, `.sqlite`, `.log`, compare txt |
| `workspace/setup_clangd_cursor.sh` | portable clangd bootstrap |
| `workspace/clangd_setup.md` | clangd docs |
| `/Volumes/case_sensitive_workspace/.tmux.conf` | durable tmux scroll config (§0c) |
| system `graphviz` / `dot` | render partition-scheduling `.dot` dumps (§0d) |
| `workspace/smoke_test.md` | this summary |
