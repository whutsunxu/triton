# ncu blocked: ERR_NVGPUCTRPERM

## Result
- Command exit code: **1**
- No `.ncu-rep` produced (GPU performance counters inaccessible)
- Bench itself ran (printed `batch_per_expt: 64 | MS: 3.06 | ...`)
- Log: `ncu_bench_mlp.log`

## In-container diagnosis
| Check | Value |
|-------|--------|
| `/proc/driver/nvidia/params` → `RmProfilingAdminOnly` | **1** (read-only from container) |
| `CAP_SYS_ADMIN` in process CapBnd | **absent** (Docker drop) |
| `sudo ncu` / `setcap cap_sys_admin` | **does not help** without CapBnd |
| `/sys/module/nvidia/parameters` | **not mounted** (driver owned by host) |

## Host fix (pick one)

### Option A — allow non-admin profiling (preferred for containers)
On the **host**:
```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
```
Then reload the NVIDIA driver (requires stopping GPU workloads; often a **reboot**):
```bash
# after reboot, verify on host:
grep RmProfilingAdminOnly /proc/driver/nvidia/params   # expect: 0
```

### Option B — keep RestrictProfilingToAdminUsers=1, grant admin in container
Re-run the container with:
```bash
docker run --gpus all --cap-add=SYS_ADMIN ...
# or --privileged
```
With `RmProfilingAdminOnly=1`, profiling still requires effective `CAP_SYS_ADMIN`.

## Re-run after host fix
```bash
/Volumes/case_sensitive_workspace/triton/work_space/MLP-MOE-B64-2048-1440/run_ncu.sh
# or: NCU_SET=full .../run_ncu.sh
```

## Retry with --set basic (2026-07-29)
- Same **ERR_NVGPUCTRPERM** — confirms failure is host/container GPU counter permissions, not metric-set weight.
- Export target was `ncu_bench_mlp_basic`; no `.ncu-rep` produced.
- Log: `ncu_bench_mlp_basic.log`
- Helpers default now: `--set basic` (`run_ncu.sh`, `work_space/run_command.sh`).
