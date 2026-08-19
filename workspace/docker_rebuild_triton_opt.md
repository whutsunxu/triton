# Docker: enter container + rebuild `triton-opt`

## Container: `triton-cpu-dev`

On the local server, `triton-cpu-dev` runs from:

- Image: `quay.io/triton-dev-containers/cpu:latest`
- Cmd: `tail -f /dev/null`
- Mounts:
  - `/Volumes/case_sensitive_workspace/triton` → `/workspace/triton` (rw)
  - `/Users/sunxu/.gitconfig` → `/etc/gitconfig` (ro)

**SSH keys are not bind-mounted.** Copy them into the container once (see [SSH keys](#ssh-keys-host--container-copy-not-mount)).

Enter:

```bash
docker exec -it triton-cpu-dev bash
```

(Re)create if needed:

```bash
docker rm -f triton-cpu-dev 2>/dev/null || true
docker run -d --name triton-cpu-dev \
  -v "/Volumes/case_sensitive_workspace/triton:/workspace/triton" \
  -v "/Users/sunxu/.gitconfig:/etc/gitconfig:ro" \
  quay.io/triton-dev-containers/cpu:latest \
  tail -f /dev/null
```

## SSH keys (host → container, copy not mount)

Host path: `/Users/sunxu/.ssh`

Target inside container: **`/root/.ssh`** (permissions: directory `700`, private keys `600`).

Do **not** use `-v .../.ssh:/root/.ssh` unless you explicitly want a bind mount. Preferred flow: **`docker cp`** (or the helper script) from the Mac host.

### Find container name or ID (on Mac host)

```bash
docker ps --format 'table {{.Names}}\t{{.ID}}\t{{.Image}}'
```

For Cursor Remote / agent sessions, the hostname inside the container is often the short container ID (e.g. `3e660b62e3d6`). Use that ID or the `NAMES` column from `docker ps`.

### One-off copy (Mac host)

```bash
CONTAINER=triton-cpu-dev   # or 3e660b62e3d6, or name from docker ps
docker exec "$CONTAINER" mkdir -p /root/.ssh
docker cp /Users/sunxu/.ssh/. "$CONTAINER:/root/.ssh/"
docker exec "$CONTAINER" bash -lc '
  chmod 700 /root/.ssh
  find /root/.ssh -type f -name "id_*" ! -name "*.pub" -exec chmod 600 {} +
  chmod 644 /root/.ssh/*.pub /root/.ssh/known_hosts 2>/dev/null || true
'
docker exec -it "$CONTAINER" bash -lc 'ssh -T git@github.com'
```

Helper script (run on Mac, from repo root):

```bash
./workspace/scripts/copy_ssh_from_mac_host.sh /Users/sunxu/.ssh "$CONTAINER"
```

### `triton-cpu-dev` (docker run)

After (re)creating the container **without** an `.ssh` mount, run the [one-off copy](#one-off-copy-mac-host) once. Keys persist until the container is removed.

Verify inside the container:

```bash
docker exec -it triton-cpu-dev bash -lc 'ls -la /root/.ssh && ssh -T git@github.com'
```

### Cursor / VS Code Dev Container

`.devcontainer/devcontainer.json` mounts **only** `${localEnv:HOME}/.gitconfig` — **not** `.ssh`. After **Rebuild Container**, copy keys from the Mac host using the commands above (container name from `docker ps`).

Optional: re-enable bind-mount by adding back to `mounts` in `devcontainer.json` (not recommended if you prefer copy-only).

### Cursor Remote SSH (current agent session)

This session is **not** controlled by `.devcontainer/devcontainer.json`. Typical state:

- Container ID / hostname: check with `hostname` inside the container (e.g. `3e660b62e3d6`)
- `/Users/sunxu/.ssh` is **not** visible inside the container (no host path mount)
- `~/.ssh` may contain only `known_hosts` (no private keys)
- `SSH_AUTH_SOCK` may point at a Cursor proxy socket; `ssh-add -l` often shows **no identities**

**Fix:** on the Mac host, run [one-off copy](#one-off-copy-mac-host) with `CONTAINER` set to the ID from `hostname` inside this session, then reconnect or run `ssh -T git@github.com` inside the container.

Alternative (agent forwarding, no copy): on Mac before connecting, `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` — only helps if Cursor forwards the agent with identities loaded.

## Rebuild `triton-opt`

Use the known-good ubi9 build dir (incremental; avoids the broken `cmake.linux-x86_64-*` dir). Safest choice inside `triton-cpu-dev`.

| | |
|---|---|
| Container | `triton-cpu-dev` |
| Working directory | `/workspace/triton` |
| Build dir | `build/cmake.ubi9-cpu-cpython-3.12` |
| Binary (not on `PATH`) | `/workspace/triton/build/cmake.ubi9-cpu-cpython-3.12/bin/triton-opt` |

```bash
cd /workspace/triton
ninja -C build/cmake.ubi9-cpu-cpython-3.12 triton-opt
```

Linking can take several minutes (large binary + Docker bind-mount I/O).

## Print guides (`ClusterBarrierInsertion.cpp`)

`DEBUG_TYPE` is `"cluster-barrier-insertion"`. File already includes `"llvm/Support/Debug.h"` (for `LDBG` / `LLVM_DEBUG`). This tree is **RelWithDebInfo** with **`-DNDEBUG`**.

### Primary (this RelWithDebInfo ubi9 `triton-opt`): `llvm::errs()`

Always prints to stderr. Add if missing:

```cpp
#include "llvm/Support/raw_ostream.h"
```

Example:

```cpp
llvm::errs() << "[cluster-barrier-insertion] Hello World (llvm::errs smoke)\n";
```

Rebuild after editing:

```bash
cd /workspace/triton
ninja -C build/cmake.ubi9-cpu-cpython-3.12 triton-opt
```

Run on the repro MLIR:

```bash
cd /workspace/triton
BIN=build/cmake.ubi9-cpu-cpython-3.12/bin/triton-opt

$BIN workspace/pass_analysis/cluster_barrier_insertion/repro_direct_alloc.mlir \
  --allocate-shared-memory -test-print-membar
```

### Secondary: `LDBG` / `LLVM_DEBUG` / `-debug-only=cluster-barrier-insertion`

Does **not** print on this `-DNDEBUG` RelWithDebInfo `triton-opt`. Those macros compile out.

They do print on **TritonRelBuildWithAsserts** (`libtriton.so`), same pattern as `ScheduleLoops.cpp` (`DEBUG_TYPE "triton-loop-pipeline"`). Do not use `-debug-only=cluster-barrier-insertion` expecting output from this ubi9 `triton-opt`.
