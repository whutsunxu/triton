# Docker: enter container + rebuild `triton-opt`

## Container: `triton-cpu-dev`

On the local server, `triton-cpu-dev` runs from:

- Image: `quay.io/triton-dev-containers/cpu:latest`
- Cmd: `tail -f /dev/null`
- Mounts:
  - `/Volumes/case_sensitive_workspace/triton` → `/workspace/triton` (rw)
  - `/Users/sunxu/.gitconfig` → `/etc/gitconfig` (ro)

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
