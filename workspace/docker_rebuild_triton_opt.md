# Docker: enter container + rebuild `triton-opt`

## 0. (Optional) CPU-only dev container: `triton-cpu-dev`

On the local server, `triton-cpu-dev` is running from:

- Image: `quay.io/triton-dev-containers/cpu:latest`
- Cmd: `tail -f /dev/null`
- Mounts:
  - `/Volumes/case_sensitive_workspace/triton` → `/workspace/triton` (rw)
  - `/Users/sunxu/.gitconfig` → `/etc/gitconfig` (ro)

Recreate if needed:

```bash
docker rm -f triton-cpu-dev 2>/dev/null || true
docker run -d --name triton-cpu-dev \
  -v "/Volumes/case_sensitive_workspace/triton:/workspace/triton" \
  -v "/Users/sunxu/.gitconfig:/etc/gitconfig:ro" \
  quay.io/triton-dev-containers/cpu:latest \
  tail -f /dev/null
```

## 1. Start the container

```bash
docker ps -a
docker start <container_name_or_id>
```

Skip if it is already running.

## 2. Enter the container

```bash
docker exec -it <container_name_or_id> bash
```

Use `sh` if the image has no `bash`.

## 3. Rebuild `triton-opt`

Inside the container, the repo is:

```bash
cd /workspace/triton
```

### Method A (recommended): known-good ubi9 build dir

Incremental rebuild; avoids the broken `cmake.linux-x86_64-*` dir:

```bash
cd /workspace/triton
ninja -C build/cmake.ubi9-cpu-cpython-3.12 triton-opt
```

Notes:

- Incremental (not a full rebuild).
- Safest choice for `triton-cpu-dev`.
- `triton-opt` is **not** on `PATH`. Use:
  `/workspace/triton/build/cmake.ubi9-cpu-cpython-3.12/bin/triton-opt`
- Linking can take several minutes (large binary + Docker bind-mount I/O).

### Method B (workaround): recreate the linux-x86_64 build dir

Only if you need `make triton-opt` against `cmake.linux-x86_64-*`:

```bash
cd /workspace/triton
rm -rf build/cmake.linux-x86_64-cpython-3.12
make triton-opt
```

## 4. Quick sanity check

```bash
/workspace/triton/build/cmake.ubi9-cpu-cpython-3.12/bin/triton-opt --help
```

## 5. Debug prints for `ClusterBarrierInsertion.cpp`

This Method A tree is **RelWithDebInfo** (`cmake.ubi9-cpu-cpython-3.12`) with `-DNDEBUG` (`CMAKE_CXX_FLAGS_RELWITHDEBINFO=-O2 -g -DNDEBUG`). `LLVM_DEBUG` / `LDBG` are compiled out. Use `llvm::errs()` — it always prints.

### 5a. `llvm::errs()` (works on this RelWithDebInfo `triton-opt`)

Dependent header:

```cpp
#include "llvm/Support/raw_ostream.h"
```

(`llvm/Support/Debug.h` is only needed if you also keep `LDBG`.)

Example:

```cpp
llvm::errs() << "[cluster-barrier-insertion] ...\n";
```

Rebuild (Method A, inside docker `triton-cpu-dev`):

```bash
cd /workspace/triton
ninja -C build/cmake.ubi9-cpu-cpython-3.12 triton-opt
```

Run (full binary path; the pass is reached by `-test-print-membar`, see `test/lib/Analysis/TestMembar.cpp`):

```bash
cd /workspace/triton
BIN=build/cmake.ubi9-cpu-cpython-3.12/bin/triton-opt
$BIN workspace/pass_analysis/cluster_barrier_insertion/repro_direct_alloc.mlir \
  --allocate-shared-memory -test-print-membar
```

Expect the `llvm::errs()` line on stderr, then the rewritten MLIR.

### 5b. `LDBG` / `-debug-only=` (secondary; not this binary)

`LDBG` / `LLVM_DEBUG` / `-debug-only=` only work on **TritonRelBuildWithAsserts** / non-NDEBUG builds (typically the Python `libtriton.so`). They do **not** print from this RelWithDebInfo `triton-opt`.

Same scheme as `AssignLatencies.cpp`, with this pass’s `DEBUG_TYPE`:

```cpp
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cluster-barrier-insertion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
```

```cpp
LDBG("interrupt: op=" << op->getName());
```

Match `-debug-only=` to `DEBUG_TYPE` exactly. Do **not** use `export LLVM_DEBUG=...`; LLVM’s DebugFlag is the command-line flag `-debug` / `-debug-only=<DEBUG_TYPE>`.

On a non-NDEBUG binary:

```bash
$BIN workspace/pass_analysis/cluster_barrier_insertion/repro_direct_alloc.mlir \
  --allocate-shared-memory -test-print-membar \
  -debug-only=cluster-barrier-insertion
```

Verified 2026-08-18 on this Method A RelWithDebInfo tree: `-debug-only=cluster-barrier-insertion` printed only the rewritten MLIR; `"Hello World"` was compiled out of the object and binary; `-debug` / `-debug-only=` are typically not registered (`triton-opt --help` will not list them).
