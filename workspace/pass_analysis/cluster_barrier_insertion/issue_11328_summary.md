# Issue 11328: ClusterBarrierInsertion misses `memdesc_index` views

Upstream: [triton-lang/triton#11328](https://github.com/triton-lang/triton/issues/11328)

Folder: `workspace/pass_analysis/cluster_barrier_insertion/`
Date: 2026-08-17
Tree: `matmul_perf_analysis` (`ClusterBarrierInsertion.cpp` still uses `getBufferIds` in the effect-collection loop, ~line 372).

---

## 1. Which pass?

**Not** the fence dumps (`add_fence_insertion` / `add_proxy_fence_insertion`). Those insert `ttng.fence_async_shared` for generic vs async **proxy** ordering.

This bug is **cluster** SMEM ordering:

| Stage in `compiler.py` | Role |
|------------------------|------|
| `add_allocate_shared_memory_nv` | Builds `Allocation`: `valueBuffer` keyed on `ttg.local_alloc`, views (`ttg.memdesc_index`) only in `aliasBuffer`. Assigns `allocation.offset`. |
| **`add_to_llvmir`** (`ConvertTritonGPUToLLVM`) | Calls `runClusterBarrierInsertion()` then `ModuleMembarAnalysis`. **This is where the missing `ttng.cluster_barrier` should appear.** |
| `add_warp_specialize_to_llvm` | Later; dump `18_…` is after WS lowering, not this analysis. |

`test-print-membar` (used by `test/TritonNvidiaGPU/membar-cluster.mlir`) runs the same cluster-barrier + membar sequence after `--allocate-shared-memory`.

The FP8 `block_m=64` kernel is **`ttg.num-ctas = 1`**, so this cluster/multicast path does not run there.

---

## 2. Bug (short)

`ClusterBarrierAnalysis::update` walks memory effects with `allocation->getBufferIds(value)` (`valueBuffer` only). A TMA dest that is `%dst = ttg.memdesc_index %bufs[%c0]` has its buffer id only in `aliasBuffer`, so the write is **dropped**. The same file already uses `getAllBufferIdsWithAliases` at the other sites. `Membar.cpp` uses the aliasing API, so a **local** `ttg.barrier` can still appear while the **cluster** barrier does not.

Naive fix (`getAllBufferIdsWithAliases` at that loop) makes the missing barrier appear, then hits:

```text
LLVM ERROR: scratch buffer operations should not have any shared memory dependencies
```

on `@local_gather_subslice_other_cta` (`test/Conversion/tritonnvidiagpu_to_llvm.mlir`), because scratch ops can then also have explicit SMEM deps. `Membar.cpp` already has a carve-out; cluster analysis does not.

---

## 3. Command (local)

Reproduced **2026-08-17** inside container `ir_dev` (image `ir_dev_wip_ncu`), repo `/Volumes/case_sensitive_workspace/triton`, branch `matmul_perf_analysis`. No GPU needed.

```bash
TRITON_OPT=build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
DIR=workspace/pass_analysis/cluster_barrier_insertion

$TRITON_OPT $DIR/repro_indexed_view.mlir \
  --allocate-shared-memory -test-print-membar \
  -o $DIR/indexed_view.after.mlir

$TRITON_OPT $DIR/repro_direct_alloc.mlir \
  --allocate-shared-memory -test-print-membar \
  -o $DIR/direct_alloc.after.mlir
```

Same pipeline as `test/TritonNvidiaGPU/membar-cluster.mlir`. Both commands **exit 0**; the bug is a missing op, not a crash.

---

## 4. Local reproduce result

Both place the TMA buffer and the reuse buffer at **`allocation.offset = 0`** (same bytes).

| Case | File | Before reuse `local_store` |
|------|------|----------------------------|
| Direct `local_alloc` dest (control) | `direct_alloc.after.mlir` | **`ttng.cluster_barrier`** |
| `memdesc_index` view dest (bug) | `indexed_view.after.mlir` | only **`ttg.barrier local`** — **no cluster barrier** |

### 4.1 Indexed view (bug) — `indexed_view.after.mlir`

TMA dest is `%2 = ttg.memdesc_index %1[%c0]`. Reuse `%3` shares offset 0. Intra-CTA `ttg.barrier local` is inserted; **`ttng.cluster_barrier` is not** before the store (only at kernel exit).

```mlir
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @case_indexed_view_reuse(...) {
    %0 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.init_barrier %0, 1 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %2 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_barrier {relaxed = true}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %2, %0, %true {multicast} : ...
    ttng.wait_barrier %0, %c0_i32 deps %2 : ...
    ttg.local_dealloc %1 : ...
    %3 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttg.barrier local
    ttg.local_store %arg1, %3 : ...          // no ttng.cluster_barrier here
    ttg.local_dealloc %3 : ...
    ttg.local_dealloc %0 : ...
    ttng.cluster_barrier                     // only at kernel exit
    tt.return
  }
}
```

### 4.2 Direct alloc (control) — `direct_alloc.after.mlir`

TMA dest is the `local_alloc` itself. Same offset-0 reuse; **`ttng.cluster_barrier` is inserted** between reuse alloc and store.

```mlir
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @case_direct_alloc_reuse(...) {
    %0 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.init_barrier %0, 1 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_barrier {relaxed = true}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %1, %0, %true {multicast} : ...
    ttng.wait_barrier %0, %c0_i32 deps %1 : ...
    ttg.local_dealloc %1 : ...
    %2 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttng.cluster_barrier                     // present
    ttg.local_store %arg1, %2 : ...
    ttg.local_dealloc %2 : ...
    ttg.local_dealloc %0 : ...
    tt.return
  }
}
```

`wait_barrier` does not cover multicast writes into the **other** CTA’s SMEM (same comment as `membar-cluster.mlir`). A local `ttg.barrier` is not a cluster barrier.

Full dumps: `indexed_view.after.mlir`, `direct_alloc.after.mlir`.

---

## 5. Files

| File | Role |
|------|------|
| `repro_indexed_view.mlir` | Issue IR (input) |
| `repro_direct_alloc.mlir` | Control input (direct `local_alloc` dest) |
| `indexed_view.after.mlir` | Local `triton-opt` output (bug) |
| `direct_alloc.after.mlir` | Local `triton-opt` output (control) |
| `issue_11328_summary.md` | This note |
