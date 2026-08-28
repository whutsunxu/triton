# ConvertNVGPUToLLVM (`convert-nv-gpu-to-llvm`)

| | |
|--|--|
| **When** | `nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm` in `make_llir`, **after** `add_warp_specialize_to_llvm` |
| **Code** | `third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp` |
| **Dump** | IR 18 `…/irs/18_after_ttnvgpuir_add_warp_specialize_to_llvm…` → IR 19 `…/irs/19_after_ttnvgpuir_add_nvgpu_to_llvm…` |
| **Lit** | `test/Conversion/nvgpu_to_llvm.mlir` |

This pass is **not** “NVVM dialect → LLVM.” It finishes leftover **Triton NVGPU** (`nvg.*`) and `ttg.warp_id`. NVVM ops (`nvvm.read.ptx.sreg.*`, `nvvm.barrier`, …) stay; they are first-class LLVM/NVPTX.

`runOnOperation` is three independent steps:

```text
1. greedy patterns   ClusterCTAId / WGMMA / LoadAcquire / WGMMAWait / WarpId
2. lowerTensorMemoryAlloc
3. makeAllWarpGroupsIsolatedFromAbove
```

This FP8 matmul (`cuda:120`, MMAv2, `ttg.tensor_memory_size = 0`) only **hits step 1 (`WarpIdOp`)**. Steps 2–3 are no-ops here; Hopper/Blackwell TMEM kernels hit them. Examples below use IR 18→19 for warp id and the lit file for TMEM / WGMMA.

---

## Problem 1 — leftover NVGPU / `ttg.warp_id` ops (lines 663–667)

```663:667:third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp
    patterns.add<ClusterCTAIdOpPattern, WGMMAOpPattern, LoadAcquireOpPattern,
                 WGMMAWaitGroupOpPattern, WarpIdOpPattern>(context);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();
```

### Problem

`ConvertTritonGPUToLLVM` and `ConvertWarpSpecializeToLLVM` already emit LLVM+NVVM, but a few ops are **deliberately left** for this pass:

| Op | Why it still exists |
|--|--|
| `ttg.warp_id` | WS lowering rewrites it to a **relative** id (`abs − startWarpId`) but keeps the op so NVGPUToLLVM emits `tid.x/32` + `shfl` |
| `nvg.cluster_id` | Cluster rank; not used in this 1-CTA kernel |
| `nvg.wgmma` / `nvg.wgmma_wait_group` | Hopper warpgroup MMA; this kernel is **MMAv2** `mma.sync`, not WGMMA |
| `nvg.ld_acquire` | Atomic acquire load (from `tt.atomic_rmw`); not in this matmul |

LLVM/NVPTX has no `ttg.warp_id`. TMA addressing in partition0 needs a **numeric warp id** (then subtract 4 because producer warps start at 4).

### Solution

Greedy rewrite to NVVM / inline PTX:

| Pattern | Result |
|--|--|
| `WarpIdOpPattern` | `nvvm.read.ptx.sreg.tid.x` / 32; optional `nvvm.shfl.sync idx …, 0` (warp-uniform). 1-warp region → constant `0` |
| `ClusterCTAIdOpPattern` | `nvvm.read.ptx.sreg.cluster.ctarank` (`NVVM::ClusterId`) |
| `WGMMAOpPattern` | `llvm.inline_asm` `wgmma.mma_async.sync.aligned.m{M}n{N}k{K}…` |
| `WGMMAWaitGroupOpPattern` | `llvm.inline_asm` `wgmma.wait_group.sync.aligned #pendings` (C registers tied in/out) |
| `LoadAcquireOpPattern` | `llvm.inline_asm` `ld.global.{cta\|gpu\|sys}.{acquire\|relaxed}.b{width}` |

`rewriteAsPtxAsm` unpacks LLVM structs into PTX regs and patches `#pendings`-style attrs into the asm string.

### Concrete example — this matmul (IR 18 → 19)

**Before** (producer TMA, dump 18). `ConvertWarpSpecializeToLLVM` already subtracted the group start (4):

```mlir
%w_91 = ttg.warp_id
%w_92 = llvm.mlir.constant(4 : i32) : i32
%w_93 = llvm.sub %w_91, %w_92 : i32
%w_94 = nvvm.elect.sync -> i1
…  // %w_93 used in TMA shared offset
```

**After** (dump 19). Same site: warp id is **this thread’s** `tid.x/32`, then `shfl` so PTXAS treats it as warp-uniform:

```mlir
%212 = nvvm.read.ptx.sreg.tid.x : i32
%213 = llvm.udiv %212, %120 : i32          // / 32
%214 = nvvm.shfl.sync idx %117, %213, %119, %118 : i32 -> i32
%215 = llvm.sub %214, %116 : i32           // − 4  (relative to producer group)
```

From Part 1 of the type notes: every thread runs this; `shfl idx 0` broadcasts lane 0’s warp id so the value is uniform.

**Not in this dump — WGMMA** (`test/Conversion/nvgpu_to_llvm.mlir` `@wgmma`):

```mlir
// before
%acc0 = nvg.wgmma %desc, %desc, %false { m = 64, n = 256, k = 32, … }
%out = nvg.wgmma_wait_group %in {pendings = 0} : !struct_64xf32

// after
llvm.inline_asm "wgmma.mma_async.sync.aligned.m64n256k32.f32.e5m2.e5m2 …"
llvm.inline_asm "// wait for regs: $0,$1,…\n\twgmma.wait_group.sync.aligned 0;"
```

**Not in this dump — cluster id:**

```mlir
%id = nvg.cluster_id
→  nvvm.read.ptx.sreg.cluster.ctarank
```

---

## Problem 2 — tensor memory base is a placeholder (line 669)

```669:669:third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp
    lowerTensorMemoryAlloc(mod);
```

### Problem

Blackwell TMEM is a **CTA-wide** allocation (`tcgen05.alloc`), not a per-op `local_alloc`. Earlier lowering emits a dummy:

```mlir
%base = nvg.tensor_memory_base  : !llvm.ptr<6>   // addr space 6 = TMEM
```

There is no hardware address until **one** `tcgen05.alloc` at kernel entry. Size lives on the module: `ttg.tensor_memory_size`.

This matmul: `ttg.tensor_memory_size = 0` → `initTensorMemory` returns empty → **no-op**. MMAv2 keeps accumulators in registers, not TMEM.

### Solution

If any `nvg.tensor_memory_base` exists:

1. Read `ttg.tensor_memory_size`. If 0, stop.
2. At **start of the kernel** (default warp group only: `tid.x < 32`):
   - optional cluster arrive/wait if two-CTA
   - predicated `tcgen05.alloc[.exclusive].cta_group::{1\|2}.sync.aligned.shared::cta.b32 [smem], SIZE`
   - `bar.sync`; load the 32-bit handle from that smem slot; `bar.sync`
   - `inttoptr` → `!llvm.ptr<6>`
   - `tcgen05.relinquish_alloc_permit…`
3. Before every `llvm.return`: `tcgen05.dealloc…`
4. Replace all `nvg.tensor_memory_base` results with that one pointer; erase the placeholders.

### Concrete example (`@tensor_memory_base_lowering`)

**Before**

```mlir
module attributes {ttg.tensor_memory_size = 128 : i32, ttg.target = "cuda:100", …}
llvm.func @tensor_memory_base_lowering() attributes {nvvm.kernel} {
  %263 = nvg.tensor_memory_base
  %264 = llvm.ptrtoint %263 : !llvm.ptr<6> to i32
  llvm.return %264 : i32
}
```

**After** (kernel entry + matching dealloc on return):

```mlir
%tid = nvvm.read.ptx.sreg.tid.x : i32
%pred = llvm.icmp "ult" %tid, 32
%smem = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
llvm.inline_asm "@$0 tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$1], 128;",
                "b,r" %pred, %smem
nvvm.barrier
%handle = llvm.load %smem : !llvm.ptr<3> -> i32
nvvm.barrier
llvm.inline_asm "@$0 tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "b" %pred
%ptr6 = llvm.inttoptr %handle : i32 to !llvm.ptr<6>
// … uses of %ptr6 …
// at return:
llvm.inline_asm "@$0 tcgen05.dealloc.cta_group::1.sync.aligned.b32 $1, 128;", "b,r" %pred, %ptr6
```

`SIZE == max TMEM columns` on Rubin can add `.exclusive`. Two-CTA modules use `cta_group::2` and cluster arrive/wait around alloc.

---

## Problem 3 — TMEM alloc lives above warp partitions (line 670)

```670:670:third_party/nvidia/lib/NVGPUToLLVM/NVGPUToLLVMPass.cpp
    makeAllWarpGroupsIsolatedFromAbove(mod);
```

**Code:** `lib/Conversion/TritonGPUToLLVM/Utility.cpp` (`makeWarpGroupsIsolatedFromAbove`).

### Problem

`ttg.warp_specialize.partitions` is **`IsolatedFromAbove`**: a partition region may not use SSA defined outside it except via **explicit captures**.

Step 2 allocates TMEM at **kernel entry** (outside partitions) and **replaces** in-partition `nvg.tensor_memory_base` with that outer `%ptr6`. That is exactly a forbidden capture.

(Lowerings in `ConvertTritonGPUToLLVM::finalizeModule` can do the same with function arguments; that is why isolate is also called there.)

On **this** matmul pipeline, `add_warp_specialize_to_llvm` already **inlined** partitions into one `llvm.func` (dump 18 has the mailbox switch, no `ttg.warp_specialize`). Walking WS ops finds nothing → **no-op**. Isolate still matters for:

- lit tests that run `--convert-nv-gpu-to-llvm` **with WS still present**
- any path that allocates TMEM before WS is inlined

### Solution

For each remaining `ttg.warp_specialize`:

```text
captures = values used in partition regions but defined above
for each capture:
  append it to warp_specialize.partitions operands
  add a matching block argument in every partition region
  replace in-region uses with that argument
```

Default region is not IsolatedFromAbove the same way; this helper only walks **partition** regions via `getPartitionOp()`.

### Concrete example (`@tensor_memory_base_warpgroup`)

**Before** (placeholder used inside partition, no captures):

```mlir
ttg.warp_specialize()
default {
  ttg.warp_yield
}
partition0() num_warps(1) {
  %0 = nvg.tensor_memory_base
  "use"(%0) : (!llvm.ptr<6>) -> ()
  ttg.warp_return
} : () -> ()
```

**After** step 2 + step 3: alloc at kernel entry, then the pointer is an **explicit capture**:

```mlir
// kernel entry: tcgen05.alloc … → [[PTR]] : !llvm.ptr<6>
ttg.warp_specialize([[PTR]])
default { ttg.warp_yield }
partition0(%arg0: !llvm.ptr<6>) num_warps(1) {
  "use"(%arg0) : (!llvm.ptr<6>) -> ()
  ttg.warp_return
}
```

Without this, the module would be invalid IsolatedFromAbove IR (or later WS-to-LLVM would miss packing that pointer into the capture scratch at `allocation.offset`).

---

## What this kernel’s dump 19 actually changed

| Step | This matmul (`cuda:120`, TMEM size 0, WS already inlined) |
|--|--|
| 1 WarpId | **Yes** — `ttg.warp_id` → `tid.x/32` + `shfl` (TMA producer offsets) |
| 1 WGMMA / cluster / ld_acquire | No matching ops |
| 2 TMEM alloc | Skipped (`tensor_memory_size = 0`, no `nvg.tensor_memory_base`) |
| 3 Isolate WS | No `ttg.warp_specialize` left |

Dump 19 vs 18 also shows CSE (constants hoisted, args `%arg0`…) from `triton-opt` / greedy rewriter, not from TMEM.

---

## Cheat sheet

| Input | After this pass |
|--|--|
| `ttg.warp_id` | `tid.x / 32` (+ `shfl.sync idx 0` unless `omitUniformHint`) |
| `nvg.cluster_id` | `nvvm.read.ptx.sreg.cluster.ctarank` |
| `nvg.wgmma` | `llvm.inline_asm` `wgmma.mma_async…` |
| `nvg.wgmma_wait_group` | `llvm.inline_asm` `wgmma.wait_group.sync.aligned N` |
| `nvg.ld_acquire` | `llvm.inline_asm` `ld.global.*.acquire.b*` |
| `nvg.tensor_memory_base` | one `tcgen05.alloc` at entry, `!llvm.ptr<6>`, dealloc on return |
| partition use of that ptr | extra `warp_specialize` capture / block arg |
