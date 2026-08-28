# `add_optimize_partition_warps` (`OptimizePartitionWarps.cpp`)

| | |
|--|--|
| **Pass** | `tritongpu-optimize-partition-warps` (`TritonGPUOptimizePartitionWarps` / `add_optimize_partition_warps`) |
| **When** | Blackwell (`sm_100+`): after `add_warp_specialize` + `add_pipeline` |
| **Role** | Shrink **partition** warp counts when a crude register model says tensors still fit; **never** resizes **default** |
| **Code** | `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/OptimizePartitionWarps.cpp` |
| **Also see** | AWS Rest2 (`LowerWarpGroup` / default vs partitions); `AllocateWarpGroups.cpp` (warp ID assignment, later) |

Pipeline position (`compiler.py` → `make_ttgir`):

```text
warp_specialize (AWS) → pipeline → optimize_partition_warps → …
        … much later (ttgir→llvm) …
allocate_warp_groups   ← assigns warpGroupStartIds / total-num-warps
```

---

## 1. What the pass does

### General idea

After warp specialization, some partitions are light (e.g. TMA-only) but still carry as many warps as the default. This pass **only shrinks those partition warp counts** when a crude model says their largest tensor still fits in registers if the SM’s **64K regs are shared evenly across all warps** (default + partitions). It does **not** change tensor shapes, does **not** touch the default’s warps, and does **not** assign warp IDs. If a partition’s warp count actually changes and it has tensors, layouts are reassigned for the new count.

### Entry (`runOnOperation`)

1. Collect every `ttg.warp_specialize`.
2. Build **`ModuleAxisInfoAnalysis`** once (used only if a partition is relayouted — capture contiguity/divisibility/constancy copied onto temp func args).
3. For each `wsOp`, call `optimizePartitionNumWarps`.

### Per `ttg.warp_specialize`

1. **Estimate tensor regs per partition**  
   Walk partition region types → `getTensorNumI32Regs` on each `RankedTensorType` (layout-aware: `elemsPerThread × threadsPerWarp × warpsPerCTA`, not raw `product(dims)`) → take **max**, then **`×2`** (“largest tensor ≈ half of warpgroup regs”).  
   This is intentionally coarse (no liveness / no PTXAS).

2. **Read warp counts**  
   - `defaultNumWarps = lookupNumWarps(wsOp)` → enclosing module `ttg.num-warps` (e.g. **4**).  
   - `partitionNumWarps = wsOp.getPartitionNumWarps()` → **only partitions**, e.g. **`[2]`** for one `partition0(... ) num_warps(2)`.  
   - Not `[4, 2]`: default’s 4 is **not** stored in `partitionNumWarps`.

3. **Per-partition floors `minWarps`** (start at 1):  
   - TMA load-like → **2**  
   - TMEM load/store/alloc → **4**

4. **Greedy shrink loop** (halve one partition per iteration while possible):  
   Assumption: SM RF **`nTotalRegs = 1<<16` (65536)** is split **evenly** across **all** warps (`default + sum(partitions)`), because the pass cannot rely on `nvvm.setmaxnreg` / known kernel `maxnreg`.

   ```text
   curTotal = defaultNumWarps + sum(partitionNumWarps)
   for each partition with numWarps > minWarps:
     reqRegs/thread  = tensorRegs / 32 / (numWarps / 2)   // need after halving
     nextTotal       = curTotal - numWarps/2
     nextRegs/thread = 65536 / 32 / nextTotal             // supply after even split
     if req <= next:
       numWarps /= 2; break   // only one shrink per outer iteration
   ```

   **Meaning:** shrink partition warps toward `minWarps` when, under “64K shared equally by every warp,” the largest tensor still needs ≤ that budget.  
   Not “maximize register usage”; not a hard no-spill guarantee.

5. **Write estimates + maybe relayout**  
   - `requestedRegisters[i] = tensorRegs ? 88 : 24` (fixed guess).  
   - If `newNumWarps != prev` **and** partition had tensors → **`relayoutWarps`**: lift body to temp module, strip encodings, re-run convert-to-ttgpu (`newNumWarps`) + coalesce / remove-layout-conversions / thread-locality / accelerate-matmul, put body back (AxisInfo on captures preserved).  
   - `setPartitionNumWarps` / `setRequestedRegisters`. Default warp count unchanged.

It does **not** assign warp IDs / start indices. That is `AllocateWarpGroups`.

### Concrete dump (WS matmul fp8)

IR: `.../05_after_ttgpuir_add_pipeline.ttgir___p_matmul_...`

| Quantity | Value |
|--|--|
| `ttg.num-warps` (default) | 4 |
| `partitionNumWarps` | `[2]` (`partition0`) |
| `curTotalNumWarps` | 4+2 = **6** |
| After this pass | still `num_warps(2)`; often `requestedRegisters = [24]` (producer partition, little/no fat tensor) |

Heavy `tensor<64x256xf32, #mma>` sits in **default**, so this pass usually cannot shrink `partition0` below `minWarps=2` (TMA).

---

## 2. Default vs partitions: not the same warps

`ttg.warp_specialize` runs **concurrent** warp groups:

| Region | Warps | How sized |
|--|--|--|
| **default** | enclosing kernel warps (`ttg.num-warps` / `lookupNumWarps`) | Fixed by this pass — **never shrunk** |
| **partitions** | **extra** warps (`partitionNumWarps`) | Tuned here (halve toward floors) |

They do **not** share a pool of the same physical warps. CTA total ≈ `baseNumWarps + sum(partitionNumWarps)` (later padded to full warp groups).

**Initial assignment** (`PartitionLoops`): every `nvws.warp_group` region gets the same `lookupNumWarps(loop)`. After `LowerWarpGroup`, region 0 becomes `default` when its count matches module `num_warps`; remaining regions keep explicit `num_warps(...)`.

**Typical AWS matmul:** default holds wait-full / load / MMA (keeps full base, e.g. 4); TMA-only partitions sit at 2 (floor) or shrink toward 1–2. Heavy compute usually lives in **default** — so “compute needs more warps” means default stays large while light partitions give warps back under the even-RF model.

---

## 3. Why floors: TMA → 2, TMEM → 4

```cpp
// OptimizePartitionWarps.cpp
if (isa<ttng::TMALoadLikeOpInterface>(op))
  *minWarps = 2;
else if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TMEMAllocOp>(op))
  *minWarps = 4;
```

| Floor | Reason |
|--|--|
| **1** (default) | Scalar / tiny partitions can collapse to a single warp |
| **2** async TMA load-like | Throughput: enough warps to issue TMA usefully |
| **4** TMEM load/store/alloc | Layout legality — need enough warps to cover TMEM lanes |

TMEM↔register layouts require **`numWarps >= 4`** (power of 2). Asserted in `getDistributedLayoutForTmemLdSt` (`TritonNvidiaGPU/IR/Dialect.cpp`): TMEM access is laid out over a warp-group-sized pattern. With fewer than 4 warps you cannot cover all lanes / produce a valid distributed encoding for `tmem_load` / `tmem_store`.

So a partition with TMEM will not shrink below 4 even if the register heuristic would allow 1–2.

---

## 4. Warp IDs are assigned later (`AllocateWarpGroups`)

After optimize you only know **counts**. Indices come in `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp` (near LLVM lowering):

1. Pad each `warp_specialize` so extra warps form full warp groups (multiples of 4); empty padding partitions get low `requestedRegisters`.
2. Set `warpGroupStartIds`: partitions sorted **largest-first**; starts at `baseNumWarps`, contiguous ranges of each size.
3. Default implicitly owns `[0, baseNumWarps)`.
4. Module attr `ttg.total-num-warps = base + numExtraWarpGroups * 4` (frontend may overwrite metadata `num_warps` from this).

```text
Example (base = 4, after optimize partition0 = 2):
  default:     warps 0..3
  partition0:  startId = 4, size 2  → warps 4..5
  (+ padding to fill warp groups as needed)
```

---

## 5. End-to-end sketch

```text
PartitionLoops          all regions num_warps = N (e.g. 4)
LowerWarpGroup          default = region0; partitions keep N
pipeline                async / SWP per region
OptimizePartitionWarps  estimate maxTensorRegs×2;
                        set minWarps (1/2/4);
                        greedy-halve partitions under even 64K split;
                        requestedRegisters = 88|24;
                        relayout if warps changed + tensors;
                        default still N; no start IDs yet
…
AllocateWarpGroups      warpGroupStartIds + ttg.total-num-warps
warp_specialize_to_llvm emit per-warp-group LLVM / setmaxnreg etc.
```

### Register model (keep in mind)

| Claim | Reality in this pass |
|--|--|
| Changes tensor dims | **No** |
| Shrinks partition warps | **Yes** (when heuristic allows) |
| Touches default warps | **No** |
| Maximizes reg usage | **No** — reduces warps when safe |
| Guarantees no spill | **No** — even-split heuristic only |
| Accurate RF accounting | **No** — max tensor ×2; fixed 88/24 hints |

SM90/100/110/120: classical RF is still **65536 × 32-bit regs/SM** (`1<<16`); SM100 may also use **TMEM** for MMA accum (orthogonal to this RF pool).

Lit coverage: `test/TritonGPU/optimize-partition-warps.mlir`, OPT checks in `test/TritonGPU/automatic-warp-specialization.mlir`.
