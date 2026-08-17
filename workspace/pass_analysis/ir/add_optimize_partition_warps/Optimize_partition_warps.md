# `add_optimize_partition_warps` (`OptimizePartitionWarps.cpp`)

| | |
|--|--|
| **Pass** | `tritongpu-optimize-partition-warps` (`TritonGPUOptimizePartitionWarps` / `add_optimize_partition_warps`) |
| **When** | Blackwell (`sm_100+`): after `add_warp_specialize` + `add_pipeline` |
| **Role** | shrink **specialized partition** warp counts when register pressure allows; never resizes **default** |
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

For each `ttg.warp_specialize`:

1. **Estimate** regs per partition from the largest tensor in that region (`getTensorNumI32Regs`, then `×2`).
2. **Halve** each partition’s `num_warps` (power-of-2) while a rough “even PTXAS share of 64K regs” still fits the tensors — iterate across partitions.
3. Enforce **per-op floors** (`minWarps`) so throughput / layout constraints are not violated.
4. If a partition’s warp count changed and it has tensors → **`relayoutWarps`**: lift region into a temp module, strip encodings, re-run convert-to-ttgpu + layout/matmul passes, put body back.
5. Write back `partitionNumWarps` and `requestedRegisters` guesses (88 if tensors, else 24).

It does **not** assign warp IDs / start indices. That is `AllocateWarpGroups`.

---

## 2. Default vs partitions: not the same warps

`ttg.warp_specialize` runs **concurrent** warp groups:

| Region | Warps | How sized |
|--|--|--|
| **default** | enclosing kernel warps (`ttg.num-warps` / `lookupNumWarps`) | Fixed by this pass — **never shrunk** |
| **partitions** | **extra** warps (`partitionNumWarps`) | Start equal to module `num_warps`; then optimized |

They do **not** share a pool of the same physical warps. CTA total ≈ `baseNumWarps + sum(partitionNumWarps)` (later padded to full warp groups).

**Initial assignment** (`PartitionLoops`): every `nvws.warp_group` region gets the same `lookupNumWarps(loop)`. After `LowerWarpGroup`, region 0 becomes `default` when its count matches module `num_warps`; remaining regions keep explicit `num_warps(...)`.

**Typical AWS matmul:** default holds wait-full / load / MMA (keeps full base, e.g. 4); TMA-only partitions shrink to 1–2. Heavy compute usually lives in **default**, not in a specialized partition — so “compute needs more warps” usually means default stays large while light partitions give warps/regs back.

---

## 3. Why floors: TMA → 2, TMEM → 4

```cpp
// OptimizePartitionWarps.cpp
if (isa<AsyncTMAGatherOp, AsyncTMAScatterOp, AsyncTMACopyGlobalToLocalOp>(op))
  *minWarps = 2;
else if (isa<TMEMLoadOp, TMEMStoreOp, TMEMAllocOp>(op))
  *minWarps = 4;
```

| Floor | Reason |
|--|--|
| **1** (default) | Scalar / tiny partitions can collapse to a single warp |
| **2** async TMA | Throughput: enough warps to issue TMA usefully |
| **4** TMEM load/store/alloc | Layout legality |

TMEM↔register layouts require **`numWarps >= 4`** (power of 2). Asserted in `getDistributedLayoutForTmemLdSt` (`TritonNvidiaGPU/IR/Dialect.cpp`): TMEM access is laid out over a warp-group-sized pattern (warp bases cover large chunks of the 128-row TMEM space). With fewer than 4 warps you cannot cover all lanes / produce a valid distributed encoding for `tmem_load` / `tmem_store`.

So a partition with TMEM will not shrink below 4 even if regs would allow 1–2.

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
OptimizePartitionWarps  shrink partitions (floors 1/2/4); relayout if needed
                        default still N; no start IDs yet
…
AllocateWarpGroups      warpGroupStartIds + ttg.total-num-warps
warp_specialize_to_llvm emit per-warp-group LLVM / setmaxnreg etc.
```

Lit coverage: `test/TritonGPU/optimize-partition-warps.mlir`, OPT checks in `test/TritonGPU/automatic-warp-specialization.mlir`.
