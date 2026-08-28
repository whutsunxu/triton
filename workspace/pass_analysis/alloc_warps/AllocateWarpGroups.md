# AllocateWarpGroups

| | |
|--|--|
| **Pass** | `triton-gpu-allocate-warp-groups` |
| **Code** | `lib/Conversion/TritonGPUToLLVM/AllocateWarpGroups.cpp` |
| **When** | Late `make_ttgir` (dump **14** after dump **13** fence insertion) |
| **Example IR** | `.../irs/13_after_ttnvgpuir_add_fence_insertion...` → `.../irs/14_after_ttgpuir_add_allocate_warp_groups...` |

---

## 1. General idea

### Problem

`ttg.warp_specialize` runs a **default** region (module `ttg.num-warps`) plus **extra partitions** with their own `num_warps`. On Hopper+, hardware and `setmaxnreg` work in **warpgroups of 4 warps**. Before LLVM lowering, the compiler must:

1. Make the partition side use a **whole number of warpgroups** (pad if needed).
2. Assign each partition a **starting warp ID** in the CTA.
3. Optionally **redistribute registers**: lean partitions keep low limits; surplus goes to the default region (matmul / consumer).

### Key hinge

A **warpgroup (WG)** = **4 consecutive warps**. Partition warps are counted **separately** from default warps:

```text
baseNumWarps          = ttg.num-warps          // default region
maxExtraWarps         = Σ partition num_warps  // non-default only
numExtraWarpGroups    = ceil(maxExtraWarps / 4)
ttg.total-num-warps   = baseNumWarps + numExtraWarpGroups × 4
```

### What the pass does (three phases)

| Phase | Action | IR effect (this matmul) |
|-------|--------|-------------------------|
| **Warp padding** | Pad partitions to `numExtraWarpGroups × 4` warps with dummy regions | `partition1 num_warps(2)` added |
| **Warp index set** | Sort partitions large→small; assign `warpGroupStartIds` from `baseNumWarps` | `warpGroupStartIds = [4, 6]` |
| **Register budget** | If `requestedRegisters` exists: group into WGs, compute surplus, set `actualRegisters` + `ttg.maxnreg` | `actualRegisters = [488, 24, 24]`, `maxnreg = 256` |

### Running example (fp8 matmul dumps 13 → 14)

**Before (IR 13):**

```mlir
module attributes {"ttg.num-warps" = 4 : i32, ...}

%0 = ttg.warp_specialize(...) attributes {
  requestedRegisters = array<i32: 24>
}
default { ... }                    // uses 4 warps
partition0(...) num_warps(2) { ... }  // TMA producer only
```

**After (IR 14):**

```mlir
module attributes {
  ttg.maxnreg = 256 : i32,
  "ttg.num-warps" = 4 : i32,
  "ttg.total-num-warps" = 8 : i32, ...
}

%0 = ttg.warp_specialize(...) attributes {
  actualRegisters = array<i32: 488, 24, 24>,
  requestedRegisters = array<i32: 24, 16>,
  warpGroupStartIds = array<i32: 4, 6>
}
default { ... }
partition0(...) num_warps(2) { ... }   // real TMA work
partition1(...) num_warps(2) { ... }   // pad: warp_return only
```

CTA layout:

```text
warps 0..3  → default (matmul / local_load consumer)
warps 4..5  → partition0 (TMA)
warps 6..7  → partition1 (padding)
```

---

## 2. Warp padding

### Goal

Ensure every `ttg.warp_specialize`’s **partition side** uses the same number of full warpgroups as the **largest** partitioning in the module, so all threads can be present to **surrender registers** (`setmaxnreg`).

### Steps

```cpp
maxExtraWarps = max over WS ops of getTotalPartitionWarps()
              // = sum of partition num_warps only (not default)

numExtraWarpGroups = divideCeil(maxExtraWarps, 4)   // ceiling to whole WGs

for each WS op:
  padToMaxWarpGroups(op, numExtraWarpGroups)
```

`getTotalPartitionWarps()` = `Σ partitionNumWarps` (default warps **excluded**).

### Target and how pads are sized

```text
target partition warps = numExtraWarpGroups × 4
warpsToAdd             = target − current total partition warps
```

Pads are **not** always a single partition. The pass splits `warpsToAdd` into **power-of-two** sizes:

```cpp
while (warpsToAdd > 0) {
  paddingSize = NextPowerOf2(warpsToAdd) / 2;  // largest power-of-2 ≤ remaining
  paddingPartitionSizes.push_back(paddingSize);
  warpsToAdd -= paddingSize;
}
```

Each size becomes one new region on `ttg.warp_specialize.partitions` with only `ttg.warp_return`, and `requestedRegisters` gets a trailing **16** per pad.

| `warpsToAdd` | `paddingPartitionSizes` |
|--------------|-------------------------|
| 2 | `[2]` |
| 3 | `[2, 1]` |
| 5 | `[4, 1]` |

### IR walkthrough

| Quantity | Value |
|----------|-------|
| `partition0` warps | 2 |
| `maxExtraWarps` | **2** (not 6 = 4+2) |
| `numExtraWarpGroups` | `ceil(2/4)` = **1** |
| target | 4 |
| `warpsToAdd` | 2 → pad list `[2]` |

**Where partitions live:** `WarpSpecializeOp` holds `defaultRegion` plus a `partitionOpHolder` region whose only op is `ttg.warp_specialize.partitions`. Pretty-printed `partition0` / `partition1` are **regions** of that container, not the return value of `getPartitionOpHolder().front().front()` (that returns the **container**).

**Unchanged by padding:** `ttg.num-warps` / `baseNumWarps` stay **4**.

---

## 3. Warp index set (`warpGroupStartIds`)

### Goal

Assign each partition a **starting warp ID** in the CTA so hardware knows which physical warps execute which region. Larger partitions get **lower** start IDs (closer to the default block).

### Algorithm (after padding)

```cpp
baseNumWarps = lookupNumWarps(mod)   // ttg.num-warps = 4

arr = getPartitionNumWarps()         // after pad: [2, 2]
idxAndSize = [(0, 2), (1, 2)]        // (partition index, size)

sort idxAndSize by size descending

startId = baseNumWarps
for (i, size) in sorted order:
  startIds[i] = startId
  startId += size

op.setWarpGroupStartIds(startIds)
mod["ttg.total-num-warps"] = baseNumWarps + numExtraWarpGroups * 4
```

### IR walkthrough

```text
sorted: [(0, 2), (1, 2)]
startIds[0] = 4; startId = 6
startIds[1] = 6; startId = 8
```

Stored on the **parent** `ttg.warp_specialize` (easy to miss — not printed on `partitionN` lines):

```mlir
warpGroupStartIds = array<i32: 4, 6>
"ttg.total-num-warps" = 8 : i32
```

```text
default:     warps [0, 4)
partition0:  warps [4, 6)
partition1:  warps [6, 8)
```

---

## 4. Warp groups and register allocation / budget check

### Goal

Hopper `setmaxnreg` applies **per warpgroup (4 warps)**, not per warp. Lean partitions (TMA, pads) need few registers; the **default** region (dot / consumers) wants as many as possible.

This phase:

1. Group padded partitions into hardware warpgroups.
2. Account **surplus / deficit** vs a CTA-wide per-thread ceiling (`maxnreg`).
3. If the budget closes, write **`actualRegisters`**: low limits on partitions, leftover on default.

If estimates are missing or the math does not close, **skip** this phase only — padding and start IDs stay.

### Steps (with IR)

**Inputs after padding + index set (this matmul):**

```mlir
// module
"ttg.num-warps" = 4 : i32
"ttg.total-num-warps" = 8 : i32
"ttg.threads-per-warp" = 32 : i32

// warp_specialize (after pad)
requestedRegisters = array<i32: 24, 16>   // p0 estimate, pad=16
warpGroupStartIds  = array<i32: 4, 6>
partition0 num_warps(2)
partition1 num_warps(2)
```

`requestedRegisters` comes from `OptimizePartitionWarps` (heuristic **24** / **88**). The `i32` in `array<i32: …>` is attribute encoding; values are **registers per thread**, not “regs of type i32.”

---

**Step 1 — Enter only if estimates exist**

```cpp
if any WS has requestedRegisters → continue
else return   // no actualRegisters
```

Also per op: `totalPartitionWarps % 4 == 0` (true after pad).

IR 13 already had `requestedRegisters = array<i32: 24>` → this matmul **enters**.

---

**Step 2 — Compute per-thread ceiling `maxnreg`**

```text
maxnreg = existing ttg.maxnreg
       or (64*1024) / totalNumWarps / threadsPerWarp   // snap to ×8
```

```text
(64*1024) / 8 / 32 = 256
```

Later written on the module:

```mlir
ttg.maxnreg = 256 : i32    // max registers per thread
```

---

**Step 3 — Bucket partitions into warpgroups**

Sort partitions by `startId` ascending. Open a new `WarpGroupInfo` when `startId % 4 == 0`. Within a WG: `maxRequestedRegs = max(ceil8(est))`, `numWarps += size`.

Why WG not per-warp: one register ceiling per 4-warp hardware group.

```text
partition0 startId=4, req=24  → 4%4==0 → WG#0
partition1 startId=6, req=16  → same WG#0

WG#0: numWarps=4, maxRequestedRegs=24
```

---

**Step 4 — Build CTA register budget (surplus / deficit)**

Units = register × thread:

```text
registerBudget = maxnreg × baseNumWarps × threadsPerWarp
for each partition WG:
  registerBudget += (maxnreg − maxRequestedRegs) × wg.numWarps × threadsPerWarp
```

| `(maxnreg − requested)` | Meaning |
|-------------------------|---------|
| **> 0** | Surplus from lean WG → can give default more |
| **< 0** | Deficit → shrinks default |
| **= 0** | Neutral |

```text
budget = 256×4×32 + (256−24)×4×32
       = 32768 + 29696
       = 62464
```

Equivalent view: full pool at `maxnreg` for all 8 warps, minus what partition WG actually keeps.

---

**Step 5 — Budget checks (safe abort)**

```cpp
if (registerBudget <= 0) return;   // nothing left for default — do not invent a split
leftover = registerBudget / (baseNumWarps × threadsPerWarp)
leftover = leftover / 8 * 8        // snap down
if (leftover < minRegsForDefault) return;  // default would be too lean
```

```text
leftover = 62464 / (4×32) = 488   // already ×8
```

`return` here only skips **`actualRegisters`** for this op. Abort when `budget ≤ 0` means no surplus to redistribute — not “leave free regs unused.”

---

**Step 6 — Write `actualRegisters`**

```text
actualRegisters[0]     = leftover              // default (dot / consumers)
actualRegisters[1 + i] = WG.maxRequestedRegs   // each partition (from its WG)
```

**IR 14 result:**

```mlir
module attributes {ttg.maxnreg = 256 : i32, "ttg.num-warps" = 4 : i32,
                   "ttg.total-num-warps" = 8 : i32, ...}

ttg.warp_specialize(...) attributes {
  actualRegisters = array<i32: 488, 24, 24>,  // default, p0, p1
  requestedRegisters = array<i32: 24, 16>,
  warpGroupStartIds = array<i32: 4, 6>
}
```

| Slot | Who | Regs/thread |
|------|-----|-------------|
| `[0]=488` | default (matmul / `local_load` / `tt.dot`) | surplus |
| `[1]=24` | `partition0` (TMA) | WG max request |
| `[2]=24` | `partition1` (pad) | same WG ceiling |

Later lowering emits `setmaxnreg` from `actualRegisters`: partitions stay low; default gets the surplus.

---

## 5. Quick reference

```text
IR 13 → AllocateWarpGroups → IR 14

maxExtraWarps=2, numExtraWarpGroups=1, baseNumWarps=4
pad [2] → partition1
warpGroupStartIds [4, 6]
total-num-warps = 8
maxnreg = 256 (per thread)
actualRegisters = [488, 24, 24]
```

| Attribute | Scope | Meaning |
|-----------|--------|---------|
| `ttg.num-warps` | module | Default / base warps (unchanged by pad) |
| `ttg.total-num-warps` | module | Base + padded partition warps |
| `ttg.maxnreg` | module | Max registers **per thread** (NVVM `maxnreg`) |
| `requestedRegisters` | WS op | Per-partition estimates (+ 16 for pads) |
| `warpGroupStartIds` | WS op | Start warp ID per partition |
| `actualRegisters` | WS op | `[default, p0, p1, …]` after budget |

**Mental checklist:**

1. Extra warps = **partitions only**; round up to ×4 via padding.
2. Start IDs live on `warp_specialize`, not on `partitionN` syntax.
3. Register math is **per warpgroup (4 warps)**; surplus from lean WGs feeds default.
4. No `requestedRegisters` ⇒ skip register phase; pad + indices still apply.
