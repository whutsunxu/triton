# TritonGPU → LLVM type conversion

| | |
|--|--|
| **When** | `ConvertTritonGPUToLLVM::runOnOperation`, after `prepareModule` / `lowerFunctions` |
| **Code** | `lib/Conversion/TritonGPUToLLVM/TypeConverter.cpp`, `include/triton/Conversion/TritonGPUToLLVM/TypeConverter.h` |
| **NVIDIA addr spaces** | `TargetInfo::getAddressSpace` / `getSharedAddressSpace` (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.cpp`) |
| **NVVM enum** | `mlir/Dialect/LLVMIR/NVVMOps.td` (`NVVMMemorySpace`), `llvm/Support/NVPTXAddrSpace.h` |
| **Dot / MMA** | `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM.cpp`, `DotOpToLLVM/MMAv2.cpp` |
| **CF / WS / barriers** | `lowerControlFlow` (`cf`→LLVM); `ConvertWarpSpecializeToLLVM.cpp` (after this pass); `BarrierOpToLLVM.cpp` (`populateBarrierOpToLLVMPatterns`) |
| **Example IR** | TTGIR `.../irs/17_after_ttnvgpuir_add_proxy_fence_insertion...` → LLVM `.../irs/18_after_ttnvgpuir_add_warp_specialize_to_llvm...` |

**Part 1 is the root rule** every later mapping obeys: TTGIR describes a **distributed tile for the program** (CGA / cluster); LLVM describes **what this thread executes**. Types, memdescs, and `tt.dot` are just that split applied to values and ops.

The type converter is **not** op lowering. Dialect conversion asks it “what LLVM type does this TTGIR value become?” Shape, layout, and contiguity stay on the **MLIR type of the op**; they are **not** stored in the LLVM type.

```text
TTGIR value type                    LLVM value type
────────────────────────────────────────────────────────────────
tensor<T, #layout>              →   !llvm.struct<(T, T, …)>     this thread’s registers
!ttg.memdesc<…, #smem>          →   !llvm.struct<(ptr<3>, i32×rank)>   fat pointer + view offsets
!ttg.memdesc<…, #ttng.tensor_memory, TMEM enc>
                                →   !llvm.ptr<3>                bare TMEM addr (see below)
!tt.ptr<T>                      →   !llvm.ptr<1>                global
!tt.ptr<T, 0> / descriptor      →   !llvm.ptr                    generic
!tt.tensordesc<…>               →   !llvm.ptr                    generic (TMA object)
!ttg.async.token                →   i32
f8E4M3 / f8E5M2 / …             →   i8
scalars (i32, f32, …)           →   themselves; index → i32
```

Index width is forced to 32 (`LowerToLLVMOptions.overrideIndexBitwidth(32)`). Triton IR does not use MLIR `index` for tensors.

---

## Part 1 — Root rule: TTGIR is a cluster tile; LLIR is this thread

TTGIR (TritonGPU / TTNGIR) describes **one distributed tile for the whole program**. LLVM / NVVM describes **the SIMT kernel each thread runs**. MMA is warp-level hardware; the tile is already split to **this thread’s registers** before the opcode is emitted. The 64×256 object is **not** an LLVM value.

### Two description levels

```text
TTGIR  —  “the program owns tensor<64×256>”
          encoding says how CTA / warp / lane / register slice it

LLIR   —  “I am one thread; I hold struct<(f32, f32, …)>”
          mma.sync / bar.sync / cluster ops are the only collectives
```

Same kernel text is launched on **every thread**. TTGIR hides that. LLVM **is** that.

| | TTGIR | LLIR |
|--|--|--|
| What an SSA value is | Whole logical tile | **This thread’s** registers or one fat ptr |
| Who the op is for | All participants implied by the encoding | This thread, unless the opcode is collective |
| Where layout lives | On the type (`#mma`, `#blocked`, `CGALayout`) | **Gone** — baked into GEPs, shuffles, which MMA fragment |
| Shape 64×256 | Cluster tile (1 CTA ⇒ same as CTA) | Not a type; exists only if you **union all threads’ structs** |

Example module: `"ttg.num-ctas" = 1`, `"ttg.num-warps" = 4` → TTGIR tile = CTA tile. Multi-CTA is still **one** `tt.dot` on a larger tensor; each CTA’s LLVM only sees `shapePerCTA = shape / CTASplitNum`.

### How the tile is split (already done before LLVM)

```text
cluster tile   (TTGIR shape, CGALayout)
    ÷ CTASplitNum          →  this CTA’s 64×256
    ÷ warpsPerCTA [1,4]    →  this warp’s 64×64  (4 warps along N)
    ÷ 32 lanes + MMA map   →  this thread’s fragments
    remaining M/N/K        →  unrolled mma.sync  (repM×repN×repK)
```

LLVM does **not** re-split the cluster. It emits **this thread’s** leftover work.

### How to read LLIR: three nested “who”

Every instruction is one of:

**1. Thread-private (default).** `add`, `fmul`, `ld.global`, `gep @global_smem`, `extractvalue` of the tensor struct. Each thread has a different `tid.x` → different registers / pointers. A `tensor<256xf32>` load is 128 threads each issuing `ld.v2.f32` into **their** two f32s. There is no `tensor<256xf32>` in LLVM.

**2. Warp-collective (uniform in the warp).** `mma.sync.aligned.m16n8k32` — warp-level hardware. All 32 lanes execute the same instruction; each lane supplies its A/B/C **fragment**. One issue is a **16×8×32** chunk, not 64×256. Also `shfl.sync`, named `bar.sync` (WS warp-group).

**3. CTA / cluster collective.** Full-CTA `bar.sync`, `mbarrier.*`, `barrier.cluster`, `ld.shared::cluster`. `@global_smem` is **CTA-private** (plus a DSMEM window). Cluster is extra ops + `cluster.cta_id`, not a bigger LLVM tensor.

### Mental model

```text
TTGIR:  SPMD over a tensor   “everyone does dot(tile)”
          │  type conversion + layout lowering
          ▼
LLIR:   SIMT kernel          “I do my registers;
                              when I hit mma.sync / bar.sync I join my warp/CTA”
```

| Question | TTGIR | LLIR |
|--|--|--|
| Where is element (i,j)? | Encoding | Not stored; would recompute `toLinearLayout` |
| What does this SSA mean? | The tile | **My** fragment of the tile |
| Is MMA warp-level? | Hidden in `#mma` | **Yes** — `mma.sync`; tile split is **unroll + register packing** |

Parts 2–7 are this rule applied to address spaces, value types, and `tt.dot`.

---

## Part 2 — Memory buffer spaces

**From Part 1.** TTGIR names **which bank the tile lives in** (`#smem`, `#ttng.tensor_memory`, or a global `!tt.ptr`). LLVM has no tile: it only has a **per-thread pointer** with an NVVM integer `addr_space`. The bank mapping is that integer; layout stays on the MLIR type (Part 1: layout is gone from the LLVM type).

Three numbering systems. Do not mix them.

### Problem

LLVM / PTX pointers carry `addr_space`. TTGIR does **not**: a memdesc uses a **memory-space attribute**, and `!tt.ptr` uses a **different** enum that only covers global-like pointers. Shared and TMEM are never `!tt.ptr`.

### Solution — NVVM / NVPTX (`N` in `!llvm.ptr<N>`)

Same numbers in `NVVM::NVVMMemorySpace` and `llvm::NVPTXAS`:

| N | Name | PTX | Used for |
|--|--|--|--|
| **0** | Generic | generic | Unified / TMA descriptor object |
| **1** | Global | global | Device DRAM |
| **3** | Shared | shared | CTA smem (`@global_smem`) — Part 1 CTA-private blob |
| **4** | Constant | const | `__constant__` |
| **5** | Local | local | Per-thread stack |
| **6** | Tensor | tensor | Blackwell TMEM (`sm100+`) |
| **7** | SharedCluster | shared::cluster | DSMEM (`sm90+`) |

There is no `2`. `@global_smem` is `addr_space = 3`.

NVIDIA `TargetInfo` maps **both** `#smem` and `#ttng.tensor_memory` to **3**. Cluster DSMEM still uses `ptr<3>` plus **cluster-collective** ops (Part 1 §3), not NVVM 7. TMEM hardware is 6; lowering currently `inttoptr`s into `ptr<3>` (`TensorMemoryToLLVM.cpp` TODO).

### Solution — memdesc `memorySpace` (named attribute, not an integer)

```
!ttg.memdesc<shape, encoding, memorySpace, mutable?>
```

| Printed | Attribute | LLVM ptr space (NVIDIA) |
|--|--|--|
| `#smem` / `#ttg.shared_memory` | `SharedMemorySpaceAttr` | `ptr<3>` |
| `#ttng.tensor_memory` | `TensorMemorySpaceAttr` | `ptr<3>` (should be 6) |

`encoding` (`#shared`, `#ttg.nvmma_shared`, TMEM encoding, …) is **layout** (Part 1: who in the CTA owns which byte), not the bank.

### Solution — `!tt.ptr` `PtrAddrSpace` (`TritonTypeEnums.td`)

Only for scalar / tensor-of-ptr **global-side** pointers:

| Triton | Value | LLVM |
|--|--|--|
| `descriptor` | 0 | `!llvm.ptr` (generic 0) |
| `global` | 1 | `!llvm.ptr<1>` |
| `constant` | 4 | `!llvm.ptr<1>` on NVIDIA (AMD-ish; mapped to global) |

No shared `!tt.ptr`. Shared is always memdesc + `#smem`.

### Concrete example

```text
TTGIR                                          LLVM (this thread)
!tt.ptr<f8E5M2>                            →   !llvm.ptr<1>
!tt.tensordesc<1x64x256xf8E5M2, #shared>   →   !llvm.ptr          (generic)
!ttg.memdesc<2x64x128xf8E5M2, #shared1, #smem>
                                           →   !llvm.struct<(ptr<3>, i32, i32, i32)>
```

Every thread in the CTA holds the **same** smem fat pointer (CTA-shared bytes). Global `!tt.ptr` is also the same bit pattern per thread unless it is a **tensor of ptrs** (Part 3).

---

## Part 3 — `tensor<T, #layout>` → this thread’s registers

**From Part 1.** TTGIR `tensor<64x256xf32, #mma>` is the **cluster/CTA tile**. LLVM has no layout and no 64×256 buffer: conversion keeps **only this thread’s unique elements**.

### Problem

A TTGIR tensor is distributed: 16384 f32s across the CTA, one SSA value. LLVM SSA is per-thread.

### Solution

`convertTritonTensorType`:

```text
n = getUniqueElemsPerThread(type)     // LinearLayout “register” dim, broadcast bases stripped
→  !llvm.struct<(convert(T), …)>      // n fields, same converted element type
```

Layout / shape stay on the **MLIR type of the still-unconverted op** so load/store / MMA can recompute who owns which coord. The LLVM struct is a bag of scalars.

`getUniqueElemsPerThread` ≠ `getTotalElemsPerThread` when the layout **broadcasts**. Conversion packs **unique** registers.

### Concrete example

TTGIR (`#blocked1` = `{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4]}`):

```mlir
%cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked1>
```

Part 1 split: 256 elements ÷ 128 threads = **2 per thread**. LLVM:

```mlir
!llvm.struct<(f32, f32)>
```

A 2-D MMA accumulator `tensor<64x256xf32, #mma>` with `warpsPerCTA = [1, 4]`: CTA tile 64×256, 4 warps along N, 32 lanes → 128 unique f32s **per thread** → `struct<(f32, …)>` with 128 fields (not a 64×256 alloc). Union of all threads’ structs **is** the TTGIR tile.

Element conversion is recursive: `tensor<…xf8E5M2, #…>` → struct of `i8`.

---

## Part 4 — `!ttg.memdesc` → fat pointer (not the bytes)

**From Part 1.** TTGIR memdesc is the **CTA’s (or cluster-view of) buffer tile** plus layout. LLVM cannot hold the tile: every thread gets the **same** handle `{ smem base, view origin }`. Addressing at `local_load` still uses the **MLIR encoding** (Part 1: layout not in the LLVM type). Bytes live in `@global_smem`, which is CTA-private.

### Problem

`local_alloc` / TMA / TMEM need a handle to a buffer **view**. Putting shape/strides in the LLVM type would duplicate compile-time layout and would not model swizzle (XOR of bit bases).

### Solution

`convertMemDescType`:

1. `ptrType = !llvm.ptr< targetInfo.getAddressSpace(memorySpace) >` → `#smem` ⇒ `ptr<3>`.
2. **TMEM encoding** (`TensorMemoryEncodingAttr` / `TensorMemoryScalesEncodingAttr`): return **bare** `ptrType` (no offset fields). TMEM addressing is a packed i32 (col | row<<16), not dim offsets.
3. Else:
   - `numBases = 1`, or `PartitionedSharedEncodingAttr.getNumPartitions()`;
   - then **rank** `i32`s = **logical view offsets** (origin of this subview), **not** extents.

```text
!ttg.memdesc<D0xD1x…xT, enc, #smem>
    →  !llvm.struct<(ptr<3> × numBases, i32 × rank)>

alloc:     offsets all 0; ptr = @global_smem + allocation.offset
index[i]:  GEP ptr by i × sizeof(one slice); drop dim 0
subslice:  add into the i32s; ptr usually unchanged
```

Contiguity / swizzle come from **encoding + shape on the memdesc type** at `local_load` / `local_store` (`toLinearLayout` → physical element offset **for this thread’s coords**). The i32s are tensor indices.

### Concrete example

TTGIR (CTA-wide tile in the 99396-byte blob):

```mlir
%x = ttg.local_alloc {allocation.offset = 65536 : i32}
  : () -> !ttg.memdesc<2x64x128xf8E5M2, #shared1, #smem, mutable>
```

LLVM — **every thread** in the CTA builds the same handle:

```mlir
%37 = llvm.mlir.addressof @global_smem : !llvm.ptr<3>
%x_22 = llvm.getelementptr %37[65536] : (!llvm.ptr<3>) -> !llvm.ptr<3>, i8
// pack { ptr, off0=0, off1=0, off2=0 }
!llvm.struct<(ptr<3>, i32, i32, i32)>
```

`local_load` then uses `#shared1` + this thread’s lane/warp to pick **which bytes** of that tile go into **this thread’s** register struct (Part 3).

A later `memdesc_index %x[%c1]` → rank-2 `memdesc<64x128x…>` → `struct<(ptr<3>, i32, i32)>` with the pointer advanced by one 64×128 slot.

Barrier objects `memdesc<2x1xi64, #shared2, #smem>` → `struct<(ptr<3>, i32, i32)>`.

---

## Part 5 — `!tt.ptr<T>` → LLVM pointer

**From Part 1.** A scalar `!tt.ptr` is **not** a distributed tile: every thread sees the same kernel argument (or the same bit pattern after `splat`). A **tensor of pointers** is a tile (Part 3): LLVM becomes a struct of `ptr<1>` — **this thread’s** pointers only. AxisInfo (contiguity / divisibility) is computed on the TTGIR tensor so `tt.load` can emit `ld.v2` vs scalar; that analysis cannot live on the LLVM struct.

### Problem

Kernel arguments and `tt.load` / `tt.store` use scalar (or tensor-of) pointers into **global** memory. Those must become NVPTX `ptr<1>` (or generic 0 for a TMA descriptor **object**).

### Solution

```text
!tt.ptr<T>                 (PtrAddrSpace::Global = 1)   →  !llvm.ptr<1>
!tt.ptr<T, 0> / descriptor (PtrAddrSpace::Descriptor)   →  !llvm.ptr      // generic
!tt.ptr<T> constant        (PtrAddrSpace::Constant = 4) →  !llvm.ptr<1>   // NVIDIA
```

Pointee type is **erased**: LLVM pointers are opaque. `tt.pointee_type` may remain as an attribute on kernel args.

`tensor<!tt.ptr<T>, #layout>` is Part 3: struct of `ptr<1>`s (one per unique element **this thread** holds).

### Concrete example

```mlir
// TTGIR — same pointer for every thread in the grid
%YPtr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}

// LLVM
%YPtr: !llvm.ptr<1> {tt.divisibility = 16 : i32, tt.pointee_type = f8E5M2}
```

TMA desc **byval** kernel args are not `!tt.ptr`; they are `!tt.tensordesc` (Part 6) lowered with `llvm.byval = array<128 x i8>`, `nvvm.grid_constant`.

---

## Part 6 — Others

**From Part 1.** Anything that is not a distributed tile maps 1:1 per thread (scalar, token, TMA handle). Anything that **is** a tile still follows Part 3 (struct of this thread’s pieces).

### `!tt.tensordesc` / `!ttng.tensor_desc_im2col` → generic `ptr`

TMA descriptor **handle** (128-byte object). The **block shape / swizzle** on the MLIR type is the cluster/CTA TMA tile; LLVM only keeps a generic pointer. Every thread in the CTA typically holds the same handle.

```mlir
// TTGIR
%Y: !tt.tensordesc<1x64x256xf8E5M2, #shared>

// LLVM (type); kernel arg also byval 128 bytes
%Y: !llvm.ptr {llvm.byval = !llvm.array<128 x i8>, nvvm.grid_constant, tt.nv_tma_desc = 1 : i32}
```

`TensorDescType` and `TensorDescIm2ColType` both convert to `LLVM::LLVMPointerType::get(ctx, 0)`.

### `!ttg.async.token` → `i32`

SSA token linking `async_copy` / `async_wait`. No hardware object; lowering uses an integer phase / dummy. Per-thread dummy; the **wait** is CTA/async collective.

### FP8 → `i8`

`f8E4M3FN`, `f8E4M3FNUZ`, `f8E5M2`, `f8E5M2FNUZ` → `i8`. NVPTX has no first-class f8; ops bitcast.

### Builtin scalars

`i1`/`i8`/`i16`/`i32`/`i64`, `f16`/`bf16`/`f32`/`f64` pass through the default LLVM type converter. `index` → `i32`. These are already Part 1 “thread-private.”

### What is **not** a converted value type

| IR thing | Where it goes |
|--|--|
| `#ttg.blocked` / `#ttg.nvmma_shared` / `#mma` | Stays on the MLIR tensor/memdesc type; patterns read it (Part 1 layout) |
| `ttg.shared = 99396` | Module attr → launch dynamic smem size (CTA blob) |
| `allocation.offset` | Op attr → `gep @global_smem, offset` |
| `@global_smem` | Module `llvm.mlir.global` in addrspace 3; not a TTGIR type |

---

## Part 7 — `tt.dot` mapping (same rule, warp MMA)

**From Part 1.** TTGIR `tt.dot` is one op on the **cluster/CTA tile**. LLVM has no such value: each thread holds A/B/C **fragments** (Part 3 structs). `mma.sync` is **warp-collective**; the leftover tile after CTA×warp split is an **unrolled** loop of those instructions. The PTX mnemonic is a **dtype table lookup**; the loop counts are not.

### TTGIR (cluster / CTA story)

```mlir
%acc = tt.dot %x, %w, %acc
  : tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
  * tensor<128x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  -> tensor<64x256xf32, #mma>
```

One op = whole CTA tile when `num-ctas = 1` (`CGALayout` omitted / `block = []`). Operand types are Part 3 tensors; lowering unpacks them to this thread’s registers.

### Split (not invented in LLVM)

```text
shapePerCTA = TTGIR shape / CTASplitNum     // still 64×256×128 here
repM = M / (16 × warps_along_M) = 64 / (16×1) = 4
repN = N / ( 8 × warps_along_N) = 256 / (8×4) = 8   // this warp owns N=64
repK = K / (32 × 1)               = 128 / 32   = 4   // fp8 m16n8k32; K-warps forced to 1
```

`getRepForOperand` (`NvidiaMmaEncodingAttr`) × `warpsPerCTA` on `#mma`. CTA/warp **placement** was chosen earlier (`AccelerateMatmul`); LLVM only **unrolls this warp’s** leftover.

### LLIR (thread + warp story)

Opcode from `mmaInstrPtxAmpere`:

```text
FP32_FP8E5M2_FP8E5M2_FP32
  →  "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"
```

Each thread’s A/B/C structs are packed into fragment tables `ha` / `hb` / `fc`. Then:

```text
this warp owns an N-slice of 64 (of 256)
this thread holds a handful of f8 A, f8 B, f32 C

for k in 0..3:          // repK
  for m in 0..3:        // repM
    for n in 0..7:      // repN
      mma.sync m16n8k32   // 32 threads together; line mma(ret, a, b, c)
```

- **Description:** per-thread function + warp MMA (Part 1 §2).
- **Execution:** 4 warps × 32 threads all run this; N is partitioned by `tid` / `warpId` already compiled into **which registers** went into `ha`/`hb`/`fc`.
- You do **not** see a 64×256 object. You see **128** `mma.sync`s **per warp**.

`mma(retArgs, aArgs, bArgs, cArgs)` only binds **this** m16n8k32’s registers to a `PTXInstr` created as `builder.create(mmaInstruction)`; `builder.launch` → `llvm.inline_asm`.

### What this is not

| | |
|--|--|
| Kernel K over the full matmul | Already outside this op (pipeline of 64×256×128 tiles) |
| LLVM inventing the 16×8×32 tile | Hardware `mma.sync` shape; `#mma` `instrShape = [16, 8]`, K from bitwidth |
| Table lookup of the **loop** | Table is **only** the opcode string |

Elementwise / load / `convert_layout` use the **same Part 1 split** (per-thread structs, layout on the MLIR op) but **not** this MMA unroll: elementwise loops unique registers; `tt.load` vector width comes from AxisInfo on the TTGIR ptr tensor; `convert_layout` rebuilds `toLinearLayout(srcTy)` / `dstTy` and shuffles via registers, `shfl`, or smem.

---

## Part 8 — SCF / warp_specialize / barriers (`populateConversionPatterns` and after)

**From Part 1.** These are **control and sync**, not tiles. LLVM is still **this thread’s CFG**. Warp/CTA join only on collective ops (`bar.sync`, `mbarrier.*`). Structured `scf.for` does not survive to this pass in the matmul dump: it is already a **CFG of `cf.br` / `cf.cond_br`** whose block arguments are the old `scf.yield` values (now per-thread structs).

`populateConversionPatterns` (lines 212–278) does **not** convert everything the user named:

| Construct | In this function? | Actual pass |
|--|--|--|
| `scf.if` / `scf.for` / `scf.yield` | **No** (already gone in IR 17) | Earlier `scf-to-cf` (or Triton `tt.for`→CF). Yield → **successor block args** |
| `cf.br` / `cf.cond_br` | **No** here | `lowerControlFlow` **after** this (Part below) |
| `tt.call` / `tt.return` | Yes | `populateControlFlowOpToLLVMPattern` |
| `ttg.warp_specialize` / `partition` / `warp_yield` | **Kept legal** | `ConvertWarpSpecializeToLLVM` **after** `ConvertTritonGPUToLLVM` |
| `ttng.init/wait/arrive_barrier`, `barrier_expect`, `fence_async_shared` | Yes | `populateBarrierOpToLLVMPatterns` |
| `ttg.barrier local` (membar / inserted before arrive) | Yes | → `NVVM::BarrierOp` / `bar.sync` (named inside WS) |

Dump **18** is named `after_…_warp_specialize_to_llvm`, so it includes that later pass.

---

### 8.1 `scf.if` / `scf.for` / `scf.yield` → CF → LLVM

**Problem.** SCF is structured (`for` with iter_args, `yield`). LLVM only has blocks + `br`/`cond_br`. AxisInfo needs a stable CFG while tensors convert (see `lowerControlFlow`).

**Solution.** By IR 17 there is **no `scf.*`**. A Python/TT loop is already:

```text
scf.for %i = … iter_args(%acc) → %acc_next {
  …
  scf.yield %acc_next
}
        ↓  scf-to-cf (before this pass)
^bb1(%acc):                          // yield = block argument
  cf.cond_br %i_lt, ^bb_body, ^bb_exit
^bb_body:
  …  %acc_next = tt.dot …
  cf.br ^bb1(%acc_next)              // back-edge carries yield
```

**This pass** rewrites ops **inside** those blocks (`tt.dot` → MMA, tensors → structs). Block args change type with the type converter (`tensor<64x256xf32,#mma>` → `!llvm.struct<(f32,…)>` — Part 3). Terminators stay `cf.*`.

**Then** `lowerControlFlow`:

```text
cf.cond_br %p, ^t, ^f(%acc)   →   llvm.cond_br %p, ^t, ^f
cf.br ^bb1(%i, %acc, …)       →   llvm.br ^bb1
```

**IR 17 (default region of warp_specialize) — loop already CF:**

```mlir
cf.br ^bb1(%c0, …, %cst_4, …)   // %cst_4 : tensor<64x256xf32, #mma>
^bb1(%local_absmax_32, …, %14: tensor<64x256xf32, #mma>, …):
  %local_absmax_35 = arith.cmpi slt, %local_absmax_32, %local_absmax_12
  cf.cond_br %local_absmax_35, ^bb2, ^bb7
  …
  %acc = tt.dot %x_52, %w_56, %14 → tensor<64x256xf32, #mma>
  cf.cond_br %local_absmax_58, ^bb5, ^bb6(…, %acc)
^bb6(…, %local_absmax_94: tensor<64x256xf32, #mma>):
  cf.br ^bb1(…, %local_absmax_94, …)   // yield of the loop
```

**IR 18:** same graph, `llvm.cond_br` / `llvm.br`, `%acc` is a register struct. From Part 1: **this thread** diverges on `%local_absmax_35`; the 64×256 tile is only the structs on the back-edge.

`scf.if` → `cf.cond_br` + two blocks; `scf.yield` from each arm → args of the join block. Same story.

---

### 8.2 `ttg.warp_specialize` / `partition` / `warp_yield`

**Problem.** TTGIR still has **one function** whose **warps** run different regions (consumer MMA vs TMA producer). LLVM has **one SIMT kernel**: every thread of all 8 warps enters the same `llvm.func`. Need a **per-thread branch on warp id**.

**Not in 212–278.** `TritonLLVMConversionTarget` leaves `WarpSpecializeOp`, `WarpYieldOp`, `WarpSpecializePartitionsOp`, `WarpReturnOp` **legal**. Bodies **are** converted (dot, barriers, CF). The **op itself** is lowered by `ConvertWarpSpecializeToLLVM`.

**IR 17:**

```mlir
%local_absmax_26 = ttg.warp_specialize(… captures …)
  attributes {actualRegisters = array<i32: 488, 24, 24>,
              allocation.offset = 81920,
              warpGroupStartIds = array<i32: 4, 6>}
default {          // warps 0–3, 4 warps, MMA consumer
  … loop, tt.dot, ttg.warp_yield %13 …
}
partition0(…) num_warps(2) {   // warps 4–5, TMA
  … wait_barrier, barrier_expect, async_tma_copy … ttg.warp_return
}
partition1(…) num_warps(2) {   // warps 6–7, empty
  ttg.warp_return
}
```

Module: `"ttg.num-warps" = 4` (default group), `"ttg.total-num-warps" = 8`.

**Solution (`lowerWarpSpecialize`):**

```text
%tid = nvvm.read.ptx.sreg.tid.x
%wid = %tid / 32          // shfl to mark warp-uniform
llvm.cond_br (%wid < 4), ^default, ^switchLoop

^default:  nvvm.setmaxregister increase 256   // steal worker regs
           … converted default body …
           warp_yield → write captures, named bar, jump back to switch

^switchLoop:   // workers
  barrier.cta.sync  (switch barrier)
  load mailbox byte from @global_smem + allocation.offset (99392)
  llvm.switch %state :
    0 → partition0   setmaxregister decrease 24
    1 → partition1
    _ → spin / exit
```

**IR 18 entry (this thread):**

```mlir
%0 = nvvm.read.ptx.sreg.tid.x
%2 = llvm.udiv %0, 32
%6 = nvvm.shfl.sync idx %5, %2, 0, 31   // uniform warp id
%8 = llvm.icmp "ult" %6, 4
llvm.cond_br %8, ^bb11, ^bb1            // default vs workers

^bb11:  // default, warps 0–3
  nvvm.setmaxregister increase 256     // steal worker regs for MMA
  … converted default body (wait, local_load, fence, arrive, mma.sync) …

^bb1:   // worker switch loop, warps 4–7
  nvvm.setmaxregister increase 256     // back at kernel maxnreg between tasks
  llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.all"(1)
  // mailbox byte at @global_smem + 99392, index (wid - 4)
  llvm.switch %16 : i8, ^bb2 [
    0: ^bb4,   // partition0
    1: ^bb10,  // partition1
    2: ^bb3    // exit
  ]
^bb4:
  nvvm.setmaxregister decrease 24      // partition0 TMA, actualRegisters[1]
  // reload captures from smem+81920, then wait / expect / TMA
^bb10:  // partition1 was empty warp_return
^bb3:   llvm.return
```

Captures (`memdesc` fat ptrs, i32s) are packed into WS scratch (`allocation.offset = 81920`) so workers reload them — still **the same CTA smem**, Part 4. The mailbox itself is the kernel `allocation.offset = 99392` (one byte per worker warp).

`ttg.barrier local` **inside** a partition becomes a **named** `bar.sync` whose count is **that group’s threads** (4×32 default, 2×32 partition0), not the full CTA. Full-CTA `bar.sync` in partition0 would hang (workers would not arrive).

#### Mailbox (how workers learn which partition to run)

**Problem.** After the `wid < 4` branch, default and workers are in **different regions of the same function**. Workers need a **state id** (“run partition0 / partition1 / exit / idle”). TTGIR has no mailbox op; `ttg.warp_specialize` only has regions + `warpGroupStartIds`.

**Solution.** `lowerWarpSpecializeCommon` (`lib/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp`) allocates a **scratch byte array in CTA smem**, one `i8` per **worker warp** (`total-num-warps − num-warps` = 8 − 4 = 4 bytes). Offset is the **kernel** `allocation.offset` (`getSharedMemoryBase(func)`), **99392** in this dump — not the WS capture buffer at **81920**.

State ids are assigned in partition order:

| Worker warp (absolute) | Mailbox index `wid − 4` | Byte written | Switch dest |
|--|--|--|--|
| 4 | 0 | `0` | partition0 (TMA) |
| 5 | 1 | `0` | partition0 |
| 6 | 2 | `1` | partition1 (empty) |
| 7 | 3 | `1` | partition1 |
| (all, at kernel exit) | 0..3 | `2` | `llvm.return` |
| unused slot | — | `-1` (never stored as i8; default switch dest) | spin: two `barrier.cta.sync.all(1)` then loop |

`warpGroupStartIds = [4, 6]` + `num_warps(2)` fills `warpToState[start − 4 : start − 4 + n]`.

**Handshake** uses **full-CTA** `llvm.nvvm.barrier.cta.sync.all(1)` (`kSwitchLoopBarrierIdx`), not a partition-named `bar.sync`:

```text
default (before entering default region):
  store mailbox[0..3]
  store captures @ 81920
  barrier.cta.sync.all 1     // workers may now load mailbox
  setmaxregister increase 256
  barrier.cta.sync.all 1     // workers have copied captures
  br ^default_body

workers (^switchLoop):
  barrier.cta.sync.all 1
  %state = load i8 mailbox[wid - 4]
  switch %state

workers (enter partition):
  setmaxregister decrease 24
  load captures @ 81920
  barrier.cta.sync.all 1     // second rendezvous with default
  … partition body …
  warp_return → barrier 1, br ^switchLoop

default llvm.return:
  store 2 to mailbox[0..3]
  barrier.cta.sync.all 1
  llvm.return
```

**IR 18 — default writes the map** (end of `^bb11`, before `^bb12` MMA loop):

```mlir
%local_absmax_307 = llvm.getelementptr @global_smem[99392]
llvm.store 0, mailbox[0]   // warp 4 → partition0
llvm.store 0, mailbox[1]   // warp 5 → partition0
llvm.store 1, mailbox[2]   // warp 6 → partition1
llvm.store 1, mailbox[3]   // warp 7 → partition1
llvm.store %capture, @global_smem[81920]
llvm.nvvm.barrier.cta.sync.all(1)
nvvm.setmaxregister increase 256
llvm.nvvm.barrier.cta.sync.all(1)
llvm.br ^bb12
```

**IR 18 — workers read it** (`^bb1`):

```mlir
%12 = getelementptr @global_smem[99392]
%14 = sub %wid, 4
%16 = load i8, getelementptr %12[%14]
llvm.switch %16, ^bb2 [ 0: ^bb4, 1: ^bb10, 2: ^bb3 ]
```

**IR 18 — exit** (after absmax atomics):

```mlir
llvm.store 2, mailbox[0..3]
llvm.nvvm.barrier.cta.sync.all(1)
llvm.return
```

Do not mix with **captures**: mailbox = “which region”; 81920 packed struct = partition **arguments** (the `warp_specialize(...)` operands that were not rematerialized). Both are CTA smem (Part 4 types).

---

### 8.3 Barrier-related ops (`populateBarrierOpToLLVMPatterns`)

**From Part 1.** Empty/full mbarriers are **CTA smem objects** (Part 4 fat ptrs). Arrive/wait are **CTA (or named-group) collectives**; only **thread 0 of the group** issues `mbarrier.arrive`. All threads `try_wait`.

Your kernel’s producer/consumer slot (Part 5 of cluster-barrier notes):

| TTGIR | LLVM / PTX |
|--|--|
| `ttng.init_barrier %bar, 1` | Predicated `mbarrier.init.shared::cta.b64 [ptr], 1` (elect warp0 / tid0) |
| `ttng.wait_barrier %bar, %phase` | Inline loop: `mbarrier.try_wait.parity.shared::cta.b64 …, [ptr], phase` (SM90+) |
| `ttng.arrive_barrier %bar, 1` | Insert `ttg.barrier local` then tid==0 `mbarrier.arrive.shared::cta.b64` |
| `ttng.barrier_expect %bar, 8192` | Named/local `bar.sync` then tid==0 `mbarrier.arrive.expect_tx …, 8192` (TMA byte count) |
| `ttng.fence_async_shared {bCluster=false}` | `nvvm.fence.proxy` async_shared / `shared_cta` (generic↔TMA proxy) |
| `ttng.inval_barrier` | Predicated `mbarrier.inval.shared::cta.b64` (after WS, teardown) |
| `ttg.barrier local` (membar / inserted before arrive) | `NVVM::BarrierOp` → full-CTA or **named** `bar.sync` inside WS |

**IR 17 consumer (default):**

```mlir
ttng.wait_barrier %x_50, %x_49
%x_52 = ttg.local_load %x_51 → tensor<…, #dot_op>
ttng.fence_async_shared {bCluster = false}
ttng.arrive_barrier %x_53, 1     // empty: allow producer to refill
```

**IR 17 producer (partition0):**

```mlir
ttng.wait_barrier %x_74, %x_73          // empty
ttng.barrier_expect %x_76, 8192, %true
ttng.async_tma_copy_global_to_local …   // HW arrives on full
```

Arrive always inserts a **partition-local** `bar.sync` first so all warps in **this** WS region have finished the previous smem use before thread 0 ticks the mbarrier (Part 5).

`ttng.cluster_barrier` (not in this 1-CTA dump) is `populateClusterOpsToLLVMPatterns`, not this list.

---

### 8.4 How this sits in `populateConversionPatterns`

```text
populateBarrierOpToLLVMPatterns     // init/wait/arrive/expect/fence  (this pass)
populateControlFlowOpToLLVMPattern  // tt.call / tt.return only
populateGpuToNVVMConversionPatterns // gpu.barrier leftover → nvvm
         │
         ▼
lowerControlFlow                    // cf.* → llvm.br / cond_br
         │
         ▼
ConvertWarpSpecializeToLLVM         // warp_specialize → tid/wid switch (dump 18)
```

SCF was already CF **before** line 212.

---

## Cheat sheet (matmul dump)

| TTGIR (cluster/CTA tile) | LLVM (this thread) |
|--|--|
| `tensor<256xf32, #blocked1>` | `!llvm.struct<(f32, f32)>` |
| `tensor<64x256xf32, #mma>` | `!llvm.struct<(f32, …)>` (unique elems/thread) |
| `!ttg.memdesc<2x128x256xf8E5M2, #shared1, #smem>` | `!llvm.struct<(ptr<3>, i32, i32, i32)>` @ offset 0 (same in every thread) |
| `!ttg.memdesc<2x64x128xf8E5M2, #shared1, #smem>` | same struct @ `gep 65536` |
| `!ttg.memdesc<2x1xi64, #shared2, #smem>` | `!llvm.struct<(ptr<3>, i32, i32)>` |
| `!tt.ptr<f8E5M2>` | `!llvm.ptr<1>` |
| `!tt.tensordesc<1x64x256xf8E5M2, #shared>` | `!llvm.ptr` (generic, byval 128B on kernel) |
| `f8E5M2` | `i8` |
| `tt.dot` 64×256×128 fp8 | 128× `mma.sync.m16n8k32` **per warp** on that warp’s fragments |
| `scf.for`/`yield` (already CF in IR 17) | `llvm.cond_br` / `llvm.br`; iter_args = block args (per-thread structs) |
| `ttg.warp_specialize` default + p0/p1 | `tid/32 < 4` vs worker `llvm.switch` mailbox; `setmaxregister` 256 vs 24 |
| mailbox (no TTGIR op) | 4× `i8` @ `@global_smem+99392`; default stores `0,0,1,1` then `2` on exit |
| `ttng.wait_barrier` | `mbarrier.try_wait.parity` spin loop |
| `ttng.arrive_barrier` | named/local `bar.sync` then tid0 `mbarrier.arrive` |
| `ttng.init_barrier` | tid0/elect `mbarrier.init` |
| `ttng.fence_async_shared` | `nvvm.fence.proxy` async_shared |
