# `add_optimize_dot_operands` (`OptimizeDotOperands.cpp`)

| | |
|--|--|
| **Pass** | `tritongpu-optimize-dot-operands` / `add_optimize_dot_operands` |
| **When** | `capability >= 80`; runs **twice** in `make_ttgir` (early + after pipeline on SM100+) |
| **Code** | `lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp` |
| **Also see** | [Canonicalizer](../canonicalizer/Canonicalizer.md) (run at start of this pass) |

Pipeline (SM100+ matmul path, simplified):

```text
… pipeline / warp_specialize / remove_tmem_tokens …
  → canonicalizer (module)
  → optimize_dot_operands          ← IR dump: 12_after_ttgpuir_add_optimize_dot_operands
  → coalesce_async_copy …
```

Pass structure:

```text
1. createCanonicalizerPass()          // broad MLIR cleanup
2. applyPatternsGreedily:
     SwizzleShmemConvert
     FuseTransMMAV3Plus, ReshapeMemDesc
     RewriteMmaOperandViewsToMemDescForDotOp
     + ConvertLayoutOp canonicalization patterns (mop-up)
```

---

## Part 1 — General idea (all four patterns)

### Problem

Dot/MMA operands often arrive as **tensor** ops (`tt.trans`, `tt.reshape`, `ttg.convert_layout`) on register layouts. That forces expensive register shuffles or non-MMA-friendly shared layouts. MMAv3/v5 hardware expects operands in **specific shared-memory encodings**; transposes/reshapes should happen as **memdesc views** on SMEM, not as tensor ops before the dot when possible.

### Key hinge

Move **layout + view work onto shared memory**:

```text
tensor views + convert_layout  →  local_alloc (SMEM store)
                                    → memdesc_trans / memdesc_reshape (views)
                                    → local_load or direct MMA memdesc operand
```

Four patterns implement different slices of that migration:

| Pattern | Matches | Target ops | Encoding focus |
|---------|---------|------------|----------------|
| **SwizzleShmemConvert** | `ttg.convert_layout` | `tt.dot` (classic) | `SwizzledSharedEncodingAttr` + `needTrans` |
| **FuseTransMMAV3Plus** | `ttg.local_alloc` | `WarpGroupDotOp`, MMAv5 | `NVMMASharedEncodingAttr` |
| **ReshapeMemDesc** | `ttg.local_alloc` | MMAv3/v5 (via NVMMA shared) | `NVMMASharedEncodingAttr` |
| **RewriteMmaOperandViewsToMemDescForDotOp** | `DotOpInterface` | TCGen5 / WGMMA | `SharedLinearEncodingAttr` → replay views on memdesc |

### Swizzle (cross-cutting)

**Swizzled shared** (`#ttg.swizzled_shared<{vec, perPhase, maxPhase, order}>`) reorders how logical tile elements map to SMEM addresses to **avoid bank conflicts** and match MMA load patterns. Parameters:

- `vec`, `perPhase`, `maxPhase`, `order` — XOR-based row/column phase patterns (see `TritonGPUAttrDefs.td`).
- **`needTrans=true`** (SwizzleShmemConvert only): swizzle is computed from the **pre-transpose** layout/shape, because `memdesc_trans` only permutes the descriptor, not `vec`/`maxPhase`.

**NVMMA shared** (`NVMMASharedEncodingAttr`) — encoding used for Hopper/Blackwell WGMMA / TCGen5 operands; `inferTransOpEncoding` / `MemDescReshapeOp::inferReturnTypes` derive the **pre-view** alloc encoding.

### `memdesc_trans` vs `local_load` (view vs “real” transpose)

| Op | Role |
|----|------|
| **`ttg.memdesc_trans`** | **View only** (`Pure`, `MemDescViewTrait`). New descriptor, **same underlying SMEM bytes**. Lowers to permuted offsets, no data movement. |
| **`ttg.local_alloc %tensor`** | **Physical store** — writes tensor into SMEM with chosen encoding (swizzled or NVMMA). |
| **`ttg.local_load`** | **Physical read** — maps **logical** output indices (e.g. transposed 256×128 dot layout) to SMEM addresses via descriptor + encoding. The **transpose effect** for the consumer is realized here (read path), not by shuffling SMEM in `memdesc_trans`. |

So: **`memdesc_trans` does not transpose bytes**; it declares “interpret this buffer with permuted shape/order.” Combined with **`needTrans` swizzle at alloc** and **`local_load`’s layout**, the dot sees a correctly transposed operand.

---

## Part 2 — `SwizzleShmemConvert` (detail)

### Problem

Classic pattern `dot(convert_layout(trans(x)))` keeps transpose + layout change on the **tensor/register** path before `tt.dot`.

### Key hinge

Insert **swizzled SMEM** path: `local_alloc → memdesc_trans → local_load`, rewire outer `convert_layout` to take the load.

### Match conditions

```text
tt.dot ← ttg.convert_layout ← tt.trans {order=[1,0]} ← (optional inner cvt) ← x
```

- Outer `convert_layout` → **single use** → `tt.dot`
- Transpose order exactly **`[1, 0]`** (2D swap)
- Outer cvt **result** type has `DotOperandEncodingAttr`

### Rewrite (schematic)

**Before:**

```mlir
%inner = ttg.convert_layout %x : #blocked -> #blocked1
%t     = tt.trans %inner {order = array<i32: 1, 0>} : 128x256 -> 256x128
%dot_op = ttg.convert_layout %t : #blocked1 -> #ttg.dot_op<...>
%acc  = tt.dot %dot_op, %w, %c0 : ...
```

**After:**

```mlir
%inner = ttg.convert_layout %x : #blocked -> #blocked1

%smem = ttg.local_alloc %inner
    : tensor<128x256xf8E5M2, #blocked1>
   -> !ttg.memdesc<128x256xf8E5M2, #swizzled_shared, #smem>   // needTrans=true

%smem_t = ttg.memdesc_trans %smem {order = array<i32: 1, 0>}
    : memdesc<128x256x...> -> memdesc<256x128x...>             // view only

%loaded = ttg.local_load %smem_t
    : memdesc<256x128x...> -> tensor<256x128xf8E5M2, #ttg.dot_op<...>>

%dot_op = ttg.convert_layout %loaded : #dot_op -> #dot_op      // often trivial → folded
%acc  = tt.dot %dot_op, %w, %c0 : ...
```

Effective end state after canonicalizer: `tt.dot %loaded, …`.

### Code walkthrough

| Step | Code | Meaning |
|------|------|---------|
| `srcTy` | `trans.getSrc()` type; if inner cvt, use **its source** | Layout **before** transpose (for swizzle geometry) |
| `sharedLoadTy` | `cvtOp.getType()` | **Result** type of outer cvt = dot-operand tensor type |
| `newInnerCvtEnc` | `SwizzledSharedEncodingAttr::get(..., needTrans=true)` | SMEM encoding anticipating transposed read |
| `local_alloc` | `trans.getSrc()` initializer | Allocate + **store** tensor into swizzled SMEM |
| `memdesc_trans` | `[1,0]` | Transposed **view** of same buffer |
| `local_load` | type = `sharedLoadTy` | Read with dot-operand layout; **logical transpose at load** |
| `modifyOpInPlace(cvtOp)` | `src = local_load` | Outer cvt stays; only **operand** rewired |

### Why `needTrans = true`

Comment in source: `MemDescTransOp` swaps descriptor order/shape but **does not fix** `vec` / `maxPhase` for the transposed MMA tile. Swizzle must be derived from **pre-transpose** `srcTy` + transposed CGA linear layout, with `needTrans` flag so codegen uses correct M/N/K geometry.

### Dataflow diagram

```text
BEFORE (registers):
  x ──► [optional cvt] ──► tt.trans ──► convert_layout ──► tt.dot
                              ↑ expensive layout on tensor path

AFTER (SMEM):
  x ──► [optional cvt] ──► local_alloc (#swizzled, needTrans)
                              └──► memdesc_trans (view)
                                     └──► local_load (#dot_op) ──► tt.dot
                              ↑ store once; transpose semantics at load
```

### When it does not apply

- Outer cvt not sole user of `tt.dot`
- Transpose order ≠ `[1, 0]`
- Result encoding not `DotOperandEncodingAttr`
- `newInnerCvtEnc == cvtEncoding` (no change)

---

## Part 3 — `FuseTransMMAV3Plus` (detail)

### Problem

For WGMMA / MMAv5, operands are often **`memdesc` in NVMMA shared layout**, but still filled via **`local_alloc(trans(x))`** — transpose happens on the **tensor** before the store. MMAv3/v5 can consume a **transposed memdesc view** directly, so the transpose should be a **`memdesc_trans` view**, not `tt.trans` on registers.

### Key hinge

Replace **`local_alloc(trans(x))`** with **`memdesc_trans(local_alloc(x))`**, using **`inferTransOpEncoding`** to pick the **pre-transpose** NVMMA encoding so the view matches what the MMA already expected.

### Match conditions

```text
WarpGroupDot / MMAv5 ← ttg.local_alloc ← tt.trans {order=[1,0]} ← x
```

- `local_alloc` has **`src`** (initializer tensor) — not a mutable empty buffer
- Alloc has **exactly one use** → `ttng.warp_group_dot` or MMAv5 op
- `src` defined by `tt.trans` with order **`[1, 0]`**
- Alloc memdesc encoding is **`NVMMASharedEncodingAttr`** (`#nvmma_shared<...>`)

### Rewrite (schematic)

**Before:**

```mlir
%t = tt.trans %x {order = array<i32: 1, 0>} : 128x256 -> 256x128

%md = ttg.local_alloc %t
    : tensor<256x128xf32, ...>
   -> !ttg.memdesc<256x128xf32, #nvmma_shared<{transposed=true, ...}>, #smem>

%acc = ttng.warp_group_dot %md, %b, %c0 ...    // sole user of %md
```

**After:**

```mlir
%md0 = ttg.local_alloc %x
    : tensor<128x256xf32, ...>
   -> !ttg.memdesc<128x256xf32, #nvmma_shared<{transposed=false, ...}>, #smem>

%md = ttg.memdesc_trans %md0 {order = array<i32: 1, 0>}
    : !ttg.memdesc<128x256x...> -> !ttg.memdesc<256x128x...>   // same type as old %md

%acc = ttng.warp_group_dot %md, %b, %c0 ...   // unchanged; operand now from memdesc_trans
```

Old `local_alloc %t` is **erased immediately** by `replaceOpWithNewOp` (not left for canonicalizer). Dead **`tt.trans`** may be removed later by canonicalizer/DCE.

### Code walkthrough

| Step | Code | Meaning |
|------|------|---------|
| `getSrc()` check | line 97 | Need `alloc(trans(...))`, not empty mutable alloc |
| `hasOneUse()` + MMA user | lines 97–100 | Only fuse when alloc feeds WGMMA/MMAv5 directly |
| `allocEncoding` | `NVMMASharedEncodingAttr` on **current** alloc | Post-transpose encoding MMA sees today |
| `inferTransOpEncoding(...)` | lines 118–120 | **Backward infer** pre-transpose encoding (`newInnerEnc`); toggles `transposed`, permutes CGA layout |
| `innerTy` | `srcTy` shape + `newInnerEnc` | Memdesc type for **pre-transpose** alloc |
| `newAlloc` | `local_alloc(trans.getSrc())` | Store **`x`**, not **`trans(x)`** |
| `replaceOpWithNewOp<MemDescTransOp>` | lines 129–130 | Create `memdesc_trans(newAlloc)`; **replace** old alloc uses with trans result; **erase** old alloc |

### How `replaceOpWithNewOp` wires ops

```cpp
rewriter.replaceOpWithNewOp<MemDescTransOp>(allocOp, newAlloc, {1, 0});
```

MLIR equivalent:

1. `memdesc_trans = MemDescTransOp(newAlloc, order=[1,0])` — **`newAlloc` is `$src`**
2. `replaceOp(allocOp, memdesc_trans)` — MMA users switch from old alloc → trans result; **old alloc erased now**

```text
BEFORE:  x ──► tt.trans ──► allocOp (%md) ──► warp_group_dot

AFTER:   x ──► newAlloc (%md0) ──► memdesc_trans (%md) ──► warp_group_dot
         tt.trans may become dead
```

### Difference from `SwizzleShmemConvert`

| | **SwizzleShmemConvert** | **FuseTransMMAV3Plus** |
|--|---------------------------|-------------------------|
| Matches | `convert_layout` → `tt.dot` | `local_alloc` → WGMMA/MMAv5 |
| Encoding | `SwizzledSharedEncodingAttr` + `needTrans` | `NVMMASharedEncodingAttr` |
| Inserts | alloc + memdesc_trans + **local_load** | alloc + **memdesc_trans only** |
| Consumer | Classic `tt.dot` (register operands) | MMA takes **memdesc** directly |
| Old op | Rewires `convert_layout` src | **Replaces** `local_alloc` with trans view |

Both push transpose from tensor path to **`memdesc_trans` view**; this pattern skips `local_load` because MMA reads SMEM via memdesc.

### Why MMAv3/v5 only

Comment in source: Hopper/Blackwell WGMMA and TCGen5 MMA ops accept shared operands in NVMMA layout and can use **transposed descriptor views** in the instruction path. Classic `tt.dot` on register tensors uses SwizzleShmemConvert instead.

### Dataflow diagram

```text
BEFORE:
  x ──► tt.trans ──► local_alloc (#nvmma transposed) ──► warp_group_dot
              ↑ tensor transpose before SMEM store

AFTER:
  x ──► local_alloc (#nvmma pre-trans) ──► memdesc_trans (view) ──► warp_group_dot
              ↑ store untransposed tile; MMA sees transposed view
```

### When it does not apply

- No `src` on alloc (TMA / mutable pipeline buffer)
- Alloc used by more than one op, or not WGMMA/MMAv5
- Transpose order ≠ `[1, 0]`
- Encoding not `NVMMASharedEncodingAttr`
- `inferTransOpEncoding` fails

---

## Part 4 — `ReshapeMemDesc` (detail)

### Problem

Same family as `FuseTransMMAV3Plus`, but for **`tt.reshape`**: operand setup does **`local_alloc(reshape(x))`** — reshape on the **tensor** before storing to NVMMA shared mem. MMAv3/v5 can take a **reshaped memdesc view** instead.

### Key hinge

Replace **`local_alloc(reshape(x))`** with **`memdesc_reshape(local_alloc(x))`**, using **`MemDescReshapeOp::inferReturnTypes`** (backward) to get the **pre-reshape** alloc type.

### Match conditions

```text
MMA consumer ← ttg.local_alloc ← tt.reshape ← x
```

- `local_alloc` has **`src`** (initializer)
- `src` defined by **`tt.reshape`** (not necessarily single-use check — unlike FuseTrans)
- Current alloc encoding is NVMMA-related; **`inferReturnTypes`** yields **`innerTy`** whose encoding is **`NVMMASharedEncodingAttr`**

### Rewrite (schematic)

**Before:**

```mlir
%r = tt.reshape %x : 64x128 -> 8192                              // example flatten
%md = ttg.local_alloc %r
    : tensor<8192xf32, ...>
   -> !ttg.memdesc<8192xf32, #nvmma_shared<...>, #smem>

%acc = ttng.warp_group_dot %md, %b, %c0 ...
```

**After:**

```mlir
%md0 = ttg.local_alloc %x
    : tensor<64x128xf32, ...>
   -> !ttg.memdesc<64x128xf32, #nvmma_shared<{...}>, #smem>   // innerTy (pre-reshape enc)

%md = ttg.memdesc_reshape %md0
    : !ttg.memdesc<64x128x...> -> !ttg.memdesc<8192x...>       // same type as old %md

%acc = ttng.warp_group_dot %md, %b, %c0 ...
```

Old alloc **erased immediately** by `replaceOpWithNewOp`. Dead **`tt.reshape`** may be cleaned later by canonicalizer.

### Code walkthrough

| Step | Code | Meaning |
|------|------|---------|
| `reshapeOp` | `allocOp.getSrc()` | Tensor reshape before store |
| `srcShape` | `reshapeOp.getSrc().getType().getShape()` | Shape **before** reshape |
| `inferReturnTypes(ctx, loc, allocType, srcShape, innerTy)` | lines 163–164 | **Backward infer**: if post-reshape memdesc is `allocType`, what pre-reshape memdesc + encoding fits? |
| NVMMA check | `isa<NVMMASharedEncodingAttr>(innerTy.getEncoding())` | Only apply when inner encoding is MMAv3/v5 compatible |
| `newAlloc` | `local_alloc(reshapeOp.getSrc())` with `innerTy` | Store **unreshaped** tensor `x` |
| `replaceOpWithNewOp<MemDescReshapeOp>` | lines 175–176 | `memdesc_reshape(newAlloc)` → same memdesc type as old alloc; replace uses; erase old alloc |

### `inferReturnTypes` (backward reshape layout)

Uses **`inferMemDescReshapeOpEncoding`**: given post-reshape memdesc type + pre-reshape **element count** shape, compute encoding for the inner alloc. Comment in source: forward and backward inference coincide for `MemDescReshapeOp`, so this call is safe for “what inner alloc produces outer shape after view.”

### Relation to Part 3

| | **FuseTransMMAV3Plus** | **ReshapeMemDesc** |
|--|------------------------|---------------------|
| Tensor op | `tt.trans [1,0]` | `tt.reshape` |
| Memdesc view | `memdesc_trans` | `memdesc_reshape` |
| Layout inference | `inferTransOpEncoding` | `MemDescReshapeOp::inferReturnTypes` |
| Single-use check | **yes** (MMA user) | **no** |

### Dataflow diagram

```text
BEFORE:
  x ──► tt.reshape ──► local_alloc (#nvmma) ──► MMA
              ↑ tensor reshape before SMEM store

AFTER:
  x ──► local_alloc (#nvmma pre-reshape) ──► memdesc_reshape (view) ──► MMA
```

### When it does not apply

- No `src` on alloc
- `src` not from `tt.reshape`
- `inferReturnTypes` fails (element count mismatch, encoding inference failure)
- `innerTy` encoding is not `NVMMASharedEncodingAttr`

---

## Part 5 — `RewriteMmaOperandViewsToMemDescForDotOp` (detail)

### Problem

TCGen5 / WGMMA operands often look like:

```text
tt.reshape / tt.trans → … → local_alloc → [memdesc views] → MMA
```

Tensor **reshape/trans** happen **before** `local_alloc`, while MMA already consumes a **memdesc** with trailing **`memdesc_reshape` / `memdesc_trans`** views. That duplicates view logic across tensor and memdesc domains.

### Key hinge

**Hoist** `local_alloc` to the **base tensor** (before tensor view chain), **replay** those tensor views as **memdesc views**, preserving the **same final memdesc type** the MMA already uses. Layout is anchored on the **sink memdesc** at the MMA and propagated backward.

### Match conditions

Pattern matches **`DotOpInterface`** on:

- `ttng.tc_gen5_mma` / `TCGen5MMAScaledOp`
- `ttng.warp_group_dot`

Per operand (`A`, `B`, scales):

```text
MMA ← [memdesc_reshape | memdesc_trans]* ← local_alloc ← [tt.reshape | tt.trans]* ← base_tensor
```

Requirements:

- Operand type is **`MemDescType`**
- Operand encoding is **`SharedLinearEncodingAttr`** only (not `NVMMAShared` — comment: backward through tensor views is **not encoding-stable** for NVMMA)
- Walk back through **`memdesc_reshape` / `memdesc_trans`** to a **`local_alloc` with `src`**
- **`local_alloc.getSrc()`** has at least one **`tt.reshape` or `tt.trans`** in its defining chain (else nothing to hoist)

### Rewrite (schematic)

**Before:**

```mlir
%x = ... : tensor<64x128xf32, #blocked>
%r = tt.reshape %x : 64x128 -> 8192
%t = tt.trans %r {order = [1, 0]} : ...                                    // optional chain

%md0 = ttg.local_alloc %t : ... -> !ttg.memdesc<..., #shared_linear, #smem>
%md1 = ttg.memdesc_reshape %md0 : ... -> !ttg.memdesc<8192x..., ...>
%md  = ttg.memdesc_trans %md1 {order = [1, 0]} : ... -> !ttg.memdesc<...>  // MMA operand

%acc = ttng.tc_gen5_mma %md, %b, %c0 ...
```

**After:**

```mlir
%x = ... : tensor<64x128xf32, #blocked>

// Hoisted alloc at base tensor; encoding back-propagated from original local_alloc
%md0' = ttg.local_alloc %x : ... -> !ttg.memdesc<64x128, #shared_linear', #smem>

// Replay tensor views as memdesc views (same order as tt.reshape / tt.trans chain)
%md1' = ttg.memdesc_reshape %md0' : ... -> ...
%md   = ttg.memdesc_trans %md1' {order = [1, 0]} : ...   // same type as before rewrite

%acc = ttng.tc_gen5_mma %md, %b, %c0 ...
```

`%md` type unchanged → MMA op needs no change. **`local_alloc` is replaced** (`rewriter.replaceOp(localAlloc, rewritten)`); tensor **`tt.reshape` / `tt.trans`** may become dead.

### Algorithm (two-phase walk)

**Phase A — walk down from MMA operand (memdesc side):**

```text
operand → … → memdesc_reshape/trans → local_alloc
```

Skip trailing memdesc views; stop at `local_alloc` with `src`.

**Phase B — walk up from `local_alloc.getSrc()` (tensor side):**

Collect `tt.reshape` / `tt.trans` ops into `tensorReplaySteps`; for each step **backward-update** `baseMemTy`:

| Tensor op | Update `baseMemTy` |
|-----------|-------------------|
| `tt.reshape` | `MemDescReshapeOp::inferReturnTypes(baseMemTy, pre-reshape shape)` |
| `tt.trans` | `inferSrcEncoding(trans, baseMemTy.getEncoding())` + pre-trans shape |

Reverse `tensorReplaySteps` to get root-to-leaf order.

**Phase C — rebuild:**

```text
rewritten = local_alloc(baseMemTy, baseTensor)
for each step in order:
  rewritten = memdesc_reshape(rewritten) or memdesc_trans(rewritten)
replaceOp(localAlloc, rewritten)
```

### Code walkthrough

| Step | Code | Meaning |
|------|------|---------|
| `rewriteOperand` | lines 224–299 | Per memdesc MMA operand |
| `SharedLinearEncodingAttr` gate | lines 229–233 | Restrict to shared-linear (TCGen5 path) |
| Trailing memdesc walk | lines 235–246 | Skip views above alloc |
| Tensor replay collection | lines 252–277 | Record reshape/trans chain + infer inner mem types |
| `tensorReplaySteps.empty()` | line 278 | No hoisting benefit |
| `replaceOp(localAlloc, rewritten)` | line 298 | Same outer memdesc type; **alloc op replaced**, not modified in place |

### Difference from Parts 2–4

| | **Parts 2–4** | **Part 5** |
|--|---------------|------------|
| Trigger | Single `convert_layout` or `local_alloc` pattern | Whole MMA operand chain |
| Encoding | Swizzled / NVMMA | **SharedLinear** only |
| Scope | One rewrite (trans **or** reshape) | **Chain** of tensor views → memdesc views |
| MMA types | `tt.dot` or WGMMA with NVMMA | TCGen5 / WGMMA with shared-linear memdesc |

Parts 3–4 are **local peephole** on one tensor op before alloc. Part 5 is **global operand restructuring** for pipelines that already use memdesc views at the MMA.

### Dataflow diagram

```text
BEFORE:
  base ──► tt views (tensor) ──► local_alloc ──► memdesc views ──► MMA
                ↑ redundant domain split

AFTER:
  base ──► local_alloc (hoisted, enc back-prop)
              └──► memdesc views (replay) ──► MMA
                ↑ tensor views eliminated; one SMEM view chain
```

### When it does not apply

- MMA op not TCGen5 / WGMMA / scaled variant
- Operand not `MemDescType` or not `SharedLinearEncodingAttr`
- No `local_alloc` with `src` behind memdesc views
- No `tt.reshape` / `tt.trans` on tensor path before alloc
- `inferSrcEncoding` / `inferReturnTypes` fails for some step in chain

---

