# Triton Dialect (`tt`) — Structure & Definition Layout

This note explains how the **Triton IR dialect** is defined and wired together in this repo,
verified against sources under `include/triton/Dialect/Triton/IR/` and `lib/Dialect/Triton/IR/`.

---

## 0. Big picture

A MLIR dialect packages:

| Kind | Role |
| ---- | ---- |
| **Dialect class** | Namespace + load/init + attribute helpers |
| **Types** | Values ops consume/produce (`!tt.ptr`, `!tt.tensordesc`, …) |
| **Operations** | IR instructions (`tt.load`, `tt.dot`, …) |
| **Attributes / enums** | Op / dialect metadata |
| **Traits / interfaces** | Shared verify / polymorphic query APIs |
| **Manual C++** | Behavior not fully expressible in TableGen |

```text
TritonDialect.td  ──► Dialect.h.inc / Dialect.cpp.inc
TritonTypes.td    ──► Types.h.inc / Types.cpp.inc
TritonOps.td      ──► Ops.h.inc / Ops.cpp.inc
  (+ TritonAttrDefs.td) ──► OpsEnums.h.inc / OpsEnums.cpp.inc
TritonInterfaces.td     ──► NativeOpTrait names used by ops
TritonOpInterfaces.td   ──► OpInterfaces.h.inc / .cpp.inc
TritonTypeInterfaces.td ──► TypeInterfaces.h.inc / .cpp.inc

Dialect.h (manual)  #includes Dialect.h.inc + Ops.h.inc + Types.h + Traits.h …
Dialect.cpp (manual) initialize(): registerTypes(); addOperations; addInterfaces
```

Build: `include/triton/Dialect/Triton/IR/CMakeLists.txt` (`mlir_tablegen`) and
`lib/Dialect/Triton/IR/CMakeLists.txt` (`add_triton_library(TritonIR …)`).

---

## 1. Component inventory

| Component | Path(s) | TableGen / Manual / Combo | Motivation |
| --------- | ------- | ------------------------- | ---------- |
| **TritonDialect** | `TritonDialect.td` → `Dialect.*.inc`; `Dialect.h`; `Dialect.cpp` | Combo | Namespace `tt`, deps, attr helpers, registration |
| **Types** | `TritonTypes.td` → `Types.*.inc`; `Types.h`; `Types.cpp` | Combo | Real types + TableGen type *constraints* |
| **Ops** | `TritonOps.td` → `Ops.*.inc`; `Ops.cpp` | Combo | All `tt.*` ops |
| **Attr enums** | `TritonAttrDefs.td` → `OpsEnums.*.inc` | TableGen | Load/store/atomic enums |
| **Native op traits** | `TritonInterfaces.td` + `Traits.h` / `Traits.cpp` | Combo | Shared verify (`TensorSizeTrait`, …) |
| **Op interfaces** | `TritonOpInterfaces.td` → `OpInterfaces.*.inc` | Combo | Polymorphic op APIs (`DotOpInterface`, …) |
| **Type interfaces** | `TritonTypeInterfaces.td` → `TypeInterfaces.*.inc` | Combo | Polymorphic type APIs (`TensorDescInterface`) |
| **Dialect interfaces** | `Interfaces.h` + `Dialect.h` extras | Manual | Inliner / layout infer-verify |
| **Discardable attrs** | dialect TD + generated helpers + `DiscardableAttributes.*` | Combo | `tt.num_stages`, `tt.latency`, … |
| **Utility** | `Utility.h` / `Utility.cpp` | Manual | Shared helpers |

---

## 1.5 Example: how `Ops.td`, `Traits.h`, and OpInterfaces relate

Think of three layers on one op (e.g. `tt.dot` / `tt.load`):

| Layer | What it is | Analogy |
| ----- | ---------- | ------- |
| **Op definition** (`TritonOps.td`) | Concrete class: operands, attrs, asm | A specific car model |
| **Traits** (`TritonInterfaces.td` + `Traits.h`) | Mix-in *verification* / properties, no new query API | Safety checks every car must pass (seatbelt, size limit) |
| **OpInterface** (`TritonOpInterfaces.td`) | Shared *methods* usable without knowing the concrete op | A “has steering wheel” interface so code can drive *any* vehicle that implements it |

### Concrete wiring for `tt.load`

Full trait / interface list on the op:

```213:224:include/triton/Dialect/Triton/IR/TritonOps.td
def TT_LoadOp : TT_Op<"load", [
  SameLoadStoreOperandsAndResultShape,
  SameLoadStoreOperandsAndResultEncoding,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<PredicatedOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"result matches ptr type", "ptr", "result",
                 "getPointeeType($_self)">,
  TypesMatchWith<"mask type matches ptr type", "ptr", "mask",
                 "getI1SameShape(getPointeeType($_self))",
                 "($_op.getOperands().size() <= 1) || std::equal_to<>()">,
  TypesMatchWith<"other matches ptr type", "ptr", "other",
                 "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 2) || std::equal_to<>()">
]> {
```

Base `TT_Op` also always adds `TensorSizeTrait` and `VerifyTensorLayoutsTrait`.

---

#### 1) Native traits → shared verify (`Traits.h`)

| Trait on `tt.load` | Role |
| ------------------ | ---- |
| `SameLoadStoreOperandsAndResultShape` | ptr / mask / other / result shapes agree |
| `SameLoadStoreOperandsAndResultEncoding` | layouts agree when present |
| `TensorSizeTrait` (via `TT_Op`) | element-count / power-of-two cap |
| `VerifyTensorLayoutsTrait` (via `TT_Op`) | layout validity vs module |

TD name (`NativeOpTrait<"TensorSizeTrait">`) → C++ `OpTrait::TensorSizeTrait::verifyTrait` → `impl::verifyTensorSize`.

---

#### 2) `AttrSizedOperandSegments` — optional operands packing

`tt.load` has **optional** SSA operands `$mask` and `$other`:

```td
TT_PtrLike:$ptr,
Optional<TT_BoolLike>:$mask,
Optional<TT_Type>:$other,
```

So a single `tt.load` may have 1, 2, or 3 operands. MLIR cannot know from the op name alone which operand slot is which when some are missing.

`AttrSizedOperandSegments` (from MLIR `OpBase.td`) attaches an inherent attribute / property **`operandSegmentSizes`**: a small int array, one entry per ODS-declared operand group, giving how many SSA values that group contributes (0 or 1 here).

Example:

| IR form | Rough segment sizes `[ptr, mask, other]` |
| ------- | ---------------------------------------- |
| `tt.load %p : !tt.ptr<f32>` | `[1, 0, 0]` |
| `tt.load %p, %m : …` | `[1, 1, 0]` |
| `tt.load %p, %m, %o : …` | `[1, 1, 1]` |

Generated `LoadOp` then exposes `getMask()` / `getOther()` that return null/`Value()` when the segment size is 0. Without this trait, optional operands on ops with several optional slots are ambiguous.

(Generated header lists `"operandSegmentSizes"` among `LoadOp` attr names.)

---

#### 3) `TypesMatchWith` — type relationships between named args

`TypesMatchWith` is a **predicate trait** from MLIR ODS (`OpBase.td`):

```td
TypesMatchWith<summary, lhsArg, rhsArg, transform, comparator = "std::equal_to<>()">
```

Meaning (default):  
`comparator( transform(lhs.getType()) , rhs.getType() )` must hold.

For `tt.load`:

| Constraint | English |
| ---------- | ------- |
| `TypesMatchWith<…, "ptr", "result", "getPointeeType($_self)">` | `result` type == pointee of `ptr` (scalar load: `!tt.ptr<f32>` → `f32`; tensor-of-ptrs → tensor of elements) |
| `… "ptr", "mask", "getI1SameShape(getPointeeType($_self))", "operands<=1 \|\| equal"` | If mask present, it is `i1` with same shape as the loaded value; if only `ptr` exists, skip |
| `… "ptr", "other", "getPointeeType($_self)", "operands<=2 \|\| equal"` | If other present, same type as result/pointee; skip when absent |

`$_self` in the transform string is substituted with `$ptr.getType()`. The optional 5th arg is a **guard** so missing optional operands do not fail the check.

These fire during op verification (parse / builder / `--verify-diagnostics`), same family as other `PredOpTrait`s.

---

#### 4) OpInterfaces on `tt.load`

| Interface | Why on load |
| --------- | ----------- |
| `PredicatedOpInterface` | mask is the predicate; `Ops.cpp` implements `getPredicateOperand()` → `getMask()` |
| `MemoryEffectsOpInterface` | models read from global memory |
| `InferTypeOpInterface` | result type can be inferred from `ptr` |

Polymorphic example (same idea as `DotOpInterface` for dots): passes can talk to any predicated op via the interface without hard-coding `LoadOp` / `StoreOp` / `AtomicRMWOp`.

---

#### 5) Inherent attrs + builders + canonicalize

```td
DefaultValuedAttr<TT_CacheModifierAttr, ...>:$cache,
DefaultValuedAttr<TT_EvictionPolicyAttr, ...>:$evict,
DefaultValuedAttr<BoolAttr, "false">:$isVolatile
let hasCanonicalizer = 1;   // e.g. mask=splat(1) → drop mask; mask=splat(0) → fold to `other`
```

Custom `LoadOp::build(...)` overloads in `Ops.cpp` fill missing mask/other as empty while setting segment sizes correctly.

---

**Summary relation for `tt.load`**

```text
TritonOps.td (TT_LoadOp)
  │
  ├─ Triton NativeOpTraits ──► Traits.h/cpp (shape/encoding/size/layout verify)
  ├─ AttrSizedOperandSegments ──► operandSegmentSizes for Optional mask/other
  ├─ TypesMatchWith ──► ptr ↔ result / mask / other type laws
  ├─ DeclareOpInterfaceMethods ──► Predicated / MemoryEffects / InferType
  └─ arguments + assemblyFormat + hasCanonicalizer ──► Ops.*.inc + Ops.cpp
```

Traits = *must-check rules*. `TypesMatchWith` = *typed coupling between args*. `AttrSizedOperandSegments` = *how optional operands are packed*. Interfaces = *shared callable API*. Ops = *concrete schema*.

---

## 2. Roles, members, and how they work together

### 2.1 `TritonDialect` — who does what

#### 2.1.0 How the four files co-work (start here)

| Artifact | Role |
| -------- | ---- |
| **`TritonDialect.td`** | *Authoritative dialect description*: name `tt`, deps, discardable attrs, flags, extra decls (`registerTypes`, `getLoaded`) |
| **`Dialect.h.inc` / `Dialect.cpp.inc`** | *TableGen output*: class skeleton, attr helpers, ctor that loads deps + calls `initialize()`, optional parse/print/materialize decls |
| **`Dialect.h`** | *Manual public umbrella*: `#include`s `.inc`, Types, Traits, Ops, plus Triton-only dialect interfaces |
| **`Dialect.cpp`** | *Manual behavior*: body of `initialize()`, `materializeConstant()`, includes `Dialect.cpp.inc` |

Co-work flow:

```text
TritonDialect.td
      │  mlir-tblgen -gen-dialect-decls / -gen-dialect-defs
      ▼
Dialect.h.inc  (class + helpers + void initialize();)
Dialect.cpp.inc (ctor: loadDialect…; initialize();)
      ▲                              ▲
      │ #include                     │ #include
Dialect.h  ──────────────────►  Dialect.cpp
  (extra interfaces)              void TritonDialect::initialize() {
                                    registerTypes();
                                    addOperations<…>();
                                    addInterfaces<…>();
                                  }
```

Consumers always `#include "triton/Dialect/Triton/IR/Dialect.h"`, never the `.inc` alone.

---

#### 2.1.1 `TritonDialect.td`

```6:54:include/triton/Dialect/Triton/IR/TritonDialect.td
def Triton_Dialect : Dialect {
  let name = "tt";
  let cppNamespace = "::mlir::triton";
  let dependentDialects = [ "arith::ArithDialect", ... ];
  let extraClassDeclaration = [{
    void registerTypes();
    static TritonDialect *getLoaded(MLIRContext *ctx) { ... }
    static TritonDialect *getLoaded(Operation *op) { ... }
  }];
  let discardableAttrs = (ins ... $num_stages, $latency, $self_latency);
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}
include "triton/Dialect/Triton/IR/TritonTypes.td"
```

- **`dependentDialects`**: not a member list in `Dialect.h.inc`; emitted as `getContext()->loadDialect<...>()` in **`Dialect.cpp.inc`**.
- **`extraClassDeclaration`**: pasted into the generated class → that is where **`registerTypes()`** is *declared*.
- **`include TritonTypes.td`**: TableGen-world include (docs / type defs share dialect). It does **not** put type classes into `Dialect.h.inc`, and it does **not** define `registerTypes()`. Types are registered later in C++ via `registerTypes()` + `Types.cpp.inc` list.

---

#### 2.1.2 `Dialect.h.inc` / `Dialect.cpp.inc` (generated)

**Why `initialize()` appears even though it is not in `TritonDialect.td`:**

MLIR’s dialect TableGen (`mlir/tools/mlir-tblgen/DialectGen.cpp`) **always** emits this skeleton for every dialect:

```cpp
// From dialectDeclBeginStr in DialectGen.cpp
class TritonDialect : public ::mlir::Dialect {
  explicit TritonDialect(::mlir::MLIRContext *context);
  void initialize();          // ← always generated; YOU implement it
  friend class ::mlir::MLIRContext;
public:
  ...
};
```

And the generated ctor always ends with `initialize();`:

```14:24:build/.../Dialect.cpp.inc
TritonDialect::TritonDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(...)
    , numStagesAttrName(context), ...
{
  getContext()->loadDialect<arith::ArithDialect>();
  ...
  initialize();   // calls your Dialect.cpp implementation
}
```

So: **declaration of `initialize()` is TableGen boilerplate**; **definition is manual** in `Dialect.cpp`. Same pattern as op `hasVerifier`: framework reserves the hook; you fill the body.

Flags from the `.td` add more generated decls:

- `useDefaultTypePrinterParser = 1` → `parseType` / `printType`
- `hasConstantMaterializer = 1` → `materializeConstant`
- `discardableAttrs` → `NumStagesAttrHelper`, `getNumStagesAttrHelper()`, …

---

#### 2.1.3 `Dialect.h` (manual umbrella)

```15:22:include/triton/Dialect/Triton/IR/Dialect.h
#include "triton/Dialect/Triton/IR/Dialect.h.inc"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/OpsEnums.h.inc"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.h.inc"
```

Plus manual types like `DialectInferLayoutInterface`, `GlobalMemory`, verify helpers. This is the header the rest of Triton includes.

---

#### 2.1.4 `Dialect.cpp` (manual) — including `registerTypes()`

**Where is `registerTypes()` defined?**  
Declared via `extraClassDeclaration` in the `.td` (appears in `Dialect.h.inc`). **Defined** in `Types.cpp`:

```20:25:lib/Dialect/Triton/IR/Types.cpp
void TritonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/Triton/IR/Types.cpp.inc"
      >();
}
```

**Why call it from `initialize()`?**

```61:71:lib/Dialect/Triton/IR/Dialect.cpp
void TritonDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonInlinerInterface>();
}
```

So: TD `include TritonTypes.td` defines the *TypeDefs* for TableGen; CMake generates `Types.cpp.inc`; `registerTypes()` **hooks those types into the live dialect**. Without that call, `!tt.ptr` would not be a registered dialect type at runtime.

Also implements `materializeConstant` → delegates to `arith::ConstantOp`.

---

### 2.2 Types — `TritonTypes.td` + `Types.{h,cpp}`

#### 2.2.1 Two kinds of “defs”

**A) Real C++ types (`TypeDef` / `TritonTypeDef`)** → appear in `Types.h.inc`

Examples: `PointerType` (`!tt.ptr`), `TensorDescType` (`!tt.tensordesc`).

**B) Type constraints / sugar (`AnyTypeOf`, `RankedTensorOf`, …)** → **no C++ class**

```19:21:include/triton/Dialect/Triton/IR/TritonTypes.td
def TT_Float : AnyTypeOf<[F8E4M3FN, ..., F64], "floating-point">;
def TT_FloatTensor : RankedTensorOf<[TT_Float]>;
def TT_FloatLike : AnyTypeOf<[TT_Float, TT_FloatTensor]>;
```

**Why you cannot find `TT_FloatLike` in `Types.h.inc` or `Types.cpp`:**  
It is only a TableGen predicate used when writing op signatures, e.g. `TT_FloatLike:$src`. ODS expands it to “accept f16/f32/… or a ranked tensor of those.” There is nothing to construct at runtime named `TT_FloatLike`.

Same for `TT_Int`, `TT_PtrLike`, `TT_FpIntTensor`, etc.

#### 2.2.2 Parameters of real types (what C++ type each is)

**`PointerType`** (`TT_PtrType`):

```td
let parameters = (ins "Type":$pointeeType, "int":$addressSpace);
```

| Parameter | C++ type | Meaning |
| --------- | -------- | ------- |
| `pointeeType` | `mlir::Type` | What the pointer points to (scalar element type, e.g. `f32`) |
| `addressSpace` | `int` | Address space id (default often `1` for global) |

IR look: `!tt.ptr<f32>` or `!tt.ptr<f32, 1>`.

**`TensorDescType`** (`TT_TensorDescType`):

```td
let parameters = (ins
  ArrayRefParameter<"int64_t">:$shape,
  "Type":$elementType,
  OptionalParameter<"Attribute">:$sharedLayout
);
```

| Parameter | C++ type | Meaning |
| --------- | -------- | ------- |
| `shape` | `llvm::ArrayRef<int64_t>` (stored as array) | Block tile dims, e.g. `128×64` |
| `elementType` | `mlir::Type` | Element dtype of the block, e.g. `f16` |
| `sharedLayout` | `mlir::Attribute` (optional) | Shared-memory encoding / swizzle assigned in lowering |

IR look: `!tt.tensordesc<128x64xf16>` or `!tt.tensordesc<128x64xf16, #shared…>`.

Custom parse/print/verify for both live in **`Types.cpp`** (`hasCustomAssemblyFormat = 1`, `genVerifyDecl = 1`).

---

### 2.3 Attributes — three kinds (difference + when to use)

Analogy: a restaurant order slip.

| Kind | Analogy | Owned by | Survives arbitrary transforms? | Example in Triton |
| ---- | ------- | -------- | ------------------------------ | ----------------- |
| **Inherent op attributes** | Required fields on the order form (meal size) | Op schema in `arguments = (ins … Attr …)` | Yes — part of the op | `tt.load`’s `$cache`, `$evict`; `tt.dot`’s `$inputPrecision` |
| **Op enum attributes** | Typed choice for those fields (S/M/L) | `TritonAttrDefs.td` → `OpsEnums.*.inc`; used *as* inherent attrs | Same as inherent | `TT_CacheModifierAttr`, `TT_InputPrecisionAttr` |
| **Discardable dialect attributes** | Sticky note someone added (“rush order”) | Dialect (`discardableAttrs`); any op may carry them | **No** — generic passes may drop unless preserved | `tt.num_stages`, `tt.latency`, `tt.self_latency` on an `scf.for` |

#### Inherent (+ enum) — use when the op *needs* the info

```td
// part of tt.load schema
DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
```

IR: `tt.load %p cacheModifier = ca : !tt.ptr<f32>`  
API: `loadOp.getCache()`.

#### Discardable — use when a *pass* attaches temporary / optional metadata

Pipeliner writes stage counts on loops without changing the op’s inherent schema:

```cpp
auto helper = TritonDialect::getLoaded(forOp)->getNumStagesAttrHelper();
if (auto attr = helper.getAttr(forOp))
  return attr.getInt();  // read tt.num_stages if present
```

**Rule of thumb**

- Needed to *define* the op’s meaning → **inherent** (often an **enum attr**).
- Optional annotation for later passes, OK if lost → **discardable**.

---

### 2.4 Operations — brief (see §1.5 for traits/interfaces)

`TritonOps.td` defines schema; `hasVerifier` / `hasFolder` / `hasCanonicalizer` / `extraClassDeclaration` / `DeclareOpInterfaceMethods` mark **manual** C++ hooks in `Ops.cpp`.

---

### 2.5 Traits — verify mix-ins (what they enable)

Covered in §1.5 for `tt.load` wiring. Key: traits do **not** add polymorphic getters; they add `verifyTrait` hooks (and some ODS constraints). Triton native traits live in `TritonInterfaces.td` + `Traits.h` / `Traits.cpp`. Below: usual traits when defining an op, **what you can do if the op has it**, and an example.

#### Base op classes (implicit — you don’t list these per op)

| Base | Always appended | What you can do | Example |
| ---- | --------------- | --------------- | ------- |
| `TT_Op` | `TensorSizeTrait`, `VerifyTensorLayoutsTrait` | Reject invalid tensor sizes/layouts at verify; passes can assume those invariants | `tt.dot %a, %b, %c` — verifier checks blocked/MMA layouts vs shape |
| `TTG_Op` | `VerifyTensorLayoutsTrait`, `VerifyMemDescLayoutsTrait` | Same for GPU tensors + shared memdesc encodings | `ttg.local_load %smem` — `#shared` encoding must match load result layout |
| `TTNG_Op` | same as TTG | Same for TMEM/TMA ops | `ttng.tmem_load %acc` — TMEM encoding must match distributed result |

#### 1) Side effects / purity

| Trait | When | What you can do | Example |
| ----- | ---- | --------------- | ------- |
| **`Pure`** | No memory effects | CSE, DCE, hoist, speculate (if nothing else blocks) | `tt.addptr %p, %i` — two identical `addptr` on same `%p,%i` → one can be reused |
| **`NoMemoryEffect`** | Explicit “no effects” marker | Same as Pure; used when semantics are view-like, not compute | `tt.cat %a, %b` — concatenation is a shape/view op with no mem traffic |
| **`MemoryEffectsOpInterface`** | Custom read/write/barrier semantics | Query `getEffects()` for legal reordering, alias analysis, fence insertion | `tt.load %ptr` — pass won’t hoist load above a `tt.store` to same memory |
| **`MemRead` / `MemWrite` on operand** | Effect tied to one value | Fine-grained mod/ref without a full custom interface | `tt.store %ptr, %val` — only `%ptr` is `MemWrite<GlobalMemory>` |
| **`RecursiveMemoryEffects`** | Region body may have effects | Effect analysis walks into regions | `tt.reduce` — if body stores to smem, outer op is treated as having those effects |

Loads/stores usually **don’t** use `Pure`; they declare effects via `MemoryEffectsOpInterface` or operand resources.

#### 2) Shape / encoding / type constraints

| Trait | Role | What you can do | Example |
| ----- | ---- | --------------- | ------- |
| **`Elementwise`** | Per-element, same mapping | Fuse elementwise chains; broadcast/mask propagation | `tt.fp_to_fp %x` — same op on every lane; can fold into a `map_elementwise` region |
| **`SameOperandsAndResultShape`** | All shapes match | Rewrite without reshaping | `tt.bitcast %tensor` — `tensor<128xf32>` in → `tensor<128xf32>` out |
| **`SameOperandsAndResultEncoding`** | All encodings match | Layout-preserving rewrites | `tt.mulhiui %a, %b` — inputs and result stay `#blocked` |
| **`SameLoadStoreOperandsAndResultShape`** | Load/store shape rules | Prove load result shape = ptr pointee shape | `tt.load %ptr` where `%ptr: !tt.ptr<tensor<64x64xf16>>` → `tensor<64x64xf16>` |
| **`SameLoadStoreOperandsAndResultEncoding`** | Load/store encoding rules | Safe layout propagation through memory ops | Masked load keeps same `#blocked` on ptr lanes and result |
| **`TypesMatchWith<...>`** | Exact cross-operand type relation | Aggressive patterns; verifier catches mistakes early | `tt.load`: `result` must equal `getPointeeType(ptr)` — can't load `f32` from `!tt.ptr<f16>` |
| **`OptionalTypesMatchWith<...>`** | Optional operand type rules | Same, when operand may be absent | `ttg.async_copy_global_to_local` — optional `mask` type inferred only if present |
| **`InferTypeOpInterface`** | Structural type inference | Builders/parsers create op without spelling result type | `tt.load %ptr` — result type inferred from pointer in `inferReturnTypes` |
| **`InferTensorTypeOpWithLayoutEquivalence`** | Infer with layout equivalence | Canonicalize layouts without breaking semantics | Two `#blocked` layouts that are equivalent but not pointer-equal still verify |

Simple pure elementwise op minimum: `[Elementwise, SameOperandsAndResultType, Pure]`.

#### 3) Operand structure

| Trait | When | What you can do | Example |
| ----- | ---- | --------------- | ------- |
| **`AttrSizedOperandSegments`** | Optional / grouped operands | Parser + patterns treat operand groups correctly | `tt.load %ptr, %mask, %other` — segments: 1 ptr + 0–1 mask + 0–1 other (see §1.5) |

#### 4) Interfaces (passes query behavior)

Declared as `DeclareOpInterfaceMethods<…>` (sometimes with a method subset). Interfaces **do** add polymorphic getters — contrast with native traits. Detail in §2.6.

| Interface | Typical ops | What you can do | Example |
| --------- | ----------- | --------------- | ------- |
| **`PredicatedOpInterface`** | load, store, masked ops | Uniformly get/set predicate; merge predicates in pipeline | SWP replaces `mask` with `mask & phase_pred` via `setPredicateOperand` |
| **`DotOpInterface`** | `dot`, `dot_scaled`, MMA | Generic matmul transforms | `AccelerateMatmul` walks `DotOpInterface` to pick MMA layout |
| **`MemoryEffectsOpInterface`** | loads, allocs, barriers | Scheduling, LICM, fence insertion | `ttng.wait_barrier` — can't move past dependent TMA |
| **`InferTypeOpInterface`** | ops with inferred results | Rewrites create new ops without manual types | Canonicalizer rebuilds `tt.load` and re-infers result from new ptr |
| **`RegionBranchOpInterface`** | `warp_specialize`, multi-region | CFG-like analysis across regions | AWS pass treats partition regions as successors of `warp_specialize` |
| **`DestinationStyleOpInterface`** | in-place mem writers | DPS-style bufferization patterns | Some memdesc writers treated as “write into destination operand” |
| **`ViewLikeInterface`** | memdesc views | Fold/view-chain simplification | `memdesc_subslice(memdesc_subslice(%x))` → single subslice |

#### 5) Control flow / regions

| Trait | When | What you can do | Example |
| ----- | ---- | --------------- | ------- |
| **`Terminator`** | Ends a block/region | Region transforms know where control ends | `tt.reduce.return %v` — must be last op in reduce body |
| **`ReturnLike`** | Returns values from region | Same + return-value plumbing | `ttg.warp_yield %acc` — yields to `warp_specialize` results |
| **`HasParent<"ReduceOp">`** | Only valid inside parent | Prevents illegal hoisting/outlining | `tt.reduce.return` cannot appear in func body directly |
| **`IsolatedFromAbove`** | No implicit captures | Partition is self-contained; safe relayout per region | `warp_specialize` partition gets explicit captures only |
| **`AsyncRegions`** | Async execution model | Async-aware scheduling / speculation rules | `ttg.warp_specialize` — regions start concurrently, join at end |
| **`RecursivelySpeculatable`** | Safe to speculate through regions | More aggressive motion into/out of regions | Used with async/recursive ops when body is speculatable |

#### 6) Domain-specific native traits (TTG / TTNG)

| Trait | When | What you can do | Example |
| ----- | ---- | --------------- | ------- |
| **`MemDescViewTrait`** | memdesc view ops | Shared verify/lowering for views | `ttg.memdesc_subslice %buf[%i]` — new descriptor, same backing smem |
| **`LocalLoadTrait`** | local_load / local_gather | Shared rules for smem → register | `ttg.local_load %smem` — encoding must match what shared layout allows |
| **`MemWaitOpTrait`** | async wait ops | Classify as wait/barrier family | `ttg.async_wait` — grouped with other wait ops in pipeline lowering |

#### Worked example: `TT_LoadOp` (see also §1.5)

| Trait on `tt.load` | What it enables | Concrete example |
| ------------------ | --------------- | ---------------- |
| `SameLoadStoreOperandsAndResultShape/Encoding` | Result layout follows ptr | load from `tensor<128x64xf16,#blocked>` ptr → same-shaped `#blocked` result |
| `AttrSizedOperandSegments` | Optional mask/other | `tt.load %p` vs `tt.load %p, %mask, %other` — same op, different segments |
| `PredicatedOpInterface` | Pipeline can rewrite mask | SWP merges loop predicate into load mask |
| `MemoryEffectsOpInterface` | No illegal motion | Can't CSE two loads if another thread may store between them (without alias proof) |
| `InferTypeOpInterface` | Parser/builder convenience | `tt.load %p : !tt.ptr<f32>` → result `f32` or `tensor<...,f32>` auto |
| `TypesMatchWith` (×3) | ptr/mask/other consistency | `mask: tensor<128xi1>` must match pointee shape `128` |

**How to choose for a new op:** (1) Pure vs memory effects. (2) Tensor layout: `Same*Shape/Encoding` or load/store variants + `TypesMatchWith`. (3) Optional operands → `AttrSizedOperandSegments`. (4) Passes need generic access → declare an interface. (5) Region/terminator → `Terminator` / `HasParent` / region-branch. (6) TTG memdesc-specific → `MemDescViewTrait` / `LocalLoadTrait`.

---


### 2.6 Op interfaces — polymorphism in detail

#### What “polymorphic” means here

Several **different concrete ops** expose the **same method set**. Callers use the interface type, not `DotOp` vs `DotScaledOp` vs `ttng.warp_group_dot`.

`DotOpInterface` methods (`TritonOpInterfaces.td`): `getA`, `getB`, `getD`, `verifyDims`, `verifyOutputDims`.

Ops that implement it (examples):

- `tt.dot` — `DeclareOpInterfaceMethods<DotOpInterface>`
- `tt.dot_scaled` — `DeclareOpInterfaceMethods<DotOpInterface, ["verifyDims", "verifyOutputDims"]>`
- Nvidia GPU dialect ops also attach `DotOpInterface`

#### Example use

```cpp
// Pass / analysis — works for any dot-like op
mod.walk([&](Operation *op) {
  auto dot = dyn_cast<triton::DotOpInterface>(op);
  if (!dot)
    return;
  Value a = dot.getA();
  Value b = dot.getB();
  Value d = dot.getD();
  if (!dot.verifyDims())
    op->emitError("bad K dimension");
});
```

From real code patterns in-tree: `FenceInsertion.cpp` walks `DotOpInterface`; AMD/NVIDIA lowering uses `dyn_cast<DotOpInterface>`.

#### Why declare an interface when each op already has `getA()`?

Because **each op’s API is a different C++ type**:

```cpp
DotOp d = ...;           d.getA();   // OK
DotScaledOp s = ...;     s.getA();   // OK, different class
```

Without an interface, shared code must be:

```cpp
if (auto d = dyn_cast<DotOp>(op)) { ... d.getA(); }
else if (auto s = dyn_cast<DotScaledOp>(op)) { ... s.getA(); }
else if (auto w = dyn_cast<WarpGroupDotOp>(op)) { ... }
// breaks every time a new backend adds another dot op
```

The interface is the **stable abstraction** for “anything that is a matmul-like op.” Concrete ops still keep their own inherent attrs (`inputPrecision` on `tt.dot` only, scales on `dot_scaled` only); the interface only covers the **shared** surface.

Also: interface `verify` (in the TD) can run shared checks (`impl::verifyDotOpInterface`) in addition to each op’s `hasVerifier` body.

---

### 2.7 Type interfaces — same idea for types

#### Why `TensorDescInterface` when `TensorDescType` already has `getShape()`?

Because **more than one C++ type** is a “tensor descriptor”:

- `tt.TensorDescType` — tiled TMA-style desc (`!tt.tensordesc<…>`)
- `ttng.TensorDescIm2ColType` — im2col desc (extra conv params), also `[TT_TensorDescInterface]`

Callers that only need shape / element type / shared layout write:

```cpp
auto desc = dyn_cast<triton::TensorDescInterface>(someType);
if (!desc)
  return;
ArrayRef<int64_t> shape = desc.getShape();
Type elem = desc.getElementType();
Attribute layout = desc.getSharedLayout();
```

Used e.g. in TMA utilities / verify helpers (`getTMASwizzleMode(loc, TensorDescInterface ty)`).

Without the interface, every helper would hard-code `TensorDescType` and break for im2col (or duplicate logic).

**Ops API vs Type interface:** same motivation — polymorphism across *sibling* concrete definitions that share a concept.

---

### 2.8 Dialect interfaces

- `TritonInlinerInterface` (`Interfaces.h`): inlining policy for `tt.func`
- `DialectInferLayoutInterface` / `DialectVerifyTensorLayoutInterface` (`Dialect.h`): implemented by GPU dialects; used by layout traits/transforms

Registered in `initialize()` via `addInterfaces<TritonInlinerInterface>()`.

---

### 2.9 Dependency diagram

```text
TritonDialect (ctor loads arith/math/scf/cf/ub → initialize)
        │
        ├─ registerTypes() ──► PointerType, TensorDescType (+ TensorDescInterface)
        ├─ addOperations ──► tt.* (traits + op interfaces + inherent attrs)
        └─ addInterfaces ──► TritonInlinerInterface

Passes attach discardable attrs (tt.num_stages) via dialect helpers
Passes query DotOpInterface / TensorDescInterface without concrete switches
```

---

## 3. TableGen vs manual — cheat sheet

| Artifact | Generated | Manual |
| -------- | --------- | ------ |
| Dialect skeleton + `void initialize();` + attr helpers + dep loads | Always / from flags | `initialize()` **body**, `registerTypes()` **body**, `materializeConstant` |
| TypeDefs | `Types.*.inc` | parse/print/verify in `Types.cpp` |
| `AnyTypeOf` constraints | Only in TableGen | No C++ |
| Ops schema | `Ops.*.inc` | verify/fold/canonicalize/interface methods |
| Enum attrs | `OpsEnums.*.inc` | — |
| Native traits | name in TD | `Traits.h/.cpp` |
| Op/Type interfaces | `.inc` decls | verify / non-default methods in `.cpp` |

---

## 4. TableGen codegen and `triton-opt` tests

### 4.1 How TableGen generates C++

You do **not** run `mlir-tblgen` by hand in normal development. CMake wraps it:

```text
*.td  ──mlir-tblgen -gen-…──►  *.h.inc / *.cpp.inc
         (mlir_tablegen)           included by Dialect.h / Ops.cpp / …
```

#### CMake wiring (`include/triton/Dialect/Triton/IR/CMakeLists.txt`)

`LLVM_TARGET_DEFINITIONS` names the `.td` file; each `mlir_tablegen` line is one generator:

| `.td` | Generator flags | Output |
| ----- | --------------- | ------ |
| `TritonOps.td` | `-gen-op-decls` / `-gen-op-defs` | `Ops.h.inc` / `Ops.cpp.inc` |
| `TritonOps.td` | `-gen-enum-decls` / `-gen-enum-defs` | `OpsEnums.h.inc` / `OpsEnums.cpp.inc` |
| `TritonDialect.td` | `-gen-dialect-decls` / `-gen-dialect-defs` | `Dialect.h.inc` / `Dialect.cpp.inc` |
| `TritonTypes.td` | `-gen-typedef-decls` / `-gen-typedef-defs` | `Types.h.inc` / `Types.cpp.inc` |
| `TritonOpInterfaces.td` | `-gen-op-interface-decls` / `-gen-op-interface-defs` | `OpInterfaces.h.inc` / `.cpp.inc` |
| `TritonTypeInterfaces.td` | `-gen-type-interface-decls` / `-gen-type-interface-defs` | `TypeInterfaces.h.inc` / `.cpp.inc` |

`add_public_tablegen_target(TritonTableGen)` is the ninja target that actually runs `mlir-tblgen`. `lib/Dialect/Triton/IR/CMakeLists.txt` lists `DEPENDS TritonTableGen` so `TritonIR` rebuilds after `.td` changes.

Other dialects use the same pattern (`TTG_Op`, `TTNG_Op`, …). Extra generators: `-gen-rewriters` (`TritonCanonicalize.inc`), `-gen-pass-decls` (pass registry).

#### How generated files are consumed

Headers include **decls**; `.cpp` files include **defs** behind macros:

```cpp
// Dialect.h — class LoadOp, getters, verify hooks
#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.h.inc"

// Dialect.cpp — register every tt.* op
addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"
>;

// Ops.cpp — parse/print/builders/verifyTrait glue
#include "triton/Dialect/Triton/IR/Ops.cpp.inc"
```

`.inc` files live under the **build tree** (e.g. `BUILD_DIR/include/triton/Dialect/Triton/IR/`), not the source tree. Manual C++ (`Ops.cpp` `verify()`, `fold()`, interface methods) fills what TableGen cannot.

#### Rebuild after a `.td` change

From the CMake build dir (`BUILD_DIR := $(shell PYTHONPATH="./python" python3 -c 'from build_helpers import get_cmake_dir; print(get_cmake_dir())')`):

```bash
cd BUILD_DIR
ninja TritonTableGen          # only regenerate .inc
ninja triton-opt              # relink the tester (depends on TritonIR)
```

Equivalent of the generator itself (CMake already sets include paths for MLIR + Triton `.td`):

```text
mlir-tblgen -gen-op-decls TritonOps.td -I <mlir/include> -I include/ …
```

If `triton-opt` is stale after a dialect edit, parse/print/verify tests will lie.

---

### 4.2 `triton-opt` + lit: how dialect tests run and verify

`triton-opt` is Triton’s `mlir-opt`: parse MLIR, run passes, print IR. It is **not** the Python compiler pipeline.

```text
.mlir  →  parse (dialects from registry)  →  verify  →  optional passes  →  print
              │                                  │
              └─ assemblyFormat + types          └─ traits + hasVerifier + interfaces
```

#### Tool construction

| Piece | Role |
| ----- | ---- |
| `bin/triton-opt.cpp` | `MlirOptMain` + `registerTritonDialects` |
| `bin/RegisterTritonDialects.h` | Registers `tt` / `ttg` / `ttng` / NVGPU / AMD + **all passes** (`-canonicalize`, `-tritongpu-pipeline`, …) |
| Linked libs | `TritonIR` (generated + manual op impl) so parse/print/verify match TableGen |

Default (no pass flags): parse → **module verify** → print. That is enough to test “this IR is well-formed.”

#### Lit harness

| File | Role |
| ---- | ---- |
| `test/lit.cfg.py` | Suite `TRITON`; suffixes `.mlir` / `.ll`; substitutes `triton-opt` from `BUILD_DIR/bin` |
| `test/lit.site.cfg.py.in` | CMake fills `triton_obj_root`, LLVM tools, FileCheck path |
| `test/CMakeLists.txt` | `check-triton-lit-tests` depends on `triton-opt`; `lit` runs the suite |

Each test file starts with `// RUN:` — lit executes that shell line. `%s` is the test file. `FileCheck` matches stdout against `CHECK:` comments in the same file.

#### Three verification styles (dialect IR)

**1. Round-trip / parse-print** (`test/Triton/ops.mlir`)

```mlir
// RUN: triton-opt %s | FileCheck %s
// CHECK: tt.load %{{.*}} : !tt.ptr<f32>
%a = tt.load %ptr : !tt.ptr<f32>
```

Mechanism: parse using generated `assemblyFormat` → verify traits (`TypesMatchWith`, `TensorSizeTrait`, …) → print → FileCheck. Proves the op **parses, verifies, and pretty-prints**. `| FileCheck` (or `>/dev/null`) fails if verify emits an error.

**2. Negative diagnostics** (`test/Triton/invalid.mlir`)

```mlir
// RUN: triton-opt --split-input-file %s --verify-diagnostics

// expected-error @+1 {{Cannot bitcast data-type of size}}
%a = tt.bitcast %arg0 : tensor<128xf32> -> tensor<128xi16>
```

`--split-input-file` treats `// -----` as separate modules (one bad op per chunk). `--verify-diagnostics` matches `expected-error` / `expected-note` against verifier messages from traits / `hasVerifier`. That is how §2.5 constraints are regression-tested.

**3. Pass transform + FileCheck** (`test/Triton/canonicalize.mlir`)

```mlir
// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s
// CHECK-NOT: tt.addptr
%0 = tt.addptr %arg, %c0 : …
```

Parse → run named pass(es) → print → FileCheck. Tests folders/canonicalizers (`hasFolder` / `hasCanonicalizer`) and later compiler passes (`-tritongpu-pipeline`, `-tritongpu-optimize-partition-warps`, …).

#### Commands

```bash
# from BUILD_DIR after ninja triton-opt
ninja triton-opt
lit -v test/Triton/ops.mlir
lit -v test/Triton/invalid.mlir

# or one-shot without lit
./bin/triton-opt ../../test/Triton/ops.mlir >/dev/null
./bin/triton-opt --split-input-file --verify-diagnostics ../../test/Triton/invalid.mlir

ninja check-triton-lit-tests   # whole suite
```

No GPU required. Python pytest (`python/test/`) is a different layer (compile kernels); lit is the dialect/pass IR contract.

```text
.td change → ninja TritonTableGen + triton-opt → lit .mlir
  generated parse/print/verify ──┐
  Traits.h / Ops.cpp verify     ─┼─► triton-opt verify / expected-error
  folders / passes              ─┘     FileCheck on printed IR
```

---

## 5. Quick answers index (Q1–Q9)

1. **Ops.td / Traits / OpInterface** — §1.5  
2. **`initialize()` not in TD** — always emitted by `DialectGen.cpp`; you implement body — §2.1.2  
3. **`registerTypes()`** — declared in TD `extraClassDeclaration`; defined in `Types.cpp`; TD `include Types.td` does *not* register at runtime — §2.1.4  
4. **Reframed 2.1** — §2.1.0–2.1.4  
5. **`TT_FloatLike` missing in Types.h.inc** — constraint only — §2.2.1  
6. **pointeeType / addressSpace / shape / …** — §2.2.2  
7. **Inherent vs discardable vs enum attrs** — §2.3  
8. **OpInterface polymorphism + why** — §2.6  
9. **TypeInterface polymorphism + why** — §2.7  
