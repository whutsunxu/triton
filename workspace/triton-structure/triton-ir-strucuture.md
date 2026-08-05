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

### 2.5 Traits — verify mix-ins

Covered in §1.5. Key: traits do **not** add polymorphic getters; they add `verifyTrait` hooks. Implemented in `Traits.h` / `Traits.cpp`.

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

## 4. Tests

Under `test/Triton/` (`ops.mlir`, `invalid.mlir`, `canonicalize.mlir`, …).

```bash
OPT=build/cmake.linux-x86_64-cpython-3.12/bin/triton-opt
$OPT test/Triton/ops.mlir >/dev/null
$OPT --split-input-file --verify-diagnostics test/Triton/invalid.mlir

# or
ninja check-triton-lit-tests   # from the CMake build dir
```

Lit config: `test/lit.cfg.py`; suite target in `test/CMakeLists.txt`.

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
