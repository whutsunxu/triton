# AxisInfo Basics — Contiguity, Divisibility, Constancy

This note summarizes what `AxisInfo` means in Triton IR analysis, with concrete
examples (especially `arith.muli`) and why the compiler needs it.

Sources: `include/triton/Analysis/AxisInfo.h`, `lib/Analysis/AxisInfo.cpp`.

---

## 1. General idea

**AxisInfo describes features of the values stored in an SSA tensor (or scalar),
not features of a physical buffer’s base address / allocation.**

For each dimension `d` of an SSA value, AxisInfo tracks three properties:

| Property | Meaning |
| --- | --- |
| **contiguity[d]** | Length of the shortest run of consecutive integers (`…, x, x+1, x+2, …`) along dim `d` |
| **divisibility[d]** | Largest power of 2 that divides the **first element of each contiguity group** along dim `d` |
| **constancy[d]** | Length of the shortest run of **identical** values along dim `d` |

Optional: `constantValue` when every element is the same known integer.

So for:

```mlir
%off = ... : tensor<64xi32>           // integer offsets
%p   = ... : tensor<64x!tt.ptr<f32>>  // pointer values
```

- AxisInfo on `%off` = facts about the **offset integers**.
- AxisInfo on `%p` = facts about the **pointer values** (addresses as values).

It is **not**:

- “this `local_alloc` / global buffer starts at an aligned physical address”, nor
- “this SSA was written into a newly allocated buffer, so alignment resets”.

Most SSA values never get `tt.contiguity` / `tt.divisibility` / `tt.constancy`
attrs attached on the IR. The analysis keeps them in memory
(`ModuleAxisInfoAnalysis` map). Those attrs are mainly used as **hints on
function arguments** (and sometimes forwarded to callees / relayout containers).

---

## 2. Details with examples

### 2.1 How to read the three fields

Example 1D tensor of integers:

```text
values: [4, 5, 6, 7, 12, 13, 14, 15]
```

- Contiguous groups of length 4: `[4,5,6,7]` and `[12,13,14,15]`
  → `contiguity = [4]`
- Group starts are `4` and `12`, both divisible by 4
  → `divisibility = [4]`
- No repeated runs
  → `constancy = [1]`

Important nuance: **divisibility is about group starts**, not every element.
In `[4,5,6,7]`, only `4` is guaranteed ÷4; `5` is not.

Lengths are powers of two (or `1` = unknown / minimal). Contiguity like `[1,5]`
is not a valid AxisInfo shape; use e.g. `[1,4]`.

### 2.2 Example A — contiguous lhs × splat rhs (`arith.muli`)

```text
lhs = [[ 4,  5,  6,  7],
       [12, 13, 14, 15]]     shape [2, 4]

rhs = [[ 2,  2,  2,  2],
       [ 2,  2,  2,  2]]     shape [2, 4]   (splat of 2)

out = lhs * rhs
    = [[ 8, 10, 12, 14],
       [24, 26, 28, 30]]
```

Operand AxisInfo:

| | contiguity | divisibility | constancy | constantValue |
| --- | --- | --- | --- | --- |
| **lhs** | `[1, 4]` | `[1, 4]` | `[1, 1]` | none |
| **rhs** | `[1, 1]` | `[2, 2]` | `[2, 4]` | `2` |

`arith.muli` visitor rules (simplified):

1. **Contiguity** — keep an operand’s contiguity only if the other side is
   constant `1`; otherwise → `1`.
2. **Divisibility** — if an operand has `contiguity > 1` and the other side is
   **not** constant `1`, force that operand’s divisibility to `1` first
   (group-start-only fact is unsafe after multiply); then
   `out.div = lhsDiv * rhsDiv`.
3. **Constancy** — `gcd(lhs.constancy, rhs.constancy)` per dim.

Applying to dim 1:

- Contiguity: rhs is constant **2**, not 1 → `out.contiguity[1] = 1`
- Divisibility: lhs contig=4 → force `lhsDiv=1`; rhs contig=1 → keep `rhsDiv=2`
  → `out.divisibility[1] = 1 * 2 = 2`
- Constancy: `gcd(1, 4) = 1`

Result AxisInfo:

```text
out:
  contiguity   = [1, 1]
  divisibility = [2, 2]
  constancy    = [1, 1]
```

Why drop lhs’s divisibility `4`? If we kept `4` after `*2`, we would claim
alignment by 4, but `10` is not ÷4. Forcing `lhsDiv=1` then `×2` correctly
yields “every element even”.

### 2.3 Example B — two uniform splats (`arith.muli` amplifies divisibility)

`splat(8)` means: broadcast scalar `8` into a tensor where **every element is 8**
(e.g. shape `[2,4]` → `[[8,8,8,8],[8,8,8,8]]`). Same for `splat(16)`.

```text
lhs = splat(8)    // every element is 8
  contig [1,1]       // neighbors are equal, not x,x+1,x+2
  div    [8,8]       // every value ÷8
  constancy full     // e.g. shape [2,4] → constancy [2,4]
  const = 8

rhs = splat(16)   // every element is 16
  contig [1,1], div [16,16], constancy full, const=16

out = splat(128)  // every element is 128
  contig     [1, 1]
  div        [128, 128]   ← 8 * 16 (both contig==1, no forced-to-1)
  constancy  full
  constantValue = 128
```

Here both sides are **uniform**, so “divisible by D” applies to every element.
Multiplying divisibilities is safe and useful (e.g. `pid * BLOCK` style offsets).

### 2.4 Example C — two contiguous ranges multiplied

```text
lhs = [4, 5, 6, 7]     contig [4], div [4], constancy [1]
rhs = [8, 9,10,11]     contig [4], div [8], constancy [1]

out = [32, 45, 60, 77]
  contig     [1]        ← neither side is *1
  div        [1]        ← both contig>1 → both div forced to 1 → 1*1
  constancy  [1]
```

Product is neither contiguous nor commonly divisible — analysis stays conservative.

### 2.5 Example D — `tt.make_range` (seed, not multiply)

```mlir
%0 = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
```

Visitor sets roughly:

```text
contiguity   = [64]          // 0,1,2,...,63
divisibility = [highestPowOf2Divisor(start)]  // start=0 → very large / max
constancy    = [1]
```

This is a common **seed** for later `addi` / `muli` / `addptr` chains that build
load/store pointers.

### 2.6 Mental model for `arith.muli`

`arith.muli` produces a **new SSA value**, but AxisInfo still asks:

> Given algebraic facts about elements of `%lhs` and `%rhs`, what facts hold
> for elements of `%lhs * %rhs`?

That is forward **value** analysis (like constant / range propagation), **not**
“new buffer → reset alignment to 1”.

### 2.7 Real case — `tt.divisibility` on matmul kernel args

From a real pipelined matmul dump:

`workspace/test_cases/.../irs/05_after_ttgpuir_add_pipeline.ttgir___p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1`

The public kernel starts roughly like:

```mlir
tt.func public @_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1(
    %Y: !tt.tensordesc<1x64x256xf8E5M2, #shared>,
    ...
    %YPtr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32},
    %stride_y_k: i32 {tt.divisibility = 16 : i32},
    %stride_y_z: i32 {tt.divisibility = 16 : i32},
    %stride_y_m: i32 {tt.divisibility = 16 : i32},
    %YExpectedScale: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    ...
    %XPtr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32},
    ...
    %WPtr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32},
    ...
    %B: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %M: i32 {tt.divisibility = 16 : i32},
    %N: i32 {tt.divisibility = 16 : i32},
    %K: i32 {tt.divisibility = 16 : i32},
    %grid_m: i32 {tt.divisibility = 16 : i32},
    %batch_size: i32,          // no tt.divisibility
    %grid_n: i32,              // no tt.divisibility
    ...
  )
```

Here `{tt.divisibility = 16 : i32}` is a **frontend hint on the function
argument**. `ModuleAxisInfoAnalysis` seeds that block arg as roughly:

```text
contiguity   = [1]
divisibility = [16]
constancy    = [1]
```

Two meanings on this signature:

| Arg kind | Example | Meaning of `tt.divisibility = 16` |
| --- | --- | --- |
| **Pointer** | `%YPtr`, `%XPtr`, `%WPtr`, `%B`, scale ptrs | The **pointer value** (base address) is 16-byte aligned: `YPtr % 16 == 0` |
| **Integer** | `%M`, `%N`, `%K`, `%stride_y_k`, `%grid_m` | The **integer value** is a multiple of 16: `M % 16 == 0` |

Args **without** the attr (`%batch_size`, `%grid_n`, many shape/stride `i64`s)
start pessimistic (`divisibility = 1`) unless later ops prove more.

This is only the **seed**. Propagation through `addi` / `muli` / `addptr` in the
body builds AxisInfo for intermediate SSAs used by coalesce / vectorized load-store.

---

## 3. Why analyze AxisInfo in IR? (motivation)

Triton lowers tensors of pointers / offsets into hardware loads and stores.
Hardware vectorization and coalescing need answers like:

- Are neighboring lanes’ addresses consecutive? → **vectorized load/store**
- Are addresses / offsets aligned to 16 bytes? → **wider accesses, fewer transactions**
- Is a mask uniform across a warp segment? → **simpler predicated codegen**

AxisInfo answers those questions **conservatively from IR**, without executing the kernel.

Typical consumers:

| Consumer | How AxisInfo helps |
| --- | --- |
| **Coalesce** (`TritonGPU/Transforms/Coalesce.cpp`) | Pick layouts so consecutive threads hit consecutive addresses |
| **Load/Store → LLVM** (e.g. NVIDIA `LoadStoreOpToLLVM.cpp`) | Choose vector width from pointer contiguity / alignment |
| **AMD buffer ops / similar** | Separate base + offset; use offset AxisInfo with pointee bitwidth |
| **Warp-specialize relayout** (`OptimizePartitionWarps`) | Copy capture AxisInfo onto temporary func args so layout reassignment keeps alignment hints |

Without AxisInfo, the compiler would often fall back to scalar / unaligned accesses
even when the program is clearly structured as `arange + pid * BLOCK`, masks that
are constant across a tile edge, etc.

**Bottom line:** AxisInfo is a cheap static summary of **value structure**
(contiguous / aligned / constant runs) so later passes can emit efficient memory
accesses and layouts — it is not a model of physical buffer allocation addresses.
