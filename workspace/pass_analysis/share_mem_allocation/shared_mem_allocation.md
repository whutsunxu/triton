# Shared-memory allocation analysis — general idea

| | |
|--|--|
| **Core analysis** | `ModuleAllocation` / `Allocation` / `AllocationAnalysis` (`include/triton/Analysis/Allocation.h`, `lib/Analysis/Allocation.cpp`) |
| **NVIDIA pass** | `AllocateSharedMemoryNv` (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp`) |
| **Example IR** | `.../irs/15_after_ttgpuir_add_allocate_shared_memory_nv...` (`ttg.shared`, `allocation.offset`) |

---

## 1. General idea

### Problem

GPU kernels share one CTA-wide **shared memory** arena. Many ops need slices of it at different times: explicit tiles and barriers, short-lived staging for layout converts, space to hand values into warp-specialized partitions, and room for callees that themselves use shared memory. The compiler must decide **how large** that arena is and **where** each live use sits, so non-overlapping lifetimes can reuse the same bytes and the final binary can address SMEM with fixed offsets.

### What this analysis does

Shared-memory allocation analysis answers those questions **before** LLVM lowering. For a module it produces, per function, a complete picture of every shared buffer that function needs and a packed layout (offsets and a total size). Across the module it then takes a single CTA size large enough for the entry kernels. A backend pass writes that into IR as module shared size and per-op offsets so later lowering can emit concrete addresses.

### General flow

The work has three stages.

**Stage 1 — Walk functions in call-graph order.**  
Analysis follows who calls whom. A function is processed only after every function it calls has already been processed. That bottom-up order matters because a caller must know how much shared memory each callee needs before it can finish its own layout. Results are kept one record per function.

**Stage 2 — Analyze each function.**  
Inside a function the analysis enumerates everything that needs shared memory, figures out which uses can overlap in time (liveness), then packs those ranges into a linear address space and records the high-water size for that function.

**Stage 3 — Publish to IR.**  
The NVIDIA allocate-shared-memory pass runs this analysis (with NVIDIA-specific scratch sizing where needed) and attaches the outcomes: the module’s total shared size and `allocation.offset` on the relevant ops. Downstream codegen reads those attributes instead of re-deriving the layout.

### End-to-end picture

```text
module (call graph)
  └─ for each function, callees first
       collect SMEM needs → liveness → pack offsets → function total
  └─ module total = max over entry functions
       ↓
  annotate IR (ttg.shared, allocation.offset)
       ↓
  LLVM / PTX lowering uses fixed SMEM addresses
```

Later sections cover how those needs are classified and stored, then how packing works. Part 1 is only this outer story: **why** allocation exists, **what** it produces, and **in what order** the module and each function are handled.

---

## 2. How shared-buffer uses are classified and stored

Inside each function’s `Allocation` record, every shared-memory need is turned into a **buffer** with a kind, size, and (after packing) an offset. Uses fall into a few classes; each class is stored in a dedicated map so later liveness and offset assignment can find them uniformly.

All concrete buffers also live in one `bufferSet` (id → buffer). The maps below are the **indexes** from IR entities into that set. After packing, `sharedMemorySize` is the function’s high-water mark.

### Explicit — allocated tiles and barriers

**Meaning:** The IR allocates a real shared object with an SSA result (`ttg.local_alloc` into `#smem`): matrix tiles, double buffers, mbarrier slots, and similar.

**Storage:** `valueBuffer`: **Value → list of buffers** (usually one; partitioned encodings may create several neighbor pieces).

**Example (IR 15):**

```mlir
%w = ttg.local_alloc {allocation.offset = 0 : i32}
    : () -> !ttg.memdesc<2x128x256xf8E5M2, #shared1, #smem, mutable>
%x_14 = ttg.local_alloc {allocation.offset = 99328 : i32}
    : () -> !ttg.memdesc<2x1xi64, #shared2, #smem, mutable>   // barrier object
```

Here `%w` / `%x_14` are keys in `valueBuffer`; each maps to an Explicit buffer whose size comes from shape × encoding × element width.

### Scratch — temporary space owned by an operation (or the function)

**Meaning:** Something in *this* function needs a private SMEM temp for its own lowering, not a user-facing `local_alloc`. Typical cases: layout converts that stage through SMEM; warp-specialize packing captures into a contiguous “struct”; the function reserving one byte per partition warp for WS control/state.

**Storage:** `opScratch`: **Operation → buffer**.

**Examples (IR 15):**

```mlir
// Staging for layout rewrite
%acc_tile = ttg.convert_layout %bias {allocation.offset = 98304 : i32} : ...

// Capture struct: pack warp_specialize operands for isolated partitions
%ws = ttg.warp_specialize(%pid_z, %width, %x, %w, ...) 
    attributes {allocation.offset = 81920 : i32, ...}

// Per partition-warp control bytes (size = total partition warps)
tt.func public @...(...) attributes {allocation.offset = 99392 : i32, ...}
```

The convert and the `warp_specialize` each own a Scratch buffer keyed by that op. The function’s Scratch is keyed by the `tt.func` itself (4 bytes after pad in this matmul: two partitions × two warps).

### Virtual — call-site reservation for a callee’s footprint

**Meaning:** A call does not allocate a new tile in the callee’s sense; while the callee runs, the CTA still needs enough shared memory for that callee’s whole layout. The caller records a **Virtual** buffer at the call whose size equals the callee’s already-computed total. That lets packing in the caller treat “callee is running” as a live range that consumes that much SMEM.

**Storage:** `opVirtual`: **Operation (the call) → buffer**.

**Example (illustrative — this matmul is a single public func):**

```mlir
tt.func @leaf() { ... local_alloc ... }          // analyzed first → size A

tt.func @mid() {
  %t = ttg.local_alloc ...
  tt.call @leaf()    // Virtual buffer on this call, size = A
}
```

`opVirtual[call]` holds one Virtual buffer of size A; it participates in `@mid`’s liveness graph with `%t` and any Scratch temps.

### Alias — views that do not allocate again

**Meaning:** Ops such as `memdesc_index` produce a **view** onto an existing Explicit alloc. They must not get a second tile; liveness should attribute uses of the view to the underlying buffer.

**Storage:** `aliasBuffer`: **Value (view) → set of Explicit buffers** it may refer to (from shared-memory alias analysis).

**Example (IR 15 pattern):**

```mlir
%x = ttg.local_alloc ... : !ttg.memdesc<2x64x128xf8E5M2, ...>
%x_44 = ttg.memdesc_index %x[%slot] : ... -> !ttg.memdesc<64x128xf8E5M2, ...>
%tile = ttg.local_load %x_44 : ...
```

`%x` is in `valueBuffer` (Explicit). `%x_44` is recorded in `aliasBuffer` pointing at `%x`’s buffer(s), so the load’s live range extends the same physical allocation.

### How the maps fit together

```text
per-function Allocation
  valueBuffer[alloc Value]  → Explicit buffers   (local_alloc tiles / barriers)
  opScratch[op or func]     → Scratch buffers    (cvt, WS captures, warp-index bytes, …)
  opVirtual[call]           → Virtual buffers    (callee total size at call site)
  aliasBuffer[view Value]   → same Explicit bufs (memdesc views, etc.)
  bufferSet[id]             → all BufferT (kind, size, align, offset)
  sharedMemorySize          → packed high-water mark
```

Classification answers *what kind of SMEM use*; the maps answer *how to look it up from IR*. Liveness and offset packing (next sections) consume these maps, not the raw ops directly.

---

## 3. Per-function pipeline (after classification)

For each function, `Allocation::run` → `AllocationAnalysis`:

```text
getValuesAndSizes()     // Explicit + Scratch/Virtual + aliases
resolveLiveness()       // live intervals for buffers
computeOffsets()        // pack non-overlapping buffers; set sharedMemorySize
```

Then NVIDIA `AllocateSharedMemoryNv`:

```text
ModuleAllocation(mod, getNvidiaAllocationAnalysisScratchSizeFn(...))
attachAllocationSizeAndOffsetAttr(mod, allocation)
→ module ttg.shared = …, ops get allocation.offset
```

**IR 15 module:**

```mlir
module attributes {ttg.shared = 99396 : i32, "ttg.total-num-warps" = 8 : i32, ...}
```

Module-level storage is `funcMap: Func → Allocation`; `ttg.shared` is roughly the max over entry functions. Totals are per function; buffer identity and offsets live in that function’s maps above.

### 3.1 Buffer recognition: type and size

`getValuesAndSizes()` is where the analysis first turns IR into buffers. Recognition has two halves:

1. **Type recognize** — decide *what kind* of buffer this op/value needs (or that it only aliases an existing one).
2. **Size recognize** — decide *how many bytes* (and alignment) that buffer occupies. Aliases skip this: they attach to an already-sized Explicit buffer.

```text
getValuesAndSizes()
  ① PreOrder over ops in this tt.func (nested regions yes; callee bodies no)
       getExplicitValueSize(op)  → type Explicit + size from memdesc
       getScratchValueSize(op)   → type Scratch or Virtual + size
  ② SharedMemoryAliasAnalysis (dataflow)
  ③ PreOrder: getValueAlias → type Alias (no new size)
```

Every concrete buffer still lands in `bufferSet[id]`. The maps below are **indexes** from IR into that set. Zero-byte scratch/virtual requests are not stored (`maybeAddScratchBuffer` only inserts when `bytes > 0`).

#### Type recognize (dispatch)

| Question | How | Result |
|----------|-----|--------|
| Shared `local_alloc`? | `LocalAllocOp` + `isSharedMemoryAlloc()` | **Explicit** |
| Call site? | `CallOpInterface` | **Virtual** |
| WS / func / cvt / reduce / …? | remaining branches of `getScratchValueSize` + `scratchSizeGetter` | **Scratch** |
| View / select / loop-carried memdesc? | alias analysis says “these allocs” | **Alias** (link only) |

`getScratchValueSize` checks in a fixed order: **call → warp_specialize → FunctionOpInterface → else `scratchSizeGetter`**. First match wins.

#### Size recognize (where bytes come from)

| Kind | Size source |
|------|-------------|
| **Explicit** | encoding + alloc shape (+ padding); partitioned encodings create *N* buffers of `pieceSize × numGroups` each |
| **Virtual** | callee’s finished `Allocation::getSharedMemorySize()` from `funcAllocMap` |
| **Scratch** | op-specific: WS → `scratchSizeGetter` / `getCaptureSize`; func → `max(getTotalPartitionWarps())` (1 byte each); else → `scratchSizeGetter` (cvt, reduce, …) |
| **Alias** | none — reuses Explicit sizes |

#### How Scratch / Virtual / Alias are stored

`addBuffer` always creates a `BufferT` in `bufferSet`, then indexes it by kind:

```text
Scratch  → opScratch[Operation*] = BufferT*
Virtual  → opVirtual[Operation*] = BufferT*
Alias    → aliasBuffer[Value]    = Set of BufferT*   (pointers into Explicit buffers)
```

**Scratch — `opScratch: Operation → BufferT*`**

- Key is the **owning op** (the `convert_layout`, the `warp_specialize`, or the `tt.func` itself for warp-index bytes).
- One Scratch buffer per such op (when size > 0).
- Kind field on the `BufferT` is `Scratch`. Owner is that same `Operation*`.
- Later packing reads `opScratch` for scratch liveness (live range ≈ that op; func-owned scratch is live for the whole function).

**Virtual — `opVirtual: Operation → BufferT*`**

- Key is the **call op** (`tt.call`).
- Size = callee total SMEM; kind is `Virtual`.
- Same map type as Scratch (`OpScratchMapT`), but a separate `opVirtual` container so Virtual is not mixed with Scratch in classification APIs (`getBufferId` checks both).
- Liveness treats Virtual like a short-lived reservation at the call site: while the callee runs, that many bytes are “in use” in the caller’s packing.

**Alias — `aliasBuffer: Value → SetVector<BufferT*>`**

- Key is a **view / SSA Value** (e.g. `memdesc_index` result), not a new allocation.
- Values are pointers to **existing Explicit** buffers from `valueBuffer[alloc]` — `addAlias(value, alloc)` copies those pointers.
- No new `BufferT`, no size, no kind of its own. Alias is only an index so liveness of the view can extend the underlying Explicit buffer(s).

```text
%x = local_alloc          → valueBuffer[%x] = { Explicit buf }
%x_44 = memdesc_index %x  → aliasBuffer[%x_44] = { same Explicit buf }
                            (no second buffer in bufferSet)
```

**Contrast with Explicit:** Explicit is keyed by **Value** (`valueBuffer: Value → SmallVector<BufferT*>`) because the alloc *result* owns the tile; Scratch/Virtual are keyed by **Operation** because there is no dedicated alloc SSA value for those temps.

### 3.2 Liveness analysis (`resolveLiveness`)

After buffers exist with sizes, the analysis decides **when** each buffer is live. Result is `bufferRange: BufferT* → Interval[opIdStart, opIdEnd)` — a half-open interval over operation IDs. Later packing treats overlapping intervals as interfering (cannot reuse the same SMEM offset).

```text
resolveLiveness()
  ① Prework: PostOrder walk → operationId[op]
  ② Explicit: MLIR Liveness on alloc Values → bufferRange
  ③ Alias: liveness of views ∪ into the same Explicit buffers
  ④ Scratch / Virtual: special ranges from owning ops
```

#### Prework — PostOrder op-ID assignment

Every op under the function root gets a dense ID in **post-order**:

```text
operation->walk<PostOrder>([&](op) { operationId[op] = operationId.size(); });
```

Children are numbered **before** their parent, so a parent’s ID is always **greater** than any nested op’s ID. That matters when a value defined outside a region is used inside it (e.g. `%5` used inside `scf.for %6`): the loop op’s ID is past the body, so the value’s live interval can stretch through the whole nest instead of ending too early.

Live ranges for Values are then built as:

```text
liveOps = Liveness(operation).resolveLiveness(value)
range   = [min(operationId[op]), max(operationId[op]) + 1)   over liveOps
```

#### Explicit buffer liveness

For each entry in `valueBuffer` (alloc Value → one or more Explicit buffers):

1. Compute that Value’s liveness interval via the helper above.
2. Write the **same** interval onto every buffer for that Value (partitioned encodings share one live range across all neighbor pieces).

```text
%x = local_alloc → buffers {B0, B1, …}
bufferRange[B0] = bufferRange[B1] = live(%x)
```

No separate range per partition piece: they are allocated and freed together from the allocator’s point of view.

#### Alias — union into Explicit ranges

Aliases do not get their own `bufferRange` entry as a new buffer. For each `aliasBuffer[view] → { Explicit buffers… }`:

1. Compute `live(view)` the same way (uses of `memdesc_index`, selects, loop-carried args, …).
2. For each underlying Explicit buffer, **union** that interval into the existing range:

```text
bufferRange[B] = [min(old.start, live(view).start),
                  max(old.end,   live(view).end))
```

So a tile stays reserved from its alloc through every view that still needs it. Loop-carried and `scf.yield` aliases (already recorded in §3.1) are why this step can stretch liveness far past the defining `local_alloc`.

#### Scratch / Virtual — special process

Scratch and Virtual are **not** Value-based. `resolveScratchBufferLiveness` walks `opScratch` and `opVirtual` with the same rule:

| Owner | Live interval | Why |
|-------|---------------|-----|
| `op ==` function root (`tt.func`) | `[0, ∞)` (whole function) | WS warp-index bytes must stay for the entire kernel |
| any other Scratch / Virtual op | `[operationId[op], operationId[op] + 1)` | temp is needed only for that op (cvt staging, WS capture struct, call-site Virtual reservation) |

So Scratch/Virtual liveness is **point-like** on the owning op (except func-owned Scratch), unlike Explicit ranges that span many uses. Virtual at a call is treated like a one-op scratch reservation sized to the callee total — it interferes with anything live across that call.

```text
convert_layout %c          → Scratch live ≈ [id(%c), id(%c)+1)
tt.call @leaf              → Virtual live ≈ [id(call), id(call)+1)
tt.func (warp-index bytes) → Scratch live ≈ entire function
```

### 3.3 Offset packing (`computeOffsets`)

After liveness, each buffer has a size and a live interval. `computeOffsets` assigns a byte `offset` so buffers that are live together do not share overlapping addresses (plus optional partition / async-WS rules). Outline:

```text
computeOffsets()
  sort buffers by size descending
  calculateStarts(buffers)              // §3.3.1 — heuristic first offsets
  loop: buildInterferenceGraph → allocate  // fix remaining conflicts
```

#### 3.3.1 `calculateStarts` — triplet slots and first offsets

Based on *Algorithms for Compile-Time Memory Optimization*. Goal: give every buffer an initial `BufferT.offset` by repeatedly claiming the lowest free **(address tip, free-time)** slot that uniquely covers that buffer’s life.

##### General idea

A **slot** (stored in `tripleMap`) is a pair:

```text
(offset, freeLiveInterval)   // “at this SMEM start address, this op-ID window is still free to try”
```

Not an exact byte-range allocator. It only suggests **where to start**; the buffer then occupies `[alignOffset, alignOffset + size)`.

Key operations:

| Concept | Role |
|---------|------|
| **triplet / slot** | `(offset → Interval)` in `tripleMap`; lowest `offset` is tried first |
| **intersect** | buffer is a candidate only if `live(buffer)` intersects the popped slot’s free interval; exclusive if no *other* open slot also intersects that live |
| **`setOffsetAligned`** | `buffer.offset = alignTo(slot.offset, buffer.alignment)` — this is the real placement |
| **free-slot update** | after place: insert **above** (higher address) + optional **before/after** sides (same address, leftover time hints) |

`tripleMap` is temporary scratch for this function only. Final offsets live on `BufferT`; leftover slots are discarded when the loop finishes.

##### Main control flow

```text
tripleMap ← {(0, [-∞,+∞))}
xBuffers  ← buffers   // size-sorted, still unplaced
placed    ← {}

while xBuffers not empty:
  ① Pop lowest-offset slot (offset, range); erase it from tripleMap

  ② Build livenessEligible from xBuffers:
       live intersects range
       AND no remaining tripleMap slot’s interval intersects live
       // → this slot is the buffer’s unique remaining free-time cover

  ③ Among livenessEligible, first partition-eligible at alignTo(offset, align)
       → chosen  (else if only partition-blocked: requeue range at higher offset;
                  if eligible empty: drop slot)

  ④ If chosen:
       alignOffset = setOffsetAligned(offset)
       insert free slots (above + maybe before/after sides)
       erase chosen from xBuffers
```

**Slot search (②–③):** buffers that fail exclusivity stay in `xBuffers` and wait. Non-exclusive slots are often dropped until only one covering slot remains; then someone becomes eligible.

**New slot update (④)** after placing `chosen` with live `xRange` into old slot `(offset, range)`:

```text
// Above — next address tip, time = range ∩ xRange
tripleMap.insert({alignOffset + size,
                  [max(range.start, xRange.start), min(range.end, xRange.end))});

// Before side — same offset, coarse leftover early time  (if range.start < xRange.start)
tripleMap.insert({offset, [range.start, xRange.end)});

// After side  — same offset, coarse leftover late time   (if xRange.end < range.end)
tripleMap.insert({offset, [xRange.start, range.end)});
```

- **Above:** space split — “try starting just past this buffer during the accounted life.”  
- **Before / after:** time split at the **same** `offset` — heuristic free-time hints, **not** exact free intervals (see walkthrough).

##### Walkthrough 1 — happy path A / B / C (no leftover conflict)

Toy buffers (align = 1, no hardware partitions):

| Buf | size | live | Meaning |
|-----|------|------|---------|
| **A** | 100 | `[0, 6)` | long-lived tile |
| **C** | 80 | `[4, 7)` | late tile |
| **B** | 40 | `[1, 3)` | short early scratch |

Sorted: A, C, B. Start: `tripleMap = {(0, [-∞,+∞))}`.

**Place A** — pop `(0, all)` → `A.offset = 0`, occupies `[0,100)` while live `[0,6)`.

```text
tripleMap:
  (0,   [-∞, 6))     // before side  — COARSE (overlaps A's live at addr 0)
  (0,   [0,  +∞))    // after side   — COARSE
  (100, [0,  6))     // above A
```

Exact free time at tip 0 would be only `[-∞,0)` and `[6,+∞)`. The inserted `[-∞,6)` / `[0,+∞)` are **over-approx** hints (in-tree comment: either `[rs,xs)` or `[rs,xe)` is OK; coloring fixes bad places).

**C/B never claim those coarse slots:** while `(100,[0,6))` still intersects their lives, exclusivity fails → offset-0 sides are **dropped**. Then `(100,[0,6))` → place **C** at 100 → then **B** at 100.

| Buf | offset | address | live |
|-----|--------|---------|------|
| A | 0 | `[0, 100)` | `[0, 6)` |
| B | 100 | `[100, 140)` | `[1, 3)` |
| C | 100 | `[100, 180)` | `[4, 7)` |

No live+address overlap → phase-2 interference empty.

##### Walkthrough 2 — concrete conflict after `calculateStarts` succeeds

Here every buffer gets an offset, but **A and C still conflict**. A short seed **S** is needed to shape slots (pure 3-buffer random search rarely hits this; coarse sides + `setOffsetAligned` do).

| Buf | size | align | live | Role |
|-----|------|-------|------|------|
| **S** | 100 | 1 | `[15, 16)` | short late seed at addr 0 |
| **B** | 80 | 1 | `[5, 9)` | places on **coarse before** of S at addr 0 |
| **A** | 64 | 64 | `[3, 14)` | `alignTo(80,64)=128` |
| **C** | 96 | 128 | `[12, 27)` | `alignTo(100,128)=128` — clashes with A |

Size order: S, C, B, A. Start: `{(0, [-∞,+∞))}`.

**① Place S** at 0 (`setOffsetAligned(0)`):

```text
S.offset = 0    // occupies [0, 100) during [15, 16)
insert above  (100, [15, 16))
insert before (0,   [-∞, 16))   // COARSE — includes times when S is live
insert after  (0,   [15, +∞))
```

**② Place B on the coarse before** — pop `(0, [-∞,16))`.  
B’s live `[5,9)` does **not** intersect `(100,[15,16))` or `(0,[15,+∞))`, so B is exclusive → `B.offset = 0`.

```text
B.offset = 0    // occupies [0, 80) during [5, 9)
// no live overlap with S → OK at same tip
insert above (80, [5, 9))
…
```

This is the coarse-before path for real: B reused tip 0 in a **disjoint** time window (good). The same mechanism can still leave **unsafe** tips for later buffers.

**③ Drop** remaining non-exclusive offset-0 slots until `(80,[5,9))` and `(100,[15,16))` remain.

**④ Place A** — pop `(80, [5,9))`. A is exclusive →

```text
alignedOffset = alignTo(80, 64) = 128
A.offset = 128    // occupies [128, 192) during [3, 14)
```

**⑤ Place C** — pop `(100, [15,16))`. C is exclusive →

```text
alignedOffset = alignTo(100, 128) = 128
C.offset = 128    // occupies [128, 224) during [12, 27)
```

**Phase-1 result — all placed, but illegal:**

| Buf | offset | address | live |
|-----|--------|---------|------|
| S | 0 | `[0, 100)` | `[15, 16)` |
| B | 0 | `[0, 80)` | `[5, 9)` |
| **A** | **128** | **`[128, 192)`** | **`[3, 14)`** |
| **C** | **128** | **`[128, 224)`** | **`[12, 27)`** |

```text
A ∩ C live  = [12, 14)   ≠ ∅
A ∩ C addr  = [128, 192) ≠ ∅
→ CONFLICT  (calculateStarts still finished)
```

Why: both claimed different free **tips** (80 vs 100), then `setOffsetAligned` rounded both up to **128**, so their byte ranges overlap while still live together. Slot liveness only required `intersects(slot.range)`, not “size fits without hitting another buffer.”

Phase 2 must fix this: `buildInterferenceGraph` adds edge A—C; `allocate` bumps one of them (non-color-0) past `128+size`.

##### Another hole `calculateStarts` cannot see — WS async regions

`calculateStarts` only reasons about **sequential** live intervals (`bufferRange` over op-IDs). That misses warp specialization:

```text
ttg.warp_specialize          // OpTrait::AsyncRegions
├─ default {  buffer X  }    // may run at the same time as
└─ partition0 { buffer Y }   // partition warps
```

X and Y can have **non-overlapping** live intervals (different regions → different op-IDs) yet still be in use concurrently. If phase 1 gave them overlapping addresses, that is also illegal.

`calculateStarts` does **not** model `AsyncRegions`. Phase 2’s `buildInterferenceGraph` adds a second edge rule: same parent WS, **different** child regions, address ranges overlap → interfere (no live-interval check). `allocate` then bumps one side, same as for ordinary live+addr conflicts.

```text
takeaway:
  setOffsetAligned(slot.offset)  → real placement (can merge different tips!)
  above slot                     → next address tip (range ∩ xRange)
  before/after sides             → same offset, coarse time hints
  exclusivity                    → often drops bad sides; not enough vs alignment
  interference / allocate        → fix live+addr overlaps AND WS async-region addr overlaps
```

#### 3.3.2 `buildInterferenceGraph` — three conflict cases

After `calculateStarts`, every buffer has a candidate `offset`. `buildInterferenceGraph` marks pairs that still cannot safely share SMEM. Result type:

```text
GraphT = DenseMap<BufferT*, DenseSet<BufferT*>>
// adjacency: interference[x] contains every y that conflicts with x
```

##### The three cases

| # | Conflict | Condition | Why |
|---|----------|-----------|-----|
| **1** | Live + address | `live(x) ∩ live(y) ≠ ∅` **and** `addr(x) ∩ addr(y) ≠ ∅` | Classic: same time, same bytes |
| **2** | WS async regions | same parent `AsyncRegions` (`warp_specialize`), **different** child regions, **and** `addr(x) ∩ addr(y) ≠ ∅` (**no** live check) | Default / partition warps run concurrently; sequential liveness underestimates overlap |
| **3** | Partitioned-encoding banks | `partitionSize > 0`, `y ∈ x->neighbors`, and physical bank index ranges overlap | `#partitioned_shared` pieces must sit in **different** hardware SMEM banks |

**Case 3 bank size:** `partitionSize` is the physical bank stride in bytes. On AMD GFX1250-style targets this is typically **65536 (64 KB)** (`test-print-allocation="partition-size=65536"`, `TargetInfo::getSharedMemoryPartitionSize()`). NVIDIA’s allocate-shared-memory path usually passes `0` → case 3 off. Bank index:

```text
bankLo = offset / partitionSize
bankHi = (offset + size - 1) / partitionSize
// overlap if bankLo_x ≤ bankHi_y && bankLo_y ≤ bankHi_x
```

##### How edges are stored — is interference recorded twice?

Yes, for the pairwise loops. The graph is **directed-as-stored / undirected-in-meaning**:

```text
for x in buffers:
  for y in buffers:          // y ≠ x
    if case1 or case2:
      interference[x].insert(y)   // when (x,y) is visited
      // later iteration also does interference[y].insert(x)
  for neighbor in x->neighbors:   // case 3
      interference[x].insert(neighbor)
      // when neighbor is the outer x, reverse edge is inserted
```

So a conflict between A and C appears as **both** `interference[A]∋C` and `interference[C]∋A`. `DenseSet` makes repeated inserts for the same pair idempotent if both case 1 and case 2 fire. There is no separate undirected edge list — consumers of `GraphT` (e.g. `allocate`) look up `interference.lookup(x)` as the neighbor set of `x`.

```text
interference:
  A → {C, …}
  C → {A, …}    // same conflict, stored both ways
```

#### 3.3.3 `allocate` — resolve interference by coloring + bump

##### Primitive idea

- Split buffers into **color groups**. Same color ⇒ no interference edge inside the group (an independent set).
- **Color 0** = **anchor** group: keep the offset from `calculateStarts` (“stay put”).
- **Color ≥ 1** = must **bump** to a new offset past interferers (or to the next hardware bank for partition neighbors).
- The first buffer in the size-sorted list is seeded as color **0**.
- End goal of the outer `computeOffsets` loop: **interference graph empty** (conflict-free layout). That is *not* the same as an explicit final pass that puts everyone into one color-0 group — when there are no edges, the loop simply stops.

##### Outer control flow (multi-round)

```text
calculateStarts(...)                    // candidate offsets
do:
  buildInterferenceGraph → GraphT
  allocate(buffers, GraphT)             // color ALL buffers, bump ALL non-0
while GraphT is not empty
```

One `allocate` does **not** fix a single edge. It recolors the whole buffer list and moves every non-anchor in one shot; then the graph is rebuilt. Bumping can create **new** edges, so rounds repeat. Offsets only increase → converges.

##### Inside one `allocate` — two stages

**Stage 1 — assign colors (first-fit)**

`available[c]` means “color ID `c` is still allowed **for this buffer**” (not “SMEM free”).

For each buffer `x` in size order:

1. Reset `available[*] = true`.
2. For each interferer `y` already colored (`colors[y] >= 0`), set `available[colors[y]] = false`.
3. `colors[x] =` index of the first `true` in `available`  
   (`std::find` + `std::distance` = that index; “first-fit” = smallest free color ID).

Uncolored neighbors (`-1`) do not block anything yet — **order matters**.

Nuance vs “interfere with anchor ⇒ always color 1”:

- You only cannot reuse colors of **already-colored** neighbors.
- If those neighbors only used 0 → you get **1** (typical).
- If they also used 1 → you get **2**, etc.
- If you interfere only with a color-**1** neighbor (not with any color-0), color **0 is still free** → you get **0** and stay put; the color-1 side bumps.

Several buffers may share color 0 if they do not conflict with each other.

**Stage 2 — bump non-color-0**

For each buffer `x`:

```text
newOffset = max over interferers y of:
  partition-neighbor && partitionSize > 0  → next bank past y   // e.g. 64KB
  else                                     → y.offset + y.size

if colors[x] != 0:
  x.offset = alignTo(newOffset)    // bump
// else: stay put

sharedMemorySize = max(offset + size)
```

##### Mini example

```text
edges: A↔C;  B isolated
order: A, C, B
offsets in: A@0, C@0, B@100
```

| Buffer | Already-colored interferers | Color | Action |
|--------|-----------------------------|-------|--------|
| A | — | **0** | stay @ 0 |
| C | A has 0 → `available[0]=false` | **1** | bump to `0+100=100` |
| B | none | **0** | stay @ 100 |

If that creates C↔B, next round rebuilds the graph and `allocate` runs again (often B becomes ≠0 and moves past C).

```text
takeaway:
  color groups = interference-free sets
  color 0 = anchors (keep phase-1 offset)
  color ≥1 = bump past interferers
  one allocate = color all + bump all non-0; then rebuild until no edges
  first-fit color = smallest ID not used by already-colored neighbors
```

---

## 4. Quick reference

```text
ModuleAllocation (call-graph post-order)
  └─ Allocation per func
       Explicit  ← local_alloc          → valueBuffer[Value]
       Scratch   ← cvt / WS / func / …  → opScratch[Op]
       Virtual   ← call                 → opVirtual[Op]  (size = callee total)
       Alias     ← memdesc views        → aliasBuffer[Value]
       → liveness → offsets → sharedMemorySize
→ attach attrs (ttg.shared, allocation.offset)
```

**Mental checklist**

1. Bottom-up on **functions** so calls see finished callee sizes.  
2. **Virtual ≠ Scratch** — only calls use Virtual today.  
3. WS captures and function warp-index bytes are **Scratch**.  
4. `valueBuffer` keys are **Values** (alloc results); aliases point views at those buffers.  
5. Final IR numbers are annotations; the analysis object holds the full maps during the pass.
