# Shared-memory proxies, memory effects, and fences

| | |
|--|--|
| **Topic** | Generic vs async SMEM proxies, op memory effects, `fence_async_shared` |
| **Passes** | `FenceInsertion.cpp`, `ProxyFenceInsertion.cpp`, NVWS `LowerAref.cpp` |
| **Example IR** | `.../irs/13_after_ttnvgpuir_add_fence_insertion.ttgir___p_matmul_...` |
| **Related** | [Canonicalizer](../canonicalizer/Canonicalizer.md) |

---

## 1. General idea

### Problem

On Hopper+ (SM90+), the same shared-memory (SMEM) address can be touched through **different hardware paths** (“proxies”). Program order in one warp does **not** automatically order:

- a **generic** SMEM access (`ld.shared` / `st.shared` from `local_load` / `local_store` / `local_alloc` with `src`), vs
- an **async** SMEM access (TMA, WGMMA reading memdesc, etc.).

### Key hinge

Cross-proxy ordering needs an explicit **`ttng.fence_async_shared`** (`fence.proxy.async.shared::cta` at LLVM), which drains / orders **prior generic** traffic **with respect to** subsequent **async** traffic on SMEM.

Without it: async can start while generic access is still outstanding → **race** (stale/torn data), not a clean deadlock.

### Motivation in Triton

Pipelined / warp-specialized kernels mix:

- **TMA** producers/consumers (async),
- **generic** loads/stores into the same double-buffered tiles,
- **dots** that may read SMEM via async (WGMMA) or via generic (`local_load` → `tt.dot`).

Compiler passes must insert fences at proxy boundaries. Triton tracks this in two layers:

1. **Declared memory effects** on ops (`MemoryEffectOpInterface` + `SharedKind`).
2. **Fence insertion passes** that consume those effects (or NVWS async-kind metadata) to place `fence_async_shared`.

### Why program order is not enough (hardware root cause)

This is **not** “different warp schedulers.” It is **different memory-access proxies** — separate hardware paths into the same physical SMEM:

| Proxy | Issuer | Hardware path | Triton ops |
|-------|--------|---------------|------------|
| **Generic** | CUDA threads | Normal `ld.shared` / `st.shared` | `local_load`, `local_store`, `local_alloc` (+ `src`) |
| **Async** | Async/tensor units | TMA engine, WGMMA, `cp.async.bulk` | `async_tma_copy_*`, `warp_group_dot` on memdesc |

On Hopper (SM90+), the CUDA memory model treats these as **separate proxy domains**. Two ops that touch the **same SMEM address** but use **different proxies** are **not “morally strong”** w.r.t. each other — source-level program order on one path does **not** guarantee visibility on the other. Generic and async traffic can be **buffered in different pipelines** and overlap for performance.

`fence.proxy.async.shared::cta` is the explicit **bridge**: it makes prior **generic** SMEM effects ordered **with respect to** subsequent **async** SMEM ops (for that thread / CTA). It is **not** a completion wait — it only fixes cross-proxy ordering.

### Cross-proxy asymmetry (important)

Ordering is **not symmetric** between the two directions:

| Direction | Fence usually required? | Why |
|-----------|---------------------------|-----|
| **Generic → Async** | **Yes** — explicit `fence_async_shared` | Async units can start before generic ops are visible on the async path. Compiler must insert fence. |
| **Async → Generic** | **Often no** — completion waits suffice | `wait_barrier`, `async_tma_store_wait`, TMA `wait_group` carry **acquire** semantics: after the wait, generic `local_load` can safely read SMEM filled by async. |

**Evidence from Triton compiler (all Generic → Async only):**

1. **`ProxyFenceInsertion.cpp`** — tracks a `generic` frontier vs `async` frontier; inserts a fence only when `state.generic.hasHazard(effects.async)` — i.e. generic traffic must be ordered before a **later async** op, never the reverse.

2. **`LowerAref.cpp`** — `rewriteGetExitOp`: fence when consumer is **generic** (`local_load`) and producer is **TMA**. `rewritePutExitOp`: fence when producer is **generic** and consumer is **MMAv5**. No symmetric rule for async consumer after generic producer via wait alone.

3. **`TMAStoresPipeline.cpp`** — hard-coded sequence: `local_store` → `fence_async_shared` → `async_tma_copy_local_to_global`. No fence after TMA before a later generic read.

4. **`FenceInsertion.cpp`** — inserts fence before dot only when dot **async-reads** memdesc after **generic write** upstream.

**Evidence from matmul IR dump 13:**

*Async → Generic (no fence):*

```mlir
ttng.wait_barrier %x_43, %x_42 : ...          // async producer completed (mbarrier acquire)
%x_45 = ttg.local_load %x_44 : ... -> ...     // generic read — NO fence between wait and load
ttng.fence_async_shared {bCluster = false}    // fence comes AFTER load, before next async handoff
ttng.arrive_barrier %x_46, 1 : ...
```

TMA producer partition (lines ~302–306): `wait_barrier` → `async_tma_copy_global_to_local` — also **no** fence before issuing async after generic emptied the slot via `arrive_barrier` + fence on the consumer side.

*Generic → Async (fence required):*

```mlir
ttg.local_load %w_48 : ...                    // generic read (consumer done with tile)
ttng.fence_async_shared {bCluster = false}    // REQUIRED before telling TMA it may refill
ttng.arrive_barrier %w_50, 1 : ...            // async producer may overwrite buffer
```

```mlir
ttg.local_store %out_85, %11 : ...            // generic write
ttng.fence_async_shared {bCluster = false}    // REQUIRED before TMA reads SMEM
ttng.async_tma_copy_local_to_global %Y_6[...] %11 : ...
```

**External / memory-model evidence:** NVIDIA’s Hopper programming notes and CUDA dev discussions state that transitioning **from generic to async** on the same SMEM location requires an explicit `fence.proxy.async`, while transitioning **from async to generic** is often covered by completion waits (`mbarrier.wait`, TMA wait) that implicitly provide acquire ordering. A fence is **per-thread** on the generic side — which is why Example A places fence **before** `arrive_barrier`, not after `syncthreads`-style coordination alone.

**Common mistake:** `wgmma.fence` orders **register** traffic for WGMMA; it does **not** replace `fence.proxy.async` for a generic SMEM write followed by `wgmma.mma_async` read.

---

## 2. Memory effects

### What is a memory effect?

In MLIR, a **memory effect** is a declared **side effect** of an operation on a **memory resource**. It answers:

> “Does this op read from, write to, allocate, or free some memory — and *which* memory?”

Effects are metadata attached to ops. Compiler passes query them instead of pattern-matching op names. For proxy/fence work, the important question is:

> “Does this op touch SMEM through the **generic** proxy or the **async** proxy?”

That distinction is encoded in Triton’s effect model.

### MLIR building blocks

Every effect is a triple:

```text
(effect type, resource, affected SSA value)
```

**Effect types** (from `MemoryEffects` in MLIR):

| Effect | Meaning |
|--------|---------|
| `Read` | Op may read current contents |
| `Write` | Op may modify contents |
| `Allocate` / `Free` | Op creates / releases a buffer |

**Resource** — which memory domain the effect applies to. Triton defines `SharedMemory` for SMEM (`TritonGPUMemoryEffects.td`).

Ops expose effects through **`MemoryEffectsOpInterface::getEffects`**. Passes call that interface (directly or via helpers) to collect what an op does.

### Triton’s extension: proxy kind on SMEM

Plain “read/write shared memory” is not enough on Hopper+: the **same SMEM address** can be accessed through different hardware paths. Triton therefore tags SMEM effects with a **`SharedKind`** (proxy):

| Kind | Value | Proxy | Typical ops |
|------|-------|-------|-------------|
| **Generic** | 0 | Normal thread SMEM (`ld.shared` / `st.shared`) | `local_load`, `local_store`, `local_alloc` (+ `src`) |
| **Async** | 1 | TMA / WGMMA / async engines | `async_tma_copy_*`, `warp_group_dot` on memdesc |
| **Barrier** | 2 | Mbarrier objects in SMEM | `wait_barrier`, `arrive_barrier` |

TableGen defines separate resources per kind, e.g.:

```text
GenericSharedRead  / GenericSharedWrite   → MemRead/Write<GenericSharedMemory>
AsyncSharedRead    / AsyncSharedWrite     → MemRead/Write<AsyncSharedMemory>
BarrierSharedRead  / BarrierSharedWrite   → MemRead/Write<BarrierSharedMemory>
```

So a **`GenericSharedWrite`** on `%dst` means: “this op does a normal thread store into `%dst`” — not an async TMA write.

### How passes consume effects

`getMemoryAccesses(op, kind, rw)` in `lib/Analysis/BufferRegion.cpp` is the main query API:

1. If the op does not implement `MemoryEffectsOpInterface` → return **`[]`**.
2. Call `getEffects`, keep only `Read` / `Write` on `MemDescType` operands/results.
3. Extract `SharedKind` from Triton’s `SharedMemoryEffect`.
4. Optionally filter by kind (`Generic` / `Async`) and direction (`Read` / `Write`).

Result type — `MemoryAccess` (`include/triton/Analysis/BufferRegion.h`):

```cpp
struct MemoryAccess {
  Value value;                              // memdesc SSA value
  bool isWrite, isRead;
  std::optional<gpu::SharedKind> sharedKind;
};
```

**Example:** `FenceInsertion.cpp` calls:

```cpp
getMemoryAccesses(dot, SharedKind::Async, RW::Read)
```

to ask: “Does this dot **async-read** any memdesc?” Only then does it look for upstream **generic** writers.

### Does every op define memory effects?

**No.** Declaring effects is opt-in:

| Mechanism | Example ops | Effects |
|-----------|-------------|---------|
| **`Pure` trait / no interface** | `tt.dot`, `arith.addi` | **None** — compiler assumes no memory side effects |
| **TableGen operand traits** | `ttg.local_store` → `GenericSharedWrite` on `$dst` | Auto-generated `getEffects` |
| **Custom `getEffects`** | `ttng.warp_group_dot`, `ttg.local_alloc` | Hand-written logic |

If an op has no effects, `getMemoryAccesses` returns empty and passes that rely on effects **cannot see** its memory traffic. That is intentional for pure register ops (`tt.dot`), but it also means fence passes only fire when the relevant ops **declare** the right proxy kind.

### How to read an op’s effects (examples)

**`ttg.local_alloc %tensor`:**

- `Allocate` on result (when mutable / offset known)
- **`Generic` `Write`** on result when `src` is present (register → SMEM store)

**`ttg.local_store %t, %md`:**

- **`Generic` `Write`** on `%md` (via `GenericSharedWrite` trait on `$dst`)

**`ttg.local_load %md`:**

- **`Generic` `Read`** on `%md`

**`ttng.warp_group_dot %a, %b, %c`** (`lib/Dialect/TritonNvidiaGPU/IR/Ops.cpp`):

- **`Async` `Read`** on each memdesc operand (`%a`, `%b` if memdesc types)

**`tt.dot %a, %b, %c`:**

- **`Pure`** — register tensors only; **no** memdesc effects → invisible to `getMemoryAccesses`

**Takeaway for fences:** a sequence like `local_store` → `async_tma_copy_local_to_global` is a **Generic write** followed by an **Async read** on the same buffer. Passes that understand effects (or hard-coded pipeline rules) insert `fence_async_shared` between them. An op that omits the correct effect declaration can be **missed** by effect-driven passes — which is why `FenceInsertion` targets `warp_group_dot` (declares async reads) but not `tt.dot` (pure registers).

---

## 3. `FenceInsertion` pass (`FenceInsertion.cpp`)

### Why this pass inserts fences (recap)

Hopper SMEM is accessed through **generic** (thread `ld.shared`/`st.shared`) and **async** (TMA/WGMMA) **proxies**. These are different hardware pipelines into the same bytes; the memory model does **not** order them automatically. **`FenceInsertion`** (and sibling passes) insert `ttng.fence_async_shared` at **Generic → Async** boundaries so prior generic reads/writes are visible **with respect to** the next async op. See §1 for asymmetry: **Async → Generic** usually relies on `wait_barrier` / TMA waits instead.

| | |
|--|--|
| **Pass name** | `triton-nvidia-gpu-fence-insertion` |
| **When** | `make_ttgir`, SM90+ (`computeCapability >= 90`) |
| **IR dump** | `13_after_ttnvgpuir_add_fence_insertion` (but see note below) |
| **Code** | `lib/Dialect/TritonNvidiaGPU/Transforms/FenceInsertion.cpp` |

### What it does (narrow pattern)

Walk every `DotOpInterface`. For each dot:

```cpp
for (access : getMemoryAccesses(dot, SharedKind::Async, RW::Read))
  copyRegToSharedOps += findCopyRegToSharedOps(access.value);

if (!copyRegToSharedOps.empty())
  insert fence_async_shared before dot;
```

**Intent:** if the dot **async-reads** SMEM that was **generic-written** upstream (`local_alloc` with `src`, or `local_store`), insert **`fence_async_shared` immediately before the dot**.

`findCopyRegToSharedOps` walks backward on the memdesc def chain (through views, loop iter args, WS captures).

### example — fence inserted by this pass

Uses **`ttng.warp_group_dot`** (async memdesc reads), not `tt.dot`.

**Before:**

```mlir
%tile = ... : tensor<64x128xf32, #blocked>

%a_smem = ttg.local_alloc %tile
    : tensor<64x128xf32, #blocked>
   -> !ttg.memdesc<64x128xf32, #nvmma_shared, #smem>
// Generic Write (store tensor into SMEM)

%b_smem = ... : !ttg.memdesc<128x256xf32, #nvmma_shared, #smem>

%acc = ttng.warp_group_dot %a_smem, %b_smem, %c0
    : memdesc * memdesc -> tensor<64x256xf32, #mma>
// Async Read on %a_smem, %b_smem
```

**Pass analysis:**

```text
getMemoryAccesses(warp_group_dot, Async, Read)
  → [{value: %a_smem}, {value: %b_smem}]

findCopyRegToSharedOps(%a_smem)
  → [ local_alloc op ]   // generic write via src
```

**After pass:**

```mlir
%a_smem = ttg.local_alloc %tile : ...
%b_smem = ...

ttng.fence_async_shared {bCluster = false}   // inserted HERE (before dot)
%acc = ttng.warp_group_dot %a_smem, %b_smem, %c0 : ...
```

**Proxy story:** generic store into `%a_smem` must complete with respect to the async proxy before WGMMA reads `%a_smem`.

### Why your fp8 matmul is different

Kernel uses **`tt.dot`** on **register** tensors after `local_load`. `getMemoryAccesses(tt.dot, Async, Read)` → **`[]`**. **`FenceInsertion` is a no-op** on that dot.

Fences in IR dump **13** already appear in dump **12** (before `add_fence_insertion`) — they come from **earlier** NVWS lowering, not this pass.

---

## 4. Real IR examples (matmul dump 13)

File:
`workspace/test_cases/Matmul-.../irs/13_after_ttnvgpuir_add_fence_insertion.ttgir___p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1`

These illustrate cross-proxy fences. They are **not** from `FenceInsertion.cpp` (that pass is a no-op on this kernel’s `tt.dot`). They come from **earlier** pipeline lowering:

| Example | Insertion site |
|---------|----------------|
| **A** (consumer) | NVWS `LowerAref::rewriteGetExitOp` — generic consumer + TMA producer |
| **B** (epilogue) | `TMAStoresPipeline::createTMAAsyncCopy` — generic `local_store` + async TMA read |

### Example A — Consumer: generic read → fence → release buffer to TMA producer

```mlir
%w_49 = ttg.local_load %w_48
    : !ttg.memdesc<128x256xf8E5M2, #shared1, #smem, mutable>
   -> tensor<128x256xf8E5M2, #ttg.dot_op<...>>
ttng.fence_async_shared {bCluster = false}
%w_50 = ttg.memdesc_index %w_20[%x_40] : ...
ttng.arrive_barrier %w_50, 1 : !ttg.memdesc<1xi64, #shared2, #smem, mutable>
...
%acc = tt.dot %x_45, %w_49, %arg50 : ...
```

| Step | Op | Proxy | Role |
|------|-----|-------|------|
| 1 | `local_load` | **Generic read** | Consumer reads W tile from SMEM |
| 2 | `fence_async_shared` | Ordering | Drain generic read before async path sees “empty” |
| 3 | `arrive_barrier` | Sync | Tell TMA **producer** it may refill `%w` slot |

**Problem without fence:** `arrive_barrier` could signal “buffer free” while `local_load` still has outstanding generic reads → TMA overwrites tile under live consumer.

**LowerAref rule:** consumer exit is **generic** (`AsyncOp::NONE` = `local_load`) and matching producer is **TMA load** → insert fence before empty-barrier arrive.

(Same pattern for X tile at lines ~181–184.)

```text
TMA fill → wait → local_load (generic) → FENCE → arrive (empty) → TMA refill
                      ↑ consumer              ↑ unlock producer
```

### Example B — Producer epilogue: generic write → fence → TMA read

```mlir
ttng.async_tma_store_wait {pendings = 0 : i32}
ttg.local_store %out_85, %11
    : tensor<1x64x256xf8E5M2, #linear>
   -> !ttg.memdesc<1x64x256xf8E5M2, #shared, #smem, mutable>
ttng.fence_async_shared {bCluster = false}
ttng.async_tma_copy_local_to_global %Y_6[...] %11
    : !tt.tensordesc<...>, !ttg.memdesc<1x64x256xf8E5M2, #shared, #smem, mutable>
```

| Step | Op | Proxy | Role |
|------|-----|-------|------|
| 0 | `async_tma_store_wait` | Async sync | Wait for **prior** TMA stores (different issue) |
| 1 | `local_store` | **Generic write** | Threads write scaled fp8 output into `%11` |
| 2 | `fence_async_shared` | Ordering | Generic writes visible before async TMA **read** |
| 3 | `async_tma_copy_local_to_global` | **Async read** (SMEM) | TMA reads `%11`, stores to global |

**Problem without fence:** TMA can read `%11` while `local_store` still in flight → wrong global output.

**Insertion:** `TMAStoresPipeline::createTMAAsyncCopy` (`lib/Dialect/TritonGPU/Transforms/Pipeliner/TMAStoresPipeline.cpp`) emits this exact triple: `TMAStoreWait` → `local_store` → `fence_async_shared` → `async_tma_copy_local_to_global`.

```text
local_store (generic write) → FENCE → TMA local_to_global (async read from SMEM)
```

Mirror of Example A:

| Path | Generic | Async | Fence between |
|------|---------|-------|---------------|
| **Consumer (A)** | `local_load` read | TMA write (after arrive) | load → arrive |
| **Producer (B)** | `local_store` write | TMA read (local_to_global) | store → TMA |

---

## 5. Other fence passes (context)

| Pass | When | Scope |
|------|------|-------|
| **`FenceInsertion`** | `make_ttgir` (~dump 13) | Dot-centric: async memdesc read on dot + generic producer |
| **NVWS `LowerAref`** | Warp-specialize / pipeline lowering (before dump 12) | TMA ↔ generic `local_load` at get-exit barrier handoff |
| **`TMAStoresPipeline`** | Software pipeliner (before dump 12) | Descriptor store → `local_store` + fence + `async_tma_copy_local_to_global` |
| **`ProxyFenceInsertion`** | `make_llir` (after shared mem alloc) | Buffer-region analysis: any generic↔async hazard on SMEM |

For this matmul kernel, examples **A** and **B** in dump 13 come from **LowerAref** and **TMAStoresPipeline** respectively; `FenceInsertion` adds nothing here.

---

## 6. Quick reference

```text
Proxy pair needing explicit fence (Generic → Async):
  Generic SMEM op  →  fence_async_shared  →  Async SMEM op
  (same buffer, different hardware path)

Often no fence needed (Async → Generic):
  async_tma_copy / TMA issue  →  wait_barrier / TMA wait  →  local_load
  (completion wait provides acquire ordering for generic consumer)

Memory effect query:
  getMemoryAccesses(op, SharedKind::Async|Generic|nullopt, RW::Read|Write)

Fence lowers to:
  fence.proxy.async.shared::cta   (bCluster=false)
  fence.proxy.async.shared::cluster (bCluster=true)
```

**Mental checklist when reading IR:**

1. Identify **generic** vs **async** op on the **same memdesc**.
2. Determine direction: **Generic → Async** (needs fence) vs **Async → Generic** (wait often enough).
3. Ask: does program order alone order the two proxies? → **No** on SM90+ for cross-proxy.
4. Is there `fence_async_shared` on the Generic → Async edge? If not, suspect missing fence bug.
5. Check **which pass** should have inserted it (dot vs TMA pipeline vs global proxy analysis).
