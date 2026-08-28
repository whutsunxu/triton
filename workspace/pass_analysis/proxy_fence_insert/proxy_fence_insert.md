# Proxy fence insertion (`ProxyFenceInsertion`)

| | |
|--|--|
| **Pass name** | `triton-nvidia-gpu-proxy-fence-insertion` (`add_proxy_fence_insertion`) |
| **Code** | `lib/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp` |
| **Analysis** | `BufferRegionAnalysis` (`include/triton/Analysis/BufferRegion.h`), `ScopedMemoryFrontier` (`include/triton/Analysis/MemoryFrontier.h`) |
| **When** | `make_llir`, SM90+, after `AllocateSharedMemoryNv` (needs `allocation.offset`) |
| **Tests** | `test/TritonGPU/proxy_fence_insertion.mlir` |
| **Example IR** | `.../irs/17_after_ttnvgpuir_add_proxy_fence_insertion...` |
| **Related** | [proxy memory effects](../proxy_memory_effect/proxy_memeffect.md) |

On Hopper+, generic SMEM (`ld.shared` / `st.shared`) and async SMEM (TMA, MMAv5, CLC, `tmem_copy`) use **different hardware proxies**. Program order does not order them. This pass inserts `ttng.fence_async_shared` on the **Generic → Async** edge when the two accesses **physically overlap**.

```text
module
  ├─ 1. Fence scope     which async ops need CTA vs cluster fences
  ├─ 2. BufferRegion    exact physical bytes of every memdesc view
  └─ 3. Fence insert    walk CFG, detect generic/async overlap, insert fence
```

---

## 1. Fence scope — CTA vs cluster

A proxy fence is not one instruction. It lowers to either:

| IR | PTX | What it orders |
|----|-----|----------------|
| `ttng.fence_async_shared {bCluster = false}` | `fence.proxy.async.shared::cta` | Prior generic SMEM vs later async, **this CTA only** |
| `ttng.fence_async_shared {bCluster = true}` | `fence.proxy.async.shared::cluster` | Same, across **all CTAs in the cluster** |

Cluster **includes** CTA: a cluster fence also kills the CTA generic frontier. A CTA fence does **not** cover a later cluster-scope async.

### How an async op gets a scope

```cpp
// ProxyFenceInsertion.cpp
bool cluster =
    (TMA load/gather with {multicast}) ||
    (MMAv5 with {two_ctas}) ||
    (tmem_copy in a two-CTA module) ||
    (clc_try_cancel with num-ctas > 1);
return cluster ? kClusterScope : kCTAScope;
```

| Async op | Scope | Fence |
|----------|--------|--------|
| Ordinary TMA, single-CTA `clc_try_cancel` | CTA | `{bCluster = false}` |
| Multicast TMA / TMA gather, two-CTA MMA, two-CTA `tmem_copy`, multi-CTA `clc_try_cancel` | cluster | `{bCluster = true}` |

Generic accesses are tagged with **every scope the module uses**. A `local_load` can therefore conflict with both CTA and cluster async.

### Module-level scope mask

Before analysis, the pass walks the module once:

```text
if any async shared access exists:
    scopes |= CTA
    if any of those async ops is cluster-scoped:
        scopes |= CLUSTER
if scopes == 0:  return   // nothing to fence (or CC < 90)
```

Only those bits are considered later. A 1-CTA kernel never tries to insert a cluster fence.

### Why the two scopes are not interchangeable

CTA-local generic traffic (this CTA’s `ld.shared`) is visible to a non-multicast TMA on the same CTA after a **CTA** fence. A multicast TMA / two-CTA MMA can consume **another CTA’s** SMEM, so a CTA fence is not enough — a later cluster async still needs `{bCluster = true}`.

Lit tests encode that:

- `clc_try_cancel_multi_cta_after_cta_fence`: existing CTA fence is kept, a **cluster** fence is added before `clc_try_cancel`.
- `multicast_tma_after_cluster_barrier_and_cta_fence`: CTA fence does not cover `{multicast}` TMA → extra cluster fence.
- `non_multicast_tma_after_cluster_barrier_and_cta_fence`: same CTA fence **is** enough; no cluster fence.

---

## 2. Buffer region analysis — continuous vs strided views

Fence insertion does **not** key off SSA buffer identity (`%a` vs `%b`). It asks BufferRegionAnalysis: **which physical SMEM bytes (per CTA) does this memdesc touch?**

That analysis must already have run (`allocation.offset` on `local_alloc`). It is a sparse forward dataflow over memdesc values.

### What a region is

```text
BufferRegionView
  allocationFrame     which function’s SMEM arena
  region.baseOffset   runtime key (NOT used for aliasing)
  region.length       runtime key (NOT used for aliasing)
  region.ctaAddresses [{ctaId, AddressSet}, ...]
```

`AddressSet` is a **sparse bitvector of exact bytes** (TMEM uses packed row/col words). Two views overlap only if:

1. they share the same `allocationFrame`, **and**
2. some **same CTA id** has intersecting byte sets.

Same numeric offset in two different CTAs does **not** alias (`same_bytes_in_different_ctas_do_not_alias`). Unknown views (`std::nullopt`) alias **everything**.

`baseOffset` / `length` are a descriptor key only. Distinct sparse views can share that key and still occupy different bytes — geometry lives in `ctaAddresses`.

### How views are built

| IR | What the lattice does |
|----|------------------------|
| `ttg.local_alloc {allocation.offset = N}` | Root view: storage base `N`, footprint of the full type |
| `ttg.memdesc_index %parent[%i]` | Advance storage by a **page** (`getMemDescStorageOffset`) |
| `ttg.memdesc_subslice %parent [o0, o1, …]` | Map logical offsets through the **inverse linear layout** → byte / partition / CTA offsets |
| `ttg.memdesc_trans` / `reshape` | Same physical view |
| `ttg.memdesc_reinterpret` | Recompute footprint for the new type |
| `arith.select` | Join both views (may-alias) |
| function args / unknown | pessimistic **unknown** (aliases all) |

Scratch (`allocation.size` / `allocation.offset` on `convert_layout`, reduce, atomics) is modeled as a generic R+W of `[offset, offset+size)`. Cross-CTA scratch (`hasCrossCTAScratch`) is replicated into every CTA’s address set.

### Continuous view

A **continuous** (dense) view occupies a compact storage slice: a full alloc, or one pipeline stage of a `memdesc_index`.

```mlir
%parent = ttg.local_alloc {allocation.offset = 0 : i32}
    : () -> !ttg.memdesc<2x64x64xf32, #shared, #smem, mutable>
%first  = ttg.memdesc_index %parent[%c0]   // page 0: bytes [0, 16384)
%second = ttg.memdesc_index %parent[%c1]   // page 1: bytes [16384, 32768)
```

`getMemDescStorageOffset` is `index * (elems * elementBytes)` (then padding). Each stage is a dense `AddressSet::fromRange`. Generic load of `%first` and async TMA into `%second` **do not overlap** → no fence (`disjoint_pipeline_stages_do_not_require_proxy_fence`).

If the index is not a constant, every stage is joined — `%selected = select %first, %second` may alias stage 0 → fence (`selected_pipeline_stage_may_alias`).

Scratch is the same idea: a dense interval, optionally copied into every CTA.

### Strided / sparse view

A **strided** view occupies a **subset** of the parent, possibly with holes, a different CTA, or a different partition.

`memdesc_subslice` does **not** add a byte interval. It inverts the linear layout at the logical offsets and records:

- `byteOffset` — element offset in the layout’s `offset` dimension
- `partitionOffset` — which partitioned-SMEM piece
- `ctaOffset` — which CGA `block` (CTA)

`getMemDescAddresses` then walks **every tensor element**, XORs layout bases, and **sets individual bytes**. Padding (`applySharedPadding`) and swizzle insert holes; CGA `block` bases split the same logical tile across `ctaAddresses`.

```text
parent 16x32, CGALayout = [[1, 0]]   (row 8 selects the other CTA)
  %local  = subslice [0, 0]   → CTA 0 bytes [...]
  %remote = subslice [8, 0]   → CTA 1 bytes [same numeric offsets]
  local ∩ remote = empty     (different CTA ids)
```

So a generic load of `%local` plus TMA into `%remote` needs **no** fence, even though both come from one `local_alloc`.

Partitioned encodings (`allocation.offset = [0, 1024]`) are the 1-CTA analogue: a subslice can sit on a **different partition base**. Overlap is still exact bytes, not “same SSA parent.”

### Picture

```text
continuous (pipeline stage / full alloc / scratch)
  AddressSet = dense [base, base+size)
  overlap  <=>  intervals intersect (same CTA, same frame)

strided (subslice / padded / swizzled / CGA / partitioned)
  AddressSet = sparse bits of only the occupied bytes
  overlap  <=>  bitwise AND of those bits (same CTA, same frame)
               holes and other-CTA copies do not conflict
```

Both kinds go through the same intersection API. The continuous case is just the dense special case of the sparse one.

---

## 3. Fence insert — overlap → proper fence

After regions are known, the pass walks the **call graph callees-first**. Each function is a forward CFG dataflow (`ProxyFenceFunctionAnalysis`) that **inserts fences while it runs**.

### State per program point

```text
ProxyBlockInfo
  generic              generic accesses since the last fence
  async                async accesses from function entry until the first fence
  entryGenericFenced   scopes that a fence covers on every path from entry
```

Each access in a frontier is a `(BufferRegionAccess, scope mask)` plus read vs write.

### Classify each op

Skip barriers (`SharedKind::Barrier`). Everything else with a shared effect:

```text
SharedKind::Async  →  effects.async   (scope = getProxyFenceScope(op))
otherwise          →  effects.generic (scope = module mask)
```

Typical generic: `local_load`, `local_store`, `local_alloc` with `src`, `async_shared_store` (async completion, **generic** SMEM path), scratch. Typical async: TMA, MMAv5, `clc_try_cancel`, `tmem_copy`.

### Hazard: physical overlap, Generic → Async only

```cpp
if state.generic.hasHazard(effects.async, scope):
    insert fence_async_shared before the op
    clear generic for that scope
```

`hasHazard` is **WAR / WAW / RAW** on overlapping regions at the **same scope**. **RAR is not a hazard.**

```text
hasHazard(generic, async) =
    (generic write ∩ async read)   WAR
 || (generic read  ∩ async write)  RAW
 || (generic write ∩ async write)  WAW
 // NOT (generic read ∩ async read)
```

Unknown region ∩ anything = hazard.

Direction is one-way. Async then generic does **not** insert a fence here; wait/acquire (`wait_barrier`, TMA wait) covers that side.

After the check, the op’s async is joined into `state.async` only if that scope is **not** yet fenced from entry (so a **caller** can still fence before the call). Then the op’s generic is joined.

### Which fence is inserted

Try **cluster first, then CTA**. Insert at most one fence at this op:

```text
if module has cluster scope AND generic ∩ async at cluster:
    insert {bCluster = true}     // also clears CTA generic
else if module has CTA scope AND generic ∩ async at CTA:
    insert {bCluster = false}
```

Existing `fence_async_shared` ops just run `fenceGeneric` (cluster fence ⇒ CTA+cluster).

### Calls

Callee summaries are translated into the caller’s arena (`allocation.offset` on the call). If the callee’s **entry-reachable async** overlaps the caller’s outstanding generic, the fence is placed **before `tt.call`**. If the callee already fences on every path from entry, that inner fence is treated as covering the caller’s generic too.

Callee-local allocs stay in the callee frame and do not alias the caller’s buffers unless translation puts them on the same bytes.

### End-to-end examples

**Overlap, CTA fence** (`fence_write_after_write`):

```mlir
ttg.local_store %value, %buffer          // generic write, region R
ttng.fence_async_shared {bCluster = false}
ttng.async_tma_copy_global_to_local ... %buffer   // async write, same R
```

**No overlap, no fence** (`no_fence_for_disjoint_allocations`, disjoint `memdesc_index` stages): different `AddressSet`s, even if they share a parent alloc.

**RAR, no fence** (`no_fence_read_after_read`): generic load then TMA **read** of the same region.

**Strided / other-CTA, no fence**: `local_load` of `subslice [0,0]` then TMA into `subslice [8,0]` on a CGA layout — same parent, different CTA address sets.

**Cluster after CTA**: generic load, existing `{bCluster = false}`, then multicast TMA → still a cluster hazard → insert `{bCluster = true}`.

**Scratch vs remote CTA**: CTA-local `convert_layout` scratch does not alias a remote-CTA subslice; `hasCrossCTAScratch` scratch does, and gets a fence before the remote TMA.

### Matmul dump 17

Most `fence_async_shared` ops in this kernel already exist from NVWS / TMA-store lowering. This pass adds any remaining Generic → Async edge that those earlier passes missed, using exact regions after shared-memory allocation. Example: producer TMA into a `memdesc_index` stage of `%w` after generic traffic that still aliases that stage.

---

## Quick reference

```text
1. Scope
     async op → CTA or cluster
     generic  → all scopes used in the module
     cluster fence covers CTA; CTA fence does not cover cluster

2. BufferRegion
     continuous: dense [base, size)  (alloc, pipeline index, scratch)
     strided:    sparse bytes from layout / subslice / padding / CGA / partition
     overlap  =  same allocationFrame AND same CTA AND AddressSet ∩ ≠ ∅
     unknown aliases all

3. Insert
     outstanding generic ∩ incoming async  (WAR/WAW/RAW, not RAR)
     → fence before the async op (or before the call)
     cluster hazard → {bCluster = true}
     else CTA hazard → {bCluster = false}
```
