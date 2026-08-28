# Cluster / CTA barriers before LLIR

| | |
|--|--|
| **When** | `ConvertTritonGPUToLLVM::prepareModule` (`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp`) |
| **Order** | `runClusterBarrierInsertion` → `runCrossCTAMBarrierInitSyncInsertion` → `ModuleMembarAnalysis` → `runClusterBarrierMbarAllocator` |
| **Code** | `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierInsertion.cpp`, `lib/Analysis/Membar.cpp`, `lib/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.cpp`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ClusterOpsToLLVM.cpp`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ConvertWarpSpecializeToLLVM.cpp`, `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp`, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/BarrierOpToLLVM.cpp` |
| **Tests** | `test/Analysis/test-membar.mlir`, `test/TritonNvidiaGPU/membar-cluster.mlir`, `test/TritonNvidiaGPU/cluster-barrier-mbar-allocator.mlir` |

These mechanisms solve **different sync directions**. Do not mix them.

```text
Part 1  CTA-local smem             all warps, one CTA              ttg.barrier local  → bar.sync (full CTA)
Part 2  Cluster (no WS)            all threads, all CTAs           ttng.cluster_barrier  → barrier.cluster.arrive/wait
Part 3  WS partition × cluster     same WS region, across CTAs     ttng.cluster_barrier {ttg.mbar_offset}  → mbarrier.arrive.cluster + try_wait
Part 4  WS partition, same CTA     warps inside one WS region      ttg.barrier local inside warp_specialize  → named bar.sync (that group only)
Part 5  WS produce/consume slot    producer partition ↔ consumer   empty/full mbarrier; named bar.sync before arrive
```

---

## Part 1 — Normal CTA barrier

**Pass:** `ModuleMembarAnalysis` (`lib/Analysis/Membar.cpp`), called from `prepareModule` with NVIDIA `canSkipBarSync`. Lowering: `ttg.barrier` → `NVVM::BarrierOp` / `bar.sync`.


### Problem (sync direction)

Warps in one CTA issue `st.shared` / `ld.shared` **independently**. A store from warp 0 is **not** visible to a load from warp 1 (and a later store may overwrite a load still in flight) until those warps execute the same CTA `bar.sync`.

Membar does not barrier every smem op. It records each op’s **physical slice** (`allocation.offset` + size, plus memdesc subslice when known) and inserts a barrier only when a later op’s slice **intersects** an earlier one and the pair is a real hazard:

| Hazard | Earlier op | Later op | Why a barrier |
|--|--|--|--|
| **RAW** | smem write | smem read, overlapping bytes | consumer warp may load before producer warp has stored |
| **WAR** | smem read | smem write, overlapping bytes | producer warp may overwrite bytes another warp has not finished reading |
| **WAW** | smem write | smem write, overlapping bytes | two writers to the same bytes (scratch / reused alloc) |

`bar.sync` is **CTA-wide**, not a warp0–warp1 pairwise handshake. The hazard is between two **ops** whose layouts spread those bytes across many warps; every thread on that path must take the barrier so the earlier op’s smem effects complete before the later op starts.

Concrete producers/consumers this pass sees:

- **Explicit buffer:** `ttg.local_alloc %x` / `ttg.local_store` write smem; a later `ttg.local_load` of the same memdesc (or an aliasing alloc) reads it → RAW.
- **Scratch:** intra-CTA `convert_layout` / `reduce` is modeled as write scratch then read scratch. If that scratch interval intersects a live slice, barrier **before** the scratch op. (A convert that reshuffles **across CTAs** is Part 2, not this.)

```text
What must wait:  all threads of this CTA that execute the path
                 (earlier smem op finished  →  later overlapping smem op)

What must not:   CTA0’s smem  ↔  CTA1’s smem          Part 2 cluster / DSMEM
                 generic sts/lds  ↔  async TMA         proxy fence
                 partition0(CTA0)  ↔  partition0(CTA1) Part 3
```

### Solution

**TTGIR.** Walk each function; track smem read/write slices. On RAW / WAR (and some WAW) of the same bytes, insert one SSA **before** the later access:

```mlir
ttg.barrier local
```

NVIDIA `canSkipBarSync` skips pairs already ordered (TMA then its `wait_barrier`, consecutive `init`/`expect`, commutative local atomics). Existing `ttng.cluster_barrier` counts as a local sync, so membar may not add another barrier on the same path.

**PTX.** That SSA lowers to a CTA-wide (or current warp-group) hardware barrier:

```
bar.sync;
```

Every participating thread in **this CTA** must execute it. After it returns, prior smem stores in this CTA are visible to later smem loads in this CTA.

### Concrete example

**Level 1 — TTGIR** (`test/Analysis/test-membar.mlir` `@raw_single_block`):

```mlir
%buf = ttg.local_alloc %x : (tensor<128x32xf16, #AL>)
    -> !ttg.memdesc<128x32xf16, #shared, #smem>
ttg.barrier local
%y = ttg.local_load %buf : !ttg.memdesc<...> -> tensor<128x32xf16, #AL>
```

**Level 2 — PTX** of that barrier:

```
                    TTGIR (one SSA)
              ttg.barrier local

                       │  lowers to
                       ▼
              CTA 0  all warps
              bar.sync
```

```
CTA0 warp0:  st.shared buf     CTA0 warp1:  (will ld.shared buf)
                    \_______ bar.sync _______/
```

---

## Part 2 — Cross-CTA cluster barrier (no warp specialize)

**Pass:** `runClusterBarrierInsertion`. Lowering: `ClusterBarrierOpConversion` when **`ttg.mbar_offset` is absent**. If the kernel has extra WS worker warps, `lowerClusterSyncForAllWarps` wraps the PTX so **every** warp group still runs it.

Sibling (not this SSA): `runCrossCTAMBarrierInitSyncInsertion` only publishes `mbarrier.init` (`fence.mbarrier_init.release.cluster` + relaxed cluster barrier if needed).

### Problem (sync direction)

**All threads in all CTAs** of the cluster, so DSMEM stores are visible cluster-wide.

```text
Need:    every warp(CTA0)  ↔  every warp(CTA1)     full cluster

Not:     warp0(CTA0)  ↔  warp1(CTA0) only          CTA smem (Part 1)
Not:     partition0(CTA0)  ↔  partition0(CTA1)     WS subset (Part 3)
```

Inserted when an op is **distributed multi-CTA** (`isDistributedMultiCTAOp`: cross-CTA `convert_layout`, reduce on a split axis, multicast TMA, two-CTA MMA, …) and aliases prior distributed smem, and conservatively before kernel `return`.

Hardware `barrier.cluster` requires **every** thread in **every** CTA. Miss any warp → hang. Legal **outside** `ttg.warp_specialize` (or wrapped so all warp groups run it). Illegal as-is inside one WS region → Part 3.

### Solution

**TTGIR.** Insert one SSA at the hazard / kernel exit:

```mlir
ttng.cluster_barrier
```

**PTX.** That SSA (no `ttg.mbar_offset`) lowers to the hardware cluster unit:

```
barrier.cluster.arrive;    // or arrive.relaxed
barrier.cluster.wait;
```

Every thread in every CTA executes both. After wait returns, prior DSMEM stores are visible cluster-wide.

### Concrete example (`num-ctas = 2`)

Split-M → split-N convert: CTA 0’s M-half must become CTA 0’s N-half → DSMEM exchange (`test/TritonNvidiaGPU/membar-cluster.mlir` `@convert_layout_cluster_barrier`).

**Level 1 — TTGIR:**

```mlir
%cvt = ttg.convert_layout %x
  : tensor<256x128xf16, #blockedSplitM>   // CGALayout = [[1, 0]]
 -> tensor<256x128xf16, #blockedSplitN>   // CGALayout = [[0, 1]]
ttng.cluster_barrier
%buf = ttg.local_alloc %cvt : ...
```

**Level 2 — PTX** of that cluster_barrier:

```
                    TTGIR (one SSA)
              ttng.cluster_barrier

                       │  lowers to
                       ▼
         CTA 0 all threads              CTA 1 all threads
    barrier.cluster.arrive         barrier.cluster.arrive
    barrier.cluster.wait           barrier.cluster.wait
```

Two levels:

| Level | What you see | Role |
|--|--|--|
| **TTGIR** | one SSA: `ttng.cluster_barrier` (no `ttg.mbar_offset`) | program point after distributed DSMEM use |
| **PTX** | `barrier.cluster.arrive` then `barrier.cluster.wait` | that SSA expanded |

No mbarrier object. Arrive/wait **are** this cluster_barrier after conversion.

---


## Part 3 — Warp-specialized partition cluster barrier(mailbox)

**Pass:** `runClusterBarrierMbarAllocator` (allocates smem + `ttg.mbar_offset`). Lowering: `ClusterBarrierOpConversion` when `ttg.mbar_offset` is set. Init: `InitializeWSClusterBarriers`.

### Problem (sync direction)

Warp specialization **splits the CTA** into warp groups that run **different regions**:

```text
        CTA 0                         CTA 1
   default  (warp group A)       default  (warp group A)
   partition0 (warp group B)     partition0 (warp group B)
```

If only `partition0` contains `ttng.cluster_barrier`:

```mlir
ttg.warp_specialize()
default {
  // never executes cluster_barrier
  ttg.warp_yield
}
partition0() num_warps(4) {
  ttng.cluster_barrier
  ttg.warp_return
}
```

**What must be synced:** `partition0` on CTA 0 **with** `partition0` on CTA 1
(same WS region, **across** CTAs).

**What must not be required:** `partition0` with **default** on the **same** CTA. Default is a different program; it never reaches this op.

Hardware `barrier.cluster` (Part 2) waits for **every warp on every CTA**, including default. Default never arrives → **hang**.

You cannot wrap the missing warps back in: they are already in the other region. So Part 2’s instruction is illegal **inside** an existing WS region.

```text
Need:    partition0(CTA0)  ↔  partition0(CTA1)     horizontal, same role

Not:     partition0(CTA0)  ↔  default(CTA0)        vertical, same CTA
Not:     all warps × all CTAs                       hardware barrier.cluster
```

The same applies to other cluster-sync ops **inside** a WS region (`needsClusterBarrier`: cross-CTA `convert_layout`, reduce, cluster-scoped atomics). They must use this path too, not `barrier.cluster`.

### Solution

Do not use hardware `barrier.cluster` (Part 2).

**TTGIR.** Allocator gives **this WS region** an mbarrier in **each CTA’s** smem at the **same local offset**, and stamps it on the one SSA:

```mlir
ttng.cluster_barrier {ttg.mbar_offset = 40 : i32}
```

Each CTA has its own physical bytes at 40. Same offset ≠ one overlapped buffer. Each region gets its **own** offset (default vs partition0 must not share, or one group waits on the other’s arrives). Buffer is 32 bytes (2×16-byte ping-pong slots + phase counter at `+8`). Module: `ttg.ws_cluster_barrier_count`, `ttg.shared` grown.

**PTX.** That SSA lowers to two steps on warp group B (relative thread 0 ≠ CTA `threadIdx.x == 0` if default owns the first warps):

1. **Arrive** (predicated: relative thread 0 only): `mbarrier.arrive.cluster.multicast [smem+40], peerMask`
   Local pointer + `.cluster` + mask `(all CTAs) XOR self` → +1 on **peer** copies only.
2. **Wait** (all threads in this warp group): `mbarrier.try_wait.parity ... [smem+40], parity`
   Local copy; spins until `expected_arrivals = numCTAs - 1` peer arrives have hit **this** CTA’s object.

Init (kernel entry, CTA thread 0, before WS): `mbarrier.init` on each copy with that expected count. This CTA never arrives on itself.

### Concrete example (`num-ctas = 2`)

```
default:    4 warps   // warp group A
partition0: 4 warps   // warp group B
```

`M0` = CTA 0 smem[40], `M1` = CTA 1 smem[40]. Two objects.

**Level 1 — TTGIR** (same SSA on both CTAs, only in `partition0`):

```mlir
ttg.warp_specialize()
default { ttg.warp_yield }
partition0() num_warps(4) {
  ttng.cluster_barrier {ttg.mbar_offset = 40 : i32}
  ttg.warp_return
}
```

**Level 2 — PTX** of that one op (`ClusterBarrierOpConversion`):

Init (once, not this SSA):

```
mbarrier.init M0, expected_arrivals = 1;   // 2 - 1
mbarrier.init M1, expected_arrivals = 1;
```

When warp group B executes the SSA:

```
# relative thread 0 only
CTA0:  mbarrier.arrive.cluster.multicast  [smem+40], peerMask=CTA1   // +1 on M1, not M0
CTA1:  mbarrier.arrive.cluster.multicast  [smem+40], peerMask=CTA0   // +1 on M0, not M1

# all threads of warp group B (unpredicated try_wait)
CTA0:  mbarrier.try_wait.parity.acquire.cluster.shared::cta  complete, [smem+40], parity   // wait M0
CTA1:  mbarrier.try_wait.parity.acquire.cluster.shared::cta  complete, [smem+40], parity   // wait M1
```

```
                    TTGIR (one SSA)
    ttng.cluster_barrier {ttg.mbar_offset = 40}

                         │  lowers to
                         ▼
         CTA 0 warp group B              CTA 1 warp group B
    thread0: arrive → M1            thread0: arrive → M0
    all:     try_wait M0            all:     try_wait M1
```

- `M0` completes when CTA 1 has arrived → CTA 0’s `try_wait` ends.
- `M1` completes when CTA 0 has arrived → CTA 1’s `try_wait` ends.

### Three CTAs: B and C both arrive on A’s mbarrier — no overwrite

`mbarrier.arrive` is an **atomic +1** on that object’s arrival count, not a store of a value. Two peers hitting the same mailbox both count; they do not clobber each other.

For CTAs **A / B / C**, init is `expected_arrivals = numCTAs - 1 = 2`. Each copy at `smem+40` needs **two** peer arrives. Multicast mask is **all CTAs except self**, not a single `peerMask=CTA1` from the 2-CTA example:

| Who arrives | `ctaId` | `peerMask` | Objects that get +1 |
|--|--|--|--|
| A | 0 | `{B, C}` | `M_B` and `M_C` |
| B | 1 | `{A, C}` | `M_A` and `M_C` |
| C | 2 | `{A, B}` | `M_A` and `M_B` |

`M_A` is incremented by **B and C** (any order):

```
M_A arrivals:  0  →  1 (B or C)  →  2 (the other)  →  complete
A’s try_wait on M_A returns when count == 2
```

Hardware serializes those two remote atomics on the same 8 bytes so neither +1 is lost. Same for `M_B` (A and C) and `M_C` (A and B). A never arrives on `M_A`. If B and C overwrote instead of adding, the count could never reach 2 and A would hang.

Still **one** multicast arrive per CTA (relative thread 0 of that warp group), not a software loop over peers. Only the mask and the expected count change vs 2 CTAs.

Two levels:

| Level | What you see | Role |
|--|--|--|
| **TTGIR** | one SSA: `ttng.cluster_barrier {ttg.mbar_offset = 40}` | program point in `partition0` |
| **PTX** (after `ClusterBarrierOpConversion`) | `mbarrier.arrive.cluster.multicast` then `mbarrier.try_wait` | that SSA expanded |

Arrive and wait **are** the cluster_barrier after conversion.

Default never executes this SSA, so it never touches `M0`/`M1`. Horizontal sync only.

If default also needs cluster sync, allocator gives it another offset (e.g. 8): another TTGIR op with `ttg.mbar_offset = 8`, its own PTX arrive/wait pair.

IR after allocator (`test/TritonNvidiaGPU/cluster-barrier-mbar-allocator.mlir`): `ttg.shared = 5` → align 8, two 32-byte region buffers → **72**; default ops `ttg.mbar_offset = 8`, partition0 `cluster_barrier` `40`; `ttg.ws_cluster_barrier_count = 2`.

### Lowering vs Part 2

| | Part 2 (no `ttg.mbar_offset`) | Part 3 (`ttg.mbar_offset` set) |
|--|--|--|
| TTGIR | `ttng.cluster_barrier` | same SSA **plus** `ttg.mbar_offset` |
| PTX | `barrier.cluster.arrive/wait` | `mbarrier.arrive.cluster.multicast` then `try_wait.parity` |
| Who | all threads, all CTAs | one WS region, all CTAs |
| Object | none | per-CTA mbarrier at that offset |
| Arrive target | HW cluster unit | peer CTAs’ mbarriers (mask excludes self) |
| Wait | HW `barrier.cluster.wait` | local `try_wait` until `numCTAs-1` peer arrives |

---

## Part 4 — Warp-specialized partition, same-CTA barrier

**Pass:** still `ModuleMembarAnalysis` inserts the TTGIR op (Part 1). **Lowering inside WS:** `ConvertWarpSpecializeToLLVM` / `lowerKernelBarriers` rewrites `NVVM::BarrierOp` to a **named** barrier whose thread count is **only that warp group**. There is no `ttng.barrier`; the SSA is `ttg.barrier local`.

Two levels; do not mix them:

| Level | What you see | Role |
|--|--|--|
| **TTGIR** | one SSA: `ttg.barrier local` **inside** `partition0` (or `default`) | smem RAW/WAR among warps of **this** region on **this** CTA |
| **PTX** (after WS lowering) | `bar.sync <id>, <numThreads>` with `numThreads = this region’s warps × 32` | that SSA expanded; default warps are **not** in the count |

A CTA-wide `bar.sync` (barrier 0, full CTA thread count) **would hang**: default never arrives. Named-barrier rewrite is what makes Part 1 safe inside WS.

### Problem (sync direction)

Same CTA, **one** WS region: e.g. partition0 warps on CTA 1 need each other’s smem stores visible (`local_alloc` then `local_load` in that region). Default on that CTA is a **different program** and must **not** be in the rendezvous.

```text
        CTA 1
   default     (warp group A)   ← must not wait
   partition0  (warp group B)   ← these warps only
```

```text
Need:    partition0(CTA1) warps  ↔  each other     vertical, same CTA, same region

Not:     partition0(CTA1)  ↔  default(CTA1)        default never executes this op → hang if CTA-wide bar.sync
Not:     partition0(CTA0)  ↔  partition0(CTA1)     horizontal; that is Part 3
Not:     all warps × all CTAs                      Part 2 hardware cluster barrier
```

Hardware fact is the same as Part 1 (independent warps, overlapping smem). The extra constraint is: **arrival count must equal the threads that actually execute the op**.

### Solution

Do **not** use `ttng.cluster_barrier` for this (that is Part 2/3). Do **not** use a full-CTA `bar.sync`.

**TTGIR.** Same SSA as Part 1, but it lives in the partition (membar still inserts it):

```mlir
ttg.barrier local
```

**PTX.** `WarpSpecializeToLLVM` maps each WS region to a **named barrier index** and a **thread count**:

| Region | Barrier id (NVIDIA helper) | `numThreads` |
|--|--|--|
| `default` | 0 (`kDefaultWarpGroupBarrierIdx`) | `defaultNumWarps * 32` |
| `partition0` | `0 + kNumReservedBarriers` (= 2) | `partition0NumWarps * 32` |
| `partition i` | `i + kNumReservedBarriers` | that partition’s warps × 32 |
| 1-warp partition | — | `bar.warp.sync` instead of `bar.sync` |

```
bar.sync  <id>,  <numThreads>;   // only this warp group
```

Default uses a **different** `<id>` and is not counted in partition0’s `numThreads`, so it cannot hang this barrier.

Part 3’s mbarrier lowering also emits local `NVVM::BarrierOp` around arrive/`try_wait`; those become this same named barrier (intra-group, same CTA), not a CTA-wide `bar.sync`.

### Concrete example

CTA 1, `partition0` has 4 warps; default has 4 warps. Only partition0 has a smem RAW.

**Level 1 — TTGIR:**

```mlir
ttg.warp_specialize()
default {
  ttg.warp_yield
}
partition0() num_warps(4) {
  %buf = ttg.local_alloc %x : ...
  ttg.barrier local
  %y = ttg.local_load %buf : ...
  ttg.warp_return
}
```

**Level 2 — PTX** of that `ttg.barrier local`:

```
                    TTGIR (one SSA)
              ttg.barrier local   // inside partition0

                       │  lowers to
                       ▼
         CTA 1 partition0 (4 warps = 128 threads)
              bar.sync  2,  128

         CTA 1 default:  does not execute this
```

```
CTA1 partition0 warp0:  st.shared     CTA1 partition0 warp1:  ld.shared
                         \___ bar.sync 2, 128 ___/

CTA1 default:  not in count 128 → does not wait, cannot hang this barrier
CTA0 partition0:  not this CTA → not this barrier (Part 3 if you need them)
```

### Lowering vs Part 1 and Part 3

| | Part 1 (no WS) | Part 4 (inside WS region) | Part 3 |
|--|--|--|--|
| TTGIR | `ttg.barrier local` | same SSA, **in** `warp_specialize` | `ttng.cluster_barrier {ttg.mbar_offset}` |
| PTX | `bar.sync` (full CTA) | `bar.sync <id>, <this group’s threads>` | `mbarrier.arrive.cluster` then `try_wait` |
| Who | all warps of the CTA | warps of **one** WS region, **one** CTA | same WS region, **all** CTAs |
| If default does not run it | hang (full CTA count) | OK (named barrier count excludes default) | OK (default never touches that mbarrier) |

---

## Part 5 — WS producer / consumer empty–full slot

**Pass:** `NVWSLowerAref` (`LowerAref.cpp`) turns `aref_put` / `aref_get` into `wait_barrier` + `arrive_barrier` on empty/full mbarriers. **Arrive lowering** (`ArriveBarrierOpConversion`) inserts a **partition-local** `ttg.barrier local` (Part 4) **before** thread 0 `mbarrier.arrive`. That is the “all my warps finished” guarantee.

Two levels; do not mix them:

| Level | What you see | Role |
|--|--|--|
| **TTGIR** | `ttng.wait_barrier` + `ttng.arrive_barrier` on **shared** empty/full mbarriers in CTA smem | slot handshake between **producer partition** and **consumer partition** |
| **PTX** | named `bar.sync` (this partition only), then **relative thread 0** `mbarrier.arrive`; all threads `mbarrier.try_wait` | arrive is one event, not one per warp |

Tile data **and** empty/full mbarriers live in **CTA smem** (visible to 2 producer warps and 4 consumer warps). Software `phase` is **not** a shared store; each partition tracks it in registers. Completing a phase = `mbarrier.arrive` (or TMA HW), not flipping a shared parity flag.

### Problem (sync direction)

Two WS partitions share a ring of smem slots. Empty/full mbarriers are **shared**. Arrival count is **1 per partition** (or TMA HW), **not** 2 or 4 warps.

If **only thread 0** arrives on empty after consume, a slow consumer warp may still be loading while thread 0 signals “slot free.” Producers would overwrite in-flight reads. Same for produce vs full.

```text
Need:    all consumer warps done with slot s  →  then one arrive on empty[s]
         all producer warps done with slot s  →  then one arrive on full[s]

Not:     bar.sync all 6 warps (producers+consumers)     hang: other partition is not at this op
Not:     consumers wait on producers via Part 4 named barrier
         (wrong group; use wait_barrier on full instead)
Not:     barrier the other partition before “flipping parity”
```

Cross-partition “the other side is done” is **`wait_barrier`** on that shared mbarrier. Intra-partition “all **my** warps are done” is **Part 4 named `bar.sync` immediately before arrive**.

### Solution

Do **not** CTA-wide barrier producers+consumers. Do **not** skip the local barrier before arrive.

**TTGIR** (`LowerAref`):

```
producers:  wait empty[s], phase     →  write slot s     →  arrive full[s]
consumers:  wait full[s], phase      →  read  slot s     →  arrive empty[s]
```

`arrive_barrier` has **block-level** meaning. Lowering always does:

```cpp
// ArriveBarrierOpConversion
ttg::BarrierOp::create(..., AddrSpace::Local);  // all warps of THIS partition
// then only relative thread 0:
mbarrier.arrive ... [empty or full];
```

Inside WS that `ttg.barrier local` is Part 4: `bar.sync <id>, <this partition’s threads>` — 2×32 producers or 4×32 consumers, **not** the other role.

**PTX (sync produce / `AsyncOp::NONE`):**

1. **Consumers, before empty arrive:** named `bar.sync` among the **4 consumer warps** (all loads of slot `s` done) → thread 0 `mbarrier.arrive` **empty**. That completes empty’s current phase (“flip” from the producer’s `wait` point of view).
2. **Producers, before full arrive:** named `bar.sync` among the **2 producer warps** (all stores of slot `s` done) → thread 0 `mbarrier.arrive` **full**.

TMA produce: HW arrives on **full**; producer warps do not software-arrive. `barrier_expect` still named-barriers the producer partition, then thread 0 `expect_tx`. Consumers still named-barrier themselves before empty-arrive.

### Concrete example

CTA smem: slot `s`, `empty[s]`, `full[s]`. Producer partition = 2 warps, consumer = 4 warps. Init: empty count = 1, full count = 1.

**Level 1 — TTGIR** (after `LowerAref`; partitions run concurrently):

```mlir
// producer partition
ttng.wait_barrier %empty[s], %phase
// ... st.shared into slot s ...
ttng.arrive_barrier %full[s], 1

// consumer partition
ttng.wait_barrier %full[s], %phase
// ... ld.shared from slot s ...
ttng.arrive_barrier %empty[s], 1
```

**Level 2 — PTX** of each `arrive_barrier`:

```
                    TTGIR
              ttng.arrive_barrier  empty[s]   // consumer exit

                       │  lowers to
                       ▼
         4 consumer warps:  bar.sync  <consumer_id>,  128
         consumer thread0:  mbarrier.arrive  [empty[s]]

         2 producer warps:  not in count 128 → no hang
```

```
                    TTGIR
              ttng.arrive_barrier  full[s]    // producer exit

                       │  lowers to
                       ▼
         2 producer warps:  bar.sync  <producer_id>,  64
         producer thread0:  mbarrier.arrive  [full[s]]

         4 consumer warps:  not in count 64 → no hang
```

```
producers (2 warps)                            consumers (4 warps)
wait empty[s], phase                           wait full[s], phase
st.shared slot s                               ld.shared slot s
bar.sync producer-named, 64                    bar.sync consumer-named, 128
thread0 arrive full[s]                         thread0 arrive empty[s]
```

- Empty arrive happens **only after** all consumer warps finished the load.
- Full arrive happens **only after** all producer warps finished the store.
- The **other** partition learns that via `wait_barrier`, not via joining that `bar.sync`.

### Lowering vs Part 3 and Part 4

| | Part 4 | Part 5 | Part 3 |
|--|--|--|--|
| TTGIR | `ttg.barrier local` | `wait_barrier` + `arrive_barrier` on empty/full | `cluster_barrier {mbar_offset}` |
| Local named `bar.sync` | the sync itself | **before** thread-0 `mbarrier.arrive` | around WS cluster mbarrier protocol |
| Who named-barriers | warps of that region | **only the arriving role** (consumers before empty, producers before full) | that WS region |
| Who `wait`s the mbarrier | — | the **other** role | peer CTAs’ same region |
