# `scheduleKeyOps` key variables

Companion to `schedule_loops_debug.log` (same directory).

## 0. Overview

### 0.1 Document context

- **Kernel / IR:** `_p_matmul_…_64x256x128x1` flattened `scf.for` (`%22`)
- **Pass:** `tritongpu-schedule-loops` (`ScheduleLoops.cpp` + `Schedule.cpp` / `AssignLatencies.cpp`)
- **Inputs:** `num_stages=3` → load latency `(3-1)/(0+1) = 2`
- **Scope:** full coarse schedule of the for-body through §§1–7 (latency → key ops → prologue/epilogue → deps → dist-1 → remaining/serialize). Final attrs match `schedule_loops_debug.ttir`.
- **SSA ids:** refer to the for-body in `schedule_loops_debug.log`.

### 0.2 Theory — general-case stage / cluster scheduling

Abstract loop:

```text
num_stages = S
scf.for i:
  load(i)
  dot(i)
  update(i) → carried to i+1
```

This is **coarse software pipelining**: split one logical iteration into **waves**, then overlap waves from different iterations in one body execution.

**Table 0.1: Stage vs cluster meanings**

| tag | meaning |
|-----|---------|
| **`stage`** | which pipeline **wave** / which logical iteration the op belongs to in the steady state |
| **`cluster`** | total **order** for serializing overlapped pieces inside the pipelined body (producer before consumer; prologue before compute before prefetch before epilogue) |

**Goal.** Without pipelining: `load(i); dot(i); update(i)` then next `i`. With `S` stages, **issue load for a future iter while computing the current one** so memory latency hides behind MMA.

### 0.2 Pass pipeline §§1–7

**Table 0.2: Overall schedule I/O**

| | |
|--|--|
| **In** | for-body IR + `num_stages` (and latency assignment from loads→dot) |
| **Out** | each scheduled op tagged with **`loop.stage`** (pipeline wave) and **`loop.cluster`** (order within / across waves) |

**Table 0.3: Seven scheduling parts**

| § | artifact / pass | role |
|---|-----------------|------|
| **1** | `opLatency` | stage **gap** K on pipelinable loads (not cycles) |
| **2** | `distance` → `opToStage` | longest path of gaps → which **wave** each key op rides |
| **3** | `schedule` / clusters (`scheduleKeyOps`) | place key ops into a stage+cluster list (compute before prefetch; epilogue if moved back) |
| **4** | `schedulePrologueAndEpilogue` | boundary `scf.if`s: prologue front / epilogue back |
| **5** | `scheduleDependencies` | same-iteration SSA producers of scheduled ops (`includeArg=false`) |
| **6** | `scheduleDistanceOneDependencies` | loop-carried producers via yield→iter_arg (`stage+1`, `newBefore`) |
| **7** | `scheduleRemainingToLastStage` + serialize | catch-all leftovers → last stage; stamp attrs (no-op top-level here) |

### 0.3 How the parts co-work

1. **Latency path first (§§1–3):** discover how far loads should lead the dot → assign waves → seed `schedule` with the load→convert→dot→yield-user skeleton.
2. **Boundaries (§4):** attach prologue/epilogue `scf.if`s around that skeleton without redoing the latency math.
3. **Fill the body (§§5–6):** pull in remaining for-body ops — first ordinary SSA deps (§5), then carried / next-iter updates that §5 deliberately skipped (§6).
4. **Close + emit (§7):** dump any leftover top-level ops onto the last stage (cluster-order fixed vs last-stage defs), then serialize → `.ttir` attrs.

Steady-state on this IR: body ≈ `loads(i+2)` ‖ `compute(i)`, with stage-1 holding dist-1 updates for the next iter.

### 0.4 Internal connections (dataflow)

```text
num_stages, loads→dot
        │
        ▼
   §1 opLatency ──► §2 distance ──► opToStage
                                        │
                                        ▼
                                   §3 schedule (key ops)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              §4 prologue/epilogue   §5 deps (SSA)    §6 dist-1 (carried)
                    │                   │                   │
                    └───────────────────┴───────────────────┘
                                        │
                                        ▼
                              §7 remaining → last stage
                                        │
                                        ▼
                         serialize: stage + cluster attrs
```

Upstream→downstream: each part **extends** the same `CoarseSchedule`; later parts never recompute `opLatency` / `opToStage`, they only insert missing ops (or relocate boundary ifs) relative to already-scheduled consumers.

---

## 1. `opLatency`

Source: `AssignLatencies.cpp` → `AssignLoadLatencies` + `loadOpsToIndirectionLevel`, then consumed by `scheduleKeyOps`.

### 1.1 General idea of latency

`opLatency[op]` is **not** a hardware cycle count. It is a **pipeline stage gap**:

> “This op should be scheduled **K stages earlier** than its consumer op. The root consumer is dot op by default.”

Only pipelinable loads (and later MMAV5, if any) get a non-zero entry. Everything else looks up as latency **0** (in fact they are not recorded in `opLatency`).

#### 1.1.1 How loads get their K (`AssignLoadLatencies`)

1. **Discover** loads on paths into `tt.dot` (A/B only). Each load gets an **indirection level**:
   - level **0** = feeds the dot path
   - level **1** = feeds that load, etc.
2. **Drop** loads with `level >= numStages - 1` from the map only (IR unchanged; those loads just are **not** software-pipelined via latency).
3. **Record** — divide the stage space evenly across these levels (one shared gap for every **kept** load):

```text
maxIndirectionLevel = max(level over kept loads)
loadLatency         = (numStages - 1) / (maxIndirectionLevel + 1)
opLatency[each kept load] = loadLatency   # same K for all kept loads
```

#### 1.1.2 Formula notes

- `numStages - 1`: one stage is reserved for the dot/consumer end; the rest is the gap budget between loads and use.
- `maxIndirectionLevel + 1`: levels are 0-based, so this is the **number** of distinct load levels to split that budget across.

Indirection level is **not** stored in `opLatency`. All kept loads share the same `loadLatency` K; later scheduling walks (see §2 `distance` / §2.2.4 `opToStage`) turn that into per-op stages.

### 1.2 Examples A / B and key variables

Assume `num_stages = 3` → stage budget `numStages - 1 = 2`.

#### 1.2.1 Example A — direct loads only (matches this matmul)

```mlir
%43 = tt.descriptor_load ...   // level 0 → %45 → %47
%44 = tt.descriptor_load ...   // level 0 → %46 → %47
%45 = ttg.convert_layout %43
%46 = ttg.convert_layout %44
%47 = tt.dot %45, %46, %arg51
```

**Table 1.1: Example A — load latency key variables**

| key variable | value |
|--------------|-------|
| `loadOpToIndLevel` | `%43→(0,dot)`, `%44→(0,dot)` |
| `maxIndirectionLevel` | **0** |
| `loadLatency` | `(3-1)/(0+1) =` **2** |
| `opLatency` | `%43→2`, `%44→2` |

One load hop uses the **full** stage budget (gap of 2).

#### 1.2.2 Example B — hypothetical parent load feeding `%43`

```mlir
%L1 = tt.descriptor_load ...          // level 1, finalUser=%43
%43 = tt.descriptor_load ... %L1 ...  // level 0, finalUser=%47
%45 = ttg.convert_layout %43
%47 = tt.dot %45, %46, %arg51
```

**Table 1.2: Example B — load latency key variables (parent load)**

| key variable | value |
|--------------|-------|
| `loadOpToIndLevel` | `%43→(0,dot)`, `%L1→(1,%43)` |
| `maxIndirectionLevel` | **1** |
| `loadLatency` | `(3-1)/(1+1) =` **1** |
| `opLatency` | `%L1→1`, `%43→1` (**same K**) |

Budget split across **two** load hops; both loads still get the same `opLatency`.

#### 1.2.3 Comparison

**Table 1.3: Example A vs B — loadLatency comparison**

| | maxIndLevel | `loadLatency` | intuition |
|--|-------------|---------------|-----------|
| A | 0 | **2** | one hop covers full depth |
| B | 1 | **1** each | depth split across two hops |

### 1.3 Real case from this IR (`schedule_loops_debug.log`)

In the flattened `scf.for`, A/B `descriptor_load`s are **direct** (indices from `%41`/`%42`, not from another load):

```mlir
%43 = tt.descriptor_load %arg13[%41#0, %41#1, %42]   // A
%44 = tt.descriptor_load %arg24[%41#0, %42, %41#2]   // B
%45 = ttg.convert_layout %43
%46 = ttg.convert_layout %44
%47 = tt.dot %45, %46, %arg51
```

So this is **example A** (§1.2.1):

**Table 1.4: This IR — `opLatency` entries**

| op | SSA | in `opLatency`? | value |
|----|-----|-----------------|-------|
| `tt.descriptor_load` A | `%43` | yes | **2** |
| `tt.descriptor_load` B | `%44` | yes | **2** |
| convert / dot / select / if | `%45`…`%51` | no | (0) |

```text
latOps = [%43, %44]
```

---

## 2. `distance`

Source: `ScheduleLoops.cpp` → `scheduleKeyOps` (`DenseMap<Operation *, int> distance` + recursive `computeDistance`).

### 2.1 General idea of distance

`distance[op]` = longest remaining stage-gap from `op` toward the yield. `opLatency` is the individual gap between `[this-op, consumer-op]`; `distance[op]` is the accumulation of the gaps between `[this op, yield-op]`. ops to collect is along the path to yield op.

```text
distance[op] = lat(op) + max(distance[users])
# lat(op) = opLatency[op] if present, else 0
# no in-block users (only yield) → add 0
```

Key points:

- Only **`opLatency` ops** (loads / MMA) inject a non-zero `lat`; they are the **seeds** (`latOps`).
- Downstream ops (convert, dot, …) are **not** in `opLatency` (`lat = 0`) but **are** recorded in `distance` when reached from a seed (usually `0`).
- Ops never reached from any `latOp` are simply **absent** from the map.
- Local `maxDist = -1` only means “no users yet”; after the formula, stored distances are always `>= 0`.

Build: for each `latOp`, walk users (`findAncestorOpInBlock`, skip terminator), memoize into `distance`, then `maxDistance = max(distance[latOps])`.

### 2.2 Real case from this IR (`schedule_loops_debug.log`)

```mlir
%43 = tt.descriptor_load ...          // opLatency=2
%44 = tt.descriptor_load ...          // opLatency=2
%45 = ttg.convert_layout %43          // lat=0
%46 = ttg.convert_layout %44          // lat=0
%47 = tt.dot %45, %46, %arg51         // lat=0
%50 = arith.select ..., %47           // lat=0; used by yield
%51 = scf.if ...                      // lat=0; uses %47 inside; results to yield
```

Chains from the two `descriptor_load`s:

#### 2.2.1 A chain (`%43`)

`computeDistance(%43)` walks **users first** (recurse), then writes `distance` on the way back:

```text
computeDistance(%43)                         # load, lat=2
  └─ user %45 → computeDistance(%45)         # convert, lat=0
       └─ user %47 → computeDistance(%47)    # dot, lat=0
            ├─ user %50 → computeDistance(%50)
            │    └─ user scf.yield → skip
            │    maxDist=-1 → distance[%50]=0+0=0
            └─ user %80 (inside if) → ancestor %51
                 → computeDistance(%51)
                 └─ user scf.yield → skip
                 maxDist=-1 → distance[%51]=0+0=0
            maxDist=max(0,0)=0 → distance[%47]=0+0=0
       maxDist=0 → distance[%45]=0+0=0
  maxDist=0 → distance[%43]=2+0=2
```

Call order (into recursion): `%43 → %45 → %47 → %50`, then `%51`.
Write order (memoize on return): `%50`, `%51`, `%47`, `%45`, `%43`.

**Table 2.1: A chain — `distance` from `%43`**

| op | SSA | lat | users (in-block) | `distance` |
|----|-----|-----|------------------|------------|
| `arith.select` | `%50` | 0 | only `scf.yield` → none | **0** |
| epilogue `scf.if` | `%51` | 0 | only `scf.yield` → none | **0** |
| `tt.dot` | `%47` | 0 | `%50`, `%51` (via `%80` inside if) | **0** |
| `ttg.convert_layout` | `%45` | 0 | `%47` | **0** |
| `tt.descriptor_load` | `%43` | 2 | `%45` | **2** |

#### 2.2.2 B chain (`%44`)

**Table 2.2: B chain — `distance` from `%44`**

| op | SSA | lat | users (in-block) | `distance` |
|----|-----|-----|------------------|------------|
| `tt.dot` | `%47` | 0 | (shared) | **0** |
| `ttg.convert_layout` | `%46` | 0 | `%47` | **0** |
| `tt.descriptor_load` | `%44` | 2 | `%46` | **2** |

#### 2.2.3 Result

Presentation order below = **memoize / first-insert order** from `computeDistance` (users written before producers).
In code, `distance` is a `DenseMap` (hash iteration, not IR order); do not read that as semantic order.

`latOps = [%43, %44]` → walk `%43` first, then `%44`:

```text
# from computeDistance(%43): write %50, %51, %47, %45, %43
# from computeDistance(%44): write %46, %44   (%47 already memoized)
distance = {
  %50: 0,   # select
  %51: 0,   # epilogue if
  %47: 0,   # dot
  %45: 0,   # convert A
  %43: 2,   # load A
  %46: 0,   # convert B
  %44: 2,   # load B
}
maxDistance = 2
```

#### 2.2.4 `opToStage` — pipeline waves

##### 2.2.4.1 General idea

Software pipelining does **not** run the whole for-body of iteration `i` as one atomic chunk. It:

1. **Splits one loop iteration into waves** (stages `0 … maxDistance`).
2. **Assigns each op a wave** via `opToStage` so different ops of the *same* iteration ride different waves.
3. **Overlaps waves from consecutive iterations** in the steady-state body (motivation): while iteration `i` is in a late wave, iteration `i+k` can already be in an early wave.

On this IR the key path only uses waves **0** and **2** (`maxDistance = 2`; ignore empty wave **1** for now — dist-1 ops fill it later in §6).

##### 2.2.4.2 Formula

```text
opToStage[op] = maxDistance - distance[op]
# here maxDistance = 2
# high distance (loads) → early wave (0)
# low  distance (dot / consumers) → late wave (2)
```

Same op set as `distance`. In code, `opToStage` is a `MapVector` filled by `for (op, dist : distance)` (hash order). Below we keep the **§2.2.3 memoize order**.

**Table 2.3: This IR — `opToStage` from `distance`**

| op | SSA | distance | `opToStage` (wave) |
|----|-----|----------|---------------------|
| `%50` select | `%50` | 0 | **2** |
| `%51` epilogue if | `%51` | 0 | **2** |
| `%47` dot | `%47` | 0 | **2** |
| `%45` convert A | `%45` | 0 | **2** |
| `%43` load A | `%43` | 2 | **0** |
| `%46` convert B | `%46` | 0 | **2** |
| `%44` load B | `%44` | 2 | **0** |

Without pipelining, iterations are serial: all of `i`, then all of `i+1`, …

With waves, the steady-state body overlaps an early wave of a **newer** iter with a late wave of an **older** iter:

```text
                    wave 0              wave 2
one steady body  ≈  loads(i+2)   +   convert/dot/epilogue(i)
```

**Table 2.4: Overlap of waves across iterations (ignore stage 1)**

| stage in body | iteration being served | ops (examples) |
|---------------|------------------------|----------------|
| **0** | **i+2** (prefetch) | `%43`, `%44` |
| **2** | **i** (compute) | `%45`–`%47`, `%50`, `%51` |

So when the body runs the **dot for iteration i** (wave 2), it is also issuing **loads for iteration i+2** (wave 0). That is the point of `opToStage`: place producers early enough in the pipeline that later iterations’ prefetches hide behind earlier iterations’ compute.

Notes:

- `%49` only enters if reached from a latency op’s user walk; if present, same wave-2 pattern.
- Prologue `%41` is **not** on this load→dot path; it gets a stage later (§4).
- Wave **1** stays unused on this key path until loop-carried ops (§6).
- Later passes may remap **cluster** ids; the stage-0 / stage-2 wave split for this key path stays.

---

## 3. `schedule` / cluster (after `scheduleKeyOps`)

Source: `ScheduleLoops.cpp` → end of `scheduleKeyOps` (`CoarseSchedule` + `ClusterList`).

### 3.1 General idea of schedule and cluster

Given `opToStage`, ops from one logical iteration are placed on different waves, and those waves overlap across iterations (§2.2.4). In one **steady-state body** execution we therefore see work for **different loop indices** at once (e.g. compute for `i` with loads for `i+2`).

Cluster order still has to respect producer→consumer constraints when serializing. So `schedule` also places each op in a **cluster** (a directed list): a total order *within* the loop body used later when expanding the pipeline.

> **stage** = which wave of the software pipeline

> **cluster** = relative order among ops (and across stages) inside that schedule

#### 3.1.1 How `scheduleKeyOps` builds them

1. `maxStage = max(opToStage values)` → `CoarseSchedule(maxStage + 1)`.
2. Create one cluster per stage index: `clusters[i] = newAtBack()` for `i = 0 … maxStage`.
3. **Initial insert:** for each `(op, stage)` in `opToStage`:

```text
schedule.insert(op, stage, clusters[maxStage - stage])
```

   Higher stage → **earlier** cluster id. In the overlapped view (§2.2.4): stage **2** (cluster 0) serves iter **`i`**; stage **0** (cluster 2) serves iter **`i+2`**. Within one body, cluster order runs compute-for-`i` before loads-for-`i+2`.

4. **Epilogue move:** `epilogue = newAtBack()`, then for each `scf.IfOp` in `opToStage` that is not a latency op: if `getForwardSlice(if)` has **no** other `opToStage` ops, re-insert that if into `epilogue`.

#### 3.1.2 Formula notes

```text
cluster_for_stage(s) = clusters[maxStage - s]
```

With `maxStage = 2`:

**Table 3.1: Stage → cluster id mapping (`maxStage = 2`)**

| stage | `maxStage - stage` | cluster id |
|-------|--------------------|------------|
| 2 | 0 | **0** |
| 1 | 1 | **1** (unused if no stage-1 ops) |
| 0 | 2 | **2** |
| (epilogue if) | — | **3** (`newAtBack` after the three) |

### 3.2 Real case from this IR (`schedule_loops_debug.log`)

From §2.2.4: `maxStage = 2`. Ops in memoize order:

```text
opToStage: %50→2, %51→2, %47→2, %45→2, %43→0, %46→2, %44→0
```

#### 3.2.1 Cluster creation

```text
clusters[0] = newAtBack()  → id 0   # for stage 2
clusters[1] = newAtBack()  → id 1   # for stage 1 (empty here)
clusters[2] = newAtBack()  → id 2   # for stage 0
```

#### 3.2.2 Initial insert

`schedule.insert(op, stage, clusters[maxStage - stage])`

**Table 3.2: Initial `schedule.insert` (before epilogue move)**

| op | SSA | stage | cluster |
|----|-----|-------|---------|
| `%50` select | `%50` | 2 | **0** |
| `%51` epilogue if | `%51` | 2 | **0** (temporary) |
| `%47` dot | `%47` | 2 | **0** |
| `%45` convert A | `%45` | 2 | **0** |
| `%43` load A | `%43` | 0 | **2** |
| `%46` convert B | `%46` | 2 | **0** |
| `%44` load B | `%44` | 0 | **2** |

#### 3.2.3 Epilogue move

For each `scf.IfOp` in `opToStage`: skip if it is in `opLatency`; skip if `getForwardSlice(if)` contains any other `opToStage` op (would place the if after a dependent already in the schedule). Otherwise re-insert into `epilogue = newAtBack()`.

On this IR only **`%51`** qualifies:

- `getForwardSlice(%51)` ≈ `{ scf.yield }` (users of the if’s results)
- yield ∉ `opToStage` → safe to move
- `epilogue = newAtBack()` → id **3**; re-insert `%51` at stage 2, cluster **3**

#### 3.2.4 Final `schedule` from `scheduleKeyOps`

**Table 3.3: Final `schedule` after `scheduleKeyOps`**

| op | SSA | stage | cluster |
|----|-----|-------|---------|
| `%50` select | `%50` | 2 | **0** |
| `%47` dot | `%47` | 2 | **0** |
| `%45` convert A | `%45` | 2 | **0** |
| `%46` convert B | `%46` | 2 | **0** |
| `%43` load A | `%43` | 0 | **2** |
| `%44` load B | `%44` | 0 | **2** |
| `%51` epilogue if | `%51` | 2 | **3** |

```text
cluster list order: 0 → 1 → 2 → 3   # id 1 unused
```

---

## 4. `schedulePrologueAndEpilogue` (after `scheduleKeyOps`)

Code: `ScheduleLoops.cpp` — push `scf.if`s to loop boundaries. Starts from the §3.2.4 schedule.

### 4.1 General idea

`opToStage` / `scheduleKeyOps` only cover ops on the latency path (load → … → yield users). Other for-body ops are still missing; among them, boundary `scf.if`s are handled here as prologue / epilogue.

- **Prologue:** `getBackwardSlice` from already-scheduled ops; any `scf.IfOp` found → `ifsToStage` (lowest discovering stage wins). Inserted on a **new front** cluster; stage matches the discovering consumer (here `%41` @ stage 0 with the loads).
- **Epilogue:** body `scf.IfOp`s **not** in `ifsToStage`. Code does `insertIfAbsent(..., numStages-1, newAtBack())`. If the if was **already** scheduled (like `%51` from §3.2.3), `insertIfAbsent` does **not** move it; only truly new ifs land on that back cluster.


### 4.2 Build `ifsToStage` (prologue candidates)

For `stage = 0 .. numStages-1`, for each scheduled op with that stage:

1. `getBackwardSlice(op)` — transitive defs of operands
   (`omitBlockArguments=true`, root not included)
2. Any `scf.IfOp` in that slice → `ifsToStage.insert({ifOp, stage})`

**`DenseMap::insert` does not overwrite.** Outer loop visits **increasing** stages, so the **lowest** stage that discovers an `if` wins — independent of load-vs-convert order inside `getOpsInOrder`.

#### 4.2.1 This IR

**Table 4.1: Backward-slice discovery for `ifsToStage`**

| Scheduled op | stage | `scf.if` in backward slice? |
|--------------|-------|-----------------------------|
| `%43`/`%44` loads | 0 | **`%41`** (tile-index if) |
| `%45`/`%46`/`%47`/`%50` | 2 | **`%41`** again (through the loads) |
| `%51` epilogue if | 2 | — (not a *producer* of these ops) |

```text
# after stage 0 (loads):
ifsToStage = { %41 → 0 }

# stage 2 also sees %41 → insert ignored → still stage 0
ifsToStage = { %41 → 0 }   # final
```

`%51` stays **out** of `ifsToStage` (epilogue path below).

### 4.3 Prologue cluster + schedule insert

```cpp
prologueCluster = schedule.clusters.newAtFront();
// push_front(-1); then ++ all ids → new front id = 0
schedule.insertIfAbsent(%41, stage=0, prologueCluster);
```

Cluster list after `newAtFront` (old ids shifted +1):

```text
before:  0 → 1 → 2 → 3
after:   0' → 1 → 2 → 3 → 4
         ↑ new prologue (%41)
         (old 0/1/2/3 became 1/2/3/4)
```

**Table 4.2: `schedule` after prologue `newAtFront` / insert**

| op | stage | cluster (after prologue insert) |
|----|-------|----------------------------------|
| **`%41` tile if** | **0** | **0** (new front) ← **added** |
| `%50`/`%47`/`%45`/`%46` | 2 | 1 (was 0) |
| `%43`/`%44` loads | 0 | 3 (was 2) |
| `%51` epilogue if | 2 | 4 (was 3) |
| *(unused)* | — | 2 (was 1) |

`insertIfAbsent`: `%41` was not in the schedule yet → inserted.
`afterPrologue` still refers to the **old** begin node → now cluster id **1** (convert/dot).

### 4.4 Other `IfOp`s → epilogue cluster

```cpp
epilogueCluster = schedule.clusters.newAtBack();  // new id 5
// for each scf.if in the body not in ifsToStage:
schedule.insertIfAbsent(ifOp, numStages-1, epilogueCluster);
```

- `%41` ∈ `ifsToStage` → skipped
- `%51` ∉ `ifsToStage` → `insertIfAbsent(%51, stage=2, epilogueCluster)`

But `%51` is **already** scheduled (cluster 4 from `scheduleKeyOps`). `insertIfAbsent` does **not** move it → stays stage 2 / cluster 4. The new back cluster (id 5) may be empty here.

(If `%51` had never been scheduled, it would land on this new epilogue cluster at stage `numStages-1`.)

### 4.5 Return + schedule snapshot after this function

```text
return afterPrologue;  # cluster id 1 — first non-prologue cluster
```

**`ifsToStage`:** `{ %41 → 0 }`

**`schedule` (conceptually):**

**Table 4.3: Schedule snapshot after `schedulePrologueAndEpilogue`**

| op | SSA | stage | cluster id | cluster role |
|----|-----|-------|------------|----------------|
| `%41` tile-index if | `%41` | 0 | **0** | **prologue** (new front) |
| `%50` select | `%50` | 2 | **1** | after prologue (old 0) |
| `%47` dot | `%47` | 2 | **1** | same |
| `%45`/`%46` converts | … | 2 | **1** | same |
| `%43`/`%44` loads | … | 0 | **3** | load cluster (was 2) |
| `%51` epilogue if | `%51` | 2 | **4** | epilogue (was 3; not moved again) |

```text
cluster list order: 0 (prologue) → 1 (compute) → 2 (unused) → 3 (loads) → 4 (epilogue) → 5 (new empty back)
```

Within each iteration, lower cluster id runs first among scheduled ops.

Later passes (`scheduleDependencies` §5, dist-1 §6, remaining) further split/renumber clusters — see §7 / final `.ttir` attrs.

---

## 5. `scheduleDependencies`

Code: `ScheduleLoops.cpp` calls `tt::scheduleDependencies` → for each scheduled op, `CoarseSchedule::insertDepsOfOp` in `Schedule.cpp`.

### 5.1 General idea

`scheduleDependencies` finds **unscheduled ops** recursively inside the `scf.for` body, with the start points of already-scheduled ops in `schedule` and via `getNestedOperands` connection.

Then it inserts them into the schedule at the **consumer’s stage/cluster**, preferring an **earlier** placement if the producer was already scheduled later (`insertMinimum`). It ignores block arguments / loop-carried edges via `includeArg=false`.

### 5.2 How `scheduleDependencies` drives `insertDepsOfOp`

```text
opsInOrder = schedule.getOpsInOrder(forOp)   # snapshot once (cluster id, then IR order)
for stage in 0 .. numStages-1:               # numStages = 3
  for (op, stage_, cluster) in opsInOrder:
    if stage_ != stage: continue
    insertDepsOfOp(op, stage, cluster,
                   includeArg=false, insertIfEarlier=true)
```

`insertIfEarlier=true` → `tryInsert` = `insertMinimum` (may move an already-scheduled op to an earlier stage/cluster; never to a later stage).

#### 5.2.1 Inside `insertDepsOfOp` (filter pipeline)

For each `operand` in `getNestedOperands(op)`:

1. **Select (nested operands):** operands of `op` and of nested ops, but only values **defined outside** `op` (`Utility.cpp` / `isProperAncestor`).
2. **BlockArg chase:** `while (dyn_cast<BlockArgument>(v))` — with `includeArg=false` → **break immediately** (no yield back-edge chase). `v` stays a block arg → no defining op.
3. **Same-block filter:** `defOp = v.getDefiningOp()` and `defOp->getBlock() == op->getBlock()` (must be the `scf.for` body). Outer constants / func args dropped.
4. **Insert:** `insertMinimum(defOp, stage, cluster)`; on success, **recurse** `insertDepsOfOp(defOp, …)`.

### 5.3 Starting schedule & visit order

**Table 5.1: Input schedule (§4.5) + `getOpsInOrder` visit order**

| visit order (among matching stage) | op | SSA | stage | cluster |
|------------------------------------|----|-----|-------|---------|
| stage 0 | tile if | `%41` | 0 | 0 |
| stage 0 | load A | `%43` | 0 | 3 |
| stage 0 | load B | `%44` | 0 | 3 |
| stage 2 | convert A | `%45` | 2 | 1 |
| stage 2 | convert B | `%46` | 2 | 1 |
| stage 2 | dot | `%47` | 2 | 1 |
| stage 2 | select | `%50` | 2 | 1 |
| stage 2 | epilogue if | `%51` | 2 | 4 |

(`getOpsInOrder`: cluster 0 → 1 → 3 → 4; within a cluster, for-body IR order. Stage-1 empty.)

### 5.4 Per-op call path (this IR)

Filter each root: nested operands → skip block args / outer defs → `insertMinimum` same-block defs → recurse.

```text
%41 (0,0)
  nested: %39, iter_args, outer consts/%5/%6/%10…
  same-block survivor: %39
  → insert %39 @(0,0); recurse → done
  IR: %39 = cmpi %arg46, %c0

%43/%44 (0,3)
  nested: %41#*, %42, descriptors…
  → %41 already (0,0): insertMinimum(0,3) not earlier → keep
  → insert %42 @(0,3)
       %42 = muli %40, %c128
       → insert %40 @(0,3)
            %40 = select %39, %c0, %arg50
            → %39 stays (0,0); %arg50 skipped (includeArg=false)

%45/%46 (2,1): operands %43/%44 already stage 0 → refuse later stage
%47 (2,1): %45/%46 already; %arg51 skipped
%50 (2,1):
  %50 = select %49, %cst_4, %47
  → insert %49 @(2,1); recurse → %arg46/%20 skip
%51 (2,4): %49/%41/%47 already placed earlier or same → no new inserts
```

**Inserted this pass:** `%39`, `%40`, `%42`, `%49`.

### 5.5 Schedule snapshot after `scheduleDependencies`

**Table 5.2: Schedule after deps (`insertDepsOfOp`)**

| op | SSA | stage | cluster id | note |
|----|-----|-------|------------|------|
| `%41` tile if | `%41` | 0 | **0** | prologue (unchanged) |
| `%39` cmpi (tile cond) | `%39` | 0 | **0** | **added** via `%41` |
| `%50` select | `%50` | 2 | **1** | |
| `%47` dot | `%47` | 2 | **1** | |
| `%45`/`%46` converts | … | 2 | **1** | |
| `%49` cmpi (epilogue cond) | `%49` | 2 | **1** | **added** via `%50` |
| `%43`/`%44` loads | … | 0 | **3** | |
| `%40` select (k idx) | `%40` | 0 | **3** | **added** via `%42` ← loads |
| `%42` muli (k offset) | `%42` | 0 | **3** | **added** via loads |
| `%51` epilogue if | `%51` | 2 | **4** | unchanged |

```text
cluster list (still): 0 (prologue+%39) → 1 (compute+%49) → 2 (unused) → 3 (loads+%40+%42) → 4 (epilogue) → 5 (empty)
```

**Summary of inserts this pass:** `%39`, `%40`, `%42`, `%49`.
Still **not** scheduled: loop-carried producers such as `%48`, `%52`/`%53`/`%54` — see §6. Final `.ttir` ids also reflect §6–§7.

---

## 6. `scheduleDistanceOneDependencies` (after `scheduleDependencies`)

Code: `ScheduleLoops.cpp` — schedule **loop-carried** (distance-1) producers of already-scheduled ops.

Starts from the §5.5 schedule snapshot.

### 6.1 General idea

In §5 `scheduleDependencies`, `includeArg=false` means for-body **iter_args are not followed**. So any ops that only feed the next iteration through `scf.yield` → iter_arg (and are not otherwise same-block SSA deps of the scheduled set) stay **out** of the schedule — e.g. `%48`, `%52`/`%53`/`%54` after §5.5.

`scheduleDistanceOneDependencies` fills that gap: it schedules those **loop-carried** producers.

**How it finds them**

1. Start from ops **already in the schedule** with `stage < numStages - 1` (here `numStages = 3` → only **stage 0**; need room for `stage + 1`).
2. Look at `getNestedOperands(op)` for a for-body **`BlockArgument`** (`argNumber > 0`, owner = for body).
3. Map that arg through **`scf.yield`**: `yield[argNumber - 1]` → its defining op `defOp`. If `defOp` is still unscheduled, it is the **previous-iteration / distance-1** producer of this consumer.

```text
this iteration:  consumer(stage S)         uses %argK

previous yield:  … defOp …  ──carried──►   %argK
```

**How it places them**

- **Cluster `newBefore(consumer)`:** within an iteration’s cluster order, the producer still sits *before* the consumer (producer→consumer). Does not reuse an existing earlier cluster; inserts a new one immediately before.
- **Stage `stage + 1`:** stages encode *which pipeline wave / iteration index* an op is tied to in the steady state—not only intra-iteration order.

  The consumer at stage `S` is using the value **already carried in** from the last iteration. The `defOp` we just found is the update computed **this** iteration for the **next** iteration’s consumer. In a `numStages`-deep software pipeline, that “produce for next iter” work is attributed to the **next** stage index (`S+1`), while stage `S` keeps using the older carried value. Same stage would wrongly treat the update as same-wave as the use of the previous value; `stage+1` is what marks **dependence distance 1** in this coarse schedule.

Exception: if `defOp` is a `tt.LoadOp`, keep it at the consumer’s **same** stage/cluster (special case in the code).

### 6.2 How the walk works

```text
for op in forBody (IR order), if already scheduled:
  if stage == numStages - 1: continue          # cannot go to stage+1
  for operand in getNestedOperands(op):
    if operand is for-body BlockArgument (argNumber > 0):
      defOp = defining op of yield[argNumber - 1]
      if defOp unscheduled:
        if LoadOp:  insert @ (stage, same cluster) + insertDepsOfOp(..., includeArg=true)
        else:       newBefore(consumer cluster) once per consumer cluster (cached)
                    insert @ (stage+1, that new cluster) + insertDepsOfOp(..., includeArg=true)
```

**`newBefore(C)`:** inserts a **new** list node immediately before `C` (does **not** reuse an existing predecessor such as the empty unused cluster). Ids from `C` onward bump by 1; the new node keeps the old id of `C`.

**Two inserts are not double-scheduling `defOp`:** `insertIfAbsent(defOp)` then `insertDepsOfOp(defOp)` schedules **deps of** `defOp` (with `includeArg=true` this time).

### 6.3 Which scheduled ops can fire? (this IR)

`numStages = 3` → only **stage 0** consumers are eligible (`stage < 2`).

**Table 6.1: Stage-0 consumers after §5.5**

| op | stage | cluster (pre–dist-1) | for-body iter_args in nested operands? |
|----|-------|----------------------|----------------------------------------|
| `%39` cmpi | 0 | 0 | **`%arg46`** |
| `%41` tile if | 0 | 0 | `%arg47`,`%arg52`–`%arg54` (yield defs = `%41` itself → already scheduled) |
| `%40` select | 0 | 3 | **`%arg50`** |
| `%42` muli | 0 | 3 | — |
| `%43`/`%44` loads | 0 | 3 | — (`%41#*` are if results, not block args) |

Stage-2 ops (`%45`–`%51`, `%49`) are skipped by the last-stage guard.

Yield map (for reference):

```text
scf.yield %54, %41#3, %51#0, %51#1, %48, %50, %41#0, %41#1, %41#2
        → %arg46 %arg47 %arg48 %arg49 %arg50 %arg51 %arg52 %arg53 %arg54
```

### 6.4 Per-consumer call path (IR walk order)

#### 6.4.1 `%39` @ (0, cluster 0) — first hit

| step | detail |
|------|--------|
| select | nested operand `%arg46` (for-body iter_arg) |
| historic producer | yield[0] = **`%54`** (`arith.select`), unscheduled, not a LoadOp |
| `newBefore(0)` | list `0→1→2→3→4→5` → `0(NEW)→1(old prologue)→2→3→4→5→6` |
| insert | `%54` @ **stage 1**, cluster **0** (NEW) |
| `insertDepsOfOp(%54)` | `%54 = select %53, %c0, %52` → insert **`%53`**, **`%52`** @ (1, same NEW cluster); their `%arg46` chase hits `%54` already scheduled |

**Newly scheduled:** `%52`, `%53`, `%54` @ stage 1, cluster before prologue.

#### 6.4.2 `%40` @ (0, loads cluster — id now 4 after the bump above)

| step | detail |
|------|--------|
| select | nested operand `%arg50` |
| historic producer | yield[4] = **`%48`** (`arith.addi`), unscheduled, not a LoadOp |
| `newBefore(loads)` | …→`3(unused)`→`4(NEW for %48)`→`5(loads/%40)`→… |
| insert | `%48` @ **stage 1**, cluster **4** (NEW) |
| `insertDepsOfOp(%48)` | operands `%40` (already @ stage 0), `%c1` (outside) → **no new ops** |

**Newly scheduled:** `%48` @ stage 1, cluster immediately before the load cluster.

#### 6.4.3 `%41` / `%42` / loads

- `%41`’s iter_args yield **`%41` / already-scheduled** results → `schedule.count(defOp) != 0` → skip
- `%42` / `%43` / `%44`: no eligible unscheduled carried `defOp`

### 6.5 Schedule snapshot after `scheduleDistanceOneDependencies`

**Table 6.2: Schedule after dist-1**

| op | SSA | stage | cluster id (after both `newBefore`s) | note |
|----|-----|-------|--------------------------------------|------|
| `%52`/`%53`/`%54` | … | **1** | **0** | **added** via `%39` ← `%arg46` |
| `%41` tile if | `%41` | 0 | **1** | was 0 |
| `%39` cmpi | `%39` | 0 | **1** | was 0 |
| `%50`/`%47`/`%45`/`%46`/`%49` | … | 2 | **2** | was 1 |
| *(unused)* | — | — | **3** | was 2 |
| `%48` addi (next k) | `%48` | **1** | **4** | **added** via `%40` ← `%arg50` |
| `%43`/`%44`/`%40`/`%42` | … | 0 | **5** | was 3 → 4 → **5** |
| `%51` epilogue if | `%51` | 2 | **6** | was 4 → … → **6** |
| *(empty back)* | — | — | **7** | |

```text
cluster list:
  0 (counter update %52–%54, stage 1)
  → 1 (prologue %39/%41, stage 0)
  → 2 (compute, stage 2)
  → 3 (unused)
  → 4 (%48 next-k, stage 1)
  → 5 (loads + %40/%42, stage 0)
  → 6 (epilogue %51, stage 2)
  → 7 (empty)
```

**Summary of inserts this pass:** `%48`, `%52`, `%53`, `%54`.
These cluster ids already match the final `.ttir` for these ops; §7 is the catch-all / serialize step (no-op on this IR’s top-level body).

---

## 7. `scheduleRemainingToLastStage` → serialize

Code: `ScheduleLoops.cpp` — after §6, dump any still-unscheduled **top-level** for-body ops onto the **last stage**, then `schedule.serialize(forOp)` writes `loop.stage` / `loop.cluster` attrs (what `schedule_loops_debug.ttir` shows).

Starts from the §6.5 schedule snapshot. `afterPrologue` is the cluster handle returned by §4 (the first cluster **after** the prologue node — on this IR the compute cluster; ids may have shifted after §6’s `newBefore`, but the handle still names that node).

### 7.1 General idea

§§1–6 schedule the latency path, boundary ifs, same-iter SSA deps, and loop-carried producers. Anything left at top level of the for-body is **not** on those paths.

`scheduleRemainingToLastStage` is the safety net:

1. Collect unscheduled top-level ops (`forOp.getBody()->without_terminator()`, so **not** ops nested inside `scf.if` regions — those ride with their parent if).
2. Put each at **`stage = numStages - 1`** (here **2**), default cluster **`afterPrologue`**.
3. Fix cluster order vs already-scheduled **last-stage** producers: a remaining use must not sit in an **earlier** cluster than its last-stage def (`*userCluster < *opCluster` → bump user to the producer’s cluster, BFS).
4. `insert` them; then **serialize** stamps attrs onto the IR.

Nested region ops (e.g. store/reduce inside `%51`) never appear in this map; the scheduled unit is the enclosing `scf.if`.

### 7.2 How the function walks

```text
# 1) seed remaining ops
opToCluster = {}
for op in forBody.without_terminator():
  if schedule.count(op) == 0:
    opToCluster[op] = afterPrologue          # default cluster

# 2) BFS from last-stage scheduled ops
queue = [op for (op, stage, _) in getOpsInOrder if stage == numStages-1]
while queue:
  op = queue.pop()
  for user in op.users:
    if user in opToCluster:                  # only remaining ops
      opCluster  = schedule[op].cluster if scheduled else opToCluster[op]
      userCluster = opToCluster[user]
      if *userCluster < *opCluster:          # user would run before def
        opToCluster[user] = opCluster
        queue.push(user)

# 3) commit
for (op, cluster) in opToCluster:
  schedule.insert(op, numStages-1, cluster)
```

Only last-stage producers are seeds: earlier-stage defs already run in earlier waves/clusters relative to stage `numStages-1`, so they do not constrain this catch-all’s cluster bumps.

### 7.3 This IR — remaining set is empty

After §6.5 every **top-level** for-body op is already in `schedule`:

```text
%39–%54  (cmpi / select / if / muli / loads / converts / dot / addi / …)
```

So `opToCluster` is **{}**, the BFS never inserts, and this pass is a **no-op**. Cluster list stays as in §6.5.

(If there were orphan arith/control ops only used by yield or by other leftovers, they would land at stage **2**, cluster **afterPrologue** (= compute cluster here), possibly bumped later if a stage-2 def forced them.)

### 7.4 Serialize → `schedule_loops_debug.ttir`

`schedule.serialize(forOp)` writes the coarse schedule onto ops; `scf.for` also gets `tt.scheduled_max_stage = 2`.

**Table 7.1: Final attrs (matches §6.5; §7 added nothing)**

| op | SSA | `loop.stage` | `loop.cluster` |
|----|-----|--------------|----------------|
| counter update | `%52`/`%53`/`%54` | 1 | **0** |
| prologue | `%39`, `%41` | 0 | **1** |
| compute | `%45`/`%46`/`%47`/`%49`/`%50` | 2 | **2** |
| *(unused cluster)* | — | — | **3** |
| next-k | `%48` | 1 | **4** |
| loads + k idx | `%40`/`%42`/`%43`/`%44` | 0 | **5** |
| epilogue if | `%51` | 2 | **6** |
| `scf.for` | `%22` | — | `tt.scheduled_max_stage = 2` |

```text
steady body (ignore unused 3):
  cluster0 stage1:  %52–%54     (produce next counter)
  cluster1 stage0:  %39/%41     (prologue tile idx)
  cluster2 stage2:  convert/dot/select   (compute iter i)
  cluster4 stage1:  %48         (produce next k)
  cluster5 stage0:  loads/%40/%42        (prefetch iter i+2)
  cluster6 stage2:  %51         (epilogue)
```

That is the complete coarse schedule scheme for this `scf.for` body.
