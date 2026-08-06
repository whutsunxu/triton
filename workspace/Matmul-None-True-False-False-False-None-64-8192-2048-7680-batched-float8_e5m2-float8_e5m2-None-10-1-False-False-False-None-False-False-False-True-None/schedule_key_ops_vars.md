# `scheduleKeyOps` key variables

Companion to `schedule_loops_debug.log` (same directory).

## 0. Context

- **0.1** Kernel / IR: `_p_matmul_…_64x256x128x1` flattened `scf.for` (`%22`)
- **0.2** Pass: `tritongpu-schedule-loops` → `scheduleKeyOps` in `ScheduleLoops.cpp`
- **0.3** Inputs: `num_stages=3` → load latency `(3-1)/(0+1) = 2`
- **0.4** Scope: **right after `scheduleKeyOps` returns** (before later cluster splits / prologue-epilogue / deps). Final `loop.cluster` ids in the `.log` differ.
- **0.5** SSA ids refer to the for-body in `schedule_loops_debug.log`.

---

## 1. `opLatency`

Source: `AssignLatencies.cpp` → `AssignLoadLatencies` + `loadOpsToIndirectionLevel`, then consumed by `scheduleKeyOps`.

### 1.1 General idea of latency

`opLatency[op]` is **not** a hardware cycle count. It is a **pipeline stage gap**:

> “Consumers of this op should be scheduled **K stages later** than this op.”

Only pipelinable loads (and later MMA, if any) get a non-zero entry. Everything else looks up as latency **0**.

#### 1.1.1 How loads get their K (`AssignLoadLatencies`)

1. **Discover** loads on paths into `tt.dot` (A/B only). Each load gets an **indirection level**:
   - level **0** = feeds the dot path
   - level **1** = feeds that load, etc.
2. **Drop** loads with `level >= numStages - 1` from the map only (IR unchanged; those loads just are **not** software-pipelined via latency).
3. **Record** one shared gap for every **kept** load:

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

**Table: Example A — load latency key variables**

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

**Table: Example B — load latency key variables (parent load)**

| key variable | value |
|--------------|-------|
| `loadOpToIndLevel` | `%43→(0,dot)`, `%L1→(1,%43)` |
| `maxIndirectionLevel` | **1** |
| `loadLatency` | `(3-1)/(1+1) =` **1** |
| `opLatency` | `%L1→1`, `%43→1` (**same K**) |

Budget split across **two** load hops; both loads still get the same `opLatency`.

#### 1.2.3 Comparison

**Table: Example A vs B — loadLatency comparison**

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

**Table: This IR — `opLatency` entries**

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

`distance[op]` = longest remaining stage-gap from `op` toward the yield.

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

**Table: A chain — `distance` from `%43`**

| op | SSA | lat | users (in-block) | `distance` |
|----|-----|-----|------------------|------------|
| `arith.select` | `%50` | 0 | only `scf.yield` → none | **0** |
| epilogue `scf.if` | `%51` | 0 | only `scf.yield` → none | **0** |
| `tt.dot` | `%47` | 0 | `%50`, `%51` (via `%80` inside if) | **0** |
| `ttg.convert_layout` | `%45` | 0 | `%47` | **0** |
| `tt.descriptor_load` | `%43` | 2 | `%45` | **2** |

#### 2.2.2 B chain (`%44`)

**Table: B chain — `distance` from `%44`**

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

#### 2.2.4 `opToStage`

`opToStage[op] = maxDistance - distance[op]`

Same op set as `distance`. In code, `opToStage` is a `MapVector` filled by `for (op, dist : distance)`, so its insertion order follows whatever order `DenseMap` yields — **not** a chosen semantic order.
Here we list ops in the **same memoize order as §2.2.3** so the two tables match.

**Table: This IR — `opToStage` from `distance`**

| op | SSA | distance | `opToStage` |
|----|-----|----------|-------------|
| `%50` select | `%50` | 0 | **2** |
| `%51` epilogue if | `%51` | 0 | **2** |
| `%47` dot | `%47` | 0 | **2** |
| `%45` convert A | `%45` | 0 | **2** |
| `%43` load A | `%43` | 2 | **0** |
| `%46` convert B | `%46` | 0 | **2** |
| `%44` load B | `%44` | 2 | **0** |

Notes:

- `%49` is an input to `%50` / `%51` but only enters `distance` if reached from a latency op’s user walk; if present, same stage-2 pattern.
- Prologue `%41` if is **not** on this forward path from the loads.

---

## 3. `schedule` / cluster (after `scheduleKeyOps`)

Source: `ScheduleLoops.cpp` → end of `scheduleKeyOps` (`CoarseSchedule` + `ClusterList`).

### 3.1 General idea of schedule and cluster

`opToStage` only gives **which pipeline stage** an op belongs to.
`schedule` also places each op in a **cluster**: a total order *within* the loop body used later when serializing / expanding the pipeline.

> **stage** = which wave of the software pipeline
> **cluster** = relative order among ops (and across stages) inside that schedule

Cluster ids are assigned with `clusters.newAtBack()` → ids `0, 1, 2, …` in list order.

#### 3.1.1 How `scheduleKeyOps` builds them

1. `maxStage = max(opToStage values)` → `CoarseSchedule(maxStage + 1)`.
2. Create one cluster per stage index: `clusters[i] = newAtBack()` for `i = 0 … maxStage`.
3. **Initial insert:** for each `(op, stage)` in `opToStage`:

```text
schedule.insert(op, stage, clusters[maxStage - stage])
```

   So higher stage → earlier cluster id (reverse-stage mapping).
4. **Epilogue move:** `epilogue = newAtBack()`, then for each scheduled `scf.if` that is not itself a latency op: if `getForwardSlice(if)` has no other `opToStage` ops, re-insert that if into `epilogue`.

#### 3.1.2 Formula notes

```text
cluster_for_stage(s) = clusters[maxStage - s]
```

With `maxStage = 2`:

**Table: Stage → cluster id mapping (`maxStage = 2`)**

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

**Table: Initial `schedule.insert` (before epilogue move)**

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

- `getForwardSlice(%51)` ≈ `{ scf.yield }` (ops that use the if’s results (and their users, …))
- yield ∉ `opToStage` → safe to move
- `epilogue = newAtBack()` → id **3**; re-insert `%51` at stage 2, cluster **3**

#### 3.2.4 Final `schedule` from `scheduleKeyOps`

**Table: Final `schedule` after `scheduleKeyOps`**

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

### 4.1 Setup

```text
numStages = schedule.getNumStages() = 3   # stages 0..2
afterPrologue = schedule.clusters.begin() # cluster id 0 (convert/dot bucket)
                                              # captured BEFORE newAtFront
```

`afterPrologue` keeps pointing at the **old** front node even if a prologue cluster is prepended later (that node becomes “right after prologue”).

### 4.2 Build `ifsToStage` (prologue candidates)

For `stage = 0 .. numStages-1`, for each scheduled op with that stage:

1. `getBackwardSlice(op)` — transitive defs of operands  
   (`omitBlockArguments=true`, root not included)
2. Any `scf.IfOp` in that slice → `ifsToStage.insert({ifOp, stage})`

**`DenseMap::insert` does not overwrite.** Outer loop visits **increasing** stages, so the **lowest** stage that discovers an `if` wins — independent of load-vs-convert order inside `getOpsInOrder`.

#### This IR

**Table: Backward-slice discovery for `ifsToStage`**

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

**Table: `schedule` after prologue `newAtFront` / insert**

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

**Table: Schedule snapshot after `schedulePrologueAndEpilogue`**

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

Later passes (`scheduleDependencies`, dist-1, remaining) further split/renumber clusters — see §5 / final `.ttir` attrs.

---

## 5. Note vs `schedule_loops_debug.log`

The `.log` / `.ttir` is **after the full schedule pass** (deps, prologue/epilogue, cluster splits). Example final attrs:

- loads `%43`/`%44`: `loop.stage = 0`, `loop.cluster = 5`
- convert/dot/select: `loop.stage = 2`, `loop.cluster = 2`
- prologue if `%41`: `loop.stage = 0`, `loop.cluster = 1`
- epilogue if `%51`: `loop.stage = 2`, `loop.cluster = 6`
- `scf.for`: `tt.scheduled_max_stage = 2`

Stages match §2.2.4 `opToStage` for key ops; cluster ids are remapped by later steps beyond §4.
