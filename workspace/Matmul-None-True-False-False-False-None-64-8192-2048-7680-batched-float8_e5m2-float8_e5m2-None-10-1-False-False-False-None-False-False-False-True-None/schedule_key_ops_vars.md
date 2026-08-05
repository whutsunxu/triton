# `scheduleKeyOps` key variables

Companion to `schedule_loops_debug.log` (same directory).

- Kernel / IR: `_p_matmul_…_64x256x128x1` flattened `scf.for` (`%22`)
- Pass: `tritongpu-schedule-loops` → `scheduleKeyOps` in `ScheduleLoops.cpp`
- Inputs: `num_stages=3` → load latency `(3-1)/(0+1) = 2`
- Scope below: **right after `scheduleKeyOps` returns** (before later cluster splits / prologue-epilogue / deps). Final `loop.cluster` ids in the `.log` differ.

SSA ids refer to the for-body in `schedule_loops_debug.log`.

---

## `opLatency`

Only ops with assigned latency (from `AssignLatencies` / `deserializeLatencies`):

| op | SSA | latency |
|----|-----|---------|
| `tt.descriptor_load` (A) | `%43` | **2** |
| `tt.descriptor_load` (B) | `%44` | **2** |

`tt.dot` / `convert_layout` / `arith.*` / `scf.if` → **not** in `opLatency` (latency 0 when looked up).

`latOps = [%43, %44]` (body order).

---

## `distance`

`distance[op] = lat(op) + max(distance[inBlockUser])`, terminator users skipped.

### A chain (`%43`)

| op | SSA | lat | users (in-block) | `distance` |
|----|-----|-----|------------------|------------|
| `arith.select` | `%50` | 0 | only `scf.yield` → none | **0** |
| epilogue `scf.if` | `%51` | 0 | only `scf.yield` → none | **0** |
| `tt.dot` | `%47` | 0 | `%50`, `%51` (via `%80` inside if) | **0** |
| `ttg.convert_layout` | `%45` | 0 | `%47` | **0** |
| `tt.descriptor_load` | `%43` | 2 | `%45` | **2** |

### B chain (`%44`)

| op | SSA | lat | users (in-block) | `distance` |
|----|-----|-----|------------------|------------|
| `tt.dot` | `%47` | 0 | (shared) | **0** |
| `ttg.convert_layout` | `%46` | 0 | `%47` | **0** |
| `tt.descriptor_load` | `%44` | 2 | `%46` | **2** |

`maxDistance = 2`.

---

## `opToStage`

`opToStage[op] = maxDistance - distance[op]`

| op | SSA | distance | `opToStage` |
|----|-----|----------|-------------|
| `%43` load A | `%43` | 2 | **0** |
| `%44` load B | `%44` | 2 | **0** |
| `%45` convert A | `%45` | 0 | **2** |
| `%46` convert B | `%46` | 0 | **2** |
| `%47` dot | `%47` | 0 | **2** |
| `%50` select | `%50` | 0 | **2** |
| `%51` epilogue if | `%51` | 0 | **2** |

(`%49` is an input to `%50` / `%51` but only enters `distance` if reached from a latency op’s user walk; if present, same stage-2 pattern. Prologue `%41` if is **not** on this forward path from the loads.)

---

## `schedule` (after `scheduleKeyOps`)

`maxStage = 2` → `CoarseSchedule(3)`.

Clusters created with `newAtBack()`:

| creation order | cluster id | meaning |
|----------------|------------|---------|
| `clusters[0]` | **0** | for stage 2 (`clusters[maxStage - stage]`) |
| `clusters[1]` | **1** | for stage 1 (unused here) |
| `clusters[2]` | **2** | for stage 0 |
| epilogue `newAtBack()` | **3** | moved `scf.if`s |

Initial insert: `schedule.insert(op, stage, clusters[maxStage - stage])`

| op | SSA | stage | cluster (after first insert) |
|----|-----|-------|------------------------------|
| `%45`, `%46`, `%47`, `%50`, `%51` | … | 2 | **0** |
| `%43`, `%44` | … | 0 | **2** |

Then epilogue move for `scf.if` not itself a latency op:

- `getForwardSlice(%51)` ≈ `{ scf.yield }` (body ops are **inside** the if, not in the forward slice)
- yield ∉ `opToStage` → move `%51` → cluster **3**

**Final `schedule` from `scheduleKeyOps`:**

| op | SSA | stage | cluster |
|----|-----|-------|---------|
| `%43` load A | `%43` | 0 | **2** |
| `%44` load B | `%44` | 0 | **2** |
| `%45` convert A | `%45` | 2 | **0** |
| `%46` convert B | `%46` | 2 | **0** |
| `%47` dot | `%47` | 2 | **0** |
| `%50` select | `%50` | 2 | **0** |
| `%51` epilogue if | `%51` | 2 | **3** |

Cluster list order: `0 → 1 → 2 → 3` (id 1 empty for this key-op set).

---

## Note vs `schedule_loops_debug.log`

The `.log` is **after the full schedule pass** (deps, prologue/epilogue, cluster splits). Example final attrs:

- loads `%43`/`%44`: `loop.stage = 0`, `loop.cluster = 5`
- convert/dot/select: `loop.stage = 2`, `loop.cluster = 2`
- epilogue if `%51`: `loop.stage = 2`, `loop.cluster = 6`
- `scf.for`: `tt.scheduled_max_stage = 2`

Stages match `opToStage`; cluster ids are remapped later.
