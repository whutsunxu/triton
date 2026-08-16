# Warp specialization (matmul pass analysis) — Rest of AWS

Companion notes for `_p_matmul_…_64x256x128x1`.

| | |
|--|--|
| **Pipeline** | `AutomaticWarpSpecialization` (`AutomaticWarpSpecialization.cpp`) |
| **IR spine** | `input_ir.ttir` → … → `stage_cluster.ttir` → `after_partition_scheduling.ttir` → … |
| **This doc** | AWS parts **§2–§11** (after PartitionScheduling) |
| **Sibling** | `WarpSpecialization_PartitionScheduling.md` — §1 PartitionScheduling |

**AWS order (high level):**

```text
1 PartitionScheduling          ← see sibling doc
2 NVWSHoistTmemStore           ← this file starts here
3 NVWSInsertAref / InsertTmemAref
4 SCCP
5 CSE
6 NVWSLowerAref
7 PartitionLoops          ← physical per-partition scf.for clones
8 NVWSLowerWarpGroup
9 ScheduleLoops
10 multiBufferTMADescriptors
```

---

# 2. NVWSHoistTmemStore

| | |
|--|--|
| **Pass** | `nvws-hoist-tmem-store` (`HoistTmemStore.cpp`) |
| **When** | right after PartitionScheduling in AWS |
| **This matmul** | no TMEM/MMAv5 → typically a no-op |

**Goal (alloc part).** Move `ttng.tmem_alloc` **out of** the `tt.warp_specialize` `scf.for` (and nested fors) so TMEM is allocated **once**, not every iteration. Then **thread the async token** through the loop nest like any other carried value.

Also folds some `tmem_store` into alloc (`FoldTmemStoreIntoAlloc`) and related cleanups; below focuses on `hoistTmemAlloc`.

## 2.1 Why must the token be an iter_arg / yield?

TMEM ops (`alloc` / `store` / `load` / MMAv5) form an SSA **token chain**: each op takes a dep token and produces a new one. That encodes “this buffer use happens after that one.”

If alloc stays **inside** the body, the token is local to that iteration. After hoist, alloc is **outside**, but MMA still runs **inside** every iter. The token that MMA depends on (and the token MMA produces for the next use) must therefore be:

1. passed **in** as a for **init / region iter_arg** (like `%acc` or any carried state), and
2. passed **out** via **`scf.yield`** so the next iteration / outer loop sees the updated token.

Without yield, the final in-body token would die at the end of the iteration and outer SSA / next-iter deps would be broken — same reason other loop-carried args are yielded.

## 2.2 Safety check (when hoist is refused)

Do **not** hoist across an outer loop if an inner MMA loop’s trip count **depends on that outer IV** and you cannot prove the inner loop always executes (≥1). Otherwise some outer iters would skip MMA while a single outer alloc/token still exists (mismatched lifetime). See § comment in `hoistTmemAlloc`.

## 2.3 Fake IR walkthrough (multi-tier fors)

Illustrative only (types shortened). Partition ids: MMA warp = `0`.

### Before hoist

Alloc lives **inside** the WS for. Its token’s **only** use is as **init** of the inner MMA for (`getUniqueUserLoopAndMMA`).

```mlir
// tt.warp_specialize on outer for
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32
%n = ... : i32
%k = ... : i32   // independent of %i  (safe to hoist)

scf.for %i = %c0 to %n step %c1 iter_args(%arg_other = ...)
    -> (...) {
  // still INSIDE ws for:
  %buf, %tok0 = ttng.tmem_alloc : ... -> !ttg.memdesc<...>, !ttg.async.token
      {ttg.partition = array<i32: 0>}

  %inner:1 = scf.for %kk = %c0 to %k step %c1
      iter_args(%tok = %tok0) -> (!ttg.async.token) {
    // MMA consumes/produces token (partition 0)
    %tok1 = ttng.tc_gen5_mma ..., %tok {ttg.partition = array<i32: 0>}
    scf.yield {ttg.partition = array<i32: 0>} %tok1 : !ttg.async.token
  } {ttg.partition = array<i32: 0>,
     ttg.partition.outputs = [array<i32: 0>]}

  scf.yield {ttg.partition = array<i32: 0>} %arg_other : ...
} {tt.warp_specialize, ttg.partition = array<i32: 0>, ...}
```

### What `hoistTmemAlloc` does

1. **Validate** nest + trip-count independence (or `assume` proves execute-once).
2. **`moveBefore`** outer WS for; **remove** `ttg.partition` on alloc.
3. Read **`tokenPartition`** from the for that currently consumes the token (`partition.outputs` at that init-arg index).
4. **Outer→inner:** `addIterArgsToLoop` with the hoisted token; extend `ttg.partition` / `ttg.partition.outputs` with `tokenPartition`; set inner init to the new iter_arg.
5. Walk the in-body token chain to the **last unused** token (after MMA/load/store).
6. **Inner→outer:** `appendToForOpYield` that token so each for **yields** the updated token; result becomes the next outer yield input.

### After hoist (shape)

```mlir
// OUTSIDE ws for — once:
%buf, %tok0 = ttng.tmem_alloc : ... -> !ttg.memdesc<...>, !ttg.async.token
// (no ttg.partition)

%ws:2 = scf.for %i = %c0 to %n step %c1
    iter_args(%arg_other = ..., %tok_o = %tok0)
    -> (..., !ttg.async.token) {
  %inner:1 = scf.for %kk = %c0 to %k step %c1
      iter_args(%tok = %tok_o) -> (!ttg.async.token) {
    %tok1 = ttng.tc_gen5_mma ..., %tok {ttg.partition = array<i32: 0>}
    scf.yield {ttg.partition = array<i32: 0>} %tok1
  } {ttg.partition = array<i32: 0>,
     ttg.partition.outputs = [array<i32: 0>]}

  scf.yield {ttg.partition = array<i32: 0>} %arg_other, %inner#0
} {tt.warp_specialize,
   ttg.partition = array<i32: 0>,
   ttg.partition.outputs = [..., array<i32: 0>], ...}
```

```text
before:  [ws for] { alloc;  inner for(tok=alloc_tok) { mma; yield tok } }
after:   alloc;
         [ws for](tok_o=alloc_tok) {
           inner for(tok=tok_o) { mma; yield tok };
           yield ..., inner_tok
         }
```

**One-liner:** hoist moves **storage creation** out; **token** stays a normal loop-carried SSA value (init + yield) so each iteration’s MMA still chains dependencies correctly.

---

# 3. NVWSInsertAref (`InsertAref.cpp`) / InsertTmemAref

| | |
|--|--|
| **Pass** | `nvws-insert-aref` (`NVWSArefInsertion`) |
| **When** | after HoistTmemStore in AWS |
| **Sibling** | `nvws-insert-tmem-aref` handles **TMEM / MMAv5** separately — this pass **skips** `MMAv5OpInterface`, `TMEMAllocOp`, `TMEMStoreOp` |
| **This matmul** | load partition `1` → compute/store partition `0` via aref on `tt.descriptor_load` results (and similar register values) |

## 3.1 General idea / mechanism

### Why aref?

After PartitionScheduling the IR still looks like **one** loop with shared SSA and `ttg.partition` attrs:

```mlir
%43 = tt.descriptor_load ... {ttg.partition = array<i32: 1>}
%45 = ttg.convert_layout %43 {ttg.partition = array<i32: 0>}
```

That `%43 → %45` edge is fine only while both ops share one region / register file. **PartitionLoops** later splits the WS for into **separate partition regions** (different warp groups). Then p1 keeps the load, p0 keeps convert/dot — they **do not share SSA**. A raw cross-partition SSA use cannot survive.

So the producer must leave data somewhere both sides can see: usually **shared memory** (TMEM for the sibling pass). **Aref** is the handle for that buffer; **put / get** are the protocol for using it safely.

| Piece | Role |
|--------|------|
| **`aref.create`** | Handle wrapping an allocated buffer. Created **outside** the WS loop so both partitions see the same channel. |
| **`aref.put.enter` / `put.exit`** | Producer critical section: enter → writable buf + token; write; exit → published. |
| **`aref.get.enter` / `get.exit`** | Consumer critical section: enter → readable buf + token; read; exit → done reading. |
| **token** | SSA stand-in for the handshake; **LowerAref** turns enter/exit into **mbarriers**. |

Aref does **not** compute — only names the shared buffer and brackets who may touch it when. Same-partition SSA stays as normal SSA (no aref).

### How data moves (put / get)

Same edge as this matmul (`p1` load → `p0` compute):

**1. Create channel once (outside the loop)**

```mlir
%buf  = ttg.local_alloc ...           // real smem
%aref = nvws.aref.create %buf         // handle both partitions use
```

**2. Producer (partition 1) — put = write into the channel**

```mlir
%tok_p, %buf_w = nvws.aref_put_enter %aref   // may write
nvws.descriptor_load ... into %buf_w         // TMA fills aref buffer
nvws.aref_put_exit %aref, %tok_p             // publish
```

No longer “produce register `%43` for someone else” — produce by **storing into the aref buffer**.

**3. Consumer (partition 0) — get = read from the channel**

```mlir
%tok_g, %buf_r = nvws.aref_get_enter %aref
%43' = ttg.local_load %buf_r                 // p0-local registers
%45  = ttg.convert_layout %43'
%47  = tt.dot %45, ...
nvws.aref_get_exit %aref, %tok_g
```

**4. After PartitionLoops (conceptually)**

```
warp group 1 (load):   put_enter → TMA → put_exit
warp group 0 (compute): get_enter → local_load → cvt/dot → get_exit
         \________________ aref / smem ________________/
```

No SSA edge between groups — only **shared buffer + sync**. Pipelining/multibuffering (later) can stage multiple slots on the same aref so load and compute overlap.

### What this pass processes

`runOnFunction` walks each `tt.warp_specialize` for that has partitions, then handles **three producer kinds independently** (same put/get mechanism; different where the produced value comes from). MMAv5/TMEM are skipped here (`InsertTmemAref`). Details in §3.2.

## 3.2 Three cases (where producers come from)

| # | Case | Producer value | Pass order |
|---|------|----------------|------------|
| 1 | **Loop-carried args** | Region iter_arg (tensor / float / int); producer partitions from `ttg.partition.outputs[argNumber − 1]` (IV = arg 0) | At **start of for body** |
| 2 | **Memory ops** | `local_alloc` or `local_alloc(desc_load)` | **First** among in-body ops — put can TMA straight into the aref buf and mark alloc/load stale; prefer before bare register uses of the same load |
| 3 | **Other non-TMEM ops** | Remaining partitioned ops (incl. leftover `desc_load` register uses) | **After** memory ops; insert after the producer. This FP8 matmul: `descriptor_load` (p1) → `convert_layout` / `dot` (p0) |

**MMAv5 / TMEM:** skipped (`MMAv5OpInterface`, `TMEMAllocOp`, `TMEMStoreOp`) → sibling `InsertTmemAref`.

### Case sketches

**Loop-carried** (fake; this matmul’s `%arg51` is often same-partition only):

```mlir
// partition.outputs for %acc includes p1; use in p0 → aref at body start
scf.for ... iter_args(%acc = %cst) -> (...) {
  %x = arith.addf %acc, %one {ttg.partition = array<i32: 0>}
  scf.yield ... %newAcc ...   // produced under p1
}
```

**Memory ops** — `local_alloc(desc_load)` processed before bare load uses:

```mlir
// Before
%ld = tt.descriptor_load %desc[...] {ttg.partition = array<i32: 1>}
%mem = ttg.local_alloc %ld {ttg.partition = array<i32: 1>}
%c  = ttg.local_load %mem {ttg.partition = array<i32: 0>}

// After: put TMA into aref buf (erase %ld/%mem); get retargets / loads for p0
```

If alloc and load have **different** partition ids, they are **not** fused (`isLoadAndAlloc`) — two separate producers.

**Non-tmem register** (real pattern from `after_partition_scheduling.ttir`):

```mlir
%43 = tt.descriptor_load ... {ttg.partition = array<i32: 1>}
%45 = ttg.convert_layout %43 {ttg.partition = array<i32: 0>}
// → put TMA/store into aref; get local_load; %45 uses loaded value
```

## 3.3 How we detect “producer ≠ consumer partitions”

Core gate is inside `insertArefs` → `processResultUses`:

```cpp
auto userPartitions = getPartitionIds(&use);
for (auto id : producedValue.partitions)
  userPartitions.remove(id);
for (auto id : userPartitions) {
  resultsPerPartition[id].insert(result);
  usesPerPartition[id].push_back(&use);
}
if (resultsPerPartition.empty())
  return false;  // no cross-partition consumer → no aref
```

- `getPartitionIds(&use)` = partitions of the **consumer** op (or per-operand attrs when present).
- Remove every **producer** partition id. Whatever remains is a **true cross-partition** consumer.
- Bucket uses by remaining consumer partition id → one get sequence per consumer partition.

Producer partitions for a normal op result: `getProducedValues` → `getPartitionIds(op)` (or `partition.outputs` for region ops). For loop iter_args: `getPartitionOutputs(for)[argNumber − 1]`.

### IR example (real) — from `after_partition_scheduling.ttir`

Load partition **1**, compute partition **0**:

```mlir
%43 = tt.descriptor_load %arg13[...] {ttg.partition = array<i32: 1>}
      : ... -> tensor<64x128xf8E5M2, #blocked2>
%45 = ttg.convert_layout %43 {ttg.partition = array<i32: 0>}
      : tensor<64x128xf8E5M2, #blocked2> -> tensor<64x128xf8E5M2, #ttg.dot_op<...>>
%47 = tt.dot %45, %46, %arg51 {ttg.partition = array<i32: 0>} ...
```

For produced value `%43` with `partitions = {1}`:

| Use | `userPartitions` | after `remove(1)` | action |
|-----|------------------|-------------------|--------|
| `%45` convert_layout | `{0}` | `{0}` | aref get on partition **0** |
| (if some same-p1 use existed) | `{1}` | `∅` | ignored |

Same idea for `%44` → `%46`. Loop-carried detection uses the same `remove` rule on iter_arg uses (see §3.2 sketch).

## 3.4 Create put / get, how they cowork, replace old SSA

Once `resultsPerPartition` is non-empty:

```cpp
aref = createAref(...);           // before outer WS loop: smem buffer(s)
staleOps = createArefPut(...);    // producer: enter → fill buf → exit
for each consumerPartition:
  createArefGet(...);             // consumer: enter → local_load / rewire → exit
for (op : staleOps) op->erase();  // old desc_load / local_alloc when absorbed
```

### Put (`createArefPut`) — producer side

1. `ArefPutEnterOp` → get writable buffer view + token (tagged with **producer** partition).
2. Fill buffer depending on producer kind:
   - `local_alloc(desc_load)` / bare `desc_load` → **TMA into aref buf** (`createNVWSDescriptorLoadOp`), mark load(/alloc) stale.
   - plain tensor / scalar → `LocalStoreOp` (scalar via splat).
3. `ArefPutExitOp` with async kind (`TMALoad` / `NONE`).

Producer SSA (`%43`) is no longer the channel; the aref buffer is.

### Get (`createArefGet`) — consumer side

1. Insert before earliest consumer use in the block.
2. `ArefGetEnterOp` → read-only buffer + token (**consumer** partition).
3. Replace consumer operands:
   - tensor → `LocalLoadOp` from buf, `use->set(localLoad)`;
   - `LocalAllocOp` result → retarget uses to aref memdesc (`replaceUsesAndPropagateType`);
   - scalar → load + unsplat.
4. `ArefGetExitOp` after last consumer / post-dominator.

Put publishes; get consumes (full IR sketch in §3.1). Later `LowerAref` turns enter/exit into barriers / multibuffers; PartitionLoops keeps put ops in p1’s clone and get ops in p0’s clone.

**One-liner:** same-partition SSA stays; cross-partition edges become aref put (producer fills smem) + get (consumer loads / retargets); three producer kinds are processed independently; MMAv5/TMEM use `InsertTmemAref`.

---

# 4. SCCP

SCCP (`Sparse Conditional Constant Propagation`) is an MLIR “cleanup” pipeline that does two things:

1. **Delete/ignore dead code** on paths that are proven unreachable.
2. **Propagate constants** by speculatively folding ops whose operands are known constant, leaving some ops unreachable.

In Triton AWS, this runs after aref insertion / staging so redundant math and loop-bound expressions can be simplified.

Iteration control (overall workflow):

SCCP runs its analyses in a loop until a **fixpoint** is reached: it repeatedly revisits affected ops/edges whenever the analysis discovers new information (e.g., “this SSA value becomes a known constant on live paths” or “this branch is dead”). Once the analysis facts stop changing (worklist empty), SCCP performs the **rewrite** step exactly once to materialize constants and erase trivially-dead ops.

---

## 4.1 Basic mechanism (analysis + rewrite)

SCCP is implemented as:

1. **Dead-code analysis (conditional part).**
   Using `DeadCodeAnalysis`, SCCP determines which regions/ops are actually live under current control-flow facts (so constants won’t be propagated through dead branches).
2. **Sparse constant propagation (value part).**
   `SparseConstantPropagation` maintains a **lattice** per SSA value using `ConstantValue`.
   Concretely, `ConstantValue` can be in these states:
   - **Uninitialized**: the analysis hasn’t visited/derived a fact for this SSA value yet (`getUninitialized()`).
   - **Known constant**: a specific `Attribute` is proven constant for this SSA value.
   - **Unknown/overdefined**: folding can’t prove a single constant (e.g., conflicting constants across live paths) (`getUnknownConstant()`).

   Lattice join uses this rule: if two different constants meet, it becomes **Unknown/overdefined**; `Uninitialized` acts like “no information yet”.
3. **Rewrite step.**
   After the analyses, SCCP replaces SSA uses with materialized constants and erases newly dead ops.

Key idea: SCCP first **decides** (analysis) what’s constant and what’s dead, then **rewrites** IR based on those decisions. It does not “randomly delete” code.

---

## 4.2 `SparseConstantPropagation::visitOperation()` workflow (simulate folding into a lattice)

`SparseConstantPropagation::visitOperation()` is the *value* part of SCCP: for each op that the dataflow solver decides to visit, it tries to answer:
“If my operands are known constants, can this op’s results be constants too?”

It does this by **simulating** the op via `op->fold(...)` and writing the result into a **lattice** for each SSA result.

### Stage 1 — quick skip for region ops

If the op owns regions (`op->getNumRegions() != 0`), the analysis does not try to fold its results:

- folding could be in-place or depend on internal control flow
- simulated execution can’t be guaranteed out-of-place

So it sets all results to the lattice “entry/unknown” state:

- `if (op->getNumRegions()) { setAllToEntryStates(results); return success(); }`

### Stage 2 — collect constant operands (lattice → attributes)

For ops with no regions, SCCP reads the lattice of each operand:

- if any operand lattice is still *uninitialized*, it returns early (solver will revisit later)
- otherwise it builds `constantOperands`
  - for **known constants**, `getConstantValue()` yields a real `Attribute`
  - for **unknown/overdefined**, `getConstantValue()` yields a **null** `Attribute`
    (still “initialized”, just not a proven constant)

Key variable/method:

- `ArrayRef<const Lattice<ConstantValue> *> operands`
- `SmallVector<Attribute, 8> constantOperands`
  - `8` is the **inline capacity** (performance hint), not a correctness parameter

### Stage 3 — simulate folding (speculative execution)

It then snapshots the op to protect against speculative in-place folding:

- `originalOperands(op->getOperands())`
- `originalAttrs = op->getAttrDictionary()`

Then it attempts:

```cpp
SmallVector<OpFoldResult, 8> foldResults;
if (failed(op->fold(constantOperands, foldResults)))
  setAllToEntryStates(results);  // → results are overdefined/unknown
```

This is where “overdefined / unknown” often comes from in **Stage 3** (on leaf ops with no regions):

1. **Folding fails** because some `constantOperands` entries are null (operand lattice is unknown/overdefined, not a proven constant).

Example of unknonwn op fold failure:

```mlir
%x = ... : i32        // lattice: unknown/overdefined
%y = ... : i32        // lattice: unknown/overdefined
%sum = arith.addi %x, %y : i32
```

### Stage 4 — detect in-place folding (the “foldResults.empty()” guard)

In MLIR, `op->fold(...)` is allowed to:

- **out-of-place fold**: return computed results in `foldResults`
- **in-place fold**: mutate the op and return **no** fold results (`foldResults` is empty)

SCCP treats the in-place case conservatively:

```cpp
if (foldResults.empty()) {
  op->setOperands(originalOperands);
  op->setAttrs(originalAttrs);
  setAllToEntryStates(results); // → overdefined/unknown
  return success();
}
```

Why this implies “overdefined”:

- SCCP only restores operands/attrs, not arbitrary internal mutations
- so if it can’t represent the fold solely via `foldResults`, the safe assumption is that results are **not provably constant**

### Stage 5 — merge fold results into the lattice

If folding succeeded out-of-place, it merges into lattices:

- if a `foldResult` is an `Attribute`, join lattice with that constant:
  - `propagateIfChanged(lattice, lattice->join(ConstantValue(attr, op->getDialect())))`
- otherwise the fold returned another `Value`, so it joins with that value’s lattice element:
  - `AbstractSparseForwardDataFlowAnalysis::join(lattice, *getLatticeElement(...))`

Key variables:

- `ArrayRef<Lattice<ConstantValue> *> results`
- `OpFoldResult` → either `Attribute` (constant) or `Value` (alias/another SSA)

**Successful example (Attribute path — full constant fold):**

Before:

```mlir
%c0 = arith.constant 0 : i32
%c1 = arith.constant 1 : i32
%sum = arith.addi %c0, %c1 : i32
%out = arith.muli %sum, %c1 : i32
```

On `arith.addi`:

| Step | What happens |
|------|----------------|
| Stage 2 | operand lattices: `%c0 → 0`, `%c1 → 1` → `constantOperands = [0, 1]` |
| Stage 3 | `op->fold(...)` succeeds out-of-place → `foldResults = [i32 1]` |
| Stage 5 | `%sum` lattice joins to **known constant** `1` (`Attribute` path) |

After SCCP **rewrite** (materialize known constants; erase trivially dead ops). Sketch after folding `%sum` (and possibly `%out` in a later visit):

```mlir
%c1 = arith.constant 1 : i32
%out = arith.muli %c1, %c1 : i32
// or, after further fold of %out:
// %out = arith.constant 1 : i32
```

**Successful example (Value path — alias / identity fold):**

Before:

```mlir
%c0 = arith.constant 0 : i32
%sum = arith.addi %x, %c0 : i32    // %x lattice is unknown/overdefined
%out = arith.muli %sum, %y : i32
```

On `arith.addi`:

| Step | What happens |
|------|----------------|
| Stage 2 | `constantOperands = [null, 0]` — `%x` is not a proven constant |
| Stage 3 | `addi(x, 0) → x` fold returns **Value** `%x`, not an `Attribute` |
| Stage 5 | `%sum` lattice **joins with `%x`’s lattice** (alias path), not forced to a new constant |

Conceptual IR after that identity fold (what “same as `%x`” means):

```mlir
%c0 = arith.constant 0 : i32      // may remain or be DCE’d if unused
%out = arith.muli %x, %y : i32   // uses of %sum → %x; %sum erased if dead
```

Note: Stage 5 itself only updates lattices. SCCP’s later **rewrite** materializes when the lattice is a known `Attribute`; it does **not** rewrite `%sum` → `%x` just from a Value fold. Identity rewrites like that are usually done by createOrFold / canonicalizer / CSE. In SCCP, the Value path mainly means “`%sum` has the same constant-ness as `%x`” for later users.

So Stage 5 covers both “computed constant” and “same as another SSA value” outcomes.

*(After SCCP, AWS runs CSE — see §5.)*

---

# 5. CSE (`mlir/lib/Transforms/CSE.cpp`)

| | |
|--|--|
| **Pass** | MLIR `cse` (`CSEPass` / `createCSEPass()`) |
| **When** | right after SCCP in AWS |
| **Role** | remove duplicate equivalent ops created/exposed by SCCP and earlier passes |

## 5.1 General idea — equivalence **and** dominance

CSE = **Common Subexpression Elimination**.

If two ops compute the **same expression** and the earlier result is **available** at the later site, CSE:

1. redirects uses of the later op to the earlier one
2. erases the duplicate

CSE needs **both** checks — neither alone is enough:

| Check | What it answers |
|--------|------------------|
| **Equivalence** (`OperationEquivalence`) | Same opcode, operands, attrs? → “same computation” |
| **Dominance** (`DominanceInfo` + scoped known-map) | Is the earlier SSA result **available** here? → “safe to reuse” |

### Dominance (brief)

**A dominates B** if **every control-flow path** from the region **entry** to **B** goes through **A**.

- “Path” = CFG route through blocks/branches — **not** SSA operands.
- An SSA **def** (e.g. `%a = …`) that dominates a **use** is always defined before that use on every path, so the value is legally readable there.

Dominance alone does **not** mean two ops are interchangeable. It only says: if they *are* the same expression, the earlier result can replace the later one.

Tiny example:

```mlir
%a = arith.addi %x, %c1 : i32
%b = arith.addi %x, %c1 : i32   // equivalent to %a, and %a dominates %b
%y = arith.muli %a, %b : i32
```

- Equivalence: both `addi %x, %c1`
- Dominance: `%a` is above `%b` on the only path → available
- CSE: uses of `%b` → `%a`, erase `%b`

## 5.2 Main workflow (`CSE::runOnOperation`)

```cpp
void CSE::runOnOperation() {
  IRRewriter rewriter(&getContext());
  CSEDriver driver(rewriter, &getAnalysis<DominanceInfo>());
  bool changed = false;
  driver.simplify(getOperation(), &changed);

  numCSE = driver.getNumCSE();
  numDCE = driver.getNumDCE();

  if (!changed)
    return markAllAnalysesPreserved();

  // CSE doesn't remove region ops in a CFG-breaking way → dominance still OK
  markAnalysesPreserved<DominanceInfo, PostDominanceInfo>();
}
```

| Step | What |
|------|------|
| **1. DominanceInfo** | analysis of which defs/blocks dominate which uses |
| **2. `simplify(...)`** | walk regions; find duplicates; replace uses; queue erases |
| **3. Stats** | `numCSE` = ops CSE’d; `numDCE` = trivially dead ops erased |
| **4. Preserve analyses** | if IR unchanged → preserve **all**; if changed → still preserve dominance (see below) |

Unlike SCCP, CSE **rewrites immediately** during the walk (no separate lattice fixpoint → materialize phase).

## 5.3 Details of `simplify`

`CSEDriver::simplify(op, &changed)`:

```cpp
ScopedMapTy knownValues;          // scoped hash table: equivalent-op → first def
for (auto &region : op->getRegions())
  simplifyRegion(knownValues, region);

for (auto *dead : opsToErase)
  rewriter.eraseOp(dead);
*changed = !opsToErase.empty();
```

### Region / block walk

- Prefer **dominance-tree** order for multi-block regions with SSA dominance.
- Push a **scope** on the known-map when entering a dominated block; pop when leaving.
- So known ops from **dominating** blocks are visible; sibling blocks don’t share each other’s locals incorrectly.

### Per op (`simplifyOperation`)

| Case | Action |
|------|--------|
| Terminator | skip |
| Trivially dead | queue erase (`numDCE++`) |
| Multi-block regions (complex) | skip (conservative) |
| Memory-effect free | if equivalent op already in map → `replaceUsesAndDelete`; else `insert(op, op)` |
| Only `Read` effects | CSE only if same block and no conflicting write between the two |
| Other side effects | don’t CSE |

#### What `isMemoryEffectFree(op)` means

MLIR helper (`SideEffectInterfaces.cpp`): **does this op (and nested ops if recursive) have no memory side effects?**

- **Memory-effect free** ≈ pure SSA compute — no read/write/alloc/free of memory.
  - Typical **yes**: `arith.addi`, `arith.muli`, `arith.constant`, …
  - Typical **no**: `memref.load`/`store`, `tt.load`/`store`, TMA/`descriptor_load`, …
- Decision sketch:
  - Implements `MemoryEffectOpInterface` and `hasNoEffect()` → free (then check nested if `HasRecursiveMemoryEffects`).
  - No interface and no recursive trait → **not** free (unknown → conservative).
  - Regions with recursive effects → free only if **all** nested ops are free.

Why CSE splits cases on this:

- **Free** → duplicates are interchangeable (same inputs ⇒ same result); only need equivalence + dominance.
- **Not free** → two “same-looking” ops may observe different memory state at different program points, so CSE is restricted (e.g. only simple same-block reads with no conflicting write in between).

#### How `knownValues` identifies and replaces a common op

`knownValues` is a **`ScopedHashTable<Operation*, Operation*>`** keyed by **operation shape**, not by SSA value id:

```cpp
using ScopedMapTy = llvm::ScopedHashTable<Operation *, Operation *,
                                          SimpleOperationInfo, AllocatorTy>;
// conceptually:  "expression fingerprint" → first Operation* that computed it
```

**Key = the current op pointer used as a lookup probe; Value = the earlier “keeper” op to reuse.**

##### 1) How “same expression” is hashed / compared

`SimpleOperationInfo` plugs into the hash table:

```cpp
getHashValue(op)  → OperationEquivalence::computeHash(
                      op, hashOperands, /*ignore results*/, IgnoreLocations)
isEqual(a, b)     → OperationEquivalence::isEquivalentTo(a, b, IgnoreLocations)
```

So two different `Operation*` collide as “the same key” when they have:

- same opcode / traits relevant to equivalence
- same operands (same SSA values)
- same attributes
- **locations ignored**; **result SSA names ignored** (results aren’t part of the hash)

Example: `%a = addi %x, %c1` and `%b = addi %x, %c1` hash/compare equal even though `%a ≠ %b`.

##### 2) Identify: `lookup` then `insert`

For a memory-effect-free op (core path):

```cpp
if (auto *existing = knownValues.lookup(op)) {
  replaceUsesAndDelete(knownValues, op, existing, hasSSADominance);
  return success();   // this op is a duplicate
}
knownValues.insert(op, op);   // first time we see this expression → become the keeper
```

Walkthrough on one block:

```mlir
%c1 = arith.constant 1 : i32
%a  = arith.addi %x, %c1 : i32
%b  = arith.addi %x, %c1 : i32
%y  = arith.muli %a, %b : i32
```

| Visit | `lookup` | Map after step |
|-------|----------|----------------|
| `%c1 = constant 1` | miss | `{ constant(1) → %c1_op }` |
| `%a = addi %x, %c1` | miss | `… + { addi(%x,%c1) → %a_op }` |
| `%b = addi %x, %c1` | **hit** → `%a_op` | unchanged; CSE `%b` |
| `%y = muli %a, %b` | (after replace, operands may already be `%a,%a`) | insert or CSE further |

Because the table is **scoped** with the dominance walk: when you leave a dominated block, that block’s inserts are popped. Sibling blocks don’t see each other’s local keepers — only ancestors’ (dominating) expressions stay visible.

##### 3) Replace: `replaceUsesAndDelete(op, existing)`

On a hit, `op` is the duplicate, `existing` is the keeper:

```cpp
// With SSA dominance (common case):
rewriter.replaceAllUsesWith(op->getResults(), existing->getResults());
opsToErase.push_back(op);   // erase later, after the walk
++numCSE;
```

So every use of `%b` becomes a use of `%a`; `%b`’s op is queued for erase. IR conceptually becomes:

```mlir
%c1 = arith.constant 1 : i32
%a  = arith.addi %x, %c1 : i32
%y  = arith.muli %a, %a : i32
```

Without full SSA dominance, CSE only replaces uses whose owning ops were **not yet visited** (`replaceUsesWithIf`), and erases only if the duplicate ends up use-empty — avoids rewriting ops already processed.

**One-liner:** `knownValues` is a dominance-scoped dictionary from “expression shape” → first op; `lookup` finds a common subexpression; `replaceUsesAndDelete` rewires SSA uses to that first op and deletes the duplicate.

### Example after SCCP (same story, end IR)

Before CSE:

```mlir
%c1 = arith.constant 1 : i32
%a  = arith.addi %x, %c1 : i32
%b  = arith.addi %x, %c1 : i32
%y  = arith.muli %a, %b : i32
```

After CSE:

```mlir
%c1 = arith.constant 1 : i32
%a  = arith.addi %x, %c1 : i32
%y  = arith.muli %a, %a : i32
```

## 5.4 `markAllAnalysesPreserved` / dominance preserve

Pass-manager bookkeeping after CSE:

1. **`if (!changed) markAllAnalysesPreserved()`**
   IR identical → every cached analysis (dominance, etc.) is still valid; don’t recompute.

2. **`if (changed) markAnalysesPreserved<DominanceInfo, PostDominanceInfo>()`**
   IR did change (use redirects / erases), so most analyses are stale — **but** CSE currently doesn’t remove region/control ops in a way that invalidates the CFG dominance tree, so dominance analyses can be reused by later passes.

That comment in `CSE.cpp`:

```cpp
// We currently don't remove region operations, so mark dominance as preserved.
```

---
