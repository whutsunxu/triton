# Two ways Triton overlaps memory / compute latency

Companion to the AWS notes (`WarpSpecialization_*.md`) and the coarse schedule notes (`Assign_latency_And_Schedule_loop_stage_cluster.md`).

Kernel example: `_p_matmul_…_64x256x128x1`.

Both paths hide TMA (or load) latency behind MMA/`dot`, but they overlap **different axes**.

| | **① Software pipeline (SWP)** | **② Warp specialization (WS)** |
|--|--|--|
| **Idea** | One set of warps time-multiplexes work across **iterations** | Different warps specialize (TMA vs compute) and run **concurrently** |
| **Schedule knobs** | `loop.stage` / `loop.cluster` | `ttg.partition` → physical fors in `ttg.warp_specialize` |
| **What `num_stages` mainly means** | Pipeline **depth** (how many iters in flight in one for) | Aref / barrier **ring depth** (how many tiles buffer between producer & consumer warps) |
| **Passes (Blackwell-ish)** | `assign_latencies` → `schedule_loops` → `pipeline` | `add_warp_specialize` / AWS (PartitionScheduling → … → LowerWarpGroup → …) |
| **Passes (docs)** | `Assign_latency_And_Schedule_loop_stage_cluster.md` + `Add_pipeline.md` | `WarpSpecialization_PartitionScheduling.md` + Rest1 + Rest2 |
| **Overlap looks like** | load(i+k) issued while dot(i) runs | TMA partition fills slot `s` while compute partition drains another slot |

```text
① SWP — same warps, different iters
   iter i:     [---- wait/TMA ----][==== dot ====]
   iter i+1:        [---- wait/TMA ----][==== dot ====]
   iter i+2:             [---- wait/TMA ----][==== dot ====]
                         ↑ stages / clusters stretch the for

② WS — different warps, same logical loop (ring of depth ~num_stages)
   TMA warps:     put tile → put tile → put tile → …
   Compute warps:     get/dot → get/dot → get/dot → …
                  ↑ empty/full mbarriers + smem slots
```

---

## Relationship

1. **Same goal:** keep the MMA/`dot` path busy while global→smem traffic is outstanding.
2. **Same parameter name, two roles:** `num_stages` sizes either the SWP wavefront or the WS multibuffer ring (AWS `multiBufferAref` / barrier arrays).
3. **Can compose:** on Blackwell, the pipeline often does **latencies → schedule_loops → warp_specialize → pipeline**. After AWS splits TMA vs compute into concurrent partitions, the post-WS `ScheduleLoops` often **collapses** each partition’s body to one stage — because that TMA↔dot overlap is already paid for by warps + barriers, not by multi-stage SWP inside one for.
4. **Shared IR vocabulary early on:** before PartitionLoops, one for still carries both `loop.stage`/`loop.cluster` (from the first schedule) and `ttg.partition` (from PartitionScheduling). Later AWS turns partitions into real regions and may rewrite stage/cluster per region.

---

## Differences

| Dimension | ① SWP | ② WS |
|--|--|--|
| **Axis of overlap** | Across **loop iterations** (temporal / software) | Across **warp groups** (spatial / concurrent hardware) |
| **Who runs TMA vs dot** | Same warps, different times in the expanded schedule | Different warps at the same time |
| **Communication** | Mostly SSA / iter_args inside one pipelined for | Smem + mbarriers (aref → wait/expect/arrive); no cross-partition registers |
| **Physical loops** | Still one logical for (expanded by `pipeline`) | One `scf.for` **per** partition region inside `ttg.warp_specialize` |
| **Failure mode if depth=1** | Little/no hide of load latency behind compute | Producer blocks on empty / consumer on full every tile (ping-pong) |
| **After split** | Stages still meaningful if a *single* partition has multi-stage latency ops | Stale global stage tags on each half are pruned (§9 ScheduleLoops in Rest2) |

**One-liner:** SWP overlaps **iters on one warp team**; WS overlaps **roles on two warp teams** with a buffered handoff.

---

## Is WS “more aligned” with persistent kernels?

**Related, not the same.**

| Concept | What it decides |
|--|--|
| **Persistent kernel** | How **work** is scheduled: a CTA stays alive and pulls many tiles (grid-stride / work queue), instead of one launch tile per CTA |
| **Warp specialization** | How **warps inside a CTA** are used: some warps only TMA, others only MMA, synchronized through smem |

Why people couple them:

- Persistent CTAs run **long** → amortizes setup and keeps producer/consumer warps busy over many tiles.
- WS needs **resident concurrent warps** and a live CTA to keep the TMA/MMA pipeline full — the same residency persistent kernels already assume.

What WS is *not*:

- It does not by itself assign the next output tile from a global work queue.
- SWP can also run inside a persistent CTA; persistence is orthogonal to “stage vs partition.”

So: WS fits the **persistent-CTA + specialized-warps** style of modern GEMM/Attention kernels better than classic single-role SWP, but **persistence = work distribution**, **WS = intra-CTA role split**.

---

## Pointers

| Topic | Doc |
|--|--|
| Stage / cluster / `opLatency` | `Assign_latency_And_Schedule_loop_stage_cluster.md` |
| Materialize SWP (`add_pipeline`) | `Add_pipeline.md` |
| Partition tags | `WarpSpecialization_PartitionScheduling.md` |
| Aref → barriers → clone fors → `warp_specialize` | `WarpSpecialization_Rest1.md`, `WarpSpecialization_Rest2.md` |
| AWS pass table | Rest2 §11 |
