# NCU bottleneck summary — persistent `block_m=64` FP8 matmul

**Kernel:** `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1`  
**Case:** `Matmul-None-True-…-None-64-8192-2048-7680-batched-float8_e5m2-float8_e5m2-…-True-None`  
**Date:** 2026-08-16  
**Tool:** Nsight Compute 2025.3.1 (host mount), container `--privileged`  
**Pytest:** `block_m=64`, `is_persistent=True` — PASSED  

## Artifacts

| File | Contents |
|------|----------|
| `test_matmul_ncu_main.ncu-rep` | Launch/Occupancy/SOL/TC roofline/Memory/Scheduler/WarpState/Compute/Instruction/Source (37 passes) |
| `test_matmul_ncu_main_{details,raw,summary}.txt` | Text exports of main report |
| `test_matmul_ncu_pmsampling.ncu-rep` | `--set pmsampling` (11 passes) + raw/summary |
| `test_matmul_ncu_tma_tensor.ncu-rep` | Focused TC / TMA pipe / warp-stall ratios |
| `test_matmul_ncu_tma_l1tex.ncu-rep` | TMA L1TEX byte/sector counters |

Host note: `/proc/driver/nvidia/params` had `RmProfilingAdminOnly: 1`. Profiling worked with Docker `--privileged`. Also wrote `/etc/modprobe.d/nvidia-nsight-profiling.conf` (`NVreg_RestrictProfilingToAdminUsers=0`) for future boots.

---

## Top-line SOL / occupancy

| Metric | Value |
|--------|-------|
| Duration (ncu, multi-pass) | **50.94 ms** (nsys wall ~43.4 ms; ncu inflated) |
| Compute (SM) Throughput | **33.91%** |
| Memory Throughput | **34.37%** |
| DRAM Throughput | **4.30%** (~19 GB/s) |
| Achieved Occupancy | **16.67%** (8 warps/SM; limited by **regs=255** and SMEM) |
| Issue slots busy | **12.07%** |
| Eligible warps / scheduler | **0.12** (No Eligible **87.90%**) |

NCU OPT: latency-bound (both compute and memory ≪ 60% peak). Est. local speedup from scheduler eligibility ~**65.6%**.

---

## Tensor Core

| Metric | Value |
|--------|-------|
| Pipe Tensor (FP) util (elapsed) | **28.52%** |
| `sm__inst_executed_pipe_tensor` | **314,572,800** (all HMMA subpipe) |
| `sm__pipe_tensor_cycles_active` | **5.03e9** cycles (sum) |
| TC in SOL breakdown | 2nd behind LSU (33.91%) |

Tensor pipe is active but far from peak — consistent with low eligible warps and stall mix below (TC cannot stay fed).

---

## TMA

| Metric | Value |
|--------|-------|
| Pipe TMA util (elapsed) | **0.17%** |
| `sm__inst_executed_pipe_tma_pred_on_any` | **1,904,748** |
| TMA global load bytes (L1TEX from L2) | **25.17 GB** sum |
| TMA global store bytes | **167.77 MB** sum |
| L1 Tmain Requests (SOL mem breakdown) | **0.06%** |

TMA moves large A/B traffic with very low pipe occupancy (efficient async copies). **TMA pipe itself is not the bottleneck**; stalls are dominated by sync / scoreboard / math-pipe, not TMA issue rate.

---

## Warp schedule / stalls

**Scheduler:** theoretical 12 warps/scheduler, only **2.00** active and **0.12** eligible → severe issue under-utilization (warp-specialized persistent CTA: default MMA warps + TMA producer partition).

### Stall mix (`smsp__warp_issue_stalled_*_per_warp_active.pct`)

| Stall reason | % of warp-active |
|--------------|------------------|
| **Barrier** | **25.87%** |
| **Long scoreboard** | **20.05%** |
| **Math pipe throttle** | **18.48%** |
| **Wait** (fixed latency) | **16.87%** |
| MIO throttle | 5.39% |
| Short scoreboard | 3.71% |
| LG throttle | 0% |

Warp State “cycles per issued instruction” view (inst units): Barrier 4.28, Long SB 3.32, Math pipe 3.06, Wait 2.79, MIO 0.89.

**Interpretation**

1. **Barrier (~26%)** — WS producer/consumer mbarrier / CTA sync between TMA loads and MMA; pipeline depth-2 helps but sync still dominates.  
2. **Long scoreboard (~20%)** — waiting on outstanding shared/global (TMA→SMEM→MMA) dependencies.  
3. **Math pipe throttle (~18%)** — tensor/ALU pipe backpressure when warps finally become eligible.  
4. **Low occupancy (regs=255)** — Block Limit Registers = 1; only 8 warps/SM → little latency hiding when barriers/scoreboards fire.  
5. **Local memory spilling** reported at **100%** spilling request overhead — register pressure is extreme; exacerbates long scoreboard / wait.

vs older non-persistent `block_m=128` ncu (math_pipe ~47% dominant): this persistent-64 build shifts the bottleneck toward **barrier + scoreboard** (WS/TMA sync), with math_pipe still significant but no longer the single largest term.

---

## Memory path

- L1/TEX hit **99.85%**, L2 hit **96.69%** — on-chip reuse is good; DRAM is quiet (4.3%).  
- Mem pipes busy ~34% — matches LSU-heavy SOL breakdown.  
- Bottleneck is **latency / eligibility**, not DRAM BW.

---

## Suggested next experiments (not run here)

1. Reduce register pressure (target ≪255) to raise occupancy above 1 CTA/SM worth of latency hiding.  
2. Soften WS barrier critical path (deeper software pipeline / overlap more tiles before sync).  
3. Compare same problem with non-persistent path ncu stalls side-by-side under identical ncu 2025.3.1 sections.
