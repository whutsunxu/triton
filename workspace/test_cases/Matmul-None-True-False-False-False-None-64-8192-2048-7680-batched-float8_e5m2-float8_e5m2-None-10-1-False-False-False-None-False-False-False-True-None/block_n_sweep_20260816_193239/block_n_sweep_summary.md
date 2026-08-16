# block_n sweep: 256 vs 128 vs 64

**Date:** 2026-08-16  
**Fixed:** `block_m=64`, `is_persistent=True`, `num_warps=8` → launch **block 384**, same case `8192×2048×7680` FP8 batched  
**Folder:** `block_n_sweep_20260816_193239/`

Goal: see if smaller `block_n` cuts SMEM enough for better occupancy / latency hiding.

---

## Nsys CUPTI kernel time (primary)

| block_n | Kernel tile | blockX | regs | dyn SMEM | MMA warpsPerCTA | **nsys time** | vs BN=256 |
|--------:|-------------|-------:|-----:|---------:|-----------------|-------------:|----------:|
| **256** (prior) | `64×256×128` | 384 | 168 | ~97 KiB | `[1, 8]` | **~33.35 ms** | baseline |
| **128** | `64×128×128` | 384 | 168 | **100.5 KiB** | `[2, 4]` | **39.24 ms** | **+18%** |
| **64** | `64×64×128` | 384 | 168 | **69.8 KiB** | `[2, 4]` | **52.05 ms** | **+56%** |

Raw CUPTI `(end-start)` ns: BN128=`39242707`, BN64=`52048118`.

Both new configs **PASSED** correctness.

---

## NCU (same kernels; times inflated vs nsys)

| Metric | BN=128 | BN=64 |
|--------|-------:|------:|
| Duration (ncu) | 45.85 ms | 60.77 ms |
| Theor / Achieved occ | 25% / 25% | 25% / 25% |
| Waves / SM | 1.0 | 1.0 |
| Issued warp / sched | 0.22 | 0.20 |
| Eligible warps / sched | 0.28 | 0.26 |
| DRAM throughput | 4.8% | 3.6% |
| Compute (SM) | 69.9% | 57.3% |

Occupancy stays **25% / 1 CTA per SM** even at BN=64 (~70 KiB SMEM). Still **register-bound** at 168 regs × 384 threads — smaller BN alone does **not** unlock 2 CTAs/SM with `num_warps=8`.

---

## Takeaways

1. **Best tile N remains 256** under `num_warps=8` (~33 ms). Shrinking N hurts arithmetic intensity / tile efficiency more than it helps SMEM.
2. BN=128 does **not** reduce SMEM vs BN=256 (100 KiB vs ~97 KiB) — other buffers / depth dominate; MMA becomes `[2,4]` instead of `[1,8]`.
3. BN=64 does cut SMEM (~70 KiB) but still 1 CTA/SM and **much slower** (~52 ms).
4. Latency / empty-scheduler problem persists (eligible ~0.26–0.28). Next levers are still lower regs / smaller CTA / depth, not smaller BN alone.

---

## Artifacts

- `block_n_128/` — pytest, irs, nsys, ncu  
- `block_n_64/` — same  
- Tip `test_matmul.py` restored after runs (`num_warps` + no env `block_n` hack).
