# Warp / block-dim sweep — persistent `block_m=64` FP8 matmul

Same workload: batched FP8 8192×2048×7680, `is_persistent=True`, tile 64×256×128.

## Device limits (RTX 5060 Ti)

| Limit | Value |
|-------|------:|
| max_threads_per_block | **1024** |
| max_threads_per_multiprocessor | **1536** |
| SMEM / SM (this kernel) | **~100 KiB → 1 CTA/SM** |

WS launch model: `block_threads ≈ (num_warps + 4) × 32`; `num_warps` must be power-of-2.

| num_warps | Predicted block | Legal? |
|----------:|----------------:|:------:|
| 4 (auto) | 256 | yes |
| 8 | 384 | yes |
| 16 | 640 | yes |
| 32 | **1152** | **no** |
| 48 | 1664 | no |

**Block 1024 / 1536: not reachable** (next power-of-2 asks for 1152 threads; 1536 is SM capacity only).

## Results

| Block | num_warps | MMA layout | Regs | Warps/SM | Occ. | nsys time | vs 256 | Notes |
|------:|----------:|------------|-----:|---------:|-----:|----------:|-------:|-------|
| **256** | auto | `[1,4]` | 255 | 8 | 16.7% | **42.671 ms** | — | baseline |
| **384** | 8 | `[1,8]` (+1D `[8]`) | 168 | 12 | **25.0%** | **33.360 ms** | **-21.8%** | **best** |
| **640** | 16 | `[2,8]` | 96 | 20 | 41.7% | **38.657 ms** | -9.4% | slower than 384 |
| **1024** | 32 | — | — | — | — | FAIL | — | need 1152 thr |
| **1536** | 48 | — | — | — | — | FAIL | — | >block max; not Po2 |

## NCU warp schedule

| Metric | 256 | 384 | 640 |
|--------|----:|----:|----:|
| Active warps / scheduler | 2.00 | **3.00** | 5.00 |
| Eligible / scheduler | 0.12 | **0.20** | 0.45 |
| No eligible | 87.9% | 84.3% | 77.5% |
| SM SOL | ~34% | **43.4%** | ~71% |
| Mem SOL | ~34% | **48.1%** | ~75% |
| Top stalls | barrier / longSB / math | **math 6.53** / barrier 3.97 / longSB 2.77 | math / MIO |

## Takeaways

1. **Sweet spot = block 384** (`num_warps=8`, MMA `[1,8]`): **33.36 ms** (−22% vs 256).
2. Larger 640 improves occupancy/SOL further but **loses time** vs 384 (MIO + math_pipe).
3. **1024 and 1536 cannot run** on this WS kernel; hardware block cap is 1024 and WS + Po2 jumps to 1152.

## Artifacts

| Block | Path |
|------:|------|
| 256 | `ncu_20260816_142622/`, report nsys |
| 384 | `warp_sweep_summary_20260816_150105/num_warps8/` |
| 640 | `rerun_num_warps16_20260816_145507/` |
| fails | `warp_sweep_limits_*/probes.log` |
