# Experiment: force more warps (`num_warps=16`) on persistent `block_m=64`

## Setup

- Same case: batched FP8 `8192×2048×7680`, `is_persistent=True`, `block_m=64`
- Constraint: `num_warps=16` (was auto / effectively CTA **256** threads)
- Goal check: IR `warpsPerCTA`, nsys time vs baseline, ncu warp schedule

## 1. IR — `warpsPerCTA`

From `irs/02_after_ttnvgpuir_add_lower_mma.ttgir`:

```text
#mma = #ttg.nvidia_mma<{..., warpsPerCTA = [2, 8], instrShape = [16, 8]}>
#blocked  ... warpsPerCTA = [1, 16]
#blocked1 ... warpsPerCTA = [16]          # 1-D
WS partition0 ... num_warps(2)            # TMA producers
```

| | Baseline (prior) | This run |
|--|--|--|
| MMA `warpsPerCTA` | **`[1, 4]`** (4 MMA warps) | **`[2, 8]`** (16 MMA warps; **8 along N**) |
| 1-D blocked | `[4]` / total CTA 8 warps | **`[16]`** |
| Literal `[8]` 1-D MMA | — | **Not emitted**; N-extent **8** appears as `[2, 8]` |

So the “more warps on slot_n” intent is satisfied via **`[2, 8]`**, not a 1-D `[8]`.

## 2. Nsight Systems — kernel time

Target: `_p_matmul_NNN_fp8e5xfp8e5xfp8e5_64x256x128x1`

| Build | GPU time | Δ vs report |
|-------|----------|-------------|
| Report baseline (block **256**) | **42.671 ms** | — |
| Earlier rerun (block 256) | 43.408 ms | +1.7% |
| **This run** | **38.657 ms** | **-9.4%** (−4.01 ms) |

CUPTI launch: grid **(36,1,1)**, block **(640,1,1)** = **20 warps**, regs **96**, dyn SMEM **98372 B**.

Note: requesting `num_warps=16` did **not** yield block 512; with WS the launched CTA is **640 threads (20 warps)**. Still a real win vs baseline.

**Verdict: yes — clear optimization (~9% faster).**

## 3. NCU — warp schedule vs prior (block 256 / 8 warps)

| Metric | Baseline ncu (`num_warps`~8) | This run (`num_warps=16` → 20 warps) |
|--------|------------------------------|--------------------------------------|
| Block size | 256 (8 warps) | **640 (20 warps)** |
| Regs/thread | 255 | **96** |
| Achieved occupancy | 16.7% (8/48) | **41.7% (20/48)** |
| Active warps / scheduler | 2.00 | **5.00** |
| Eligible warps / scheduler | 0.12 | **0.45** |
| No eligible | 87.9% | **77.5%** |
| Issue slots busy | 12.1% | **22.5%** |
| Compute (SM) SOL | ~34% | **~71%** |
| Memory SOL | ~34% | **~75%** |
| Tensor pipe (elapsed) | 28.5% | **32.1%** |
| Est. local speedup (scheduler rule) | ~66% | **~25%** (less headroom left) |

### Stall mix (Warp State, inst units)

| Stall | Baseline (8 warps) | This (20 warps) |
|-------|--------------------|-----------------|
| Barrier | **4.28** (largest) | 3.18 |
| Long scoreboard | 3.32 | 2.25 |
| Math pipe throttle | 3.06 | **6.22** (largest) |
| Wait | 2.79 | 1.64 |
| MIO throttle | 0.89 | **5.84** (2nd) |

**Schedule takeaway:** occupancy and eligibility improved a lot (2→5 active warps/scheduler; 0.12→0.45 eligible). Bottleneck shifts from **barrier/scoreboard** toward **math_pipe + MIO throttle** — consistent with feeding the tensor/LSU pipes harder.

## Artifacts

- `irs/` — IR dumps including MMA `[2, 8]`
- `test_matmul_nsys.*` — timing
- `test_matmul_ncu_warp.*` — Launch/Occupancy/Scheduler/WarpState/SOL
