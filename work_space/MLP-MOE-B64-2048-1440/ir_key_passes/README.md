# Key-pass IR dumps — MLP-MOE-B64-2048-1440

## Layout
```
ir_key_passes/
  <kernel_name>/
    add_loop_unroll.mlir          # IR after ttir add_loop_unroll (last make_ttir pass)
    add_lower_mma_before.mlir     # IR immediately before add_lower_mma
    add_lower_mma_after.mlir      # IR immediately after add_lower_mma
    llvm_to_module.ll             # LLVM IR right after llvm.to_module (pre-opt)
    …__vN…                        # additional specializations of the same kernel
```

## Filter
Only kernels whose names contain `matmul`, `topk`, or `reduce` were retained from the full dump.

## How produced
- Bench: `torchrun --nproc-per-node=1 python/triton_kernels/bench/bench_mlp.py` (plain bench from `work_space/run_command.sh`)
- Env: `TRITON_ALWAYS_COMPILE=1 TRITON_PROTON_DISABLE=1 TRITON_KEY_PASS_DUMP_DIR=…`
- Temporary (uncommitted) hooks in `third_party/nvidia/backend/compiler.py` dumped only these key points.

## Kernels included
- `_matmul_NNT_fp32xbf16xbf16_32x128x64x11`
- `_p_matmul_NNT_fp8e4nvxfp8e4nvxfp8e4nv_16x256x128x1`
- `_p_matmul_NNT_fp8e4nvxfp8e4nvxfp8e4nv_16x256x128x1_swiglu`
- `_reduce_forward`
- `_topk_forward`
