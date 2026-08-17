// Control case for triton-lang/triton#11328: TMA dest is the local_alloc
// itself (not a memdesc_index view). ClusterBarrierInsertion *does* emit
// ttng.cluster_barrier between the reuse alloc and the overlapping store.

#blockedTmaDst = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#nvmmaTma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#barrierEncTma = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @case_direct_alloc_reuse(%desc: !tt.tensordesc<64x128xf16, #nvmmaTma>, %v: tensor<64x128xf16, #blockedTmaDst>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.async_tma_copy_global_to_local %desc[%c0, %c0] %dst, %barrier, %true {multicast} :
        !tt.tensordesc<64x128xf16, #nvmmaTma>, !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttng.wait_barrier %barrier, %c0 deps %dst :
        !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>,
        !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %dst : !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    %reuse = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_store %v, %reuse : tensor<64x128xf16, #blockedTmaDst> -> !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %reuse : !ttg.memdesc<64x128xf16, #nvmmaTma, #smem, mutable>
    ttg.local_dealloc %barrier : !ttg.memdesc<2xi64, #barrierEncTma, #smem, mutable>
    tt.return
  }
}
