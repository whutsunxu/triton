#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, CGALayout = [[0, 0]]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16392 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @case_indexed_view_reuse(%arg0: !tt.tensordesc<64x128xf16, #shared>, %arg1: tensor<64x128xf16, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.init_barrier %0, 1 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %2 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttng.fence_mbarrier_init_release_cluster
    ttng.cluster_barrier {relaxed = true}
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %2, %0, %true {multicast} : !tt.tensordesc<64x128xf16, #shared>, !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttng.wait_barrier %0, %c0_i32 deps %2 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttg.barrier local
    ttg.local_store %arg1, %3 : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttg.local_dealloc %3 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.cluster_barrier
    tt.return
  }
}

