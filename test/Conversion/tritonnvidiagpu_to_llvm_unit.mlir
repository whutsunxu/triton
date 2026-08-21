// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90 --initialize-ws-cluster-barriers=compute-capability=90 -reconcile-unrealized-casts | FileCheck %s

#local_subslice_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 0]]}>
#local_subslice_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @local_atomic_subslice_other_cta
  // CHECK: mapa.shared::cluster.u32
  // CHECK: atom.shared::cluster.cluster.relaxed.add.u32
  tt.func @local_atomic_subslice_other_cta(%out: !tt.ptr<i32>, %vals: tensor<2x32xi32, #local_subslice_blocked>) {
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x32xi32, #local_subslice_shared, #ttg.shared_memory, mutable>
    %tile = ttg.memdesc_subslice %src [2, 0] : !ttg.memdesc<4x32xi32, #local_subslice_shared, #ttg.shared_memory, mutable> -> !ttg.memdesc<2x32xi32, #local_subslice_shared, #ttg.shared_memory, mutable, 4x32>
    %idx = arith.constant dense<0> : tensor<2x32xi32, #local_subslice_blocked>
    %old = ttg.local_atomic_scatter_rmw add, %tile[%idx], %vals {axis = 1 : i32} : (!ttg.memdesc<2x32xi32, #local_subslice_shared, #ttg.shared_memory, mutable, 4x32>, tensor<2x32xi32, #local_subslice_blocked>, tensor<2x32xi32, #local_subslice_blocked>) -> tensor<2x32xi32, #local_subslice_blocked>
    %ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2x32x!tt.ptr<i32>, #local_subslice_blocked>
    %offs = arith.constant dense<0> : tensor<2x32xi32, #local_subslice_blocked>
    %out_ptrs = tt.addptr %ptrs, %offs : tensor<2x32x!tt.ptr<i32>, #local_subslice_blocked>, tensor<2x32xi32, #local_subslice_blocked>
    tt.store %out_ptrs, %old : tensor<2x32x!tt.ptr<i32>, #local_subslice_blocked>
    tt.return
  }
}