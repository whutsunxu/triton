// kernel: _p_matmul_NNT_fp8e4nvxfp8e4nvxfp8e4nv_16x256x128x1_swiglu
// pass: llvm_to_module
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@global_smem = external addrspace(3) global [0 x i8], align 16

define ptx_kernel void @_p_matmul_NNT_fp8e4nvxfp8e4nvxfp8e4nv_16x256x128x1_swiglu(ptr byval([128 x i8]) align 64 "nvvm.grid_constant" %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i64 %6, i64 %7, i64 %8, i64 %9, i64 %10, ptr addrspace(1) %11, i32 %12, i32 %13, i32 %14, ptr byval([128 x i8]) align 64 "nvvm.grid_constant" %15, i32 %16, i32 %17, i32 %18, i32 %19, i32 %20, i64 %21, i64 %22, i64 %23, i64 %24, i64 %25, ptr addrspace(1) %26, i32 %27, i32 %28, ptr byval([128 x i8]) align 64 "nvvm.grid_constant" %29, i32 %30, i32 %31, i32 %32, i64 %33, i64 %34, i64 %35, ptr addrspace(1) %36, i32 %37, i32 %38, ptr addrspace(1) %39, ptr addrspace(1) %40, i32 %41, i32 %42, i32 %43, i32 %44, ptr addrspace(1) %45, ptr addrspace(1) %46, ptr addrspace(1) %47, ptr addrspace(1) %48, i32 %49, i32 %50, float %51, float %52, i32 %53, ptr addrspace(1) %54, ptr addrspace(1) %55) #0 !dbg !3 {
  %57 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !14
  %58 = udiv i32 %57, 32, !dbg !14
  %59 = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %58, i32 0, i32 31), !dbg !14
  %60 = icmp ult i32 %59, 4, !dbg !14
  br i1 %60, label %173, label %61, !dbg !14

61:                                               ; preds = %65, %171, %172, %56
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 256), !dbg !14
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !14
  %62 = sub i32 %59, 4, !dbg !14
  %63 = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88128), i32 %62, !dbg !14
  %64 = load i8, ptr addrspace(3) %63, align 1, !dbg !14
  switch i8 %64, label %65 [
    i8 0, label %67
    i8 1, label %172
    i8 2, label %66
  ], !dbg !14

65:                                               ; preds = %61
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !14
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !14
  br label %61, !dbg !14, !llvm.loop !15

66:                                               ; preds = %61
  ret void, !dbg !14

67:                                               ; preds = %61
  call void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 24), !dbg !17
  %68 = load i32, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), align 1, !dbg !17
  %69 = load i32, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65540), align 1, !dbg !17
  %70 = load i32, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65544), align 1, !dbg !17
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !17
  %71 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !18
  %72 = sub i32 %71, 36, !dbg !19
  %73 = add i32 %43, 127, !dbg !20
  %74 = sdiv i32 %73, 128, !dbg !25
  %75 = call i32 @llvm.smax.i32(i32 %74, i32 1), !dbg !26
  %76 = call i32 @llvm.smax.i32(i32 %75, i32 1), !dbg !17
  %77 = sub i32 %76, 1, !dbg !17
  %78 = icmp sgt i32 %75, 0, !dbg !27
  %79 = mul i32 %50, 8, !dbg !28
  br label %80, !dbg !17

80:                                               ; preds = %123, %67
  %81 = phi i32 [ %170, %123 ], [ 0, %67 ], !dbg !17
  %82 = phi i32 [ %169, %123 ], [ 0, %67 ]
  %83 = phi i32 [ %129, %123 ], [ %72, %67 ], !dbg !19
  %84 = phi i32 [ %166, %123 ], [ 0, %67 ]
  %85 = phi i32 [ %124, %123 ], [ 0, %67 ]
  %86 = phi i32 [ %125, %123 ], [ 0, %67 ]
  %87 = phi i32 [ %126, %123 ], [ 0, %67 ]
  %88 = phi i32 [ %127, %123 ], [ 0, %67 ]
  %89 = phi i32 [ %128, %123 ], [ 0, %67 ]
  %90 = phi i32 [ %136, %123 ], [ 1, %67 ]
  %91 = phi i32 [ %138, %123 ], [ 0, %67 ]
  %92 = icmp slt i32 %81, %70, !dbg !17
  br i1 %92, label %93, label %171, !dbg !17

93:                                               ; preds = %80
  %94 = icmp eq i32 %82, 0, !dbg !17
  %95 = select i1 %94, i32 0, i32 %84, !dbg !17
  br i1 %94, label %96, label %123, !dbg !17

96:                                               ; preds = %93
  %97 = add i32 %83, 36, !dbg !17
  %98 = srem i32 %97, %68, !dbg !33
  %99 = sdiv i32 %98, %79, !dbg !34
  %100 = mul i32 %99, 8, !dbg !35
  %101 = sub i32 %69, %100, !dbg !36
  %102 = call i32 @llvm.smin.i32(i32 %101, i32 8), !dbg !37
  %103 = srem i32 %98, %102, !dbg !38
  %104 = add i32 %100, %103, !dbg !39
  %105 = srem i32 %98, %79, !dbg !40
  %106 = sdiv i32 %105, %102, !dbg !41
  %107 = getelementptr i32, ptr addrspace(1) %48, i32 %104, !dbg !42
  %108 = call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l"(ptr addrspace(1) %107), !dbg !44
  %109 = bitcast i32 %108 to <1 x i32>, !dbg !44
  %110 = extractelement <1 x i32> %109, i32 0, !dbg !44
  %111 = and i32 %110, 65535, !dbg !45
  %112 = ashr i32 %110, 16, !dbg !46
  %113 = getelementptr i32, ptr addrspace(1) %46, i32 %111, !dbg !47
  %114 = call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l"(ptr addrspace(1) %113), !dbg !48
  %115 = bitcast i32 %114 to <1 x i32>, !dbg !48
  %116 = extractelement <1 x i32> %115, i32 0, !dbg !48
  %117 = mul i32 %112, 16, !dbg !49
  %118 = getelementptr i32, ptr addrspace(1) %45, i32 %111, !dbg !50
  %119 = call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l"(ptr addrspace(1) %118), !dbg !51
  %120 = bitcast i32 %119 to <1 x i32>, !dbg !51
  %121 = extractelement <1 x i32> %120, i32 0, !dbg !51
  %122 = mul i32 %106, 256, !dbg !52
  call void @llvm.assume(i1 %78), !dbg !53
  br label %123, !dbg !17

123:                                              ; preds = %96, %93
  %124 = phi i32 [ %111, %96 ], [ %85, %93 ], !dbg !17
  %125 = phi i32 [ %116, %96 ], [ %86, %93 ], !dbg !17
  %126 = phi i32 [ %117, %96 ], [ %87, %93 ], !dbg !17
  %127 = phi i32 [ %121, %96 ], [ %88, %93 ], !dbg !17
  %128 = phi i32 [ %122, %96 ], [ %89, %93 ], !dbg !17
  %129 = phi i32 [ %97, %96 ], [ %83, %93 ], !dbg !17
  %130 = mul i32 %95, 128, !dbg !54
  %131 = sub i32 1073741824, %127, !dbg !55
  %132 = add i32 %131, %126, !dbg !55
  %133 = add i32 %125, %127, !dbg !60
  %134 = add i32 %90, 1, !dbg !61
  %135 = icmp eq i32 %134, 2, !dbg !61
  %136 = select i1 %135, i32 0, i32 %134, !dbg !61
  %137 = xor i32 %91, 1, !dbg !61
  %138 = select i1 %135, i32 %137, i32 %91, !dbg !61
  %139 = mul i32 %136, 1, !dbg !61
  %140 = getelementptr i64, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88064), i32 %139, !dbg !61
  call void asm sideeffect "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r"(ptr addrspace(3) %140, i32 %138), !dbg !61
  %141 = mul i32 %136, 2048, !dbg !61
  %142 = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 81920), i32 %141, !dbg !61
  %143 = getelementptr i64, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88080), i32 %139, !dbg !61
  %144 = sub i32 %57, 128, !dbg !61
  %145 = and i32 %144, 63, !dbg !61
  %146 = icmp eq i32 %145, 0, !dbg !61
  %147 = and i1 %146, true, !dbg !61
  call void asm sideeffect "@$0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], 2048;", "b,r"(i1 %147, ptr addrspace(3) %143), !dbg !61
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 2, i32 64), !dbg !61
  %148 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %149 = extractvalue { i32, i1 } %148, 1, !dbg !61
  %150 = and i1 true, %149, !dbg !61
  %151 = icmp ult i32 %145, 32, !dbg !61
  %152 = and i1 %150, %151, !dbg !61
  %153 = getelementptr i8, ptr addrspace(3) %142, i32 0, !dbg !61
  %154 = add i32 %130, 0, !dbg !61
  %155 = add i32 %132, 0, !dbg !61
  call void asm sideeffect "@$0 cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4, $5, $6, $7}], [$8];", "b,r,l,r,r,r,r,r,r"(i1 %152, ptr addrspace(3) %153, ptr %15, i32 %154, i32 %155, i32 0, i32 %133, i32 1073741824, ptr addrspace(3) %143), !dbg !61
  %156 = getelementptr i64, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88096), i32 %139, !dbg !62
  call void asm sideeffect "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r"(ptr addrspace(3) %156, i32 %138), !dbg !62
  %157 = mul i32 %136, 32768, !dbg !62
  %158 = getelementptr i8, ptr addrspace(3) @global_smem, i32 %157, !dbg !62
  %159 = getelementptr i64, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88112), i32 %139, !dbg !62
  call void asm sideeffect "@$0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], 32768;", "b,r"(i1 %147, ptr addrspace(3) %159), !dbg !62
  call void @llvm.nvvm.fence.proxy.async.shared_cta(), !dbg !62
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 2, i32 64), !dbg !62
  %160 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %161 = extractvalue { i32, i1 } %160, 1, !dbg !62
  %162 = and i1 true, %161, !dbg !62
  %163 = and i1 %162, %151, !dbg !62
  %164 = getelementptr i8, ptr addrspace(3) %158, i32 0, !dbg !62
  %165 = add i32 %128, 0, !dbg !62
  call void asm sideeffect "@$0 cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4, $5}], [$6];", "b,r,l,r,r,r,r"(i1 %163, ptr addrspace(3) %164, ptr %29, i32 %154, i32 %165, i32 %124, ptr addrspace(3) %159), !dbg !62
  %166 = add i32 %95, 1, !dbg !17
  %167 = add i32 %82, 1, !dbg !17
  %168 = icmp eq i32 %82, %77, !dbg !17
  %169 = select i1 %168, i32 0, i32 %167, !dbg !17
  %170 = add i32 %81, 1, !dbg !17
  br label %80, !dbg !17

171:                                              ; preds = %80
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !17
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 256), !dbg !17
  br label %61, !dbg !17

172:                                              ; preds = %61
  call void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 24), !dbg !17
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !17
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !17
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 256), !dbg !17
  br label %61, !dbg !17

173:                                              ; preds = %56
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 256), !dbg !14
  %174 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !18
  %175 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y(), !dbg !18
  %176 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z(), !dbg !18
  %177 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x(), !dbg !18
  %178 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y(), !dbg !18
  %179 = mul i32 %176, %178, !dbg !18
  %180 = add i32 %175, %179, !dbg !18
  %181 = mul i32 %180, %177, !dbg !18
  %182 = add i32 %174, %181, !dbg !18
  %183 = mul i32 %182, 128, !dbg !18
  %184 = add i32 %183, 0, !dbg !18
  %185 = getelementptr i8, ptr addrspace(1) %54, i32 %184, !dbg !18
  %186 = and i32 %57, 127, !dbg !18
  %187 = icmp slt i32 %186, 32, !dbg !18
  %188 = getelementptr i32, ptr addrspace(3) @global_smem, i32 %186, !dbg !18
  call void asm sideeffect "@$2 st.shared::cta.b32 [ $0 + 0 ], $1;", "r,r,b"(ptr addrspace(3) %188, <1 x i32> zeroinitializer, i1 %187), !dbg !18
  call void @llvm.nvvm.bar.warp.sync(i32 -1), !dbg !18
  %189 = icmp eq i32 %186, 0, !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_address.shared::cta.b1024.b64 [ $0 + 0 ], $1;", "l,l,b"(ptr addrspace(3) @global_smem, ptr addrspace(1) %11, i1 %189), !dbg !18
  call void asm sideeffect "@$1 tensormap.replace.tile.rank.shared::cta.b1024.b32 [ $0 + 0 ], 0x4;", "l,b"(ptr addrspace(3) @global_smem, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 128, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 16, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x2, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 1, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x3, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 1, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x4, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 1, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 %5, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 %4, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x2, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 %3, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x3, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 %2, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [ $0 + 0 ], 0x4, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 %1, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [ $0 + 0 ], 0x0, $1;", "l,l,b"(ptr addrspace(3) @global_smem, i64 %9, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [ $0 + 0 ], 0x1, $1;", "l,l,b"(ptr addrspace(3) @global_smem, i64 %8, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [ $0 + 0 ], 0x2, $1;", "l,l,b"(ptr addrspace(3) @global_smem, i64 %7, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [ $0 + 0 ], 0x3, $1;", "l,l,b"(ptr addrspace(3) @global_smem, i64 %6, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x0, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 1, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x1, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 1, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x2, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 1, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x3, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 1, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [ $0 + 0 ], 0x4, $1;", "l,r,b"(ptr addrspace(3) @global_smem, i32 1, i1 %189), !dbg !18
  call void asm sideeffect "@$1 tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;", "l,b"(ptr addrspace(3) @global_smem, i1 %189), !dbg !18
  call void asm sideeffect "@$1 tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;", "l,b"(ptr addrspace(3) @global_smem, i1 %189), !dbg !18
  call void asm sideeffect "@$1 tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x3;", "l,b"(ptr addrspace(3) @global_smem, i1 %189), !dbg !18
  call void asm sideeffect "@$1 tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [ $0 + 0 ], 0x0;", "l,b"(ptr addrspace(3) @global_smem, i1 %189), !dbg !18
  call void asm sideeffect "@$2 tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [ $0 + 0 ], [ $1 + 0 ], 0x80;", "l,l,b"(ptr addrspace(1) %185, ptr addrspace(3) @global_smem, i1 %187), !dbg !18
  call void asm sideeffect "@$1 fence.proxy.tensormap::generic.acquire.gpu [ $0 + 0 ], 0x80;\0A\09@$2 cp.async.bulk.commit_group ;\0A\09@$3 cp.async.bulk.wait_group.read 0 ;", "l,b,b,b"(ptr addrspace(1) %185, i1 %187, i1 %187, i1 %187), !dbg !18
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !18
  %190 = addrspacecast ptr addrspace(1) %185 to ptr, !dbg !18
  %191 = getelementptr i32, ptr addrspace(1) %47, i32 128, !dbg !63
  %192 = call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l"(ptr addrspace(1) %191), !dbg !64
  %193 = bitcast i32 %192 to <1 x i32>, !dbg !64
  %194 = extractelement <1 x i32> %193, i32 0, !dbg !64
  %195 = mul i32 %194, %50, !dbg !65
  %196 = sub i32 %174, 36, !dbg !19
  %197 = mul i32 %50, 8, !dbg !28
  %198 = add i32 %43, 127, !dbg !20
  %199 = sdiv i32 %198, 128, !dbg !25
  %200 = call i32 @llvm.smax.i32(i32 %199, i32 1), !dbg !26
  %201 = icmp sgt i32 %200, 0, !dbg !27
  %202 = urem i32 %186, 32, !dbg !66
  %203 = shl i32 %202, 0, !dbg !66
  %204 = or i32 0, %203, !dbg !66
  %205 = shl i32 %58, 5, !dbg !66
  %206 = or i32 %204, %205, !dbg !66
  %207 = and i32 %206, 127, !dbg !66
  %208 = shl i32 %207, 1, !dbg !66
  %209 = or disjoint i32 %208, 0, !dbg !66
  %210 = xor i32 0, %209, !dbg !66
  %211 = xor i32 %210, 0, !dbg !66
  %212 = add i32 %211, 0, !dbg !66
  %213 = fsub float 0.000000e+00, %52, !dbg !67
  %214 = fsub float 0.000000e+00, %51, !dbg !73
  %215 = sub i32 %195, %174, !dbg !17
  %216 = sdiv i32 %215, 36, !dbg !17
  %217 = mul i32 %216, 36, !dbg !17
  %218 = icmp ne i32 %215, %217, !dbg !17
  %219 = icmp slt i32 %215, 0, !dbg !17
  %220 = icmp eq i1 %219, false, !dbg !17
  %221 = and i1 %218, %220, !dbg !17
  %222 = add i32 %216, 1, !dbg !17
  %223 = select i1 %221, i32 %222, i32 %216, !dbg !17
  %224 = call i32 @llvm.smax.i32(i32 %200, i32 1), !dbg !17
  %225 = mul i32 %223, %224, !dbg !17
  %226 = sub i32 %224, 1, !dbg !17
  %227 = icmp eq i32 %58, 0, !dbg !61
  %228 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %229 = extractvalue { i32, i1 } %228, 1, !dbg !61
  %230 = and i1 %227, %229, !dbg !61
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %230, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88064)), !dbg !61
  %231 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %232 = extractvalue { i32, i1 } %231, 1, !dbg !61
  %233 = and i1 %227, %232, !dbg !61
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %233, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88072)), !dbg !61
  %234 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %235 = extractvalue { i32, i1 } %234, 1, !dbg !61
  %236 = and i1 %227, %235, !dbg !61
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %236, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88080)), !dbg !61
  %237 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %238 = extractvalue { i32, i1 } %237, 1, !dbg !61
  %239 = and i1 %227, %238, !dbg !61
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %239, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88088)), !dbg !61
  %240 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %241 = extractvalue { i32, i1 } %240, 1, !dbg !62
  %242 = and i1 %227, %241, !dbg !62
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %242, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88096)), !dbg !62
  %243 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %244 = extractvalue { i32, i1 } %243, 1, !dbg !62
  %245 = and i1 %227, %244, !dbg !62
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %245, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88104)), !dbg !62
  %246 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %247 = extractvalue { i32, i1 } %246, 1, !dbg !62
  %248 = and i1 %227, %247, !dbg !62
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %248, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88112)), !dbg !62
  %249 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %250 = extractvalue { i32, i1 } %249, 1, !dbg !62
  %251 = and i1 %227, %250, !dbg !62
  call void asm sideeffect "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r"(i1 %251, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88120)), !dbg !62
  store i8 0, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88128), align 1, !dbg !17
  store i8 0, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88129), align 1, !dbg !17
  store i8 1, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88130), align 1, !dbg !17
  store i8 1, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88131), align 1, !dbg !17
  store i32 %195, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), align 1, !dbg !17
  store i32 %194, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65540), align 1, !dbg !17
  store i32 %225, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65544), align 1, !dbg !17
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !17
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 256), !dbg !17
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !17
  br label %252, !dbg !17

252:                                              ; preds = %173
  br label %253, !dbg !17

253:                                              ; preds = %2403, %252
  %254 = phi i32 [ %2408, %2403 ], [ 0, %252 ], !dbg !17
  %255 = phi i32 [ %2407, %2403 ], [ 0, %252 ]
  %256 = phi i32 [ %278, %2403 ], [ %196, %252 ], !dbg !19
  %257 = phi i32 [ %2404, %2403 ], [ %196, %252 ], !dbg !19
  %258 = phi { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } [ %2405, %2403 ], [ zeroinitializer, %252 ]
  %259 = phi ptr addrspace(1) [ %277, %2403 ], [ poison, %252 ]
  %260 = phi i32 [ %281, %2403 ], [ 1, %252 ]
  %261 = phi i32 [ %283, %2403 ], [ 1, %252 ]
  %262 = icmp slt i32 %254, %225, !dbg !17
  br i1 %262, label %263, label %2409, !dbg !17

263:                                              ; preds = %253
  %264 = icmp eq i32 %255, 0, !dbg !17
  br i1 %264, label %265, label %276, !dbg !17

265:                                              ; preds = %263
  %266 = add i32 %256, 36, !dbg !17
  %267 = srem i32 %266, %195, !dbg !33
  %268 = sdiv i32 %267, %197, !dbg !34
  %269 = mul i32 %268, 8, !dbg !35
  %270 = sub i32 %194, %269, !dbg !36
  %271 = call i32 @llvm.smin.i32(i32 %270, i32 8), !dbg !37
  %272 = icmp sge i32 %271, 0, !dbg !74
  call void @llvm.assume(i1 %272), !dbg !75
  %273 = srem i32 %267, %271, !dbg !38
  %274 = add i32 %269, %273, !dbg !39
  %275 = getelementptr i32, ptr addrspace(1) %48, i32 %274, !dbg !42
  call void @llvm.assume(i1 %201), !dbg !53
  br label %276, !dbg !17

276:                                              ; preds = %265, %263
  %277 = phi ptr addrspace(1) [ %275, %265 ], [ %259, %263 ], !dbg !17
  %278 = phi i32 [ %266, %265 ], [ %256, %263 ], !dbg !17
  %279 = add i32 %260, 1, !dbg !61
  %280 = icmp eq i32 %279, 2, !dbg !61
  %281 = select i1 %280, i32 0, i32 %279, !dbg !61
  %282 = xor i32 %261, 1, !dbg !61
  %283 = select i1 %280, i32 %282, i32 %261, !dbg !61
  %284 = mul i32 %281, 1, !dbg !61
  %285 = getelementptr i64, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88080), i32 %284, !dbg !61
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !61
  call void asm sideeffect "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r"(ptr addrspace(3) %285, i32 %283), !dbg !61
  %286 = mul i32 %281, 2048, !dbg !61
  %287 = getelementptr i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 81920), i32 %286, !dbg !61
  %288 = and i32 %206, 7, !dbg !76
  %289 = shl i32 %288, 7, !dbg !76
  %290 = shl i32 %288, 4, !dbg !76
  %291 = and i32 %206, 24, !dbg !76
  %292 = shl i32 %291, 1, !dbg !76
  %293 = xor i32 %289, %290, !dbg !76
  %294 = xor i32 %293, %292, !dbg !76
  %295 = or disjoint i32 0, %294, !dbg !76
  %296 = xor i32 0, %295, !dbg !76
  %297 = xor i32 %296, 0, !dbg !76
  %298 = xor i32 %297, 0, !dbg !76
  %299 = add i32 %298, 0, !dbg !76
  %300 = getelementptr inbounds i8, ptr addrspace(3) %287, i32 %299, !dbg !76
  %301 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %300), !dbg !76
  %302 = extractvalue { i32, i32, i32, i32 } %301, 0, !dbg !76
  %303 = bitcast i32 %302 to <4 x i8>, !dbg !76
  %304 = extractelement <4 x i8> %303, i32 0, !dbg !76
  %305 = extractelement <4 x i8> %303, i32 1, !dbg !76
  %306 = extractelement <4 x i8> %303, i32 2, !dbg !76
  %307 = extractelement <4 x i8> %303, i32 3, !dbg !76
  %308 = extractvalue { i32, i32, i32, i32 } %301, 1, !dbg !76
  %309 = bitcast i32 %308 to <4 x i8>, !dbg !76
  %310 = extractelement <4 x i8> %309, i32 0, !dbg !76
  %311 = extractelement <4 x i8> %309, i32 1, !dbg !76
  %312 = extractelement <4 x i8> %309, i32 2, !dbg !76
  %313 = extractelement <4 x i8> %309, i32 3, !dbg !76
  %314 = extractvalue { i32, i32, i32, i32 } %301, 2, !dbg !76
  %315 = bitcast i32 %314 to <4 x i8>, !dbg !76
  %316 = extractelement <4 x i8> %315, i32 0, !dbg !76
  %317 = extractelement <4 x i8> %315, i32 1, !dbg !76
  %318 = extractelement <4 x i8> %315, i32 2, !dbg !76
  %319 = extractelement <4 x i8> %315, i32 3, !dbg !76
  %320 = extractvalue { i32, i32, i32, i32 } %301, 3, !dbg !76
  %321 = bitcast i32 %320 to <4 x i8>, !dbg !76
  %322 = extractelement <4 x i8> %321, i32 0, !dbg !76
  %323 = extractelement <4 x i8> %321, i32 1, !dbg !76
  %324 = extractelement <4 x i8> %321, i32 2, !dbg !76
  %325 = extractelement <4 x i8> %321, i32 3, !dbg !76
  %326 = add i32 %298, 1024, !dbg !76
  %327 = getelementptr inbounds i8, ptr addrspace(3) %287, i32 %326, !dbg !76
  %328 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %327), !dbg !76
  %329 = extractvalue { i32, i32, i32, i32 } %328, 0, !dbg !76
  %330 = bitcast i32 %329 to <4 x i8>, !dbg !76
  %331 = extractelement <4 x i8> %330, i32 0, !dbg !76
  %332 = extractelement <4 x i8> %330, i32 1, !dbg !76
  %333 = extractelement <4 x i8> %330, i32 2, !dbg !76
  %334 = extractelement <4 x i8> %330, i32 3, !dbg !76
  %335 = extractvalue { i32, i32, i32, i32 } %328, 1, !dbg !76
  %336 = bitcast i32 %335 to <4 x i8>, !dbg !76
  %337 = extractelement <4 x i8> %336, i32 0, !dbg !76
  %338 = extractelement <4 x i8> %336, i32 1, !dbg !76
  %339 = extractelement <4 x i8> %336, i32 2, !dbg !76
  %340 = extractelement <4 x i8> %336, i32 3, !dbg !76
  %341 = extractvalue { i32, i32, i32, i32 } %328, 2, !dbg !76
  %342 = bitcast i32 %341 to <4 x i8>, !dbg !76
  %343 = extractelement <4 x i8> %342, i32 0, !dbg !76
  %344 = extractelement <4 x i8> %342, i32 1, !dbg !76
  %345 = extractelement <4 x i8> %342, i32 2, !dbg !76
  %346 = extractelement <4 x i8> %342, i32 3, !dbg !76
  %347 = extractvalue { i32, i32, i32, i32 } %328, 3, !dbg !76
  %348 = bitcast i32 %347 to <4 x i8>, !dbg !76
  %349 = extractelement <4 x i8> %348, i32 0, !dbg !76
  %350 = extractelement <4 x i8> %348, i32 1, !dbg !76
  %351 = extractelement <4 x i8> %348, i32 2, !dbg !76
  %352 = extractelement <4 x i8> %348, i32 3, !dbg !76
  %353 = xor i32 %297, 64, !dbg !76
  %354 = add i32 %353, 0, !dbg !76
  %355 = getelementptr inbounds i8, ptr addrspace(3) %287, i32 %354, !dbg !76
  %356 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %355), !dbg !76
  %357 = extractvalue { i32, i32, i32, i32 } %356, 0, !dbg !76
  %358 = bitcast i32 %357 to <4 x i8>, !dbg !76
  %359 = extractelement <4 x i8> %358, i32 0, !dbg !76
  %360 = extractelement <4 x i8> %358, i32 1, !dbg !76
  %361 = extractelement <4 x i8> %358, i32 2, !dbg !76
  %362 = extractelement <4 x i8> %358, i32 3, !dbg !76
  %363 = extractvalue { i32, i32, i32, i32 } %356, 1, !dbg !76
  %364 = bitcast i32 %363 to <4 x i8>, !dbg !76
  %365 = extractelement <4 x i8> %364, i32 0, !dbg !76
  %366 = extractelement <4 x i8> %364, i32 1, !dbg !76
  %367 = extractelement <4 x i8> %364, i32 2, !dbg !76
  %368 = extractelement <4 x i8> %364, i32 3, !dbg !76
  %369 = extractvalue { i32, i32, i32, i32 } %356, 2, !dbg !76
  %370 = bitcast i32 %369 to <4 x i8>, !dbg !76
  %371 = extractelement <4 x i8> %370, i32 0, !dbg !76
  %372 = extractelement <4 x i8> %370, i32 1, !dbg !76
  %373 = extractelement <4 x i8> %370, i32 2, !dbg !76
  %374 = extractelement <4 x i8> %370, i32 3, !dbg !76
  %375 = extractvalue { i32, i32, i32, i32 } %356, 3, !dbg !76
  %376 = bitcast i32 %375 to <4 x i8>, !dbg !76
  %377 = extractelement <4 x i8> %376, i32 0, !dbg !76
  %378 = extractelement <4 x i8> %376, i32 1, !dbg !76
  %379 = extractelement <4 x i8> %376, i32 2, !dbg !76
  %380 = extractelement <4 x i8> %376, i32 3, !dbg !76
  %381 = add i32 %353, 1024, !dbg !76
  %382 = getelementptr inbounds i8, ptr addrspace(3) %287, i32 %381, !dbg !76
  %383 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %382), !dbg !76
  %384 = extractvalue { i32, i32, i32, i32 } %383, 0, !dbg !76
  %385 = bitcast i32 %384 to <4 x i8>, !dbg !76
  %386 = extractelement <4 x i8> %385, i32 0, !dbg !76
  %387 = extractelement <4 x i8> %385, i32 1, !dbg !76
  %388 = extractelement <4 x i8> %385, i32 2, !dbg !76
  %389 = extractelement <4 x i8> %385, i32 3, !dbg !76
  %390 = extractvalue { i32, i32, i32, i32 } %383, 1, !dbg !76
  %391 = bitcast i32 %390 to <4 x i8>, !dbg !76
  %392 = extractelement <4 x i8> %391, i32 0, !dbg !76
  %393 = extractelement <4 x i8> %391, i32 1, !dbg !76
  %394 = extractelement <4 x i8> %391, i32 2, !dbg !76
  %395 = extractelement <4 x i8> %391, i32 3, !dbg !76
  %396 = extractvalue { i32, i32, i32, i32 } %383, 2, !dbg !76
  %397 = bitcast i32 %396 to <4 x i8>, !dbg !76
  %398 = extractelement <4 x i8> %397, i32 0, !dbg !76
  %399 = extractelement <4 x i8> %397, i32 1, !dbg !76
  %400 = extractelement <4 x i8> %397, i32 2, !dbg !76
  %401 = extractelement <4 x i8> %397, i32 3, !dbg !76
  %402 = extractvalue { i32, i32, i32, i32 } %383, 3, !dbg !76
  %403 = bitcast i32 %402 to <4 x i8>, !dbg !76
  %404 = extractelement <4 x i8> %403, i32 0, !dbg !76
  %405 = extractelement <4 x i8> %403, i32 1, !dbg !76
  %406 = extractelement <4 x i8> %403, i32 2, !dbg !76
  %407 = extractelement <4 x i8> %403, i32 3, !dbg !76
  call void @llvm.nvvm.fence.proxy.async.shared_cta(), !dbg !61
  %408 = getelementptr i64, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88064), i32 %284, !dbg !61
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !61
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !61
  call void asm sideeffect "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];", "b,r"(i1 %189, ptr addrspace(3) %408), !dbg !61
  %409 = getelementptr i64, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88112), i32 %284, !dbg !62
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !62
  call void asm sideeffect "\0A{\0A\09.reg .pred complete;\0A\09waitLoop:\0A\09mbarrier.try_wait.parity.shared::cta.b64 complete, [$0], $1;\0A\09@!complete bra.uni waitLoop;\0A}\0A", "r,r"(ptr addrspace(3) %409, i32 %283), !dbg !62
  %410 = mul i32 %281, 32768, !dbg !62
  %411 = getelementptr i8, ptr addrspace(3) @global_smem, i32 %410, !dbg !62
  %412 = and i32 %206, 15, !dbg !62
  %413 = shl i32 %412, 7, !dbg !62
  %414 = and i32 %206, 96, !dbg !62
  %415 = shl i32 %414, 6, !dbg !62
  %416 = and i32 %206, 16, !dbg !62
  %417 = xor i32 %413, %290, !dbg !62
  %418 = xor i32 %417, %416, !dbg !62
  %419 = or disjoint i32 %415, %418, !dbg !62
  %420 = xor i32 0, %419, !dbg !62
  %421 = xor i32 %420, 0, !dbg !62
  %422 = xor i32 %421, 0, !dbg !62
  %423 = add i32 %422, 0, !dbg !62
  %424 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %423, !dbg !62
  %425 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %424), !dbg !62
  %426 = extractvalue { i32, i32, i32, i32 } %425, 0, !dbg !62
  %427 = bitcast i32 %426 to <4 x i8>, !dbg !62
  %428 = extractelement <4 x i8> %427, i32 0, !dbg !62
  %429 = extractelement <4 x i8> %427, i32 1, !dbg !62
  %430 = extractelement <4 x i8> %427, i32 2, !dbg !62
  %431 = extractelement <4 x i8> %427, i32 3, !dbg !62
  %432 = extractvalue { i32, i32, i32, i32 } %425, 1, !dbg !62
  %433 = bitcast i32 %432 to <4 x i8>, !dbg !62
  %434 = extractelement <4 x i8> %433, i32 0, !dbg !62
  %435 = extractelement <4 x i8> %433, i32 1, !dbg !62
  %436 = extractelement <4 x i8> %433, i32 2, !dbg !62
  %437 = extractelement <4 x i8> %433, i32 3, !dbg !62
  %438 = extractvalue { i32, i32, i32, i32 } %425, 2, !dbg !62
  %439 = bitcast i32 %438 to <4 x i8>, !dbg !62
  %440 = extractelement <4 x i8> %439, i32 0, !dbg !62
  %441 = extractelement <4 x i8> %439, i32 1, !dbg !62
  %442 = extractelement <4 x i8> %439, i32 2, !dbg !62
  %443 = extractelement <4 x i8> %439, i32 3, !dbg !62
  %444 = extractvalue { i32, i32, i32, i32 } %425, 3, !dbg !62
  %445 = bitcast i32 %444 to <4 x i8>, !dbg !62
  %446 = extractelement <4 x i8> %445, i32 0, !dbg !62
  %447 = extractelement <4 x i8> %445, i32 1, !dbg !62
  %448 = extractelement <4 x i8> %445, i32 2, !dbg !62
  %449 = extractelement <4 x i8> %445, i32 3, !dbg !62
  %450 = add i32 %422, 8192, !dbg !62
  %451 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %450, !dbg !62
  %452 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %451), !dbg !62
  %453 = extractvalue { i32, i32, i32, i32 } %452, 0, !dbg !62
  %454 = bitcast i32 %453 to <4 x i8>, !dbg !62
  %455 = extractelement <4 x i8> %454, i32 0, !dbg !62
  %456 = extractelement <4 x i8> %454, i32 1, !dbg !62
  %457 = extractelement <4 x i8> %454, i32 2, !dbg !62
  %458 = extractelement <4 x i8> %454, i32 3, !dbg !62
  %459 = extractvalue { i32, i32, i32, i32 } %452, 1, !dbg !62
  %460 = bitcast i32 %459 to <4 x i8>, !dbg !62
  %461 = extractelement <4 x i8> %460, i32 0, !dbg !62
  %462 = extractelement <4 x i8> %460, i32 1, !dbg !62
  %463 = extractelement <4 x i8> %460, i32 2, !dbg !62
  %464 = extractelement <4 x i8> %460, i32 3, !dbg !62
  %465 = extractvalue { i32, i32, i32, i32 } %452, 2, !dbg !62
  %466 = bitcast i32 %465 to <4 x i8>, !dbg !62
  %467 = extractelement <4 x i8> %466, i32 0, !dbg !62
  %468 = extractelement <4 x i8> %466, i32 1, !dbg !62
  %469 = extractelement <4 x i8> %466, i32 2, !dbg !62
  %470 = extractelement <4 x i8> %466, i32 3, !dbg !62
  %471 = extractvalue { i32, i32, i32, i32 } %452, 3, !dbg !62
  %472 = bitcast i32 %471 to <4 x i8>, !dbg !62
  %473 = extractelement <4 x i8> %472, i32 0, !dbg !62
  %474 = extractelement <4 x i8> %472, i32 1, !dbg !62
  %475 = extractelement <4 x i8> %472, i32 2, !dbg !62
  %476 = extractelement <4 x i8> %472, i32 3, !dbg !62
  %477 = add i32 %422, 16384, !dbg !62
  %478 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %477, !dbg !62
  %479 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %478), !dbg !62
  %480 = extractvalue { i32, i32, i32, i32 } %479, 0, !dbg !62
  %481 = bitcast i32 %480 to <4 x i8>, !dbg !62
  %482 = extractelement <4 x i8> %481, i32 0, !dbg !62
  %483 = extractelement <4 x i8> %481, i32 1, !dbg !62
  %484 = extractelement <4 x i8> %481, i32 2, !dbg !62
  %485 = extractelement <4 x i8> %481, i32 3, !dbg !62
  %486 = extractvalue { i32, i32, i32, i32 } %479, 1, !dbg !62
  %487 = bitcast i32 %486 to <4 x i8>, !dbg !62
  %488 = extractelement <4 x i8> %487, i32 0, !dbg !62
  %489 = extractelement <4 x i8> %487, i32 1, !dbg !62
  %490 = extractelement <4 x i8> %487, i32 2, !dbg !62
  %491 = extractelement <4 x i8> %487, i32 3, !dbg !62
  %492 = extractvalue { i32, i32, i32, i32 } %479, 2, !dbg !62
  %493 = bitcast i32 %492 to <4 x i8>, !dbg !62
  %494 = extractelement <4 x i8> %493, i32 0, !dbg !62
  %495 = extractelement <4 x i8> %493, i32 1, !dbg !62
  %496 = extractelement <4 x i8> %493, i32 2, !dbg !62
  %497 = extractelement <4 x i8> %493, i32 3, !dbg !62
  %498 = extractvalue { i32, i32, i32, i32 } %479, 3, !dbg !62
  %499 = bitcast i32 %498 to <4 x i8>, !dbg !62
  %500 = extractelement <4 x i8> %499, i32 0, !dbg !62
  %501 = extractelement <4 x i8> %499, i32 1, !dbg !62
  %502 = extractelement <4 x i8> %499, i32 2, !dbg !62
  %503 = extractelement <4 x i8> %499, i32 3, !dbg !62
  %504 = add i32 %422, 24576, !dbg !62
  %505 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %504, !dbg !62
  %506 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %505), !dbg !62
  %507 = extractvalue { i32, i32, i32, i32 } %506, 0, !dbg !62
  %508 = bitcast i32 %507 to <4 x i8>, !dbg !62
  %509 = extractelement <4 x i8> %508, i32 0, !dbg !62
  %510 = extractelement <4 x i8> %508, i32 1, !dbg !62
  %511 = extractelement <4 x i8> %508, i32 2, !dbg !62
  %512 = extractelement <4 x i8> %508, i32 3, !dbg !62
  %513 = extractvalue { i32, i32, i32, i32 } %506, 1, !dbg !62
  %514 = bitcast i32 %513 to <4 x i8>, !dbg !62
  %515 = extractelement <4 x i8> %514, i32 0, !dbg !62
  %516 = extractelement <4 x i8> %514, i32 1, !dbg !62
  %517 = extractelement <4 x i8> %514, i32 2, !dbg !62
  %518 = extractelement <4 x i8> %514, i32 3, !dbg !62
  %519 = extractvalue { i32, i32, i32, i32 } %506, 2, !dbg !62
  %520 = bitcast i32 %519 to <4 x i8>, !dbg !62
  %521 = extractelement <4 x i8> %520, i32 0, !dbg !62
  %522 = extractelement <4 x i8> %520, i32 1, !dbg !62
  %523 = extractelement <4 x i8> %520, i32 2, !dbg !62
  %524 = extractelement <4 x i8> %520, i32 3, !dbg !62
  %525 = extractvalue { i32, i32, i32, i32 } %506, 3, !dbg !62
  %526 = bitcast i32 %525 to <4 x i8>, !dbg !62
  %527 = extractelement <4 x i8> %526, i32 0, !dbg !62
  %528 = extractelement <4 x i8> %526, i32 1, !dbg !62
  %529 = extractelement <4 x i8> %526, i32 2, !dbg !62
  %530 = extractelement <4 x i8> %526, i32 3, !dbg !62
  %531 = xor i32 %421, 32, !dbg !62
  %532 = add i32 %531, 0, !dbg !62
  %533 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %532, !dbg !62
  %534 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %533), !dbg !62
  %535 = extractvalue { i32, i32, i32, i32 } %534, 0, !dbg !62
  %536 = bitcast i32 %535 to <4 x i8>, !dbg !62
  %537 = extractelement <4 x i8> %536, i32 0, !dbg !62
  %538 = extractelement <4 x i8> %536, i32 1, !dbg !62
  %539 = extractelement <4 x i8> %536, i32 2, !dbg !62
  %540 = extractelement <4 x i8> %536, i32 3, !dbg !62
  %541 = extractvalue { i32, i32, i32, i32 } %534, 1, !dbg !62
  %542 = bitcast i32 %541 to <4 x i8>, !dbg !62
  %543 = extractelement <4 x i8> %542, i32 0, !dbg !62
  %544 = extractelement <4 x i8> %542, i32 1, !dbg !62
  %545 = extractelement <4 x i8> %542, i32 2, !dbg !62
  %546 = extractelement <4 x i8> %542, i32 3, !dbg !62
  %547 = extractvalue { i32, i32, i32, i32 } %534, 2, !dbg !62
  %548 = bitcast i32 %547 to <4 x i8>, !dbg !62
  %549 = extractelement <4 x i8> %548, i32 0, !dbg !62
  %550 = extractelement <4 x i8> %548, i32 1, !dbg !62
  %551 = extractelement <4 x i8> %548, i32 2, !dbg !62
  %552 = extractelement <4 x i8> %548, i32 3, !dbg !62
  %553 = extractvalue { i32, i32, i32, i32 } %534, 3, !dbg !62
  %554 = bitcast i32 %553 to <4 x i8>, !dbg !62
  %555 = extractelement <4 x i8> %554, i32 0, !dbg !62
  %556 = extractelement <4 x i8> %554, i32 1, !dbg !62
  %557 = extractelement <4 x i8> %554, i32 2, !dbg !62
  %558 = extractelement <4 x i8> %554, i32 3, !dbg !62
  %559 = add i32 %531, 8192, !dbg !62
  %560 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %559, !dbg !62
  %561 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %560), !dbg !62
  %562 = extractvalue { i32, i32, i32, i32 } %561, 0, !dbg !62
  %563 = bitcast i32 %562 to <4 x i8>, !dbg !62
  %564 = extractelement <4 x i8> %563, i32 0, !dbg !62
  %565 = extractelement <4 x i8> %563, i32 1, !dbg !62
  %566 = extractelement <4 x i8> %563, i32 2, !dbg !62
  %567 = extractelement <4 x i8> %563, i32 3, !dbg !62
  %568 = extractvalue { i32, i32, i32, i32 } %561, 1, !dbg !62
  %569 = bitcast i32 %568 to <4 x i8>, !dbg !62
  %570 = extractelement <4 x i8> %569, i32 0, !dbg !62
  %571 = extractelement <4 x i8> %569, i32 1, !dbg !62
  %572 = extractelement <4 x i8> %569, i32 2, !dbg !62
  %573 = extractelement <4 x i8> %569, i32 3, !dbg !62
  %574 = extractvalue { i32, i32, i32, i32 } %561, 2, !dbg !62
  %575 = bitcast i32 %574 to <4 x i8>, !dbg !62
  %576 = extractelement <4 x i8> %575, i32 0, !dbg !62
  %577 = extractelement <4 x i8> %575, i32 1, !dbg !62
  %578 = extractelement <4 x i8> %575, i32 2, !dbg !62
  %579 = extractelement <4 x i8> %575, i32 3, !dbg !62
  %580 = extractvalue { i32, i32, i32, i32 } %561, 3, !dbg !62
  %581 = bitcast i32 %580 to <4 x i8>, !dbg !62
  %582 = extractelement <4 x i8> %581, i32 0, !dbg !62
  %583 = extractelement <4 x i8> %581, i32 1, !dbg !62
  %584 = extractelement <4 x i8> %581, i32 2, !dbg !62
  %585 = extractelement <4 x i8> %581, i32 3, !dbg !62
  %586 = add i32 %531, 16384, !dbg !62
  %587 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %586, !dbg !62
  %588 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %587), !dbg !62
  %589 = extractvalue { i32, i32, i32, i32 } %588, 0, !dbg !62
  %590 = bitcast i32 %589 to <4 x i8>, !dbg !62
  %591 = extractelement <4 x i8> %590, i32 0, !dbg !62
  %592 = extractelement <4 x i8> %590, i32 1, !dbg !62
  %593 = extractelement <4 x i8> %590, i32 2, !dbg !62
  %594 = extractelement <4 x i8> %590, i32 3, !dbg !62
  %595 = extractvalue { i32, i32, i32, i32 } %588, 1, !dbg !62
  %596 = bitcast i32 %595 to <4 x i8>, !dbg !62
  %597 = extractelement <4 x i8> %596, i32 0, !dbg !62
  %598 = extractelement <4 x i8> %596, i32 1, !dbg !62
  %599 = extractelement <4 x i8> %596, i32 2, !dbg !62
  %600 = extractelement <4 x i8> %596, i32 3, !dbg !62
  %601 = extractvalue { i32, i32, i32, i32 } %588, 2, !dbg !62
  %602 = bitcast i32 %601 to <4 x i8>, !dbg !62
  %603 = extractelement <4 x i8> %602, i32 0, !dbg !62
  %604 = extractelement <4 x i8> %602, i32 1, !dbg !62
  %605 = extractelement <4 x i8> %602, i32 2, !dbg !62
  %606 = extractelement <4 x i8> %602, i32 3, !dbg !62
  %607 = extractvalue { i32, i32, i32, i32 } %588, 3, !dbg !62
  %608 = bitcast i32 %607 to <4 x i8>, !dbg !62
  %609 = extractelement <4 x i8> %608, i32 0, !dbg !62
  %610 = extractelement <4 x i8> %608, i32 1, !dbg !62
  %611 = extractelement <4 x i8> %608, i32 2, !dbg !62
  %612 = extractelement <4 x i8> %608, i32 3, !dbg !62
  %613 = add i32 %531, 24576, !dbg !62
  %614 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %613, !dbg !62
  %615 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %614), !dbg !62
  %616 = extractvalue { i32, i32, i32, i32 } %615, 0, !dbg !62
  %617 = bitcast i32 %616 to <4 x i8>, !dbg !62
  %618 = extractelement <4 x i8> %617, i32 0, !dbg !62
  %619 = extractelement <4 x i8> %617, i32 1, !dbg !62
  %620 = extractelement <4 x i8> %617, i32 2, !dbg !62
  %621 = extractelement <4 x i8> %617, i32 3, !dbg !62
  %622 = extractvalue { i32, i32, i32, i32 } %615, 1, !dbg !62
  %623 = bitcast i32 %622 to <4 x i8>, !dbg !62
  %624 = extractelement <4 x i8> %623, i32 0, !dbg !62
  %625 = extractelement <4 x i8> %623, i32 1, !dbg !62
  %626 = extractelement <4 x i8> %623, i32 2, !dbg !62
  %627 = extractelement <4 x i8> %623, i32 3, !dbg !62
  %628 = extractvalue { i32, i32, i32, i32 } %615, 2, !dbg !62
  %629 = bitcast i32 %628 to <4 x i8>, !dbg !62
  %630 = extractelement <4 x i8> %629, i32 0, !dbg !62
  %631 = extractelement <4 x i8> %629, i32 1, !dbg !62
  %632 = extractelement <4 x i8> %629, i32 2, !dbg !62
  %633 = extractelement <4 x i8> %629, i32 3, !dbg !62
  %634 = extractvalue { i32, i32, i32, i32 } %615, 3, !dbg !62
  %635 = bitcast i32 %634 to <4 x i8>, !dbg !62
  %636 = extractelement <4 x i8> %635, i32 0, !dbg !62
  %637 = extractelement <4 x i8> %635, i32 1, !dbg !62
  %638 = extractelement <4 x i8> %635, i32 2, !dbg !62
  %639 = extractelement <4 x i8> %635, i32 3, !dbg !62
  %640 = xor i32 %421, 64, !dbg !62
  %641 = add i32 %640, 0, !dbg !62
  %642 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %641, !dbg !62
  %643 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %642), !dbg !62
  %644 = extractvalue { i32, i32, i32, i32 } %643, 0, !dbg !62
  %645 = bitcast i32 %644 to <4 x i8>, !dbg !62
  %646 = extractelement <4 x i8> %645, i32 0, !dbg !62
  %647 = extractelement <4 x i8> %645, i32 1, !dbg !62
  %648 = extractelement <4 x i8> %645, i32 2, !dbg !62
  %649 = extractelement <4 x i8> %645, i32 3, !dbg !62
  %650 = extractvalue { i32, i32, i32, i32 } %643, 1, !dbg !62
  %651 = bitcast i32 %650 to <4 x i8>, !dbg !62
  %652 = extractelement <4 x i8> %651, i32 0, !dbg !62
  %653 = extractelement <4 x i8> %651, i32 1, !dbg !62
  %654 = extractelement <4 x i8> %651, i32 2, !dbg !62
  %655 = extractelement <4 x i8> %651, i32 3, !dbg !62
  %656 = extractvalue { i32, i32, i32, i32 } %643, 2, !dbg !62
  %657 = bitcast i32 %656 to <4 x i8>, !dbg !62
  %658 = extractelement <4 x i8> %657, i32 0, !dbg !62
  %659 = extractelement <4 x i8> %657, i32 1, !dbg !62
  %660 = extractelement <4 x i8> %657, i32 2, !dbg !62
  %661 = extractelement <4 x i8> %657, i32 3, !dbg !62
  %662 = extractvalue { i32, i32, i32, i32 } %643, 3, !dbg !62
  %663 = bitcast i32 %662 to <4 x i8>, !dbg !62
  %664 = extractelement <4 x i8> %663, i32 0, !dbg !62
  %665 = extractelement <4 x i8> %663, i32 1, !dbg !62
  %666 = extractelement <4 x i8> %663, i32 2, !dbg !62
  %667 = extractelement <4 x i8> %663, i32 3, !dbg !62
  %668 = add i32 %640, 8192, !dbg !62
  %669 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %668, !dbg !62
  %670 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %669), !dbg !62
  %671 = extractvalue { i32, i32, i32, i32 } %670, 0, !dbg !62
  %672 = bitcast i32 %671 to <4 x i8>, !dbg !62
  %673 = extractelement <4 x i8> %672, i32 0, !dbg !62
  %674 = extractelement <4 x i8> %672, i32 1, !dbg !62
  %675 = extractelement <4 x i8> %672, i32 2, !dbg !62
  %676 = extractelement <4 x i8> %672, i32 3, !dbg !62
  %677 = extractvalue { i32, i32, i32, i32 } %670, 1, !dbg !62
  %678 = bitcast i32 %677 to <4 x i8>, !dbg !62
  %679 = extractelement <4 x i8> %678, i32 0, !dbg !62
  %680 = extractelement <4 x i8> %678, i32 1, !dbg !62
  %681 = extractelement <4 x i8> %678, i32 2, !dbg !62
  %682 = extractelement <4 x i8> %678, i32 3, !dbg !62
  %683 = extractvalue { i32, i32, i32, i32 } %670, 2, !dbg !62
  %684 = bitcast i32 %683 to <4 x i8>, !dbg !62
  %685 = extractelement <4 x i8> %684, i32 0, !dbg !62
  %686 = extractelement <4 x i8> %684, i32 1, !dbg !62
  %687 = extractelement <4 x i8> %684, i32 2, !dbg !62
  %688 = extractelement <4 x i8> %684, i32 3, !dbg !62
  %689 = extractvalue { i32, i32, i32, i32 } %670, 3, !dbg !62
  %690 = bitcast i32 %689 to <4 x i8>, !dbg !62
  %691 = extractelement <4 x i8> %690, i32 0, !dbg !62
  %692 = extractelement <4 x i8> %690, i32 1, !dbg !62
  %693 = extractelement <4 x i8> %690, i32 2, !dbg !62
  %694 = extractelement <4 x i8> %690, i32 3, !dbg !62
  %695 = add i32 %640, 16384, !dbg !62
  %696 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %695, !dbg !62
  %697 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %696), !dbg !62
  %698 = extractvalue { i32, i32, i32, i32 } %697, 0, !dbg !62
  %699 = bitcast i32 %698 to <4 x i8>, !dbg !62
  %700 = extractelement <4 x i8> %699, i32 0, !dbg !62
  %701 = extractelement <4 x i8> %699, i32 1, !dbg !62
  %702 = extractelement <4 x i8> %699, i32 2, !dbg !62
  %703 = extractelement <4 x i8> %699, i32 3, !dbg !62
  %704 = extractvalue { i32, i32, i32, i32 } %697, 1, !dbg !62
  %705 = bitcast i32 %704 to <4 x i8>, !dbg !62
  %706 = extractelement <4 x i8> %705, i32 0, !dbg !62
  %707 = extractelement <4 x i8> %705, i32 1, !dbg !62
  %708 = extractelement <4 x i8> %705, i32 2, !dbg !62
  %709 = extractelement <4 x i8> %705, i32 3, !dbg !62
  %710 = extractvalue { i32, i32, i32, i32 } %697, 2, !dbg !62
  %711 = bitcast i32 %710 to <4 x i8>, !dbg !62
  %712 = extractelement <4 x i8> %711, i32 0, !dbg !62
  %713 = extractelement <4 x i8> %711, i32 1, !dbg !62
  %714 = extractelement <4 x i8> %711, i32 2, !dbg !62
  %715 = extractelement <4 x i8> %711, i32 3, !dbg !62
  %716 = extractvalue { i32, i32, i32, i32 } %697, 3, !dbg !62
  %717 = bitcast i32 %716 to <4 x i8>, !dbg !62
  %718 = extractelement <4 x i8> %717, i32 0, !dbg !62
  %719 = extractelement <4 x i8> %717, i32 1, !dbg !62
  %720 = extractelement <4 x i8> %717, i32 2, !dbg !62
  %721 = extractelement <4 x i8> %717, i32 3, !dbg !62
  %722 = add i32 %640, 24576, !dbg !62
  %723 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %722, !dbg !62
  %724 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %723), !dbg !62
  %725 = extractvalue { i32, i32, i32, i32 } %724, 0, !dbg !62
  %726 = bitcast i32 %725 to <4 x i8>, !dbg !62
  %727 = extractelement <4 x i8> %726, i32 0, !dbg !62
  %728 = extractelement <4 x i8> %726, i32 1, !dbg !62
  %729 = extractelement <4 x i8> %726, i32 2, !dbg !62
  %730 = extractelement <4 x i8> %726, i32 3, !dbg !62
  %731 = extractvalue { i32, i32, i32, i32 } %724, 1, !dbg !62
  %732 = bitcast i32 %731 to <4 x i8>, !dbg !62
  %733 = extractelement <4 x i8> %732, i32 0, !dbg !62
  %734 = extractelement <4 x i8> %732, i32 1, !dbg !62
  %735 = extractelement <4 x i8> %732, i32 2, !dbg !62
  %736 = extractelement <4 x i8> %732, i32 3, !dbg !62
  %737 = extractvalue { i32, i32, i32, i32 } %724, 2, !dbg !62
  %738 = bitcast i32 %737 to <4 x i8>, !dbg !62
  %739 = extractelement <4 x i8> %738, i32 0, !dbg !62
  %740 = extractelement <4 x i8> %738, i32 1, !dbg !62
  %741 = extractelement <4 x i8> %738, i32 2, !dbg !62
  %742 = extractelement <4 x i8> %738, i32 3, !dbg !62
  %743 = extractvalue { i32, i32, i32, i32 } %724, 3, !dbg !62
  %744 = bitcast i32 %743 to <4 x i8>, !dbg !62
  %745 = extractelement <4 x i8> %744, i32 0, !dbg !62
  %746 = extractelement <4 x i8> %744, i32 1, !dbg !62
  %747 = extractelement <4 x i8> %744, i32 2, !dbg !62
  %748 = extractelement <4 x i8> %744, i32 3, !dbg !62
  %749 = xor i32 %421, 96, !dbg !62
  %750 = add i32 %749, 0, !dbg !62
  %751 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %750, !dbg !62
  %752 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %751), !dbg !62
  %753 = extractvalue { i32, i32, i32, i32 } %752, 0, !dbg !62
  %754 = bitcast i32 %753 to <4 x i8>, !dbg !62
  %755 = extractelement <4 x i8> %754, i32 0, !dbg !62
  %756 = extractelement <4 x i8> %754, i32 1, !dbg !62
  %757 = extractelement <4 x i8> %754, i32 2, !dbg !62
  %758 = extractelement <4 x i8> %754, i32 3, !dbg !62
  %759 = extractvalue { i32, i32, i32, i32 } %752, 1, !dbg !62
  %760 = bitcast i32 %759 to <4 x i8>, !dbg !62
  %761 = extractelement <4 x i8> %760, i32 0, !dbg !62
  %762 = extractelement <4 x i8> %760, i32 1, !dbg !62
  %763 = extractelement <4 x i8> %760, i32 2, !dbg !62
  %764 = extractelement <4 x i8> %760, i32 3, !dbg !62
  %765 = extractvalue { i32, i32, i32, i32 } %752, 2, !dbg !62
  %766 = bitcast i32 %765 to <4 x i8>, !dbg !62
  %767 = extractelement <4 x i8> %766, i32 0, !dbg !62
  %768 = extractelement <4 x i8> %766, i32 1, !dbg !62
  %769 = extractelement <4 x i8> %766, i32 2, !dbg !62
  %770 = extractelement <4 x i8> %766, i32 3, !dbg !62
  %771 = extractvalue { i32, i32, i32, i32 } %752, 3, !dbg !62
  %772 = bitcast i32 %771 to <4 x i8>, !dbg !62
  %773 = extractelement <4 x i8> %772, i32 0, !dbg !62
  %774 = extractelement <4 x i8> %772, i32 1, !dbg !62
  %775 = extractelement <4 x i8> %772, i32 2, !dbg !62
  %776 = extractelement <4 x i8> %772, i32 3, !dbg !62
  %777 = add i32 %749, 8192, !dbg !62
  %778 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %777, !dbg !62
  %779 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %778), !dbg !62
  %780 = extractvalue { i32, i32, i32, i32 } %779, 0, !dbg !62
  %781 = bitcast i32 %780 to <4 x i8>, !dbg !62
  %782 = extractelement <4 x i8> %781, i32 0, !dbg !62
  %783 = extractelement <4 x i8> %781, i32 1, !dbg !62
  %784 = extractelement <4 x i8> %781, i32 2, !dbg !62
  %785 = extractelement <4 x i8> %781, i32 3, !dbg !62
  %786 = extractvalue { i32, i32, i32, i32 } %779, 1, !dbg !62
  %787 = bitcast i32 %786 to <4 x i8>, !dbg !62
  %788 = extractelement <4 x i8> %787, i32 0, !dbg !62
  %789 = extractelement <4 x i8> %787, i32 1, !dbg !62
  %790 = extractelement <4 x i8> %787, i32 2, !dbg !62
  %791 = extractelement <4 x i8> %787, i32 3, !dbg !62
  %792 = extractvalue { i32, i32, i32, i32 } %779, 2, !dbg !62
  %793 = bitcast i32 %792 to <4 x i8>, !dbg !62
  %794 = extractelement <4 x i8> %793, i32 0, !dbg !62
  %795 = extractelement <4 x i8> %793, i32 1, !dbg !62
  %796 = extractelement <4 x i8> %793, i32 2, !dbg !62
  %797 = extractelement <4 x i8> %793, i32 3, !dbg !62
  %798 = extractvalue { i32, i32, i32, i32 } %779, 3, !dbg !62
  %799 = bitcast i32 %798 to <4 x i8>, !dbg !62
  %800 = extractelement <4 x i8> %799, i32 0, !dbg !62
  %801 = extractelement <4 x i8> %799, i32 1, !dbg !62
  %802 = extractelement <4 x i8> %799, i32 2, !dbg !62
  %803 = extractelement <4 x i8> %799, i32 3, !dbg !62
  %804 = add i32 %749, 16384, !dbg !62
  %805 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %804, !dbg !62
  %806 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %805), !dbg !62
  %807 = extractvalue { i32, i32, i32, i32 } %806, 0, !dbg !62
  %808 = bitcast i32 %807 to <4 x i8>, !dbg !62
  %809 = extractelement <4 x i8> %808, i32 0, !dbg !62
  %810 = extractelement <4 x i8> %808, i32 1, !dbg !62
  %811 = extractelement <4 x i8> %808, i32 2, !dbg !62
  %812 = extractelement <4 x i8> %808, i32 3, !dbg !62
  %813 = extractvalue { i32, i32, i32, i32 } %806, 1, !dbg !62
  %814 = bitcast i32 %813 to <4 x i8>, !dbg !62
  %815 = extractelement <4 x i8> %814, i32 0, !dbg !62
  %816 = extractelement <4 x i8> %814, i32 1, !dbg !62
  %817 = extractelement <4 x i8> %814, i32 2, !dbg !62
  %818 = extractelement <4 x i8> %814, i32 3, !dbg !62
  %819 = extractvalue { i32, i32, i32, i32 } %806, 2, !dbg !62
  %820 = bitcast i32 %819 to <4 x i8>, !dbg !62
  %821 = extractelement <4 x i8> %820, i32 0, !dbg !62
  %822 = extractelement <4 x i8> %820, i32 1, !dbg !62
  %823 = extractelement <4 x i8> %820, i32 2, !dbg !62
  %824 = extractelement <4 x i8> %820, i32 3, !dbg !62
  %825 = extractvalue { i32, i32, i32, i32 } %806, 3, !dbg !62
  %826 = bitcast i32 %825 to <4 x i8>, !dbg !62
  %827 = extractelement <4 x i8> %826, i32 0, !dbg !62
  %828 = extractelement <4 x i8> %826, i32 1, !dbg !62
  %829 = extractelement <4 x i8> %826, i32 2, !dbg !62
  %830 = extractelement <4 x i8> %826, i32 3, !dbg !62
  %831 = add i32 %749, 24576, !dbg !62
  %832 = getelementptr inbounds i8, ptr addrspace(3) %411, i32 %831, !dbg !62
  %833 = call { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) %832), !dbg !62
  %834 = extractvalue { i32, i32, i32, i32 } %833, 0, !dbg !62
  %835 = bitcast i32 %834 to <4 x i8>, !dbg !62
  %836 = extractelement <4 x i8> %835, i32 0, !dbg !62
  %837 = extractelement <4 x i8> %835, i32 1, !dbg !62
  %838 = extractelement <4 x i8> %835, i32 2, !dbg !62
  %839 = extractelement <4 x i8> %835, i32 3, !dbg !62
  %840 = extractvalue { i32, i32, i32, i32 } %833, 1, !dbg !62
  %841 = bitcast i32 %840 to <4 x i8>, !dbg !62
  %842 = extractelement <4 x i8> %841, i32 0, !dbg !62
  %843 = extractelement <4 x i8> %841, i32 1, !dbg !62
  %844 = extractelement <4 x i8> %841, i32 2, !dbg !62
  %845 = extractelement <4 x i8> %841, i32 3, !dbg !62
  %846 = extractvalue { i32, i32, i32, i32 } %833, 2, !dbg !62
  %847 = bitcast i32 %846 to <4 x i8>, !dbg !62
  %848 = extractelement <4 x i8> %847, i32 0, !dbg !62
  %849 = extractelement <4 x i8> %847, i32 1, !dbg !62
  %850 = extractelement <4 x i8> %847, i32 2, !dbg !62
  %851 = extractelement <4 x i8> %847, i32 3, !dbg !62
  %852 = extractvalue { i32, i32, i32, i32 } %833, 3, !dbg !62
  %853 = bitcast i32 %852 to <4 x i8>, !dbg !62
  %854 = extractelement <4 x i8> %853, i32 0, !dbg !62
  %855 = extractelement <4 x i8> %853, i32 1, !dbg !62
  %856 = extractelement <4 x i8> %853, i32 2, !dbg !62
  %857 = extractelement <4 x i8> %853, i32 3, !dbg !62
  call void @llvm.nvvm.fence.proxy.async.shared_cta(), !dbg !62
  %858 = getelementptr i64, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88096), i32 %284, !dbg !62
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !62
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !62
  call void asm sideeffect "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];", "b,r"(i1 %189, ptr addrspace(3) %858), !dbg !62
  %859 = insertelement <4 x i8> undef, i8 %428, i32 0, !dbg !76
  %860 = insertelement <4 x i8> %859, i8 %429, i32 1, !dbg !76
  %861 = insertelement <4 x i8> %860, i8 %430, i32 2, !dbg !76
  %862 = insertelement <4 x i8> %861, i8 %431, i32 3, !dbg !76
  %863 = bitcast <4 x i8> %862 to i32, !dbg !76
  %864 = insertelement <4 x i8> undef, i8 %434, i32 0, !dbg !76
  %865 = insertelement <4 x i8> %864, i8 %435, i32 1, !dbg !76
  %866 = insertelement <4 x i8> %865, i8 %436, i32 2, !dbg !76
  %867 = insertelement <4 x i8> %866, i8 %437, i32 3, !dbg !76
  %868 = bitcast <4 x i8> %867 to i32, !dbg !76
  %869 = insertelement <4 x i8> undef, i8 %440, i32 0, !dbg !76
  %870 = insertelement <4 x i8> %869, i8 %441, i32 1, !dbg !76
  %871 = insertelement <4 x i8> %870, i8 %442, i32 2, !dbg !76
  %872 = insertelement <4 x i8> %871, i8 %443, i32 3, !dbg !76
  %873 = bitcast <4 x i8> %872 to i32, !dbg !76
  %874 = insertelement <4 x i8> undef, i8 %446, i32 0, !dbg !76
  %875 = insertelement <4 x i8> %874, i8 %447, i32 1, !dbg !76
  %876 = insertelement <4 x i8> %875, i8 %448, i32 2, !dbg !76
  %877 = insertelement <4 x i8> %876, i8 %449, i32 3, !dbg !76
  %878 = bitcast <4 x i8> %877 to i32, !dbg !76
  %879 = insertelement <4 x i8> undef, i8 %537, i32 0, !dbg !76
  %880 = insertelement <4 x i8> %879, i8 %538, i32 1, !dbg !76
  %881 = insertelement <4 x i8> %880, i8 %539, i32 2, !dbg !76
  %882 = insertelement <4 x i8> %881, i8 %540, i32 3, !dbg !76
  %883 = bitcast <4 x i8> %882 to i32, !dbg !76
  %884 = insertelement <4 x i8> undef, i8 %543, i32 0, !dbg !76
  %885 = insertelement <4 x i8> %884, i8 %544, i32 1, !dbg !76
  %886 = insertelement <4 x i8> %885, i8 %545, i32 2, !dbg !76
  %887 = insertelement <4 x i8> %886, i8 %546, i32 3, !dbg !76
  %888 = bitcast <4 x i8> %887 to i32, !dbg !76
  %889 = insertelement <4 x i8> undef, i8 %549, i32 0, !dbg !76
  %890 = insertelement <4 x i8> %889, i8 %550, i32 1, !dbg !76
  %891 = insertelement <4 x i8> %890, i8 %551, i32 2, !dbg !76
  %892 = insertelement <4 x i8> %891, i8 %552, i32 3, !dbg !76
  %893 = bitcast <4 x i8> %892 to i32, !dbg !76
  %894 = insertelement <4 x i8> undef, i8 %555, i32 0, !dbg !76
  %895 = insertelement <4 x i8> %894, i8 %556, i32 1, !dbg !76
  %896 = insertelement <4 x i8> %895, i8 %557, i32 2, !dbg !76
  %897 = insertelement <4 x i8> %896, i8 %558, i32 3, !dbg !76
  %898 = bitcast <4 x i8> %897 to i32, !dbg !76
  %899 = insertelement <4 x i8> undef, i8 %646, i32 0, !dbg !76
  %900 = insertelement <4 x i8> %899, i8 %647, i32 1, !dbg !76
  %901 = insertelement <4 x i8> %900, i8 %648, i32 2, !dbg !76
  %902 = insertelement <4 x i8> %901, i8 %649, i32 3, !dbg !76
  %903 = bitcast <4 x i8> %902 to i32, !dbg !76
  %904 = insertelement <4 x i8> undef, i8 %652, i32 0, !dbg !76
  %905 = insertelement <4 x i8> %904, i8 %653, i32 1, !dbg !76
  %906 = insertelement <4 x i8> %905, i8 %654, i32 2, !dbg !76
  %907 = insertelement <4 x i8> %906, i8 %655, i32 3, !dbg !76
  %908 = bitcast <4 x i8> %907 to i32, !dbg !76
  %909 = insertelement <4 x i8> undef, i8 %658, i32 0, !dbg !76
  %910 = insertelement <4 x i8> %909, i8 %659, i32 1, !dbg !76
  %911 = insertelement <4 x i8> %910, i8 %660, i32 2, !dbg !76
  %912 = insertelement <4 x i8> %911, i8 %661, i32 3, !dbg !76
  %913 = bitcast <4 x i8> %912 to i32, !dbg !76
  %914 = insertelement <4 x i8> undef, i8 %664, i32 0, !dbg !76
  %915 = insertelement <4 x i8> %914, i8 %665, i32 1, !dbg !76
  %916 = insertelement <4 x i8> %915, i8 %666, i32 2, !dbg !76
  %917 = insertelement <4 x i8> %916, i8 %667, i32 3, !dbg !76
  %918 = bitcast <4 x i8> %917 to i32, !dbg !76
  %919 = insertelement <4 x i8> undef, i8 %755, i32 0, !dbg !76
  %920 = insertelement <4 x i8> %919, i8 %756, i32 1, !dbg !76
  %921 = insertelement <4 x i8> %920, i8 %757, i32 2, !dbg !76
  %922 = insertelement <4 x i8> %921, i8 %758, i32 3, !dbg !76
  %923 = bitcast <4 x i8> %922 to i32, !dbg !76
  %924 = insertelement <4 x i8> undef, i8 %761, i32 0, !dbg !76
  %925 = insertelement <4 x i8> %924, i8 %762, i32 1, !dbg !76
  %926 = insertelement <4 x i8> %925, i8 %763, i32 2, !dbg !76
  %927 = insertelement <4 x i8> %926, i8 %764, i32 3, !dbg !76
  %928 = bitcast <4 x i8> %927 to i32, !dbg !76
  %929 = insertelement <4 x i8> undef, i8 %767, i32 0, !dbg !76
  %930 = insertelement <4 x i8> %929, i8 %768, i32 1, !dbg !76
  %931 = insertelement <4 x i8> %930, i8 %769, i32 2, !dbg !76
  %932 = insertelement <4 x i8> %931, i8 %770, i32 3, !dbg !76
  %933 = bitcast <4 x i8> %932 to i32, !dbg !76
  %934 = insertelement <4 x i8> undef, i8 %773, i32 0, !dbg !76
  %935 = insertelement <4 x i8> %934, i8 %774, i32 1, !dbg !76
  %936 = insertelement <4 x i8> %935, i8 %775, i32 2, !dbg !76
  %937 = insertelement <4 x i8> %936, i8 %776, i32 3, !dbg !76
  %938 = bitcast <4 x i8> %937 to i32, !dbg !76
  %939 = insertelement <4 x i8> undef, i8 %455, i32 0, !dbg !76
  %940 = insertelement <4 x i8> %939, i8 %456, i32 1, !dbg !76
  %941 = insertelement <4 x i8> %940, i8 %457, i32 2, !dbg !76
  %942 = insertelement <4 x i8> %941, i8 %458, i32 3, !dbg !76
  %943 = bitcast <4 x i8> %942 to i32, !dbg !76
  %944 = insertelement <4 x i8> undef, i8 %461, i32 0, !dbg !76
  %945 = insertelement <4 x i8> %944, i8 %462, i32 1, !dbg !76
  %946 = insertelement <4 x i8> %945, i8 %463, i32 2, !dbg !76
  %947 = insertelement <4 x i8> %946, i8 %464, i32 3, !dbg !76
  %948 = bitcast <4 x i8> %947 to i32, !dbg !76
  %949 = insertelement <4 x i8> undef, i8 %467, i32 0, !dbg !76
  %950 = insertelement <4 x i8> %949, i8 %468, i32 1, !dbg !76
  %951 = insertelement <4 x i8> %950, i8 %469, i32 2, !dbg !76
  %952 = insertelement <4 x i8> %951, i8 %470, i32 3, !dbg !76
  %953 = bitcast <4 x i8> %952 to i32, !dbg !76
  %954 = insertelement <4 x i8> undef, i8 %473, i32 0, !dbg !76
  %955 = insertelement <4 x i8> %954, i8 %474, i32 1, !dbg !76
  %956 = insertelement <4 x i8> %955, i8 %475, i32 2, !dbg !76
  %957 = insertelement <4 x i8> %956, i8 %476, i32 3, !dbg !76
  %958 = bitcast <4 x i8> %957 to i32, !dbg !76
  %959 = insertelement <4 x i8> undef, i8 %564, i32 0, !dbg !76
  %960 = insertelement <4 x i8> %959, i8 %565, i32 1, !dbg !76
  %961 = insertelement <4 x i8> %960, i8 %566, i32 2, !dbg !76
  %962 = insertelement <4 x i8> %961, i8 %567, i32 3, !dbg !76
  %963 = bitcast <4 x i8> %962 to i32, !dbg !76
  %964 = insertelement <4 x i8> undef, i8 %570, i32 0, !dbg !76
  %965 = insertelement <4 x i8> %964, i8 %571, i32 1, !dbg !76
  %966 = insertelement <4 x i8> %965, i8 %572, i32 2, !dbg !76
  %967 = insertelement <4 x i8> %966, i8 %573, i32 3, !dbg !76
  %968 = bitcast <4 x i8> %967 to i32, !dbg !76
  %969 = insertelement <4 x i8> undef, i8 %576, i32 0, !dbg !76
  %970 = insertelement <4 x i8> %969, i8 %577, i32 1, !dbg !76
  %971 = insertelement <4 x i8> %970, i8 %578, i32 2, !dbg !76
  %972 = insertelement <4 x i8> %971, i8 %579, i32 3, !dbg !76
  %973 = bitcast <4 x i8> %972 to i32, !dbg !76
  %974 = insertelement <4 x i8> undef, i8 %582, i32 0, !dbg !76
  %975 = insertelement <4 x i8> %974, i8 %583, i32 1, !dbg !76
  %976 = insertelement <4 x i8> %975, i8 %584, i32 2, !dbg !76
  %977 = insertelement <4 x i8> %976, i8 %585, i32 3, !dbg !76
  %978 = bitcast <4 x i8> %977 to i32, !dbg !76
  %979 = insertelement <4 x i8> undef, i8 %673, i32 0, !dbg !76
  %980 = insertelement <4 x i8> %979, i8 %674, i32 1, !dbg !76
  %981 = insertelement <4 x i8> %980, i8 %675, i32 2, !dbg !76
  %982 = insertelement <4 x i8> %981, i8 %676, i32 3, !dbg !76
  %983 = bitcast <4 x i8> %982 to i32, !dbg !76
  %984 = insertelement <4 x i8> undef, i8 %679, i32 0, !dbg !76
  %985 = insertelement <4 x i8> %984, i8 %680, i32 1, !dbg !76
  %986 = insertelement <4 x i8> %985, i8 %681, i32 2, !dbg !76
  %987 = insertelement <4 x i8> %986, i8 %682, i32 3, !dbg !76
  %988 = bitcast <4 x i8> %987 to i32, !dbg !76
  %989 = insertelement <4 x i8> undef, i8 %685, i32 0, !dbg !76
  %990 = insertelement <4 x i8> %989, i8 %686, i32 1, !dbg !76
  %991 = insertelement <4 x i8> %990, i8 %687, i32 2, !dbg !76
  %992 = insertelement <4 x i8> %991, i8 %688, i32 3, !dbg !76
  %993 = bitcast <4 x i8> %992 to i32, !dbg !76
  %994 = insertelement <4 x i8> undef, i8 %691, i32 0, !dbg !76
  %995 = insertelement <4 x i8> %994, i8 %692, i32 1, !dbg !76
  %996 = insertelement <4 x i8> %995, i8 %693, i32 2, !dbg !76
  %997 = insertelement <4 x i8> %996, i8 %694, i32 3, !dbg !76
  %998 = bitcast <4 x i8> %997 to i32, !dbg !76
  %999 = insertelement <4 x i8> undef, i8 %782, i32 0, !dbg !76
  %1000 = insertelement <4 x i8> %999, i8 %783, i32 1, !dbg !76
  %1001 = insertelement <4 x i8> %1000, i8 %784, i32 2, !dbg !76
  %1002 = insertelement <4 x i8> %1001, i8 %785, i32 3, !dbg !76
  %1003 = bitcast <4 x i8> %1002 to i32, !dbg !76
  %1004 = insertelement <4 x i8> undef, i8 %788, i32 0, !dbg !76
  %1005 = insertelement <4 x i8> %1004, i8 %789, i32 1, !dbg !76
  %1006 = insertelement <4 x i8> %1005, i8 %790, i32 2, !dbg !76
  %1007 = insertelement <4 x i8> %1006, i8 %791, i32 3, !dbg !76
  %1008 = bitcast <4 x i8> %1007 to i32, !dbg !76
  %1009 = insertelement <4 x i8> undef, i8 %794, i32 0, !dbg !76
  %1010 = insertelement <4 x i8> %1009, i8 %795, i32 1, !dbg !76
  %1011 = insertelement <4 x i8> %1010, i8 %796, i32 2, !dbg !76
  %1012 = insertelement <4 x i8> %1011, i8 %797, i32 3, !dbg !76
  %1013 = bitcast <4 x i8> %1012 to i32, !dbg !76
  %1014 = insertelement <4 x i8> undef, i8 %800, i32 0, !dbg !76
  %1015 = insertelement <4 x i8> %1014, i8 %801, i32 1, !dbg !76
  %1016 = insertelement <4 x i8> %1015, i8 %802, i32 2, !dbg !76
  %1017 = insertelement <4 x i8> %1016, i8 %803, i32 3, !dbg !76
  %1018 = bitcast <4 x i8> %1017 to i32, !dbg !76
  %1019 = insertelement <4 x i8> undef, i8 %482, i32 0, !dbg !76
  %1020 = insertelement <4 x i8> %1019, i8 %483, i32 1, !dbg !76
  %1021 = insertelement <4 x i8> %1020, i8 %484, i32 2, !dbg !76
  %1022 = insertelement <4 x i8> %1021, i8 %485, i32 3, !dbg !76
  %1023 = bitcast <4 x i8> %1022 to i32, !dbg !76
  %1024 = insertelement <4 x i8> undef, i8 %488, i32 0, !dbg !76
  %1025 = insertelement <4 x i8> %1024, i8 %489, i32 1, !dbg !76
  %1026 = insertelement <4 x i8> %1025, i8 %490, i32 2, !dbg !76
  %1027 = insertelement <4 x i8> %1026, i8 %491, i32 3, !dbg !76
  %1028 = bitcast <4 x i8> %1027 to i32, !dbg !76
  %1029 = insertelement <4 x i8> undef, i8 %494, i32 0, !dbg !76
  %1030 = insertelement <4 x i8> %1029, i8 %495, i32 1, !dbg !76
  %1031 = insertelement <4 x i8> %1030, i8 %496, i32 2, !dbg !76
  %1032 = insertelement <4 x i8> %1031, i8 %497, i32 3, !dbg !76
  %1033 = bitcast <4 x i8> %1032 to i32, !dbg !76
  %1034 = insertelement <4 x i8> undef, i8 %500, i32 0, !dbg !76
  %1035 = insertelement <4 x i8> %1034, i8 %501, i32 1, !dbg !76
  %1036 = insertelement <4 x i8> %1035, i8 %502, i32 2, !dbg !76
  %1037 = insertelement <4 x i8> %1036, i8 %503, i32 3, !dbg !76
  %1038 = bitcast <4 x i8> %1037 to i32, !dbg !76
  %1039 = insertelement <4 x i8> undef, i8 %591, i32 0, !dbg !76
  %1040 = insertelement <4 x i8> %1039, i8 %592, i32 1, !dbg !76
  %1041 = insertelement <4 x i8> %1040, i8 %593, i32 2, !dbg !76
  %1042 = insertelement <4 x i8> %1041, i8 %594, i32 3, !dbg !76
  %1043 = bitcast <4 x i8> %1042 to i32, !dbg !76
  %1044 = insertelement <4 x i8> undef, i8 %597, i32 0, !dbg !76
  %1045 = insertelement <4 x i8> %1044, i8 %598, i32 1, !dbg !76
  %1046 = insertelement <4 x i8> %1045, i8 %599, i32 2, !dbg !76
  %1047 = insertelement <4 x i8> %1046, i8 %600, i32 3, !dbg !76
  %1048 = bitcast <4 x i8> %1047 to i32, !dbg !76
  %1049 = insertelement <4 x i8> undef, i8 %603, i32 0, !dbg !76
  %1050 = insertelement <4 x i8> %1049, i8 %604, i32 1, !dbg !76
  %1051 = insertelement <4 x i8> %1050, i8 %605, i32 2, !dbg !76
  %1052 = insertelement <4 x i8> %1051, i8 %606, i32 3, !dbg !76
  %1053 = bitcast <4 x i8> %1052 to i32, !dbg !76
  %1054 = insertelement <4 x i8> undef, i8 %609, i32 0, !dbg !76
  %1055 = insertelement <4 x i8> %1054, i8 %610, i32 1, !dbg !76
  %1056 = insertelement <4 x i8> %1055, i8 %611, i32 2, !dbg !76
  %1057 = insertelement <4 x i8> %1056, i8 %612, i32 3, !dbg !76
  %1058 = bitcast <4 x i8> %1057 to i32, !dbg !76
  %1059 = insertelement <4 x i8> undef, i8 %700, i32 0, !dbg !76
  %1060 = insertelement <4 x i8> %1059, i8 %701, i32 1, !dbg !76
  %1061 = insertelement <4 x i8> %1060, i8 %702, i32 2, !dbg !76
  %1062 = insertelement <4 x i8> %1061, i8 %703, i32 3, !dbg !76
  %1063 = bitcast <4 x i8> %1062 to i32, !dbg !76
  %1064 = insertelement <4 x i8> undef, i8 %706, i32 0, !dbg !76
  %1065 = insertelement <4 x i8> %1064, i8 %707, i32 1, !dbg !76
  %1066 = insertelement <4 x i8> %1065, i8 %708, i32 2, !dbg !76
  %1067 = insertelement <4 x i8> %1066, i8 %709, i32 3, !dbg !76
  %1068 = bitcast <4 x i8> %1067 to i32, !dbg !76
  %1069 = insertelement <4 x i8> undef, i8 %712, i32 0, !dbg !76
  %1070 = insertelement <4 x i8> %1069, i8 %713, i32 1, !dbg !76
  %1071 = insertelement <4 x i8> %1070, i8 %714, i32 2, !dbg !76
  %1072 = insertelement <4 x i8> %1071, i8 %715, i32 3, !dbg !76
  %1073 = bitcast <4 x i8> %1072 to i32, !dbg !76
  %1074 = insertelement <4 x i8> undef, i8 %718, i32 0, !dbg !76
  %1075 = insertelement <4 x i8> %1074, i8 %719, i32 1, !dbg !76
  %1076 = insertelement <4 x i8> %1075, i8 %720, i32 2, !dbg !76
  %1077 = insertelement <4 x i8> %1076, i8 %721, i32 3, !dbg !76
  %1078 = bitcast <4 x i8> %1077 to i32, !dbg !76
  %1079 = insertelement <4 x i8> undef, i8 %809, i32 0, !dbg !76
  %1080 = insertelement <4 x i8> %1079, i8 %810, i32 1, !dbg !76
  %1081 = insertelement <4 x i8> %1080, i8 %811, i32 2, !dbg !76
  %1082 = insertelement <4 x i8> %1081, i8 %812, i32 3, !dbg !76
  %1083 = bitcast <4 x i8> %1082 to i32, !dbg !76
  %1084 = insertelement <4 x i8> undef, i8 %815, i32 0, !dbg !76
  %1085 = insertelement <4 x i8> %1084, i8 %816, i32 1, !dbg !76
  %1086 = insertelement <4 x i8> %1085, i8 %817, i32 2, !dbg !76
  %1087 = insertelement <4 x i8> %1086, i8 %818, i32 3, !dbg !76
  %1088 = bitcast <4 x i8> %1087 to i32, !dbg !76
  %1089 = insertelement <4 x i8> undef, i8 %821, i32 0, !dbg !76
  %1090 = insertelement <4 x i8> %1089, i8 %822, i32 1, !dbg !76
  %1091 = insertelement <4 x i8> %1090, i8 %823, i32 2, !dbg !76
  %1092 = insertelement <4 x i8> %1091, i8 %824, i32 3, !dbg !76
  %1093 = bitcast <4 x i8> %1092 to i32, !dbg !76
  %1094 = insertelement <4 x i8> undef, i8 %827, i32 0, !dbg !76
  %1095 = insertelement <4 x i8> %1094, i8 %828, i32 1, !dbg !76
  %1096 = insertelement <4 x i8> %1095, i8 %829, i32 2, !dbg !76
  %1097 = insertelement <4 x i8> %1096, i8 %830, i32 3, !dbg !76
  %1098 = bitcast <4 x i8> %1097 to i32, !dbg !76
  %1099 = insertelement <4 x i8> undef, i8 %509, i32 0, !dbg !76
  %1100 = insertelement <4 x i8> %1099, i8 %510, i32 1, !dbg !76
  %1101 = insertelement <4 x i8> %1100, i8 %511, i32 2, !dbg !76
  %1102 = insertelement <4 x i8> %1101, i8 %512, i32 3, !dbg !76
  %1103 = bitcast <4 x i8> %1102 to i32, !dbg !76
  %1104 = insertelement <4 x i8> undef, i8 %515, i32 0, !dbg !76
  %1105 = insertelement <4 x i8> %1104, i8 %516, i32 1, !dbg !76
  %1106 = insertelement <4 x i8> %1105, i8 %517, i32 2, !dbg !76
  %1107 = insertelement <4 x i8> %1106, i8 %518, i32 3, !dbg !76
  %1108 = bitcast <4 x i8> %1107 to i32, !dbg !76
  %1109 = insertelement <4 x i8> undef, i8 %521, i32 0, !dbg !76
  %1110 = insertelement <4 x i8> %1109, i8 %522, i32 1, !dbg !76
  %1111 = insertelement <4 x i8> %1110, i8 %523, i32 2, !dbg !76
  %1112 = insertelement <4 x i8> %1111, i8 %524, i32 3, !dbg !76
  %1113 = bitcast <4 x i8> %1112 to i32, !dbg !76
  %1114 = insertelement <4 x i8> undef, i8 %527, i32 0, !dbg !76
  %1115 = insertelement <4 x i8> %1114, i8 %528, i32 1, !dbg !76
  %1116 = insertelement <4 x i8> %1115, i8 %529, i32 2, !dbg !76
  %1117 = insertelement <4 x i8> %1116, i8 %530, i32 3, !dbg !76
  %1118 = bitcast <4 x i8> %1117 to i32, !dbg !76
  %1119 = insertelement <4 x i8> undef, i8 %618, i32 0, !dbg !76
  %1120 = insertelement <4 x i8> %1119, i8 %619, i32 1, !dbg !76
  %1121 = insertelement <4 x i8> %1120, i8 %620, i32 2, !dbg !76
  %1122 = insertelement <4 x i8> %1121, i8 %621, i32 3, !dbg !76
  %1123 = bitcast <4 x i8> %1122 to i32, !dbg !76
  %1124 = insertelement <4 x i8> undef, i8 %624, i32 0, !dbg !76
  %1125 = insertelement <4 x i8> %1124, i8 %625, i32 1, !dbg !76
  %1126 = insertelement <4 x i8> %1125, i8 %626, i32 2, !dbg !76
  %1127 = insertelement <4 x i8> %1126, i8 %627, i32 3, !dbg !76
  %1128 = bitcast <4 x i8> %1127 to i32, !dbg !76
  %1129 = insertelement <4 x i8> undef, i8 %630, i32 0, !dbg !76
  %1130 = insertelement <4 x i8> %1129, i8 %631, i32 1, !dbg !76
  %1131 = insertelement <4 x i8> %1130, i8 %632, i32 2, !dbg !76
  %1132 = insertelement <4 x i8> %1131, i8 %633, i32 3, !dbg !76
  %1133 = bitcast <4 x i8> %1132 to i32, !dbg !76
  %1134 = insertelement <4 x i8> undef, i8 %636, i32 0, !dbg !76
  %1135 = insertelement <4 x i8> %1134, i8 %637, i32 1, !dbg !76
  %1136 = insertelement <4 x i8> %1135, i8 %638, i32 2, !dbg !76
  %1137 = insertelement <4 x i8> %1136, i8 %639, i32 3, !dbg !76
  %1138 = bitcast <4 x i8> %1137 to i32, !dbg !76
  %1139 = insertelement <4 x i8> undef, i8 %727, i32 0, !dbg !76
  %1140 = insertelement <4 x i8> %1139, i8 %728, i32 1, !dbg !76
  %1141 = insertelement <4 x i8> %1140, i8 %729, i32 2, !dbg !76
  %1142 = insertelement <4 x i8> %1141, i8 %730, i32 3, !dbg !76
  %1143 = bitcast <4 x i8> %1142 to i32, !dbg !76
  %1144 = insertelement <4 x i8> undef, i8 %733, i32 0, !dbg !76
  %1145 = insertelement <4 x i8> %1144, i8 %734, i32 1, !dbg !76
  %1146 = insertelement <4 x i8> %1145, i8 %735, i32 2, !dbg !76
  %1147 = insertelement <4 x i8> %1146, i8 %736, i32 3, !dbg !76
  %1148 = bitcast <4 x i8> %1147 to i32, !dbg !76
  %1149 = insertelement <4 x i8> undef, i8 %739, i32 0, !dbg !76
  %1150 = insertelement <4 x i8> %1149, i8 %740, i32 1, !dbg !76
  %1151 = insertelement <4 x i8> %1150, i8 %741, i32 2, !dbg !76
  %1152 = insertelement <4 x i8> %1151, i8 %742, i32 3, !dbg !76
  %1153 = bitcast <4 x i8> %1152 to i32, !dbg !76
  %1154 = insertelement <4 x i8> undef, i8 %745, i32 0, !dbg !76
  %1155 = insertelement <4 x i8> %1154, i8 %746, i32 1, !dbg !76
  %1156 = insertelement <4 x i8> %1155, i8 %747, i32 2, !dbg !76
  %1157 = insertelement <4 x i8> %1156, i8 %748, i32 3, !dbg !76
  %1158 = bitcast <4 x i8> %1157 to i32, !dbg !76
  %1159 = insertelement <4 x i8> undef, i8 %836, i32 0, !dbg !76
  %1160 = insertelement <4 x i8> %1159, i8 %837, i32 1, !dbg !76
  %1161 = insertelement <4 x i8> %1160, i8 %838, i32 2, !dbg !76
  %1162 = insertelement <4 x i8> %1161, i8 %839, i32 3, !dbg !76
  %1163 = bitcast <4 x i8> %1162 to i32, !dbg !76
  %1164 = insertelement <4 x i8> undef, i8 %842, i32 0, !dbg !76
  %1165 = insertelement <4 x i8> %1164, i8 %843, i32 1, !dbg !76
  %1166 = insertelement <4 x i8> %1165, i8 %844, i32 2, !dbg !76
  %1167 = insertelement <4 x i8> %1166, i8 %845, i32 3, !dbg !76
  %1168 = bitcast <4 x i8> %1167 to i32, !dbg !76
  %1169 = insertelement <4 x i8> undef, i8 %848, i32 0, !dbg !76
  %1170 = insertelement <4 x i8> %1169, i8 %849, i32 1, !dbg !76
  %1171 = insertelement <4 x i8> %1170, i8 %850, i32 2, !dbg !76
  %1172 = insertelement <4 x i8> %1171, i8 %851, i32 3, !dbg !76
  %1173 = bitcast <4 x i8> %1172 to i32, !dbg !76
  %1174 = insertelement <4 x i8> undef, i8 %854, i32 0, !dbg !76
  %1175 = insertelement <4 x i8> %1174, i8 %855, i32 1, !dbg !76
  %1176 = insertelement <4 x i8> %1175, i8 %856, i32 2, !dbg !76
  %1177 = insertelement <4 x i8> %1176, i8 %857, i32 3, !dbg !76
  %1178 = bitcast <4 x i8> %1177 to i32, !dbg !76
  %1179 = insertelement <4 x i8> undef, i8 %304, i32 0, !dbg !76
  %1180 = insertelement <4 x i8> %1179, i8 %305, i32 1, !dbg !76
  %1181 = insertelement <4 x i8> %1180, i8 %306, i32 2, !dbg !76
  %1182 = insertelement <4 x i8> %1181, i8 %307, i32 3, !dbg !76
  %1183 = bitcast <4 x i8> %1182 to i32, !dbg !76
  %1184 = insertelement <4 x i8> undef, i8 %310, i32 0, !dbg !76
  %1185 = insertelement <4 x i8> %1184, i8 %311, i32 1, !dbg !76
  %1186 = insertelement <4 x i8> %1185, i8 %312, i32 2, !dbg !76
  %1187 = insertelement <4 x i8> %1186, i8 %313, i32 3, !dbg !76
  %1188 = bitcast <4 x i8> %1187 to i32, !dbg !76
  %1189 = insertelement <4 x i8> undef, i8 %316, i32 0, !dbg !76
  %1190 = insertelement <4 x i8> %1189, i8 %317, i32 1, !dbg !76
  %1191 = insertelement <4 x i8> %1190, i8 %318, i32 2, !dbg !76
  %1192 = insertelement <4 x i8> %1191, i8 %319, i32 3, !dbg !76
  %1193 = bitcast <4 x i8> %1192 to i32, !dbg !76
  %1194 = insertelement <4 x i8> undef, i8 %322, i32 0, !dbg !76
  %1195 = insertelement <4 x i8> %1194, i8 %323, i32 1, !dbg !76
  %1196 = insertelement <4 x i8> %1195, i8 %324, i32 2, !dbg !76
  %1197 = insertelement <4 x i8> %1196, i8 %325, i32 3, !dbg !76
  %1198 = bitcast <4 x i8> %1197 to i32, !dbg !76
  %1199 = insertelement <4 x i8> undef, i8 %359, i32 0, !dbg !76
  %1200 = insertelement <4 x i8> %1199, i8 %360, i32 1, !dbg !76
  %1201 = insertelement <4 x i8> %1200, i8 %361, i32 2, !dbg !76
  %1202 = insertelement <4 x i8> %1201, i8 %362, i32 3, !dbg !76
  %1203 = bitcast <4 x i8> %1202 to i32, !dbg !76
  %1204 = insertelement <4 x i8> undef, i8 %365, i32 0, !dbg !76
  %1205 = insertelement <4 x i8> %1204, i8 %366, i32 1, !dbg !76
  %1206 = insertelement <4 x i8> %1205, i8 %367, i32 2, !dbg !76
  %1207 = insertelement <4 x i8> %1206, i8 %368, i32 3, !dbg !76
  %1208 = bitcast <4 x i8> %1207 to i32, !dbg !76
  %1209 = insertelement <4 x i8> undef, i8 %371, i32 0, !dbg !76
  %1210 = insertelement <4 x i8> %1209, i8 %372, i32 1, !dbg !76
  %1211 = insertelement <4 x i8> %1210, i8 %373, i32 2, !dbg !76
  %1212 = insertelement <4 x i8> %1211, i8 %374, i32 3, !dbg !76
  %1213 = bitcast <4 x i8> %1212 to i32, !dbg !76
  %1214 = insertelement <4 x i8> undef, i8 %377, i32 0, !dbg !76
  %1215 = insertelement <4 x i8> %1214, i8 %378, i32 1, !dbg !76
  %1216 = insertelement <4 x i8> %1215, i8 %379, i32 2, !dbg !76
  %1217 = insertelement <4 x i8> %1216, i8 %380, i32 3, !dbg !76
  %1218 = bitcast <4 x i8> %1217 to i32, !dbg !76
  %1219 = insertelement <4 x i8> undef, i8 %331, i32 0, !dbg !76
  %1220 = insertelement <4 x i8> %1219, i8 %332, i32 1, !dbg !76
  %1221 = insertelement <4 x i8> %1220, i8 %333, i32 2, !dbg !76
  %1222 = insertelement <4 x i8> %1221, i8 %334, i32 3, !dbg !76
  %1223 = bitcast <4 x i8> %1222 to i32, !dbg !76
  %1224 = insertelement <4 x i8> undef, i8 %337, i32 0, !dbg !76
  %1225 = insertelement <4 x i8> %1224, i8 %338, i32 1, !dbg !76
  %1226 = insertelement <4 x i8> %1225, i8 %339, i32 2, !dbg !76
  %1227 = insertelement <4 x i8> %1226, i8 %340, i32 3, !dbg !76
  %1228 = bitcast <4 x i8> %1227 to i32, !dbg !76
  %1229 = insertelement <4 x i8> undef, i8 %343, i32 0, !dbg !76
  %1230 = insertelement <4 x i8> %1229, i8 %344, i32 1, !dbg !76
  %1231 = insertelement <4 x i8> %1230, i8 %345, i32 2, !dbg !76
  %1232 = insertelement <4 x i8> %1231, i8 %346, i32 3, !dbg !76
  %1233 = bitcast <4 x i8> %1232 to i32, !dbg !76
  %1234 = insertelement <4 x i8> undef, i8 %349, i32 0, !dbg !76
  %1235 = insertelement <4 x i8> %1234, i8 %350, i32 1, !dbg !76
  %1236 = insertelement <4 x i8> %1235, i8 %351, i32 2, !dbg !76
  %1237 = insertelement <4 x i8> %1236, i8 %352, i32 3, !dbg !76
  %1238 = bitcast <4 x i8> %1237 to i32, !dbg !76
  %1239 = insertelement <4 x i8> undef, i8 %386, i32 0, !dbg !76
  %1240 = insertelement <4 x i8> %1239, i8 %387, i32 1, !dbg !76
  %1241 = insertelement <4 x i8> %1240, i8 %388, i32 2, !dbg !76
  %1242 = insertelement <4 x i8> %1241, i8 %389, i32 3, !dbg !76
  %1243 = bitcast <4 x i8> %1242 to i32, !dbg !76
  %1244 = insertelement <4 x i8> undef, i8 %392, i32 0, !dbg !76
  %1245 = insertelement <4 x i8> %1244, i8 %393, i32 1, !dbg !76
  %1246 = insertelement <4 x i8> %1245, i8 %394, i32 2, !dbg !76
  %1247 = insertelement <4 x i8> %1246, i8 %395, i32 3, !dbg !76
  %1248 = bitcast <4 x i8> %1247 to i32, !dbg !76
  %1249 = insertelement <4 x i8> undef, i8 %398, i32 0, !dbg !76
  %1250 = insertelement <4 x i8> %1249, i8 %399, i32 1, !dbg !76
  %1251 = insertelement <4 x i8> %1250, i8 %400, i32 2, !dbg !76
  %1252 = insertelement <4 x i8> %1251, i8 %401, i32 3, !dbg !76
  %1253 = bitcast <4 x i8> %1252 to i32, !dbg !76
  %1254 = insertelement <4 x i8> undef, i8 %404, i32 0, !dbg !76
  %1255 = insertelement <4 x i8> %1254, i8 %405, i32 1, !dbg !76
  %1256 = insertelement <4 x i8> %1255, i8 %406, i32 2, !dbg !76
  %1257 = insertelement <4 x i8> %1256, i8 %407, i32 3, !dbg !76
  %1258 = bitcast <4 x i8> %1257 to i32, !dbg !76
  %1259 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 0, !dbg !76
  %1260 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 1, !dbg !76
  %1261 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 2, !dbg !76
  %1262 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 3, !dbg !76
  %1263 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 4, !dbg !76
  %1264 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 5, !dbg !76
  %1265 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 6, !dbg !76
  %1266 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 7, !dbg !76
  %1267 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 8, !dbg !76
  %1268 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 9, !dbg !76
  %1269 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 10, !dbg !76
  %1270 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 11, !dbg !76
  %1271 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 12, !dbg !76
  %1272 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 13, !dbg !76
  %1273 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 14, !dbg !76
  %1274 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 15, !dbg !76
  %1275 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 16, !dbg !76
  %1276 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 17, !dbg !76
  %1277 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 18, !dbg !76
  %1278 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 19, !dbg !76
  %1279 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 20, !dbg !76
  %1280 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 21, !dbg !76
  %1281 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 22, !dbg !76
  %1282 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 23, !dbg !76
  %1283 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 24, !dbg !76
  %1284 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 25, !dbg !76
  %1285 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 26, !dbg !76
  %1286 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 27, !dbg !76
  %1287 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 28, !dbg !76
  %1288 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 29, !dbg !76
  %1289 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 30, !dbg !76
  %1290 = extractvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %258, 31, !dbg !76
  %1291 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1259, float %1260, float %1261, float %1262, i32 %863, i32 %868, i32 %873, i32 %878, i32 %1183, i32 %1188), !dbg !76
  %1292 = extractvalue { float, float, float, float } %1291, 0, !dbg !76
  %1293 = extractvalue { float, float, float, float } %1291, 1, !dbg !76
  %1294 = extractvalue { float, float, float, float } %1291, 2, !dbg !76
  %1295 = extractvalue { float, float, float, float } %1291, 3, !dbg !76
  %1296 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1263, float %1264, float %1265, float %1266, i32 %863, i32 %868, i32 %873, i32 %878, i32 %1223, i32 %1228), !dbg !76
  %1297 = extractvalue { float, float, float, float } %1296, 0, !dbg !76
  %1298 = extractvalue { float, float, float, float } %1296, 1, !dbg !76
  %1299 = extractvalue { float, float, float, float } %1296, 2, !dbg !76
  %1300 = extractvalue { float, float, float, float } %1296, 3, !dbg !76
  %1301 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1267, float %1268, float %1269, float %1270, i32 %943, i32 %948, i32 %953, i32 %958, i32 %1183, i32 %1188), !dbg !76
  %1302 = extractvalue { float, float, float, float } %1301, 0, !dbg !76
  %1303 = extractvalue { float, float, float, float } %1301, 1, !dbg !76
  %1304 = extractvalue { float, float, float, float } %1301, 2, !dbg !76
  %1305 = extractvalue { float, float, float, float } %1301, 3, !dbg !76
  %1306 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1271, float %1272, float %1273, float %1274, i32 %943, i32 %948, i32 %953, i32 %958, i32 %1223, i32 %1228), !dbg !76
  %1307 = extractvalue { float, float, float, float } %1306, 0, !dbg !76
  %1308 = extractvalue { float, float, float, float } %1306, 1, !dbg !76
  %1309 = extractvalue { float, float, float, float } %1306, 2, !dbg !76
  %1310 = extractvalue { float, float, float, float } %1306, 3, !dbg !76
  %1311 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1275, float %1276, float %1277, float %1278, i32 %1023, i32 %1028, i32 %1033, i32 %1038, i32 %1183, i32 %1188), !dbg !76
  %1312 = extractvalue { float, float, float, float } %1311, 0, !dbg !76
  %1313 = extractvalue { float, float, float, float } %1311, 1, !dbg !76
  %1314 = extractvalue { float, float, float, float } %1311, 2, !dbg !76
  %1315 = extractvalue { float, float, float, float } %1311, 3, !dbg !76
  %1316 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1279, float %1280, float %1281, float %1282, i32 %1023, i32 %1028, i32 %1033, i32 %1038, i32 %1223, i32 %1228), !dbg !76
  %1317 = extractvalue { float, float, float, float } %1316, 0, !dbg !76
  %1318 = extractvalue { float, float, float, float } %1316, 1, !dbg !76
  %1319 = extractvalue { float, float, float, float } %1316, 2, !dbg !76
  %1320 = extractvalue { float, float, float, float } %1316, 3, !dbg !76
  %1321 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1283, float %1284, float %1285, float %1286, i32 %1103, i32 %1108, i32 %1113, i32 %1118, i32 %1183, i32 %1188), !dbg !76
  %1322 = extractvalue { float, float, float, float } %1321, 0, !dbg !76
  %1323 = extractvalue { float, float, float, float } %1321, 1, !dbg !76
  %1324 = extractvalue { float, float, float, float } %1321, 2, !dbg !76
  %1325 = extractvalue { float, float, float, float } %1321, 3, !dbg !76
  %1326 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1287, float %1288, float %1289, float %1290, i32 %1103, i32 %1108, i32 %1113, i32 %1118, i32 %1223, i32 %1228), !dbg !76
  %1327 = extractvalue { float, float, float, float } %1326, 0, !dbg !76
  %1328 = extractvalue { float, float, float, float } %1326, 1, !dbg !76
  %1329 = extractvalue { float, float, float, float } %1326, 2, !dbg !76
  %1330 = extractvalue { float, float, float, float } %1326, 3, !dbg !76
  %1331 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1292, float %1293, float %1294, float %1295, i32 %883, i32 %888, i32 %893, i32 %898, i32 %1193, i32 %1198), !dbg !76
  %1332 = extractvalue { float, float, float, float } %1331, 0, !dbg !76
  %1333 = extractvalue { float, float, float, float } %1331, 1, !dbg !76
  %1334 = extractvalue { float, float, float, float } %1331, 2, !dbg !76
  %1335 = extractvalue { float, float, float, float } %1331, 3, !dbg !76
  %1336 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1297, float %1298, float %1299, float %1300, i32 %883, i32 %888, i32 %893, i32 %898, i32 %1233, i32 %1238), !dbg !76
  %1337 = extractvalue { float, float, float, float } %1336, 0, !dbg !76
  %1338 = extractvalue { float, float, float, float } %1336, 1, !dbg !76
  %1339 = extractvalue { float, float, float, float } %1336, 2, !dbg !76
  %1340 = extractvalue { float, float, float, float } %1336, 3, !dbg !76
  %1341 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1302, float %1303, float %1304, float %1305, i32 %963, i32 %968, i32 %973, i32 %978, i32 %1193, i32 %1198), !dbg !76
  %1342 = extractvalue { float, float, float, float } %1341, 0, !dbg !76
  %1343 = extractvalue { float, float, float, float } %1341, 1, !dbg !76
  %1344 = extractvalue { float, float, float, float } %1341, 2, !dbg !76
  %1345 = extractvalue { float, float, float, float } %1341, 3, !dbg !76
  %1346 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1307, float %1308, float %1309, float %1310, i32 %963, i32 %968, i32 %973, i32 %978, i32 %1233, i32 %1238), !dbg !76
  %1347 = extractvalue { float, float, float, float } %1346, 0, !dbg !76
  %1348 = extractvalue { float, float, float, float } %1346, 1, !dbg !76
  %1349 = extractvalue { float, float, float, float } %1346, 2, !dbg !76
  %1350 = extractvalue { float, float, float, float } %1346, 3, !dbg !76
  %1351 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1312, float %1313, float %1314, float %1315, i32 %1043, i32 %1048, i32 %1053, i32 %1058, i32 %1193, i32 %1198), !dbg !76
  %1352 = extractvalue { float, float, float, float } %1351, 0, !dbg !76
  %1353 = extractvalue { float, float, float, float } %1351, 1, !dbg !76
  %1354 = extractvalue { float, float, float, float } %1351, 2, !dbg !76
  %1355 = extractvalue { float, float, float, float } %1351, 3, !dbg !76
  %1356 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1317, float %1318, float %1319, float %1320, i32 %1043, i32 %1048, i32 %1053, i32 %1058, i32 %1233, i32 %1238), !dbg !76
  %1357 = extractvalue { float, float, float, float } %1356, 0, !dbg !76
  %1358 = extractvalue { float, float, float, float } %1356, 1, !dbg !76
  %1359 = extractvalue { float, float, float, float } %1356, 2, !dbg !76
  %1360 = extractvalue { float, float, float, float } %1356, 3, !dbg !76
  %1361 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1322, float %1323, float %1324, float %1325, i32 %1123, i32 %1128, i32 %1133, i32 %1138, i32 %1193, i32 %1198), !dbg !76
  %1362 = extractvalue { float, float, float, float } %1361, 0, !dbg !76
  %1363 = extractvalue { float, float, float, float } %1361, 1, !dbg !76
  %1364 = extractvalue { float, float, float, float } %1361, 2, !dbg !76
  %1365 = extractvalue { float, float, float, float } %1361, 3, !dbg !76
  %1366 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1327, float %1328, float %1329, float %1330, i32 %1123, i32 %1128, i32 %1133, i32 %1138, i32 %1233, i32 %1238), !dbg !76
  %1367 = extractvalue { float, float, float, float } %1366, 0, !dbg !76
  %1368 = extractvalue { float, float, float, float } %1366, 1, !dbg !76
  %1369 = extractvalue { float, float, float, float } %1366, 2, !dbg !76
  %1370 = extractvalue { float, float, float, float } %1366, 3, !dbg !76
  %1371 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1332, float %1333, float %1334, float %1335, i32 %903, i32 %908, i32 %913, i32 %918, i32 %1203, i32 %1208), !dbg !76
  %1372 = extractvalue { float, float, float, float } %1371, 0, !dbg !76
  %1373 = extractvalue { float, float, float, float } %1371, 1, !dbg !76
  %1374 = extractvalue { float, float, float, float } %1371, 2, !dbg !76
  %1375 = extractvalue { float, float, float, float } %1371, 3, !dbg !76
  %1376 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1337, float %1338, float %1339, float %1340, i32 %903, i32 %908, i32 %913, i32 %918, i32 %1243, i32 %1248), !dbg !76
  %1377 = extractvalue { float, float, float, float } %1376, 0, !dbg !76
  %1378 = extractvalue { float, float, float, float } %1376, 1, !dbg !76
  %1379 = extractvalue { float, float, float, float } %1376, 2, !dbg !76
  %1380 = extractvalue { float, float, float, float } %1376, 3, !dbg !76
  %1381 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1342, float %1343, float %1344, float %1345, i32 %983, i32 %988, i32 %993, i32 %998, i32 %1203, i32 %1208), !dbg !76
  %1382 = extractvalue { float, float, float, float } %1381, 0, !dbg !76
  %1383 = extractvalue { float, float, float, float } %1381, 1, !dbg !76
  %1384 = extractvalue { float, float, float, float } %1381, 2, !dbg !76
  %1385 = extractvalue { float, float, float, float } %1381, 3, !dbg !76
  %1386 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1347, float %1348, float %1349, float %1350, i32 %983, i32 %988, i32 %993, i32 %998, i32 %1243, i32 %1248), !dbg !76
  %1387 = extractvalue { float, float, float, float } %1386, 0, !dbg !76
  %1388 = extractvalue { float, float, float, float } %1386, 1, !dbg !76
  %1389 = extractvalue { float, float, float, float } %1386, 2, !dbg !76
  %1390 = extractvalue { float, float, float, float } %1386, 3, !dbg !76
  %1391 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1352, float %1353, float %1354, float %1355, i32 %1063, i32 %1068, i32 %1073, i32 %1078, i32 %1203, i32 %1208), !dbg !76
  %1392 = extractvalue { float, float, float, float } %1391, 0, !dbg !76
  %1393 = extractvalue { float, float, float, float } %1391, 1, !dbg !76
  %1394 = extractvalue { float, float, float, float } %1391, 2, !dbg !76
  %1395 = extractvalue { float, float, float, float } %1391, 3, !dbg !76
  %1396 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1357, float %1358, float %1359, float %1360, i32 %1063, i32 %1068, i32 %1073, i32 %1078, i32 %1243, i32 %1248), !dbg !76
  %1397 = extractvalue { float, float, float, float } %1396, 0, !dbg !76
  %1398 = extractvalue { float, float, float, float } %1396, 1, !dbg !76
  %1399 = extractvalue { float, float, float, float } %1396, 2, !dbg !76
  %1400 = extractvalue { float, float, float, float } %1396, 3, !dbg !76
  %1401 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1362, float %1363, float %1364, float %1365, i32 %1143, i32 %1148, i32 %1153, i32 %1158, i32 %1203, i32 %1208), !dbg !76
  %1402 = extractvalue { float, float, float, float } %1401, 0, !dbg !76
  %1403 = extractvalue { float, float, float, float } %1401, 1, !dbg !76
  %1404 = extractvalue { float, float, float, float } %1401, 2, !dbg !76
  %1405 = extractvalue { float, float, float, float } %1401, 3, !dbg !76
  %1406 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1367, float %1368, float %1369, float %1370, i32 %1143, i32 %1148, i32 %1153, i32 %1158, i32 %1243, i32 %1248), !dbg !76
  %1407 = extractvalue { float, float, float, float } %1406, 0, !dbg !76
  %1408 = extractvalue { float, float, float, float } %1406, 1, !dbg !76
  %1409 = extractvalue { float, float, float, float } %1406, 2, !dbg !76
  %1410 = extractvalue { float, float, float, float } %1406, 3, !dbg !76
  %1411 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1372, float %1373, float %1374, float %1375, i32 %923, i32 %928, i32 %933, i32 %938, i32 %1213, i32 %1218), !dbg !76
  %1412 = extractvalue { float, float, float, float } %1411, 0, !dbg !76
  %1413 = extractvalue { float, float, float, float } %1411, 1, !dbg !76
  %1414 = extractvalue { float, float, float, float } %1411, 2, !dbg !76
  %1415 = extractvalue { float, float, float, float } %1411, 3, !dbg !76
  %1416 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1377, float %1378, float %1379, float %1380, i32 %923, i32 %928, i32 %933, i32 %938, i32 %1253, i32 %1258), !dbg !76
  %1417 = extractvalue { float, float, float, float } %1416, 0, !dbg !76
  %1418 = extractvalue { float, float, float, float } %1416, 1, !dbg !76
  %1419 = extractvalue { float, float, float, float } %1416, 2, !dbg !76
  %1420 = extractvalue { float, float, float, float } %1416, 3, !dbg !76
  %1421 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1382, float %1383, float %1384, float %1385, i32 %1003, i32 %1008, i32 %1013, i32 %1018, i32 %1213, i32 %1218), !dbg !76
  %1422 = extractvalue { float, float, float, float } %1421, 0, !dbg !76
  %1423 = extractvalue { float, float, float, float } %1421, 1, !dbg !76
  %1424 = extractvalue { float, float, float, float } %1421, 2, !dbg !76
  %1425 = extractvalue { float, float, float, float } %1421, 3, !dbg !76
  %1426 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1387, float %1388, float %1389, float %1390, i32 %1003, i32 %1008, i32 %1013, i32 %1018, i32 %1253, i32 %1258), !dbg !76
  %1427 = extractvalue { float, float, float, float } %1426, 0, !dbg !76
  %1428 = extractvalue { float, float, float, float } %1426, 1, !dbg !76
  %1429 = extractvalue { float, float, float, float } %1426, 2, !dbg !76
  %1430 = extractvalue { float, float, float, float } %1426, 3, !dbg !76
  %1431 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1392, float %1393, float %1394, float %1395, i32 %1083, i32 %1088, i32 %1093, i32 %1098, i32 %1213, i32 %1218), !dbg !76
  %1432 = extractvalue { float, float, float, float } %1431, 0, !dbg !76
  %1433 = extractvalue { float, float, float, float } %1431, 1, !dbg !76
  %1434 = extractvalue { float, float, float, float } %1431, 2, !dbg !76
  %1435 = extractvalue { float, float, float, float } %1431, 3, !dbg !76
  %1436 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1397, float %1398, float %1399, float %1400, i32 %1083, i32 %1088, i32 %1093, i32 %1098, i32 %1253, i32 %1258), !dbg !76
  %1437 = extractvalue { float, float, float, float } %1436, 0, !dbg !76
  %1438 = extractvalue { float, float, float, float } %1436, 1, !dbg !76
  %1439 = extractvalue { float, float, float, float } %1436, 2, !dbg !76
  %1440 = extractvalue { float, float, float, float } %1436, 3, !dbg !76
  %1441 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1402, float %1403, float %1404, float %1405, i32 %1163, i32 %1168, i32 %1173, i32 %1178, i32 %1213, i32 %1218), !dbg !76
  %1442 = extractvalue { float, float, float, float } %1441, 0, !dbg !76
  %1443 = extractvalue { float, float, float, float } %1441, 1, !dbg !76
  %1444 = extractvalue { float, float, float, float } %1441, 2, !dbg !76
  %1445 = extractvalue { float, float, float, float } %1441, 3, !dbg !76
  %1446 = call { float, float, float, float } asm sideeffect "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 { $0, $1, $2, $3 }, { $8, $9, $10, $11 }, { $12, $13 }, { $4, $5, $6, $7 };", "=f,=f,=f,=f,0,1,2,3,r,r,r,r,r,r"(float %1407, float %1408, float %1409, float %1410, i32 %1163, i32 %1168, i32 %1173, i32 %1178, i32 %1253, i32 %1258), !dbg !76
  %1447 = extractvalue { float, float, float, float } %1446, 0, !dbg !76
  %1448 = extractvalue { float, float, float, float } %1446, 1, !dbg !76
  %1449 = extractvalue { float, float, float, float } %1446, 2, !dbg !76
  %1450 = extractvalue { float, float, float, float } %1446, 3, !dbg !76
  %1451 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } undef, float %1412, 0, !dbg !76
  %1452 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1451, float %1413, 1, !dbg !76
  %1453 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1452, float %1414, 2, !dbg !76
  %1454 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1453, float %1415, 3, !dbg !76
  %1455 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1454, float %1417, 4, !dbg !76
  %1456 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1455, float %1418, 5, !dbg !76
  %1457 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1456, float %1419, 6, !dbg !76
  %1458 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1457, float %1420, 7, !dbg !76
  %1459 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1458, float %1422, 8, !dbg !76
  %1460 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1459, float %1423, 9, !dbg !76
  %1461 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1460, float %1424, 10, !dbg !76
  %1462 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1461, float %1425, 11, !dbg !76
  %1463 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1462, float %1427, 12, !dbg !76
  %1464 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1463, float %1428, 13, !dbg !76
  %1465 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1464, float %1429, 14, !dbg !76
  %1466 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1465, float %1430, 15, !dbg !76
  %1467 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1466, float %1432, 16, !dbg !76
  %1468 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1467, float %1433, 17, !dbg !76
  %1469 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1468, float %1434, 18, !dbg !76
  %1470 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1469, float %1435, 19, !dbg !76
  %1471 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1470, float %1437, 20, !dbg !76
  %1472 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1471, float %1438, 21, !dbg !76
  %1473 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1472, float %1439, 22, !dbg !76
  %1474 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1473, float %1440, 23, !dbg !76
  %1475 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1474, float %1442, 24, !dbg !76
  %1476 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1475, float %1443, 25, !dbg !76
  %1477 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1476, float %1444, 26, !dbg !76
  %1478 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1477, float %1445, 27, !dbg !76
  %1479 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1478, float %1447, 28, !dbg !76
  %1480 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1479, float %1448, 29, !dbg !76
  %1481 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1480, float %1449, 30, !dbg !76
  %1482 = insertvalue { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } %1481, float %1450, 31, !dbg !76
  %1483 = icmp eq i32 %255, %226, !dbg !17
  br i1 %1483, label %1484, label %2403, !dbg !17

1484:                                             ; preds = %276
  %1485 = add i32 %257, 36, !dbg !77
  %1486 = srem i32 %1485, %195, !dbg !78
  %1487 = sdiv i32 %1486, %197, !dbg !80
  %1488 = mul i32 %1487, 8, !dbg !82
  %1489 = sub i32 %194, %1488, !dbg !83
  %1490 = call i32 @llvm.smin.i32(i32 %1489, i32 8), !dbg !84
  %1491 = icmp sge i32 %1490, 0, !dbg !85
  call void @llvm.assume(i1 %1491), !dbg !86
  %1492 = srem i32 %1486, %197, !dbg !87
  %1493 = sdiv i32 %1492, %1490, !dbg !88
  %1494 = call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l"(ptr addrspace(1) %277), !dbg !89
  %1495 = bitcast i32 %1494 to <1 x i32>, !dbg !89
  %1496 = extractelement <1 x i32> %1495, i32 0, !dbg !89
  %1497 = and i32 %1496, 65535, !dbg !91
  %1498 = ashr i32 %1496, 16, !dbg !92
  %1499 = getelementptr i32, ptr addrspace(1) %46, i32 %1497, !dbg !93
  %1500 = call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l"(ptr addrspace(1) %1499), !dbg !94
  %1501 = bitcast i32 %1500 to <1 x i32>, !dbg !94
  %1502 = extractelement <1 x i32> %1501, i32 0, !dbg !94
  %1503 = mul i32 %1498, 16, !dbg !95
  %1504 = mul i32 %1493, 256, !dbg !96
  %1505 = getelementptr i32, ptr addrspace(1) %45, i32 %1497, !dbg !97
  %1506 = call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l"(ptr addrspace(1) %1505), !dbg !98
  %1507 = bitcast i32 %1506 to <1 x i32>, !dbg !98
  %1508 = extractelement <1 x i32> %1507, i32 0, !dbg !98
  %1509 = add i32 %1504, %212, !dbg !99
  %1510 = icmp slt i32 %1509, %42, !dbg !100
  %1511 = mul i32 %1497, %41, !dbg !101
  %1512 = getelementptr float, ptr addrspace(1) %40, i32 %1511, !dbg !102
  %1513 = getelementptr float, ptr addrspace(1) %1512, i32 %1509, !dbg !102
  %1514 = call { i32, i32 } asm sideeffect "mov.u32 $0, $2;\0A\09mov.u32 $1, $3;\0A\09@$5 ld.global.v2.b32 { $0, $1 }, [ $4 + 0 ];", "=r,=r,r,r,l,b"(i32 0, i32 0, ptr addrspace(1) %1513, i1 %1510), !dbg !103
  %1515 = extractvalue { i32, i32 } %1514, 0, !dbg !103
  %1516 = bitcast i32 %1515 to <1 x float>, !dbg !103
  %1517 = extractvalue { i32, i32 } %1514, 1, !dbg !103
  %1518 = bitcast i32 %1517 to <1 x float>, !dbg !103
  %1519 = extractelement <1 x float> %1516, i32 0, !dbg !103
  %1520 = extractelement <1 x float> %1518, i32 0, !dbg !103
  %1521 = call i32 asm sideeffect "mov.u32 $0, 0x0;\0A\09ld.global.b32 { $0 }, [ $1 + 0 ];", "=r,l"(ptr addrspace(1) %39), !dbg !104
  %1522 = bitcast i32 %1521 to <1 x float>, !dbg !104
  %1523 = extractelement <1 x float> %1522, i32 0, !dbg !104
  %1524 = fmul float %1412, %1523, !dbg !108
  %1525 = fmul float %1413, %1523, !dbg !108
  %1526 = fmul float %1414, %1523, !dbg !108
  %1527 = fmul float %1415, %1523, !dbg !108
  %1528 = fmul float %1417, %1523, !dbg !108
  %1529 = fmul float %1418, %1523, !dbg !108
  %1530 = fmul float %1419, %1523, !dbg !108
  %1531 = fmul float %1420, %1523, !dbg !108
  %1532 = fmul float %1422, %1523, !dbg !108
  %1533 = fmul float %1423, %1523, !dbg !108
  %1534 = fmul float %1424, %1523, !dbg !108
  %1535 = fmul float %1425, %1523, !dbg !108
  %1536 = fmul float %1427, %1523, !dbg !108
  %1537 = fmul float %1428, %1523, !dbg !108
  %1538 = fmul float %1429, %1523, !dbg !108
  %1539 = fmul float %1430, %1523, !dbg !108
  %1540 = fmul float %1432, %1523, !dbg !108
  %1541 = fmul float %1433, %1523, !dbg !108
  %1542 = fmul float %1434, %1523, !dbg !108
  %1543 = fmul float %1435, %1523, !dbg !108
  %1544 = fmul float %1437, %1523, !dbg !108
  %1545 = fmul float %1438, %1523, !dbg !108
  %1546 = fmul float %1439, %1523, !dbg !108
  %1547 = fmul float %1440, %1523, !dbg !108
  %1548 = fmul float %1442, %1523, !dbg !108
  %1549 = fmul float %1443, %1523, !dbg !108
  %1550 = fmul float %1444, %1523, !dbg !108
  %1551 = fmul float %1445, %1523, !dbg !108
  %1552 = fmul float %1447, %1523, !dbg !108
  %1553 = fmul float %1448, %1523, !dbg !108
  %1554 = fmul float %1449, %1523, !dbg !108
  %1555 = fmul float %1450, %1523, !dbg !108
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !109
  %1556 = and i32 %206, 3, !dbg !109
  %1557 = shl i32 %1556, 3, !dbg !109
  %1558 = and i32 %206, 120, !dbg !109
  %1559 = shl i32 %1558, 2, !dbg !109
  %1560 = and i32 %206, 4, !dbg !109
  %1561 = or disjoint i32 %1559, %1560, !dbg !109
  %1562 = or disjoint i32 %1561, %1557, !dbg !109
  %1563 = xor i32 0, %1562, !dbg !109
  %1564 = xor i32 %1563, 0, !dbg !109
  %1565 = xor i32 %1564, 0, !dbg !109
  %1566 = add i32 %1565, 0, !dbg !109
  %1567 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1566, !dbg !109
  %1568 = insertelement <1 x float> undef, float %1519, i32 0, !dbg !109
  %1569 = extractelement <1 x float> %1568, i32 0, !dbg !109
  %1570 = bitcast float %1569 to i32, !dbg !109
  %1571 = insertelement <1 x i32> undef, i32 %1570, i32 0, !dbg !109
  store <1 x i32> %1571, ptr addrspace(3) %1567, align 4, !dbg !109
  %1572 = xor i32 %1564, 516, !dbg !109
  %1573 = add i32 %1572, 0, !dbg !109
  %1574 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1573, !dbg !109
  %1575 = insertelement <1 x float> undef, float %1520, i32 0, !dbg !109
  %1576 = extractelement <1 x float> %1575, i32 0, !dbg !109
  %1577 = bitcast float %1576 to i32, !dbg !109
  %1578 = insertelement <1 x i32> undef, i32 %1577, i32 0, !dbg !109
  store <1 x i32> %1578, ptr addrspace(3) %1574, align 4, !dbg !109
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !109
  %1579 = lshr i32 %1558, 0, !dbg !109
  %1580 = icmp eq i32 %1560, 0, !dbg !109
  %1581 = select i1 %1580, i32 0, i32 516, !dbg !109
  %1582 = or disjoint i32 %1581, %1579, !dbg !109
  %1583 = xor i32 0, %1582, !dbg !109
  %1584 = xor i32 %1583, 0, !dbg !109
  %1585 = xor i32 %1584, 0, !dbg !109
  %1586 = add i32 %1585, 0, !dbg !109
  %1587 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1586, !dbg !109
  %1588 = load i32, ptr addrspace(3) %1587, align 4, !dbg !109
  %1589 = insertelement <1 x i32> undef, i32 %1588, i32 0, !dbg !109
  %1590 = extractelement <1 x i32> %1589, i32 0, !dbg !109
  %1591 = bitcast i32 %1590 to float, !dbg !109
  %1592 = insertelement <1 x float> undef, float %1591, i32 0, !dbg !109
  %1593 = extractelement <1 x float> %1592, i32 0, !dbg !109
  %1594 = add i32 %1585, 128, !dbg !109
  %1595 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1594, !dbg !109
  %1596 = load i32, ptr addrspace(3) %1595, align 4, !dbg !109
  %1597 = insertelement <1 x i32> undef, i32 %1596, i32 0, !dbg !109
  %1598 = extractelement <1 x i32> %1597, i32 0, !dbg !109
  %1599 = bitcast i32 %1598 to float, !dbg !109
  %1600 = insertelement <1 x float> undef, float %1599, i32 0, !dbg !109
  %1601 = extractelement <1 x float> %1600, i32 0, !dbg !109
  %1602 = add i32 %1585, 256, !dbg !109
  %1603 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1602, !dbg !109
  %1604 = load i32, ptr addrspace(3) %1603, align 4, !dbg !109
  %1605 = insertelement <1 x i32> undef, i32 %1604, i32 0, !dbg !109
  %1606 = extractelement <1 x i32> %1605, i32 0, !dbg !109
  %1607 = bitcast i32 %1606 to float, !dbg !109
  %1608 = insertelement <1 x float> undef, float %1607, i32 0, !dbg !109
  %1609 = extractelement <1 x float> %1608, i32 0, !dbg !109
  %1610 = add i32 %1585, 384, !dbg !109
  %1611 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1610, !dbg !109
  %1612 = load i32, ptr addrspace(3) %1611, align 4, !dbg !109
  %1613 = insertelement <1 x i32> undef, i32 %1612, i32 0, !dbg !109
  %1614 = extractelement <1 x i32> %1613, i32 0, !dbg !109
  %1615 = bitcast i32 %1614 to float, !dbg !109
  %1616 = insertelement <1 x float> undef, float %1615, i32 0, !dbg !109
  %1617 = extractelement <1 x float> %1616, i32 0, !dbg !109
  %1618 = xor i32 %1584, 4, !dbg !109
  %1619 = add i32 %1618, 0, !dbg !109
  %1620 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1619, !dbg !109
  %1621 = load i32, ptr addrspace(3) %1620, align 4, !dbg !109
  %1622 = insertelement <1 x i32> undef, i32 %1621, i32 0, !dbg !109
  %1623 = extractelement <1 x i32> %1622, i32 0, !dbg !109
  %1624 = bitcast i32 %1623 to float, !dbg !109
  %1625 = insertelement <1 x float> undef, float %1624, i32 0, !dbg !109
  %1626 = extractelement <1 x float> %1625, i32 0, !dbg !109
  %1627 = add i32 %1618, 128, !dbg !109
  %1628 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1627, !dbg !109
  %1629 = load i32, ptr addrspace(3) %1628, align 4, !dbg !109
  %1630 = insertelement <1 x i32> undef, i32 %1629, i32 0, !dbg !109
  %1631 = extractelement <1 x i32> %1630, i32 0, !dbg !109
  %1632 = bitcast i32 %1631 to float, !dbg !109
  %1633 = insertelement <1 x float> undef, float %1632, i32 0, !dbg !109
  %1634 = extractelement <1 x float> %1633, i32 0, !dbg !109
  %1635 = add i32 %1618, 256, !dbg !109
  %1636 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1635, !dbg !109
  %1637 = load i32, ptr addrspace(3) %1636, align 4, !dbg !109
  %1638 = insertelement <1 x i32> undef, i32 %1637, i32 0, !dbg !109
  %1639 = extractelement <1 x i32> %1638, i32 0, !dbg !109
  %1640 = bitcast i32 %1639 to float, !dbg !109
  %1641 = insertelement <1 x float> undef, float %1640, i32 0, !dbg !109
  %1642 = extractelement <1 x float> %1641, i32 0, !dbg !109
  %1643 = add i32 %1618, 384, !dbg !109
  %1644 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1643, !dbg !109
  %1645 = load i32, ptr addrspace(3) %1644, align 4, !dbg !109
  %1646 = insertelement <1 x i32> undef, i32 %1645, i32 0, !dbg !109
  %1647 = extractelement <1 x i32> %1646, i32 0, !dbg !109
  %1648 = bitcast i32 %1647 to float, !dbg !109
  %1649 = insertelement <1 x float> undef, float %1648, i32 0, !dbg !109
  %1650 = extractelement <1 x float> %1649, i32 0, !dbg !109
  %1651 = fadd float %1524, %1593, !dbg !110
  %1652 = fadd float %1525, %1593, !dbg !110
  %1653 = fadd float %1526, %1626, !dbg !110
  %1654 = fadd float %1527, %1626, !dbg !110
  %1655 = fadd float %1528, %1593, !dbg !110
  %1656 = fadd float %1529, %1593, !dbg !110
  %1657 = fadd float %1530, %1626, !dbg !110
  %1658 = fadd float %1531, %1626, !dbg !110
  %1659 = fadd float %1532, %1601, !dbg !110
  %1660 = fadd float %1533, %1601, !dbg !110
  %1661 = fadd float %1534, %1634, !dbg !110
  %1662 = fadd float %1535, %1634, !dbg !110
  %1663 = fadd float %1536, %1601, !dbg !110
  %1664 = fadd float %1537, %1601, !dbg !110
  %1665 = fadd float %1538, %1634, !dbg !110
  %1666 = fadd float %1539, %1634, !dbg !110
  %1667 = fadd float %1540, %1609, !dbg !110
  %1668 = fadd float %1541, %1609, !dbg !110
  %1669 = fadd float %1542, %1642, !dbg !110
  %1670 = fadd float %1543, %1642, !dbg !110
  %1671 = fadd float %1544, %1609, !dbg !110
  %1672 = fadd float %1545, %1609, !dbg !110
  %1673 = fadd float %1546, %1642, !dbg !110
  %1674 = fadd float %1547, %1642, !dbg !110
  %1675 = fadd float %1548, %1617, !dbg !110
  %1676 = fadd float %1549, %1617, !dbg !110
  %1677 = fadd float %1550, %1650, !dbg !110
  %1678 = fadd float %1551, %1650, !dbg !110
  %1679 = fadd float %1552, %1617, !dbg !110
  %1680 = fadd float %1553, %1617, !dbg !110
  %1681 = fadd float %1554, %1650, !dbg !110
  %1682 = fadd float %1555, %1650, !dbg !110
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !111
  %1683 = and i32 %206, 49, !dbg !111
  %1684 = shl i32 %1683, 4, !dbg !111
  %1685 = and i32 %206, 2, !dbg !111
  %1686 = icmp eq i32 %1685, 0, !dbg !111
  %1687 = select i1 %1686, i32 0, i32 2080, !dbg !111
  %1688 = shl i32 %1560, 1, !dbg !111
  %1689 = and i32 %206, 8, !dbg !111
  %1690 = icmp eq i32 %1689, 0, !dbg !111
  %1691 = select i1 %1690, i32 0, i32 4160, !dbg !111
  %1692 = and i32 %206, 64, !dbg !111
  %1693 = lshr i32 %1692, 1, !dbg !111
  %1694 = or disjoint i32 %1688, %1691, !dbg !111
  %1695 = xor i32 %1684, %1687, !dbg !111
  %1696 = xor i32 %1695, %1693, !dbg !111
  %1697 = or disjoint i32 %1694, %1696, !dbg !111
  %1698 = xor i32 0, %1697, !dbg !111
  %1699 = xor i32 %1698, 0, !dbg !111
  %1700 = xor i32 %1699, 0, !dbg !111
  %1701 = add i32 %1700, 0, !dbg !111
  %1702 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1701, !dbg !111
  %1703 = insertelement <2 x float> undef, float %1651, i32 0, !dbg !111
  %1704 = insertelement <2 x float> %1703, float %1653, i32 1, !dbg !111
  %1705 = extractelement <2 x float> %1704, i32 0, !dbg !111
  %1706 = extractelement <2 x float> %1704, i32 1, !dbg !111
  %1707 = bitcast float %1705 to i32, !dbg !111
  %1708 = bitcast float %1706 to i32, !dbg !111
  %1709 = insertelement <2 x i32> undef, i32 %1707, i32 0, !dbg !111
  %1710 = insertelement <2 x i32> %1709, i32 %1708, i32 1, !dbg !111
  store <2 x i32> %1710, ptr addrspace(3) %1702, align 8, !dbg !111
  %1711 = add i32 %1700, 128, !dbg !111
  %1712 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1711, !dbg !111
  %1713 = insertelement <2 x float> undef, float %1655, i32 0, !dbg !111
  %1714 = insertelement <2 x float> %1713, float %1657, i32 1, !dbg !111
  %1715 = extractelement <2 x float> %1714, i32 0, !dbg !111
  %1716 = extractelement <2 x float> %1714, i32 1, !dbg !111
  %1717 = bitcast float %1715 to i32, !dbg !111
  %1718 = bitcast float %1716 to i32, !dbg !111
  %1719 = insertelement <2 x i32> undef, i32 %1717, i32 0, !dbg !111
  %1720 = insertelement <2 x i32> %1719, i32 %1718, i32 1, !dbg !111
  store <2 x i32> %1720, ptr addrspace(3) %1712, align 8, !dbg !111
  %1721 = xor i32 %1699, 1040, !dbg !111
  %1722 = add i32 %1721, 0, !dbg !111
  %1723 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1722, !dbg !111
  %1724 = insertelement <2 x float> undef, float %1652, i32 0, !dbg !111
  %1725 = insertelement <2 x float> %1724, float %1654, i32 1, !dbg !111
  %1726 = extractelement <2 x float> %1725, i32 0, !dbg !111
  %1727 = extractelement <2 x float> %1725, i32 1, !dbg !111
  %1728 = bitcast float %1726 to i32, !dbg !111
  %1729 = bitcast float %1727 to i32, !dbg !111
  %1730 = insertelement <2 x i32> undef, i32 %1728, i32 0, !dbg !111
  %1731 = insertelement <2 x i32> %1730, i32 %1729, i32 1, !dbg !111
  store <2 x i32> %1731, ptr addrspace(3) %1723, align 8, !dbg !111
  %1732 = add i32 %1721, 128, !dbg !111
  %1733 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1732, !dbg !111
  %1734 = insertelement <2 x float> undef, float %1656, i32 0, !dbg !111
  %1735 = insertelement <2 x float> %1734, float %1658, i32 1, !dbg !111
  %1736 = extractelement <2 x float> %1735, i32 0, !dbg !111
  %1737 = extractelement <2 x float> %1735, i32 1, !dbg !111
  %1738 = bitcast float %1736 to i32, !dbg !111
  %1739 = bitcast float %1737 to i32, !dbg !111
  %1740 = insertelement <2 x i32> undef, i32 %1738, i32 0, !dbg !111
  %1741 = insertelement <2 x i32> %1740, i32 %1739, i32 1, !dbg !111
  store <2 x i32> %1741, ptr addrspace(3) %1733, align 8, !dbg !111
  %1742 = xor i32 %1699, 64, !dbg !111
  %1743 = add i32 %1742, 0, !dbg !111
  %1744 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1743, !dbg !111
  %1745 = insertelement <2 x float> undef, float %1659, i32 0, !dbg !111
  %1746 = insertelement <2 x float> %1745, float %1661, i32 1, !dbg !111
  %1747 = extractelement <2 x float> %1746, i32 0, !dbg !111
  %1748 = extractelement <2 x float> %1746, i32 1, !dbg !111
  %1749 = bitcast float %1747 to i32, !dbg !111
  %1750 = bitcast float %1748 to i32, !dbg !111
  %1751 = insertelement <2 x i32> undef, i32 %1749, i32 0, !dbg !111
  %1752 = insertelement <2 x i32> %1751, i32 %1750, i32 1, !dbg !111
  store <2 x i32> %1752, ptr addrspace(3) %1744, align 8, !dbg !111
  %1753 = add i32 %1742, 128, !dbg !111
  %1754 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1753, !dbg !111
  %1755 = insertelement <2 x float> undef, float %1663, i32 0, !dbg !111
  %1756 = insertelement <2 x float> %1755, float %1665, i32 1, !dbg !111
  %1757 = extractelement <2 x float> %1756, i32 0, !dbg !111
  %1758 = extractelement <2 x float> %1756, i32 1, !dbg !111
  %1759 = bitcast float %1757 to i32, !dbg !111
  %1760 = bitcast float %1758 to i32, !dbg !111
  %1761 = insertelement <2 x i32> undef, i32 %1759, i32 0, !dbg !111
  %1762 = insertelement <2 x i32> %1761, i32 %1760, i32 1, !dbg !111
  store <2 x i32> %1762, ptr addrspace(3) %1754, align 8, !dbg !111
  %1763 = xor i32 %1699, 1104, !dbg !111
  %1764 = add i32 %1763, 0, !dbg !111
  %1765 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1764, !dbg !111
  %1766 = insertelement <2 x float> undef, float %1660, i32 0, !dbg !111
  %1767 = insertelement <2 x float> %1766, float %1662, i32 1, !dbg !111
  %1768 = extractelement <2 x float> %1767, i32 0, !dbg !111
  %1769 = extractelement <2 x float> %1767, i32 1, !dbg !111
  %1770 = bitcast float %1768 to i32, !dbg !111
  %1771 = bitcast float %1769 to i32, !dbg !111
  %1772 = insertelement <2 x i32> undef, i32 %1770, i32 0, !dbg !111
  %1773 = insertelement <2 x i32> %1772, i32 %1771, i32 1, !dbg !111
  store <2 x i32> %1773, ptr addrspace(3) %1765, align 8, !dbg !111
  %1774 = add i32 %1763, 128, !dbg !111
  %1775 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1774, !dbg !111
  %1776 = insertelement <2 x float> undef, float %1664, i32 0, !dbg !111
  %1777 = insertelement <2 x float> %1776, float %1666, i32 1, !dbg !111
  %1778 = extractelement <2 x float> %1777, i32 0, !dbg !111
  %1779 = extractelement <2 x float> %1777, i32 1, !dbg !111
  %1780 = bitcast float %1778 to i32, !dbg !111
  %1781 = bitcast float %1779 to i32, !dbg !111
  %1782 = insertelement <2 x i32> undef, i32 %1780, i32 0, !dbg !111
  %1783 = insertelement <2 x i32> %1782, i32 %1781, i32 1, !dbg !111
  store <2 x i32> %1783, ptr addrspace(3) %1775, align 8, !dbg !111
  %1784 = xor i32 %1699, 8200, !dbg !111
  %1785 = add i32 %1784, 0, !dbg !111
  %1786 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1785, !dbg !111
  %1787 = insertelement <2 x float> undef, float %1667, i32 0, !dbg !111
  %1788 = insertelement <2 x float> %1787, float %1669, i32 1, !dbg !111
  %1789 = extractelement <2 x float> %1788, i32 0, !dbg !111
  %1790 = extractelement <2 x float> %1788, i32 1, !dbg !111
  %1791 = bitcast float %1789 to i32, !dbg !111
  %1792 = bitcast float %1790 to i32, !dbg !111
  %1793 = insertelement <2 x i32> undef, i32 %1791, i32 0, !dbg !111
  %1794 = insertelement <2 x i32> %1793, i32 %1792, i32 1, !dbg !111
  store <2 x i32> %1794, ptr addrspace(3) %1786, align 8, !dbg !111
  %1795 = add i32 %1784, 128, !dbg !111
  %1796 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1795, !dbg !111
  %1797 = insertelement <2 x float> undef, float %1671, i32 0, !dbg !111
  %1798 = insertelement <2 x float> %1797, float %1673, i32 1, !dbg !111
  %1799 = extractelement <2 x float> %1798, i32 0, !dbg !111
  %1800 = extractelement <2 x float> %1798, i32 1, !dbg !111
  %1801 = bitcast float %1799 to i32, !dbg !111
  %1802 = bitcast float %1800 to i32, !dbg !111
  %1803 = insertelement <2 x i32> undef, i32 %1801, i32 0, !dbg !111
  %1804 = insertelement <2 x i32> %1803, i32 %1802, i32 1, !dbg !111
  store <2 x i32> %1804, ptr addrspace(3) %1796, align 8, !dbg !111
  %1805 = xor i32 %1699, 9240, !dbg !111
  %1806 = add i32 %1805, 0, !dbg !111
  %1807 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1806, !dbg !111
  %1808 = insertelement <2 x float> undef, float %1668, i32 0, !dbg !111
  %1809 = insertelement <2 x float> %1808, float %1670, i32 1, !dbg !111
  %1810 = extractelement <2 x float> %1809, i32 0, !dbg !111
  %1811 = extractelement <2 x float> %1809, i32 1, !dbg !111
  %1812 = bitcast float %1810 to i32, !dbg !111
  %1813 = bitcast float %1811 to i32, !dbg !111
  %1814 = insertelement <2 x i32> undef, i32 %1812, i32 0, !dbg !111
  %1815 = insertelement <2 x i32> %1814, i32 %1813, i32 1, !dbg !111
  store <2 x i32> %1815, ptr addrspace(3) %1807, align 8, !dbg !111
  %1816 = add i32 %1805, 128, !dbg !111
  %1817 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1816, !dbg !111
  %1818 = insertelement <2 x float> undef, float %1672, i32 0, !dbg !111
  %1819 = insertelement <2 x float> %1818, float %1674, i32 1, !dbg !111
  %1820 = extractelement <2 x float> %1819, i32 0, !dbg !111
  %1821 = extractelement <2 x float> %1819, i32 1, !dbg !111
  %1822 = bitcast float %1820 to i32, !dbg !111
  %1823 = bitcast float %1821 to i32, !dbg !111
  %1824 = insertelement <2 x i32> undef, i32 %1822, i32 0, !dbg !111
  %1825 = insertelement <2 x i32> %1824, i32 %1823, i32 1, !dbg !111
  store <2 x i32> %1825, ptr addrspace(3) %1817, align 8, !dbg !111
  %1826 = xor i32 %1699, 8264, !dbg !111
  %1827 = add i32 %1826, 0, !dbg !111
  %1828 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1827, !dbg !111
  %1829 = insertelement <2 x float> undef, float %1675, i32 0, !dbg !111
  %1830 = insertelement <2 x float> %1829, float %1677, i32 1, !dbg !111
  %1831 = extractelement <2 x float> %1830, i32 0, !dbg !111
  %1832 = extractelement <2 x float> %1830, i32 1, !dbg !111
  %1833 = bitcast float %1831 to i32, !dbg !111
  %1834 = bitcast float %1832 to i32, !dbg !111
  %1835 = insertelement <2 x i32> undef, i32 %1833, i32 0, !dbg !111
  %1836 = insertelement <2 x i32> %1835, i32 %1834, i32 1, !dbg !111
  store <2 x i32> %1836, ptr addrspace(3) %1828, align 8, !dbg !111
  %1837 = add i32 %1826, 128, !dbg !111
  %1838 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1837, !dbg !111
  %1839 = insertelement <2 x float> undef, float %1679, i32 0, !dbg !111
  %1840 = insertelement <2 x float> %1839, float %1681, i32 1, !dbg !111
  %1841 = extractelement <2 x float> %1840, i32 0, !dbg !111
  %1842 = extractelement <2 x float> %1840, i32 1, !dbg !111
  %1843 = bitcast float %1841 to i32, !dbg !111
  %1844 = bitcast float %1842 to i32, !dbg !111
  %1845 = insertelement <2 x i32> undef, i32 %1843, i32 0, !dbg !111
  %1846 = insertelement <2 x i32> %1845, i32 %1844, i32 1, !dbg !111
  store <2 x i32> %1846, ptr addrspace(3) %1838, align 8, !dbg !111
  %1847 = xor i32 %1699, 9304, !dbg !111
  %1848 = add i32 %1847, 0, !dbg !111
  %1849 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1848, !dbg !111
  %1850 = insertelement <2 x float> undef, float %1676, i32 0, !dbg !111
  %1851 = insertelement <2 x float> %1850, float %1678, i32 1, !dbg !111
  %1852 = extractelement <2 x float> %1851, i32 0, !dbg !111
  %1853 = extractelement <2 x float> %1851, i32 1, !dbg !111
  %1854 = bitcast float %1852 to i32, !dbg !111
  %1855 = bitcast float %1853 to i32, !dbg !111
  %1856 = insertelement <2 x i32> undef, i32 %1854, i32 0, !dbg !111
  %1857 = insertelement <2 x i32> %1856, i32 %1855, i32 1, !dbg !111
  store <2 x i32> %1857, ptr addrspace(3) %1849, align 8, !dbg !111
  %1858 = add i32 %1847, 128, !dbg !111
  %1859 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1858, !dbg !111
  %1860 = insertelement <2 x float> undef, float %1680, i32 0, !dbg !111
  %1861 = insertelement <2 x float> %1860, float %1682, i32 1, !dbg !111
  %1862 = extractelement <2 x float> %1861, i32 0, !dbg !111
  %1863 = extractelement <2 x float> %1861, i32 1, !dbg !111
  %1864 = bitcast float %1862 to i32, !dbg !111
  %1865 = bitcast float %1863 to i32, !dbg !111
  %1866 = insertelement <2 x i32> undef, i32 %1864, i32 0, !dbg !111
  %1867 = insertelement <2 x i32> %1866, i32 %1865, i32 1, !dbg !111
  store <2 x i32> %1867, ptr addrspace(3) %1859, align 8, !dbg !111
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !111
  %1868 = shl i32 %1556, 5, !dbg !111
  %1869 = select i1 %1580, i32 0, i32 8200, !dbg !111
  %1870 = select i1 %1690, i32 0, i32 1040, !dbg !111
  %1871 = and i32 %206, 32, !dbg !111
  %1872 = icmp eq i32 %1871, 0, !dbg !111
  %1873 = select i1 %1872, i32 0, i32 2080, !dbg !111
  %1874 = shl i32 %1692, 1, !dbg !111
  %1875 = or disjoint i32 %1869, %1874, !dbg !111
  %1876 = xor i32 %1868, %1870, !dbg !111
  %1877 = xor i32 %416, %1873, !dbg !111
  %1878 = xor i32 %1876, %1877, !dbg !111
  %1879 = or disjoint i32 %1875, %1878, !dbg !111
  %1880 = xor i32 0, %1879, !dbg !111
  %1881 = xor i32 %1880, 0, !dbg !111
  %1882 = xor i32 %1881, 0, !dbg !111
  %1883 = add i32 %1882, 0, !dbg !111
  %1884 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1883, !dbg !111
  %1885 = load <2 x i32>, ptr addrspace(3) %1884, align 8, !dbg !111
  %1886 = extractelement <2 x i32> %1885, i32 0, !dbg !111
  %1887 = extractelement <2 x i32> %1885, i32 1, !dbg !111
  %1888 = insertelement <2 x i32> undef, i32 %1886, i32 0, !dbg !111
  %1889 = insertelement <2 x i32> %1888, i32 %1887, i32 1, !dbg !111
  %1890 = extractelement <2 x i32> %1889, i32 0, !dbg !111
  %1891 = extractelement <2 x i32> %1889, i32 1, !dbg !111
  %1892 = bitcast i32 %1890 to float, !dbg !111
  %1893 = bitcast i32 %1891 to float, !dbg !111
  %1894 = insertelement <2 x float> undef, float %1892, i32 0, !dbg !111
  %1895 = insertelement <2 x float> %1894, float %1893, i32 1, !dbg !111
  %1896 = extractelement <2 x float> %1895, i32 0, !dbg !111
  %1897 = extractelement <2 x float> %1895, i32 1, !dbg !111
  %1898 = add i32 %1882, 256, !dbg !111
  %1899 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1898, !dbg !111
  %1900 = load <2 x i32>, ptr addrspace(3) %1899, align 8, !dbg !111
  %1901 = extractelement <2 x i32> %1900, i32 0, !dbg !111
  %1902 = extractelement <2 x i32> %1900, i32 1, !dbg !111
  %1903 = insertelement <2 x i32> undef, i32 %1901, i32 0, !dbg !111
  %1904 = insertelement <2 x i32> %1903, i32 %1902, i32 1, !dbg !111
  %1905 = extractelement <2 x i32> %1904, i32 0, !dbg !111
  %1906 = extractelement <2 x i32> %1904, i32 1, !dbg !111
  %1907 = bitcast i32 %1905 to float, !dbg !111
  %1908 = bitcast i32 %1906 to float, !dbg !111
  %1909 = insertelement <2 x float> undef, float %1907, i32 0, !dbg !111
  %1910 = insertelement <2 x float> %1909, float %1908, i32 1, !dbg !111
  %1911 = extractelement <2 x float> %1910, i32 0, !dbg !111
  %1912 = extractelement <2 x float> %1910, i32 1, !dbg !111
  %1913 = add i32 %1882, 512, !dbg !111
  %1914 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1913, !dbg !111
  %1915 = load <2 x i32>, ptr addrspace(3) %1914, align 8, !dbg !111
  %1916 = extractelement <2 x i32> %1915, i32 0, !dbg !111
  %1917 = extractelement <2 x i32> %1915, i32 1, !dbg !111
  %1918 = insertelement <2 x i32> undef, i32 %1916, i32 0, !dbg !111
  %1919 = insertelement <2 x i32> %1918, i32 %1917, i32 1, !dbg !111
  %1920 = extractelement <2 x i32> %1919, i32 0, !dbg !111
  %1921 = extractelement <2 x i32> %1919, i32 1, !dbg !111
  %1922 = bitcast i32 %1920 to float, !dbg !111
  %1923 = bitcast i32 %1921 to float, !dbg !111
  %1924 = insertelement <2 x float> undef, float %1922, i32 0, !dbg !111
  %1925 = insertelement <2 x float> %1924, float %1923, i32 1, !dbg !111
  %1926 = extractelement <2 x float> %1925, i32 0, !dbg !111
  %1927 = extractelement <2 x float> %1925, i32 1, !dbg !111
  %1928 = add i32 %1882, 768, !dbg !111
  %1929 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1928, !dbg !111
  %1930 = load <2 x i32>, ptr addrspace(3) %1929, align 8, !dbg !111
  %1931 = extractelement <2 x i32> %1930, i32 0, !dbg !111
  %1932 = extractelement <2 x i32> %1930, i32 1, !dbg !111
  %1933 = insertelement <2 x i32> undef, i32 %1931, i32 0, !dbg !111
  %1934 = insertelement <2 x i32> %1933, i32 %1932, i32 1, !dbg !111
  %1935 = extractelement <2 x i32> %1934, i32 0, !dbg !111
  %1936 = extractelement <2 x i32> %1934, i32 1, !dbg !111
  %1937 = bitcast i32 %1935 to float, !dbg !111
  %1938 = bitcast i32 %1936 to float, !dbg !111
  %1939 = insertelement <2 x float> undef, float %1937, i32 0, !dbg !111
  %1940 = insertelement <2 x float> %1939, float %1938, i32 1, !dbg !111
  %1941 = extractelement <2 x float> %1940, i32 0, !dbg !111
  %1942 = extractelement <2 x float> %1940, i32 1, !dbg !111
  %1943 = xor i32 %1881, 8, !dbg !111
  %1944 = add i32 %1943, 0, !dbg !111
  %1945 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1944, !dbg !111
  %1946 = load <2 x i32>, ptr addrspace(3) %1945, align 8, !dbg !111
  %1947 = extractelement <2 x i32> %1946, i32 0, !dbg !111
  %1948 = extractelement <2 x i32> %1946, i32 1, !dbg !111
  %1949 = insertelement <2 x i32> undef, i32 %1947, i32 0, !dbg !111
  %1950 = insertelement <2 x i32> %1949, i32 %1948, i32 1, !dbg !111
  %1951 = extractelement <2 x i32> %1950, i32 0, !dbg !111
  %1952 = extractelement <2 x i32> %1950, i32 1, !dbg !111
  %1953 = bitcast i32 %1951 to float, !dbg !111
  %1954 = bitcast i32 %1952 to float, !dbg !111
  %1955 = insertelement <2 x float> undef, float %1953, i32 0, !dbg !111
  %1956 = insertelement <2 x float> %1955, float %1954, i32 1, !dbg !111
  %1957 = extractelement <2 x float> %1956, i32 0, !dbg !111
  %1958 = extractelement <2 x float> %1956, i32 1, !dbg !111
  %1959 = add i32 %1943, 256, !dbg !111
  %1960 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1959, !dbg !111
  %1961 = load <2 x i32>, ptr addrspace(3) %1960, align 8, !dbg !111
  %1962 = extractelement <2 x i32> %1961, i32 0, !dbg !111
  %1963 = extractelement <2 x i32> %1961, i32 1, !dbg !111
  %1964 = insertelement <2 x i32> undef, i32 %1962, i32 0, !dbg !111
  %1965 = insertelement <2 x i32> %1964, i32 %1963, i32 1, !dbg !111
  %1966 = extractelement <2 x i32> %1965, i32 0, !dbg !111
  %1967 = extractelement <2 x i32> %1965, i32 1, !dbg !111
  %1968 = bitcast i32 %1966 to float, !dbg !111
  %1969 = bitcast i32 %1967 to float, !dbg !111
  %1970 = insertelement <2 x float> undef, float %1968, i32 0, !dbg !111
  %1971 = insertelement <2 x float> %1970, float %1969, i32 1, !dbg !111
  %1972 = extractelement <2 x float> %1971, i32 0, !dbg !111
  %1973 = extractelement <2 x float> %1971, i32 1, !dbg !111
  %1974 = add i32 %1943, 512, !dbg !111
  %1975 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1974, !dbg !111
  %1976 = load <2 x i32>, ptr addrspace(3) %1975, align 8, !dbg !111
  %1977 = extractelement <2 x i32> %1976, i32 0, !dbg !111
  %1978 = extractelement <2 x i32> %1976, i32 1, !dbg !111
  %1979 = insertelement <2 x i32> undef, i32 %1977, i32 0, !dbg !111
  %1980 = insertelement <2 x i32> %1979, i32 %1978, i32 1, !dbg !111
  %1981 = extractelement <2 x i32> %1980, i32 0, !dbg !111
  %1982 = extractelement <2 x i32> %1980, i32 1, !dbg !111
  %1983 = bitcast i32 %1981 to float, !dbg !111
  %1984 = bitcast i32 %1982 to float, !dbg !111
  %1985 = insertelement <2 x float> undef, float %1983, i32 0, !dbg !111
  %1986 = insertelement <2 x float> %1985, float %1984, i32 1, !dbg !111
  %1987 = extractelement <2 x float> %1986, i32 0, !dbg !111
  %1988 = extractelement <2 x float> %1986, i32 1, !dbg !111
  %1989 = add i32 %1943, 768, !dbg !111
  %1990 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %1989, !dbg !111
  %1991 = load <2 x i32>, ptr addrspace(3) %1990, align 8, !dbg !111
  %1992 = extractelement <2 x i32> %1991, i32 0, !dbg !111
  %1993 = extractelement <2 x i32> %1991, i32 1, !dbg !111
  %1994 = insertelement <2 x i32> undef, i32 %1992, i32 0, !dbg !111
  %1995 = insertelement <2 x i32> %1994, i32 %1993, i32 1, !dbg !111
  %1996 = extractelement <2 x i32> %1995, i32 0, !dbg !111
  %1997 = extractelement <2 x i32> %1995, i32 1, !dbg !111
  %1998 = bitcast i32 %1996 to float, !dbg !111
  %1999 = bitcast i32 %1997 to float, !dbg !111
  %2000 = insertelement <2 x float> undef, float %1998, i32 0, !dbg !111
  %2001 = insertelement <2 x float> %2000, float %1999, i32 1, !dbg !111
  %2002 = extractelement <2 x float> %2001, i32 0, !dbg !111
  %2003 = extractelement <2 x float> %2001, i32 1, !dbg !111
  %2004 = xor i32 %1881, 4160, !dbg !111
  %2005 = add i32 %2004, 0, !dbg !111
  %2006 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %2005, !dbg !111
  %2007 = load <2 x i32>, ptr addrspace(3) %2006, align 8, !dbg !111
  %2008 = extractelement <2 x i32> %2007, i32 0, !dbg !111
  %2009 = extractelement <2 x i32> %2007, i32 1, !dbg !111
  %2010 = insertelement <2 x i32> undef, i32 %2008, i32 0, !dbg !111
  %2011 = insertelement <2 x i32> %2010, i32 %2009, i32 1, !dbg !111
  %2012 = extractelement <2 x i32> %2011, i32 0, !dbg !111
  %2013 = extractelement <2 x i32> %2011, i32 1, !dbg !111
  %2014 = bitcast i32 %2012 to float, !dbg !111
  %2015 = bitcast i32 %2013 to float, !dbg !111
  %2016 = insertelement <2 x float> undef, float %2014, i32 0, !dbg !111
  %2017 = insertelement <2 x float> %2016, float %2015, i32 1, !dbg !111
  %2018 = extractelement <2 x float> %2017, i32 0, !dbg !111
  %2019 = extractelement <2 x float> %2017, i32 1, !dbg !111
  %2020 = add i32 %2004, 256, !dbg !111
  %2021 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %2020, !dbg !111
  %2022 = load <2 x i32>, ptr addrspace(3) %2021, align 8, !dbg !111
  %2023 = extractelement <2 x i32> %2022, i32 0, !dbg !111
  %2024 = extractelement <2 x i32> %2022, i32 1, !dbg !111
  %2025 = insertelement <2 x i32> undef, i32 %2023, i32 0, !dbg !111
  %2026 = insertelement <2 x i32> %2025, i32 %2024, i32 1, !dbg !111
  %2027 = extractelement <2 x i32> %2026, i32 0, !dbg !111
  %2028 = extractelement <2 x i32> %2026, i32 1, !dbg !111
  %2029 = bitcast i32 %2027 to float, !dbg !111
  %2030 = bitcast i32 %2028 to float, !dbg !111
  %2031 = insertelement <2 x float> undef, float %2029, i32 0, !dbg !111
  %2032 = insertelement <2 x float> %2031, float %2030, i32 1, !dbg !111
  %2033 = extractelement <2 x float> %2032, i32 0, !dbg !111
  %2034 = extractelement <2 x float> %2032, i32 1, !dbg !111
  %2035 = add i32 %2004, 512, !dbg !111
  %2036 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %2035, !dbg !111
  %2037 = load <2 x i32>, ptr addrspace(3) %2036, align 8, !dbg !111
  %2038 = extractelement <2 x i32> %2037, i32 0, !dbg !111
  %2039 = extractelement <2 x i32> %2037, i32 1, !dbg !111
  %2040 = insertelement <2 x i32> undef, i32 %2038, i32 0, !dbg !111
  %2041 = insertelement <2 x i32> %2040, i32 %2039, i32 1, !dbg !111
  %2042 = extractelement <2 x i32> %2041, i32 0, !dbg !111
  %2043 = extractelement <2 x i32> %2041, i32 1, !dbg !111
  %2044 = bitcast i32 %2042 to float, !dbg !111
  %2045 = bitcast i32 %2043 to float, !dbg !111
  %2046 = insertelement <2 x float> undef, float %2044, i32 0, !dbg !111
  %2047 = insertelement <2 x float> %2046, float %2045, i32 1, !dbg !111
  %2048 = extractelement <2 x float> %2047, i32 0, !dbg !111
  %2049 = extractelement <2 x float> %2047, i32 1, !dbg !111
  %2050 = add i32 %2004, 768, !dbg !111
  %2051 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %2050, !dbg !111
  %2052 = load <2 x i32>, ptr addrspace(3) %2051, align 8, !dbg !111
  %2053 = extractelement <2 x i32> %2052, i32 0, !dbg !111
  %2054 = extractelement <2 x i32> %2052, i32 1, !dbg !111
  %2055 = insertelement <2 x i32> undef, i32 %2053, i32 0, !dbg !111
  %2056 = insertelement <2 x i32> %2055, i32 %2054, i32 1, !dbg !111
  %2057 = extractelement <2 x i32> %2056, i32 0, !dbg !111
  %2058 = extractelement <2 x i32> %2056, i32 1, !dbg !111
  %2059 = bitcast i32 %2057 to float, !dbg !111
  %2060 = bitcast i32 %2058 to float, !dbg !111
  %2061 = insertelement <2 x float> undef, float %2059, i32 0, !dbg !111
  %2062 = insertelement <2 x float> %2061, float %2060, i32 1, !dbg !111
  %2063 = extractelement <2 x float> %2062, i32 0, !dbg !111
  %2064 = extractelement <2 x float> %2062, i32 1, !dbg !111
  %2065 = xor i32 %1881, 4168, !dbg !111
  %2066 = add i32 %2065, 0, !dbg !111
  %2067 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %2066, !dbg !111
  %2068 = load <2 x i32>, ptr addrspace(3) %2067, align 8, !dbg !111
  %2069 = extractelement <2 x i32> %2068, i32 0, !dbg !111
  %2070 = extractelement <2 x i32> %2068, i32 1, !dbg !111
  %2071 = insertelement <2 x i32> undef, i32 %2069, i32 0, !dbg !111
  %2072 = insertelement <2 x i32> %2071, i32 %2070, i32 1, !dbg !111
  %2073 = extractelement <2 x i32> %2072, i32 0, !dbg !111
  %2074 = extractelement <2 x i32> %2072, i32 1, !dbg !111
  %2075 = bitcast i32 %2073 to float, !dbg !111
  %2076 = bitcast i32 %2074 to float, !dbg !111
  %2077 = insertelement <2 x float> undef, float %2075, i32 0, !dbg !111
  %2078 = insertelement <2 x float> %2077, float %2076, i32 1, !dbg !111
  %2079 = extractelement <2 x float> %2078, i32 0, !dbg !111
  %2080 = extractelement <2 x float> %2078, i32 1, !dbg !111
  %2081 = add i32 %2065, 256, !dbg !111
  %2082 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %2081, !dbg !111
  %2083 = load <2 x i32>, ptr addrspace(3) %2082, align 8, !dbg !111
  %2084 = extractelement <2 x i32> %2083, i32 0, !dbg !111
  %2085 = extractelement <2 x i32> %2083, i32 1, !dbg !111
  %2086 = insertelement <2 x i32> undef, i32 %2084, i32 0, !dbg !111
  %2087 = insertelement <2 x i32> %2086, i32 %2085, i32 1, !dbg !111
  %2088 = extractelement <2 x i32> %2087, i32 0, !dbg !111
  %2089 = extractelement <2 x i32> %2087, i32 1, !dbg !111
  %2090 = bitcast i32 %2088 to float, !dbg !111
  %2091 = bitcast i32 %2089 to float, !dbg !111
  %2092 = insertelement <2 x float> undef, float %2090, i32 0, !dbg !111
  %2093 = insertelement <2 x float> %2092, float %2091, i32 1, !dbg !111
  %2094 = extractelement <2 x float> %2093, i32 0, !dbg !111
  %2095 = extractelement <2 x float> %2093, i32 1, !dbg !111
  %2096 = add i32 %2065, 512, !dbg !111
  %2097 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %2096, !dbg !111
  %2098 = load <2 x i32>, ptr addrspace(3) %2097, align 8, !dbg !111
  %2099 = extractelement <2 x i32> %2098, i32 0, !dbg !111
  %2100 = extractelement <2 x i32> %2098, i32 1, !dbg !111
  %2101 = insertelement <2 x i32> undef, i32 %2099, i32 0, !dbg !111
  %2102 = insertelement <2 x i32> %2101, i32 %2100, i32 1, !dbg !111
  %2103 = extractelement <2 x i32> %2102, i32 0, !dbg !111
  %2104 = extractelement <2 x i32> %2102, i32 1, !dbg !111
  %2105 = bitcast i32 %2103 to float, !dbg !111
  %2106 = bitcast i32 %2104 to float, !dbg !111
  %2107 = insertelement <2 x float> undef, float %2105, i32 0, !dbg !111
  %2108 = insertelement <2 x float> %2107, float %2106, i32 1, !dbg !111
  %2109 = extractelement <2 x float> %2108, i32 0, !dbg !111
  %2110 = extractelement <2 x float> %2108, i32 1, !dbg !111
  %2111 = add i32 %2065, 768, !dbg !111
  %2112 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 65536), i32 %2111, !dbg !111
  %2113 = load <2 x i32>, ptr addrspace(3) %2112, align 8, !dbg !111
  %2114 = extractelement <2 x i32> %2113, i32 0, !dbg !111
  %2115 = extractelement <2 x i32> %2113, i32 1, !dbg !111
  %2116 = insertelement <2 x i32> undef, i32 %2114, i32 0, !dbg !111
  %2117 = insertelement <2 x i32> %2116, i32 %2115, i32 1, !dbg !111
  %2118 = extractelement <2 x i32> %2117, i32 0, !dbg !111
  %2119 = extractelement <2 x i32> %2117, i32 1, !dbg !111
  %2120 = bitcast i32 %2118 to float, !dbg !111
  %2121 = bitcast i32 %2119 to float, !dbg !111
  %2122 = insertelement <2 x float> undef, float %2120, i32 0, !dbg !111
  %2123 = insertelement <2 x float> %2122, float %2121, i32 1, !dbg !111
  %2124 = extractelement <2 x float> %2123, i32 0, !dbg !111
  %2125 = extractelement <2 x float> %2123, i32 1, !dbg !111
  %2126 = call float @llvm.minnum.f32(float %1896, float %52), !dbg !112
  %2127 = call float @llvm.minnum.f32(float %2018, float %52), !dbg !112
  %2128 = call float @llvm.minnum.f32(float %1911, float %52), !dbg !112
  %2129 = call float @llvm.minnum.f32(float %2033, float %52), !dbg !112
  %2130 = call float @llvm.minnum.f32(float %1897, float %52), !dbg !112
  %2131 = call float @llvm.minnum.f32(float %2019, float %52), !dbg !112
  %2132 = call float @llvm.minnum.f32(float %1912, float %52), !dbg !112
  %2133 = call float @llvm.minnum.f32(float %2034, float %52), !dbg !112
  %2134 = call float @llvm.minnum.f32(float %1926, float %52), !dbg !112
  %2135 = call float @llvm.minnum.f32(float %2048, float %52), !dbg !112
  %2136 = call float @llvm.minnum.f32(float %1941, float %52), !dbg !112
  %2137 = call float @llvm.minnum.f32(float %2063, float %52), !dbg !112
  %2138 = call float @llvm.minnum.f32(float %1927, float %52), !dbg !112
  %2139 = call float @llvm.minnum.f32(float %2049, float %52), !dbg !112
  %2140 = call float @llvm.minnum.f32(float %1942, float %52), !dbg !112
  %2141 = call float @llvm.minnum.f32(float %2064, float %52), !dbg !112
  %2142 = call float @llvm.maxnum.f32(float %1957, float %213), !dbg !114
  %2143 = call float @llvm.minnum.f32(float %2142, float %52), !dbg !114
  %2144 = call float @llvm.maxnum.f32(float %2079, float %213), !dbg !114
  %2145 = call float @llvm.minnum.f32(float %2144, float %52), !dbg !114
  %2146 = call float @llvm.maxnum.f32(float %1972, float %213), !dbg !114
  %2147 = call float @llvm.minnum.f32(float %2146, float %52), !dbg !114
  %2148 = call float @llvm.maxnum.f32(float %2094, float %213), !dbg !114
  %2149 = call float @llvm.minnum.f32(float %2148, float %52), !dbg !114
  %2150 = call float @llvm.maxnum.f32(float %1958, float %213), !dbg !114
  %2151 = call float @llvm.minnum.f32(float %2150, float %52), !dbg !114
  %2152 = call float @llvm.maxnum.f32(float %2080, float %213), !dbg !114
  %2153 = call float @llvm.minnum.f32(float %2152, float %52), !dbg !114
  %2154 = call float @llvm.maxnum.f32(float %1973, float %213), !dbg !114
  %2155 = call float @llvm.minnum.f32(float %2154, float %52), !dbg !114
  %2156 = call float @llvm.maxnum.f32(float %2095, float %213), !dbg !114
  %2157 = call float @llvm.minnum.f32(float %2156, float %52), !dbg !114
  %2158 = call float @llvm.maxnum.f32(float %1987, float %213), !dbg !114
  %2159 = call float @llvm.minnum.f32(float %2158, float %52), !dbg !114
  %2160 = call float @llvm.maxnum.f32(float %2109, float %213), !dbg !114
  %2161 = call float @llvm.minnum.f32(float %2160, float %52), !dbg !114
  %2162 = call float @llvm.maxnum.f32(float %2002, float %213), !dbg !114
  %2163 = call float @llvm.minnum.f32(float %2162, float %52), !dbg !114
  %2164 = call float @llvm.maxnum.f32(float %2124, float %213), !dbg !114
  %2165 = call float @llvm.minnum.f32(float %2164, float %52), !dbg !114
  %2166 = call float @llvm.maxnum.f32(float %1988, float %213), !dbg !114
  %2167 = call float @llvm.minnum.f32(float %2166, float %52), !dbg !114
  %2168 = call float @llvm.maxnum.f32(float %2110, float %213), !dbg !114
  %2169 = call float @llvm.minnum.f32(float %2168, float %52), !dbg !114
  %2170 = call float @llvm.maxnum.f32(float %2003, float %213), !dbg !114
  %2171 = call float @llvm.minnum.f32(float %2170, float %52), !dbg !114
  %2172 = call float @llvm.maxnum.f32(float %2125, float %213), !dbg !114
  %2173 = call float @llvm.minnum.f32(float %2172, float %52), !dbg !114
  %2174 = fmul float %214, %2126, !dbg !73
  %2175 = fmul float %214, %2127, !dbg !73
  %2176 = fmul float %214, %2128, !dbg !73
  %2177 = fmul float %214, %2129, !dbg !73
  %2178 = fmul float %214, %2130, !dbg !73
  %2179 = fmul float %214, %2131, !dbg !73
  %2180 = fmul float %214, %2132, !dbg !73
  %2181 = fmul float %214, %2133, !dbg !73
  %2182 = fmul float %214, %2134, !dbg !73
  %2183 = fmul float %214, %2135, !dbg !73
  %2184 = fmul float %214, %2136, !dbg !73
  %2185 = fmul float %214, %2137, !dbg !73
  %2186 = fmul float %214, %2138, !dbg !73
  %2187 = fmul float %214, %2139, !dbg !73
  %2188 = fmul float %214, %2140, !dbg !73
  %2189 = fmul float %214, %2141, !dbg !73
  %2190 = fmul float %2174, 0x3FF7154760000000, !dbg !115
  %2191 = fmul float %2175, 0x3FF7154760000000, !dbg !115
  %2192 = fmul float %2176, 0x3FF7154760000000, !dbg !115
  %2193 = fmul float %2177, 0x3FF7154760000000, !dbg !115
  %2194 = fmul float %2178, 0x3FF7154760000000, !dbg !115
  %2195 = fmul float %2179, 0x3FF7154760000000, !dbg !115
  %2196 = fmul float %2180, 0x3FF7154760000000, !dbg !115
  %2197 = fmul float %2181, 0x3FF7154760000000, !dbg !115
  %2198 = fmul float %2182, 0x3FF7154760000000, !dbg !115
  %2199 = fmul float %2183, 0x3FF7154760000000, !dbg !115
  %2200 = fmul float %2184, 0x3FF7154760000000, !dbg !115
  %2201 = fmul float %2185, 0x3FF7154760000000, !dbg !115
  %2202 = fmul float %2186, 0x3FF7154760000000, !dbg !115
  %2203 = fmul float %2187, 0x3FF7154760000000, !dbg !115
  %2204 = fmul float %2188, 0x3FF7154760000000, !dbg !115
  %2205 = fmul float %2189, 0x3FF7154760000000, !dbg !115
  %2206 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2190), !dbg !117
  %2207 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2191), !dbg !117
  %2208 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2192), !dbg !117
  %2209 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2193), !dbg !117
  %2210 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2194), !dbg !117
  %2211 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2195), !dbg !117
  %2212 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2196), !dbg !117
  %2213 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2197), !dbg !117
  %2214 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2198), !dbg !117
  %2215 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2199), !dbg !117
  %2216 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2200), !dbg !117
  %2217 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2201), !dbg !117
  %2218 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2202), !dbg !117
  %2219 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2203), !dbg !117
  %2220 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2204), !dbg !117
  %2221 = call float asm "ex2.approx.ftz.f32 $0, $1;", "=r, r"(float %2205), !dbg !117
  %2222 = fadd float %2206, 1.000000e+00, !dbg !118
  %2223 = fadd float %2207, 1.000000e+00, !dbg !118
  %2224 = fadd float %2208, 1.000000e+00, !dbg !118
  %2225 = fadd float %2209, 1.000000e+00, !dbg !118
  %2226 = fadd float %2210, 1.000000e+00, !dbg !118
  %2227 = fadd float %2211, 1.000000e+00, !dbg !118
  %2228 = fadd float %2212, 1.000000e+00, !dbg !118
  %2229 = fadd float %2213, 1.000000e+00, !dbg !118
  %2230 = fadd float %2214, 1.000000e+00, !dbg !118
  %2231 = fadd float %2215, 1.000000e+00, !dbg !118
  %2232 = fadd float %2216, 1.000000e+00, !dbg !118
  %2233 = fadd float %2217, 1.000000e+00, !dbg !118
  %2234 = fadd float %2218, 1.000000e+00, !dbg !118
  %2235 = fadd float %2219, 1.000000e+00, !dbg !118
  %2236 = fadd float %2220, 1.000000e+00, !dbg !118
  %2237 = fadd float %2221, 1.000000e+00, !dbg !118
  %2238 = call float @llvm.nvvm.div.full(float %2126, float %2222), !dbg !119
  %2239 = call float @llvm.nvvm.div.full(float %2127, float %2223), !dbg !119
  %2240 = call float @llvm.nvvm.div.full(float %2128, float %2224), !dbg !119
  %2241 = call float @llvm.nvvm.div.full(float %2129, float %2225), !dbg !119
  %2242 = call float @llvm.nvvm.div.full(float %2130, float %2226), !dbg !119
  %2243 = call float @llvm.nvvm.div.full(float %2131, float %2227), !dbg !119
  %2244 = call float @llvm.nvvm.div.full(float %2132, float %2228), !dbg !119
  %2245 = call float @llvm.nvvm.div.full(float %2133, float %2229), !dbg !119
  %2246 = call float @llvm.nvvm.div.full(float %2134, float %2230), !dbg !119
  %2247 = call float @llvm.nvvm.div.full(float %2135, float %2231), !dbg !119
  %2248 = call float @llvm.nvvm.div.full(float %2136, float %2232), !dbg !119
  %2249 = call float @llvm.nvvm.div.full(float %2137, float %2233), !dbg !119
  %2250 = call float @llvm.nvvm.div.full(float %2138, float %2234), !dbg !119
  %2251 = call float @llvm.nvvm.div.full(float %2139, float %2235), !dbg !119
  %2252 = call float @llvm.nvvm.div.full(float %2140, float %2236), !dbg !119
  %2253 = call float @llvm.nvvm.div.full(float %2141, float %2237), !dbg !119
  %2254 = call float @llvm.fma.f32(float %2238, float %2143, float %2238), !dbg !120
  %2255 = call float @llvm.fma.f32(float %2239, float %2145, float %2239), !dbg !120
  %2256 = call float @llvm.fma.f32(float %2240, float %2147, float %2240), !dbg !120
  %2257 = call float @llvm.fma.f32(float %2241, float %2149, float %2241), !dbg !120
  %2258 = call float @llvm.fma.f32(float %2242, float %2151, float %2242), !dbg !120
  %2259 = call float @llvm.fma.f32(float %2243, float %2153, float %2243), !dbg !120
  %2260 = call float @llvm.fma.f32(float %2244, float %2155, float %2244), !dbg !120
  %2261 = call float @llvm.fma.f32(float %2245, float %2157, float %2245), !dbg !120
  %2262 = call float @llvm.fma.f32(float %2246, float %2159, float %2246), !dbg !120
  %2263 = call float @llvm.fma.f32(float %2247, float %2161, float %2247), !dbg !120
  %2264 = call float @llvm.fma.f32(float %2248, float %2163, float %2248), !dbg !120
  %2265 = call float @llvm.fma.f32(float %2249, float %2165, float %2249), !dbg !120
  %2266 = call float @llvm.fma.f32(float %2250, float %2167, float %2250), !dbg !120
  %2267 = call float @llvm.fma.f32(float %2251, float %2169, float %2251), !dbg !120
  %2268 = call float @llvm.fma.f32(float %2252, float %2171, float %2252), !dbg !120
  %2269 = call float @llvm.fma.f32(float %2253, float %2173, float %2253), !dbg !120
  %2270 = sdiv i32 %1504, 2, !dbg !121
  %2271 = insertelement <1 x float> undef, float %2254, i32 0, !dbg !122
  %2272 = insertelement <1 x float> undef, float %2255, i32 0, !dbg !122
  %2273 = bitcast <1 x float> %2271 to i32, !dbg !122
  %2274 = bitcast <1 x float> %2272 to i32, !dbg !122
  %2275 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %2273, i32 %2274), !dbg !122
  %2276 = extractelement <2 x i8> %2275, i32 0, !dbg !122
  %2277 = extractelement <2 x i8> %2275, i32 1, !dbg !122
  %2278 = insertelement <1 x float> undef, float %2256, i32 0, !dbg !122
  %2279 = insertelement <1 x float> undef, float %2257, i32 0, !dbg !122
  %2280 = bitcast <1 x float> %2278 to i32, !dbg !122
  %2281 = bitcast <1 x float> %2279 to i32, !dbg !122
  %2282 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %2280, i32 %2281), !dbg !122
  %2283 = extractelement <2 x i8> %2282, i32 0, !dbg !122
  %2284 = extractelement <2 x i8> %2282, i32 1, !dbg !122
  %2285 = insertelement <1 x float> undef, float %2258, i32 0, !dbg !122
  %2286 = insertelement <1 x float> undef, float %2259, i32 0, !dbg !122
  %2287 = bitcast <1 x float> %2285 to i32, !dbg !122
  %2288 = bitcast <1 x float> %2286 to i32, !dbg !122
  %2289 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %2287, i32 %2288), !dbg !122
  %2290 = extractelement <2 x i8> %2289, i32 0, !dbg !122
  %2291 = extractelement <2 x i8> %2289, i32 1, !dbg !122
  %2292 = insertelement <1 x float> undef, float %2260, i32 0, !dbg !122
  %2293 = insertelement <1 x float> undef, float %2261, i32 0, !dbg !122
  %2294 = bitcast <1 x float> %2292 to i32, !dbg !122
  %2295 = bitcast <1 x float> %2293 to i32, !dbg !122
  %2296 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %2294, i32 %2295), !dbg !122
  %2297 = extractelement <2 x i8> %2296, i32 0, !dbg !122
  %2298 = extractelement <2 x i8> %2296, i32 1, !dbg !122
  %2299 = insertelement <1 x float> undef, float %2262, i32 0, !dbg !122
  %2300 = insertelement <1 x float> undef, float %2263, i32 0, !dbg !122
  %2301 = bitcast <1 x float> %2299 to i32, !dbg !122
  %2302 = bitcast <1 x float> %2300 to i32, !dbg !122
  %2303 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %2301, i32 %2302), !dbg !122
  %2304 = extractelement <2 x i8> %2303, i32 0, !dbg !122
  %2305 = extractelement <2 x i8> %2303, i32 1, !dbg !122
  %2306 = insertelement <1 x float> undef, float %2264, i32 0, !dbg !122
  %2307 = insertelement <1 x float> undef, float %2265, i32 0, !dbg !122
  %2308 = bitcast <1 x float> %2306 to i32, !dbg !122
  %2309 = bitcast <1 x float> %2307 to i32, !dbg !122
  %2310 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %2308, i32 %2309), !dbg !122
  %2311 = extractelement <2 x i8> %2310, i32 0, !dbg !122
  %2312 = extractelement <2 x i8> %2310, i32 1, !dbg !122
  %2313 = insertelement <1 x float> undef, float %2266, i32 0, !dbg !122
  %2314 = insertelement <1 x float> undef, float %2267, i32 0, !dbg !122
  %2315 = bitcast <1 x float> %2313 to i32, !dbg !122
  %2316 = bitcast <1 x float> %2314 to i32, !dbg !122
  %2317 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %2315, i32 %2316), !dbg !122
  %2318 = extractelement <2 x i8> %2317, i32 0, !dbg !122
  %2319 = extractelement <2 x i8> %2317, i32 1, !dbg !122
  %2320 = insertelement <1 x float> undef, float %2268, i32 0, !dbg !122
  %2321 = insertelement <1 x float> undef, float %2269, i32 0, !dbg !122
  %2322 = bitcast <1 x float> %2320 to i32, !dbg !122
  %2323 = bitcast <1 x float> %2321 to i32, !dbg !122
  %2324 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %2322, i32 %2323), !dbg !122
  %2325 = extractelement <2 x i8> %2324, i32 0, !dbg !122
  %2326 = extractelement <2 x i8> %2324, i32 1, !dbg !122
  %2327 = sub i32 1073741824, %1508, !dbg !123
  %2328 = add i32 %2327, %1503, !dbg !123
  %2329 = add i32 %1502, %1508, !dbg !126
  call void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 0), !dbg !127
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !127
  %2330 = shl i32 %207, 4, !dbg !127
  %2331 = and i32 %206, 56, !dbg !127
  %2332 = shl i32 %2331, 1, !dbg !127
  %2333 = xor i32 %2330, %2332, !dbg !127
  %2334 = or disjoint i32 0, %2333, !dbg !127
  %2335 = xor i32 0, %2334, !dbg !127
  %2336 = xor i32 %2335, 0, !dbg !127
  %2337 = xor i32 %2336, 0, !dbg !127
  %2338 = add i32 %2337, 0, !dbg !127
  %2339 = getelementptr inbounds i8, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 86016), i32 %2338, !dbg !127
  %2340 = insertelement <16 x i8> undef, i8 %2276, i32 0, !dbg !127
  %2341 = insertelement <16 x i8> %2340, i8 %2277, i32 1, !dbg !127
  %2342 = insertelement <16 x i8> %2341, i8 %2283, i32 2, !dbg !127
  %2343 = insertelement <16 x i8> %2342, i8 %2284, i32 3, !dbg !127
  %2344 = insertelement <16 x i8> %2343, i8 %2290, i32 4, !dbg !127
  %2345 = insertelement <16 x i8> %2344, i8 %2291, i32 5, !dbg !127
  %2346 = insertelement <16 x i8> %2345, i8 %2297, i32 6, !dbg !127
  %2347 = insertelement <16 x i8> %2346, i8 %2298, i32 7, !dbg !127
  %2348 = insertelement <16 x i8> %2347, i8 %2304, i32 8, !dbg !127
  %2349 = insertelement <16 x i8> %2348, i8 %2305, i32 9, !dbg !127
  %2350 = insertelement <16 x i8> %2349, i8 %2311, i32 10, !dbg !127
  %2351 = insertelement <16 x i8> %2350, i8 %2312, i32 11, !dbg !127
  %2352 = insertelement <16 x i8> %2351, i8 %2318, i32 12, !dbg !127
  %2353 = insertelement <16 x i8> %2352, i8 %2319, i32 13, !dbg !127
  %2354 = insertelement <16 x i8> %2353, i8 %2325, i32 14, !dbg !127
  %2355 = insertelement <16 x i8> %2354, i8 %2326, i32 15, !dbg !127
  %2356 = extractelement <16 x i8> %2355, i32 0, !dbg !127
  %2357 = extractelement <16 x i8> %2355, i32 1, !dbg !127
  %2358 = extractelement <16 x i8> %2355, i32 2, !dbg !127
  %2359 = extractelement <16 x i8> %2355, i32 3, !dbg !127
  %2360 = extractelement <16 x i8> %2355, i32 4, !dbg !127
  %2361 = extractelement <16 x i8> %2355, i32 5, !dbg !127
  %2362 = extractelement <16 x i8> %2355, i32 6, !dbg !127
  %2363 = extractelement <16 x i8> %2355, i32 7, !dbg !127
  %2364 = extractelement <16 x i8> %2355, i32 8, !dbg !127
  %2365 = extractelement <16 x i8> %2355, i32 9, !dbg !127
  %2366 = extractelement <16 x i8> %2355, i32 10, !dbg !127
  %2367 = extractelement <16 x i8> %2355, i32 11, !dbg !127
  %2368 = extractelement <16 x i8> %2355, i32 12, !dbg !127
  %2369 = extractelement <16 x i8> %2355, i32 13, !dbg !127
  %2370 = extractelement <16 x i8> %2355, i32 14, !dbg !127
  %2371 = extractelement <16 x i8> %2355, i32 15, !dbg !127
  %2372 = insertelement <4 x i8> undef, i8 %2356, i32 0, !dbg !127
  %2373 = insertelement <4 x i8> %2372, i8 %2357, i32 1, !dbg !127
  %2374 = insertelement <4 x i8> %2373, i8 %2358, i32 2, !dbg !127
  %2375 = insertelement <4 x i8> %2374, i8 %2359, i32 3, !dbg !127
  %2376 = bitcast <4 x i8> %2375 to i32, !dbg !127
  %2377 = insertelement <4 x i8> undef, i8 %2360, i32 0, !dbg !127
  %2378 = insertelement <4 x i8> %2377, i8 %2361, i32 1, !dbg !127
  %2379 = insertelement <4 x i8> %2378, i8 %2362, i32 2, !dbg !127
  %2380 = insertelement <4 x i8> %2379, i8 %2363, i32 3, !dbg !127
  %2381 = bitcast <4 x i8> %2380 to i32, !dbg !127
  %2382 = insertelement <4 x i8> undef, i8 %2364, i32 0, !dbg !127
  %2383 = insertelement <4 x i8> %2382, i8 %2365, i32 1, !dbg !127
  %2384 = insertelement <4 x i8> %2383, i8 %2366, i32 2, !dbg !127
  %2385 = insertelement <4 x i8> %2384, i8 %2367, i32 3, !dbg !127
  %2386 = bitcast <4 x i8> %2385 to i32, !dbg !127
  %2387 = insertelement <4 x i8> undef, i8 %2368, i32 0, !dbg !127
  %2388 = insertelement <4 x i8> %2387, i8 %2369, i32 1, !dbg !127
  %2389 = insertelement <4 x i8> %2388, i8 %2370, i32 2, !dbg !127
  %2390 = insertelement <4 x i8> %2389, i8 %2371, i32 3, !dbg !127
  %2391 = bitcast <4 x i8> %2390 to i32, !dbg !127
  %2392 = insertelement <4 x i32> undef, i32 %2376, i32 0, !dbg !127
  %2393 = insertelement <4 x i32> %2392, i32 %2381, i32 1, !dbg !127
  %2394 = insertelement <4 x i32> %2393, i32 %2386, i32 2, !dbg !127
  %2395 = insertelement <4 x i32> %2394, i32 %2391, i32 3, !dbg !127
  store <4 x i32> %2395, ptr addrspace(3) %2339, align 16, !dbg !127
  call void @llvm.nvvm.fence.proxy.async.shared_cta(), !dbg !127
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !127
  %2396 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !127
  %2397 = extractvalue { i32, i1 } %2396, 1, !dbg !127
  %2398 = icmp ult i32 %186, 32, !dbg !127
  %2399 = and i1 %2397, %2398, !dbg !127
  %2400 = add i32 %2270, 0, !dbg !127
  %2401 = add i32 %2328, 0, !dbg !127
  %2402 = add i32 %2329, 0, !dbg !127
  call void asm sideeffect "@$0 cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [$1, {$2, $3, $4, $5, $6}], [$7];", "b,l,r,r,r,r,r,r"(i1 %2399, ptr %190, i32 %2400, i32 %2401, i32 0, i32 %2402, i32 1073741824, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 86016)), !dbg !127
  call void @llvm.nvvm.cp.async.bulk.commit.group(), !dbg !127
  br label %2403, !dbg !17

2403:                                             ; preds = %1484, %276
  %2404 = phi i32 [ %1485, %1484 ], [ %257, %276 ], !dbg !17
  %2405 = phi { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float } [ zeroinitializer, %1484 ], [ %1482, %276 ], !dbg !17
  %2406 = add i32 %255, 1, !dbg !17
  %2407 = select i1 %1483, i32 0, i32 %2406, !dbg !17
  %2408 = add i32 %254, 1, !dbg !17
  br label %253, !dbg !17

2409:                                             ; preds = %253
  call void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 0), !dbg !17
  call void @llvm.nvvm.barrier.cta.sync.aligned.count(i32 0, i32 128), !dbg !17
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !17
  call void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 256), !dbg !17
  br label %2410, !dbg !17

2410:                                             ; preds = %2409
  %2411 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %2412 = extractvalue { i32, i1 } %2411, 1, !dbg !61
  %2413 = and i1 %227, %2412, !dbg !61
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %2413, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88064)), !dbg !61
  %2414 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %2415 = extractvalue { i32, i1 } %2414, 1, !dbg !61
  %2416 = and i1 %227, %2415, !dbg !61
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %2416, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88072)), !dbg !61
  %2417 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %2418 = extractvalue { i32, i1 } %2417, 1, !dbg !61
  %2419 = and i1 %227, %2418, !dbg !61
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %2419, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88080)), !dbg !61
  %2420 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !61
  %2421 = extractvalue { i32, i1 } %2420, 1, !dbg !61
  %2422 = and i1 %227, %2421, !dbg !61
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %2422, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88088)), !dbg !61
  %2423 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %2424 = extractvalue { i32, i1 } %2423, 1, !dbg !62
  %2425 = and i1 %227, %2424, !dbg !62
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %2425, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88096)), !dbg !62
  %2426 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %2427 = extractvalue { i32, i1 } %2426, 1, !dbg !62
  %2428 = and i1 %227, %2427, !dbg !62
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %2428, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88104)), !dbg !62
  %2429 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %2430 = extractvalue { i32, i1 } %2429, 1, !dbg !62
  %2431 = and i1 %227, %2430, !dbg !62
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %2431, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88112)), !dbg !62
  %2432 = call { i32, i1 } @llvm.nvvm.elect.sync(i32 -1), !dbg !62
  %2433 = extractvalue { i32, i1 } %2432, 1, !dbg !62
  %2434 = and i1 %227, %2433, !dbg !62
  call void asm sideeffect "@$0 mbarrier.inval.shared::cta.b64 [$1];", "b,r"(i1 %2434, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88120)), !dbg !62
  store i8 2, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88128), align 1, !dbg !14
  store i8 2, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88129), align 1, !dbg !14
  store i8 2, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88130), align 1, !dbg !14
  store i8 2, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @global_smem, i64 88131), align 1, !dbg !14
  call void @llvm.nvvm.barrier.cta.sync.all(i32 1), !dbg !14
  ret void, !dbg !14
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

; Function Attrs: convergent nocallback nounwind memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32) #2

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn
declare void @llvm.nvvm.setmaxnreg.inc.sync.aligned.u32(i32 immarg range(i32 24, 257)) #3

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.all(i32) #4

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn
declare void @llvm.nvvm.setmaxnreg.dec.sync.aligned.u32(i32 immarg range(i32 24, 257)) #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #6

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.count(i32, i32) #4

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare { i32, i1 } @llvm.nvvm.elect.sync(i32) #7

; Function Attrs: nocallback nounwind
declare void @llvm.nvvm.fence.proxy.async.shared_cta() #8

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, -2147483648) i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #1

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.bar.warp.sync(i32) #4

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.bulk.wait.group.read(i32 immarg) #9

; Function Attrs: convergent nocallback nounwind memory(argmem: read)
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) readonly captures(none)) #10

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minnum.f32(float, float) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare float @llvm.nvvm.div.full(float, float) #11

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #5

; Function Attrs: nounwind
declare void @llvm.nvvm.cp.async.bulk.commit.group() #9

attributes #0 = { "nvvm.maxnreg"="256" "nvvm.reqntid"="256" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
attributes #3 = { convergent nocallback nofree nosync nounwind willreturn }
attributes #4 = { convergent nocallback nounwind }
attributes #5 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #7 = { convergent nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #8 = { nocallback nounwind }
attributes #9 = { nounwind }
attributes #10 = { convergent nocallback nounwind memory(argmem: read) }
attributes #11 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "_p_matmul.py", directory: "/Volumes/case_sensitive_workspace/triton/python/triton_kernels/triton_kernels/matmul_details")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "_p_matmul_NNT_fp8e4nvxfp8e4nvxfp8e4nv_16x256x128x1_swiglu", linkageName: "_p_matmul_NNT_fp8e4nvxfp8e4nvxfp8e4nv_16x256x128x1_swiglu", scope: !1, file: !1, line: 116, type: !4, scopeLine: 116, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{null, !6, !8, !8, !8, !8, !8, !9, !9, !9, !9, !9, !10, !8, !8, !8, !6, !8, !8, !8, !8, !8, !9, !9, !9, !9, !9, !10, !8, !8, !6, !8, !8, !8, !9, !9, !9, !10, !8, !8, !11, !11, !8, !8, !8, !8, !13, !13, !13, !13, !8, !8, !12, !12, !8, !10, !10}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !7, size: 64, dwarfAddressSpace: 0)
!7 = !DIBasicType(name: "unknown_type", encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !7, size: 64, dwarfAddressSpace: 1)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !12, size: 64, dwarfAddressSpace: 1)
!12 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !8, size: 64, dwarfAddressSpace: 1)
!14 = !DILocation(line: 116, column: 1, scope: !3)
!15 = distinct !{!15, !16}
!16 = !{!"llvm.licm.disable"}
!17 = !DILocation(line: 234, column: 5, scope: !3)
!18 = !DILocation(line: 125, column: 13, scope: !3)
!19 = !DILocation(line: 224, column: 20, scope: !3)
!20 = !DILocation(line: 43, column: 13, scope: !21, inlinedAt: !23)
!21 = distinct !DILexicalBlockFile(scope: !3, file: !22, discriminator: 0)
!22 = !DIFile(filename: "standard.py", directory: "/Volumes/case_sensitive_workspace/triton/python/triton/language")
!23 = !DILocation(line: 302, column: 19, scope: !24)
!24 = distinct !DILexicalBlockFile(scope: !3, file: !1, discriminator: 0)
!25 = !DILocation(line: 43, column: 12, scope: !21, inlinedAt: !23)
!26 = !DILocation(line: 303, column: 22, scope: !3)
!27 = !DILocation(line: 304, column: 19, scope: !3)
!28 = !DILocation(line: 47, column: 13, scope: !29, inlinedAt: !31)
!29 = distinct !DILexicalBlockFile(scope: !3, file: !30, discriminator: 0)
!30 = !DIFile(filename: "_common.py", directory: "/Volumes/case_sensitive_workspace/triton/python/triton_kernels/triton_kernels/matmul_details")
!31 = !DILocation(line: 70, column: 20, scope: !29, inlinedAt: !32)
!32 = !DILocation(line: 242, column: 38, scope: !24)
!33 = !DILocation(line: 63, column: 15, scope: !29, inlinedAt: !32)
!34 = !DILocation(line: 48, column: 16, scope: !29, inlinedAt: !31)
!35 = !DILocation(line: 49, column: 31, scope: !29, inlinedAt: !31)
!36 = !DILocation(line: 49, column: 22, scope: !29, inlinedAt: !31)
!37 = !DILocation(line: 49, column: 18, scope: !29, inlinedAt: !31)
!38 = !DILocation(line: 51, column: 35, scope: !29, inlinedAt: !31)
!39 = !DILocation(line: 51, column: 13, scope: !29, inlinedAt: !31)
!40 = !DILocation(line: 52, column: 14, scope: !29, inlinedAt: !31)
!41 = !DILocation(line: 52, column: 13, scope: !29, inlinedAt: !31)
!42 = !DILocation(line: 113, column: 34, scope: !29, inlinedAt: !43)
!43 = !DILocation(line: 247, column: 96, scope: !24)
!44 = !DILocation(line: 113, column: 26, scope: !29, inlinedAt: !43)
!45 = !DILocation(line: 114, column: 19, scope: !29, inlinedAt: !43)
!46 = !DILocation(line: 115, column: 20, scope: !29, inlinedAt: !43)
!47 = !DILocation(line: 116, column: 31, scope: !29, inlinedAt: !43)
!48 = !DILocation(line: 116, column: 23, scope: !29, inlinedAt: !43)
!49 = !DILocation(line: 119, column: 19, scope: !29, inlinedAt: !43)
!50 = !DILocation(line: 257, column: 31, scope: !3)
!51 = !DILocation(line: 257, column: 23, scope: !3)
!52 = !DILocation(line: 261, column: 19, scope: !3)
!53 = !DILocation(line: 304, column: 9, scope: !3)
!54 = !DILocation(line: 311, column: 38, scope: !3)
!55 = !DILocation(line: 55, column: 9, scope: !56, inlinedAt: !58)
!56 = distinct !DILexicalBlockFile(scope: !3, file: !57, discriminator: 0)
!57 = !DIFile(filename: "ragged_tma.py", directory: "/Volumes/case_sensitive_workspace/triton/python/triton/tools")
!58 = !DILocation(line: 73, column: 18, scope: !56, inlinedAt: !59)
!59 = !DILocation(line: 325, column: 21, scope: !24)
!60 = !DILocation(line: 56, column: 9, scope: !56, inlinedAt: !58)
!61 = !DILocation(line: 74, column: 12, scope: !56, inlinedAt: !59)
!62 = !DILocation(line: 384, column: 32, scope: !3)
!63 = !DILocation(line: 192, column: 33, scope: !3)
!64 = !DILocation(line: 192, column: 25, scope: !3)
!65 = !DILocation(line: 216, column: 18, scope: !3)
!66 = !DILocation(line: 476, column: 29, scope: !3)
!67 = !DILocation(line: 9, column: 27, scope: !68, inlinedAt: !70)
!68 = distinct !DILexicalBlockFile(scope: !3, file: !69, discriminator: 0)
!69 = !DIFile(filename: "_swiglu.py", directory: "/Volumes/case_sensitive_workspace/triton/python/triton_kernels/triton_kernels/swiglu_details")
!70 = !DILocation(line: 62, column: 18, scope: !68, inlinedAt: !71)
!71 = !DILocation(line: 70, column: 12, scope: !68, inlinedAt: !72)
!72 = !DILocation(line: 545, column: 23, scope: !24)
!73 = !DILocation(line: 63, column: 29, scope: !68, inlinedAt: !71)
!74 = !DILocation(line: 50, column: 15, scope: !29, inlinedAt: !31)
!75 = !DILocation(line: 50, column: 5, scope: !29, inlinedAt: !31)
!76 = !DILocation(line: 427, column: 27, scope: !3)
!77 = !DILocation(line: 438, column: 13, scope: !3)
!78 = !DILocation(line: 63, column: 15, scope: !29, inlinedAt: !79)
!79 = !DILocation(line: 439, column: 46, scope: !24)
!80 = !DILocation(line: 48, column: 16, scope: !29, inlinedAt: !81)
!81 = !DILocation(line: 70, column: 20, scope: !29, inlinedAt: !79)
!82 = !DILocation(line: 49, column: 31, scope: !29, inlinedAt: !81)
!83 = !DILocation(line: 49, column: 22, scope: !29, inlinedAt: !81)
!84 = !DILocation(line: 49, column: 18, scope: !29, inlinedAt: !81)
!85 = !DILocation(line: 50, column: 15, scope: !29, inlinedAt: !81)
!86 = !DILocation(line: 50, column: 5, scope: !29, inlinedAt: !81)
!87 = !DILocation(line: 52, column: 14, scope: !29, inlinedAt: !81)
!88 = !DILocation(line: 52, column: 13, scope: !29, inlinedAt: !81)
!89 = !DILocation(line: 113, column: 26, scope: !29, inlinedAt: !90)
!90 = !DILocation(line: 440, column: 64, scope: !24)
!91 = !DILocation(line: 114, column: 19, scope: !29, inlinedAt: !90)
!92 = !DILocation(line: 115, column: 20, scope: !29, inlinedAt: !90)
!93 = !DILocation(line: 116, column: 31, scope: !29, inlinedAt: !90)
!94 = !DILocation(line: 116, column: 23, scope: !29, inlinedAt: !90)
!95 = !DILocation(line: 119, column: 19, scope: !29, inlinedAt: !90)
!96 = !DILocation(line: 447, column: 22, scope: !3)
!97 = !DILocation(line: 449, column: 31, scope: !3)
!98 = !DILocation(line: 449, column: 23, scope: !3)
!99 = !DILocation(line: 476, column: 20, scope: !3)
!100 = !DILocation(line: 477, column: 18, scope: !3)
!101 = !DILocation(line: 479, column: 25, scope: !3)
!102 = !DILocation(line: 479, column: 21, scope: !3)
!103 = !DILocation(line: 481, column: 24, scope: !3)
!104 = !DILocation(line: 107, column: 42, scope: !105, inlinedAt: !107)
!105 = distinct !DILexicalBlockFile(scope: !3, file: !106, discriminator: 0)
!106 = !DIFile(filename: "flexpoint.py", directory: "/Volumes/case_sensitive_workspace/triton/python/triton_kernels/triton_kernels/numerics_details")
!107 = !DILocation(line: 498, column: 23, scope: !24)
!108 = !DILocation(line: 535, column: 13, scope: !3)
!109 = !DILocation(line: 540, column: 35, scope: !3)
!110 = !DILocation(line: 540, column: 24, scope: !3)
!111 = !DILocation(line: 69, column: 29, scope: !68, inlinedAt: !72)
!112 = !DILocation(line: 11, column: 15, scope: !68, inlinedAt: !113)
!113 = !DILocation(line: 59, column: 16, scope: !68, inlinedAt: !71)
!114 = !DILocation(line: 9, column: 15, scope: !68, inlinedAt: !70)
!115 = !DILocation(line: 42, column: 9, scope: !68, inlinedAt: !116)
!116 = !DILocation(line: 63, column: 21, scope: !68, inlinedAt: !71)
!117 = !DILocation(line: 43, column: 16, scope: !68, inlinedAt: !116)
!118 = !DILocation(line: 63, column: 17, scope: !68, inlinedAt: !71)
!119 = !DILocation(line: 63, column: 9, scope: !68, inlinedAt: !71)
!120 = !DILocation(line: 64, column: 12, scope: !68, inlinedAt: !71)
!121 = !DILocation(line: 579, column: 25, scope: !3)
!122 = !DILocation(line: 628, column: 19, scope: !3)
!123 = !DILocation(line: 55, column: 9, scope: !56, inlinedAt: !124)
!124 = !DILocation(line: 90, column: 18, scope: !56, inlinedAt: !125)
!125 = !DILocation(line: 647, column: 21, scope: !24)
!126 = !DILocation(line: 56, column: 9, scope: !56, inlinedAt: !124)
!127 = !DILocation(line: 92, column: 5, scope: !56, inlinedAt: !125)
