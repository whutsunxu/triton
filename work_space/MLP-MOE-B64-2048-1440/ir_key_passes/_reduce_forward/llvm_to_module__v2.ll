// kernel: _reduce_forward
// pass: llvm_to_module
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@global_smem = external addrspace(3) global [0 x i8], align 16

define ptx_kernel void @_reduce_forward(ptr addrspace(1) %0, i64 %1, i64 %2, ptr addrspace(1) %3, i64 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11, i32 %12, i32 %13, ptr addrspace(1) %14, ptr addrspace(1) %15) #0 !dbg !3 {
  %17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x(), !dbg !10
  %18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y(), !dbg !11
  %19 = mul i32 %17, 32, !dbg !12
  %20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !dbg !13
  %21 = and i32 %20, 127, !dbg !13
  %22 = urem i32 %21, 32, !dbg !13
  %23 = udiv i32 %20, 32, !dbg !13
  %24 = shl i32 %22, 0, !dbg !13
  %25 = or i32 0, %24, !dbg !13
  %26 = shl i32 %23, 5, !dbg !13
  %27 = or i32 %25, %26, !dbg !13
  %28 = and i32 %27, 120, !dbg !13
  %29 = lshr i32 %28, 3, !dbg !13
  %30 = or disjoint i32 %29, 0, !dbg !13
  %31 = xor i32 0, %30, !dbg !13
  %32 = xor i32 %31, 0, !dbg !13
  %33 = xor i32 %31, 16, !dbg !13
  %34 = add i32 %32, 0, !dbg !13
  %35 = add i32 %33, 0, !dbg !13
  %36 = add i32 %19, %34, !dbg !12
  %37 = add i32 %19, %35, !dbg !12
  %38 = mul i32 %18, 128, !dbg !14
  %39 = and i32 %27, 7, !dbg !15
  %40 = shl i32 %39, 4, !dbg !15
  %41 = or disjoint i32 %40, 0, !dbg !15
  %42 = xor i32 0, %41, !dbg !15
  %43 = xor i32 %42, 0, !dbg !15
  %44 = add i32 %43, 0, !dbg !15
  %45 = add i32 %38, %44, !dbg !14
  %46 = icmp slt i32 %36, %11, !dbg !16
  %47 = icmp slt i32 %37, %11, !dbg !16
  %48 = icmp slt i32 %45, %12, !dbg !17
  %49 = and i1 %46, %48, !dbg !18
  %50 = and i1 %47, %48, !dbg !18
  %51 = sext i32 %36 to i64, !dbg !19
  %52 = sext i32 %37 to i64, !dbg !19
  %53 = mul i64 %51, %2, !dbg !19
  %54 = mul i64 %52, %2, !dbg !19
  %55 = getelementptr i8, ptr addrspace(1) %0, i64 %53, !dbg !20
  %56 = getelementptr i8, ptr addrspace(1) %0, i64 %54, !dbg !20
  %57 = getelementptr i8, ptr addrspace(1) %55, i32 %45, !dbg !20
  %58 = getelementptr i8, ptr addrspace(1) %56, i32 %45, !dbg !20
  %59 = call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %57, i1 %49), !dbg !21
  %60 = extractvalue { i32, i32, i32, i32 } %59, 0, !dbg !21
  %61 = bitcast i32 %60 to <4 x i8>, !dbg !21
  %62 = extractvalue { i32, i32, i32, i32 } %59, 1, !dbg !21
  %63 = bitcast i32 %62 to <4 x i8>, !dbg !21
  %64 = extractvalue { i32, i32, i32, i32 } %59, 2, !dbg !21
  %65 = bitcast i32 %64 to <4 x i8>, !dbg !21
  %66 = extractvalue { i32, i32, i32, i32 } %59, 3, !dbg !21
  %67 = bitcast i32 %66 to <4 x i8>, !dbg !21
  %68 = extractelement <4 x i8> %61, i32 0, !dbg !21
  %69 = extractelement <4 x i8> %61, i32 1, !dbg !21
  %70 = extractelement <4 x i8> %61, i32 2, !dbg !21
  %71 = extractelement <4 x i8> %61, i32 3, !dbg !21
  %72 = extractelement <4 x i8> %63, i32 0, !dbg !21
  %73 = extractelement <4 x i8> %63, i32 1, !dbg !21
  %74 = extractelement <4 x i8> %63, i32 2, !dbg !21
  %75 = extractelement <4 x i8> %63, i32 3, !dbg !21
  %76 = extractelement <4 x i8> %65, i32 0, !dbg !21
  %77 = extractelement <4 x i8> %65, i32 1, !dbg !21
  %78 = extractelement <4 x i8> %65, i32 2, !dbg !21
  %79 = extractelement <4 x i8> %65, i32 3, !dbg !21
  %80 = extractelement <4 x i8> %67, i32 0, !dbg !21
  %81 = extractelement <4 x i8> %67, i32 1, !dbg !21
  %82 = extractelement <4 x i8> %67, i32 2, !dbg !21
  %83 = extractelement <4 x i8> %67, i32 3, !dbg !21
  %84 = call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %58, i1 %50), !dbg !21
  %85 = extractvalue { i32, i32, i32, i32 } %84, 0, !dbg !21
  %86 = bitcast i32 %85 to <4 x i8>, !dbg !21
  %87 = extractvalue { i32, i32, i32, i32 } %84, 1, !dbg !21
  %88 = bitcast i32 %87 to <4 x i8>, !dbg !21
  %89 = extractvalue { i32, i32, i32, i32 } %84, 2, !dbg !21
  %90 = bitcast i32 %89 to <4 x i8>, !dbg !21
  %91 = extractvalue { i32, i32, i32, i32 } %84, 3, !dbg !21
  %92 = bitcast i32 %91 to <4 x i8>, !dbg !21
  %93 = extractelement <4 x i8> %86, i32 0, !dbg !21
  %94 = extractelement <4 x i8> %86, i32 1, !dbg !21
  %95 = extractelement <4 x i8> %86, i32 2, !dbg !21
  %96 = extractelement <4 x i8> %86, i32 3, !dbg !21
  %97 = extractelement <4 x i8> %88, i32 0, !dbg !21
  %98 = extractelement <4 x i8> %88, i32 1, !dbg !21
  %99 = extractelement <4 x i8> %88, i32 2, !dbg !21
  %100 = extractelement <4 x i8> %88, i32 3, !dbg !21
  %101 = extractelement <4 x i8> %90, i32 0, !dbg !21
  %102 = extractelement <4 x i8> %90, i32 1, !dbg !21
  %103 = extractelement <4 x i8> %90, i32 2, !dbg !21
  %104 = extractelement <4 x i8> %90, i32 3, !dbg !21
  %105 = extractelement <4 x i8> %92, i32 0, !dbg !21
  %106 = extractelement <4 x i8> %92, i32 1, !dbg !21
  %107 = extractelement <4 x i8> %92, i32 2, !dbg !21
  %108 = extractelement <4 x i8> %92, i32 3, !dbg !21
  %109 = insertelement <2 x i8> undef, i8 %68, i32 0, !dbg !21
  %110 = insertelement <2 x i8> %109, i8 %69, i32 1, !dbg !21
  %111 = bitcast <2 x i8> %110 to i16, !dbg !21
  %112 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %111), !dbg !21
  %113 = extractelement <2 x half> %112, i32 0, !dbg !21
  %114 = extractelement <2 x half> %112, i32 1, !dbg !21
  %115 = fpext half %113 to float, !dbg !21
  %116 = fpext half %114 to float, !dbg !21
  %117 = insertelement <2 x i8> undef, i8 %70, i32 0, !dbg !21
  %118 = insertelement <2 x i8> %117, i8 %71, i32 1, !dbg !21
  %119 = bitcast <2 x i8> %118 to i16, !dbg !21
  %120 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %119), !dbg !21
  %121 = extractelement <2 x half> %120, i32 0, !dbg !21
  %122 = extractelement <2 x half> %120, i32 1, !dbg !21
  %123 = fpext half %121 to float, !dbg !21
  %124 = fpext half %122 to float, !dbg !21
  %125 = insertelement <2 x i8> undef, i8 %72, i32 0, !dbg !21
  %126 = insertelement <2 x i8> %125, i8 %73, i32 1, !dbg !21
  %127 = bitcast <2 x i8> %126 to i16, !dbg !21
  %128 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %127), !dbg !21
  %129 = extractelement <2 x half> %128, i32 0, !dbg !21
  %130 = extractelement <2 x half> %128, i32 1, !dbg !21
  %131 = fpext half %129 to float, !dbg !21
  %132 = fpext half %130 to float, !dbg !21
  %133 = insertelement <2 x i8> undef, i8 %74, i32 0, !dbg !21
  %134 = insertelement <2 x i8> %133, i8 %75, i32 1, !dbg !21
  %135 = bitcast <2 x i8> %134 to i16, !dbg !21
  %136 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %135), !dbg !21
  %137 = extractelement <2 x half> %136, i32 0, !dbg !21
  %138 = extractelement <2 x half> %136, i32 1, !dbg !21
  %139 = fpext half %137 to float, !dbg !21
  %140 = fpext half %138 to float, !dbg !21
  %141 = insertelement <2 x i8> undef, i8 %76, i32 0, !dbg !21
  %142 = insertelement <2 x i8> %141, i8 %77, i32 1, !dbg !21
  %143 = bitcast <2 x i8> %142 to i16, !dbg !21
  %144 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %143), !dbg !21
  %145 = extractelement <2 x half> %144, i32 0, !dbg !21
  %146 = extractelement <2 x half> %144, i32 1, !dbg !21
  %147 = fpext half %145 to float, !dbg !21
  %148 = fpext half %146 to float, !dbg !21
  %149 = insertelement <2 x i8> undef, i8 %78, i32 0, !dbg !21
  %150 = insertelement <2 x i8> %149, i8 %79, i32 1, !dbg !21
  %151 = bitcast <2 x i8> %150 to i16, !dbg !21
  %152 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %151), !dbg !21
  %153 = extractelement <2 x half> %152, i32 0, !dbg !21
  %154 = extractelement <2 x half> %152, i32 1, !dbg !21
  %155 = fpext half %153 to float, !dbg !21
  %156 = fpext half %154 to float, !dbg !21
  %157 = insertelement <2 x i8> undef, i8 %80, i32 0, !dbg !21
  %158 = insertelement <2 x i8> %157, i8 %81, i32 1, !dbg !21
  %159 = bitcast <2 x i8> %158 to i16, !dbg !21
  %160 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %159), !dbg !21
  %161 = extractelement <2 x half> %160, i32 0, !dbg !21
  %162 = extractelement <2 x half> %160, i32 1, !dbg !21
  %163 = fpext half %161 to float, !dbg !21
  %164 = fpext half %162 to float, !dbg !21
  %165 = insertelement <2 x i8> undef, i8 %82, i32 0, !dbg !21
  %166 = insertelement <2 x i8> %165, i8 %83, i32 1, !dbg !21
  %167 = bitcast <2 x i8> %166 to i16, !dbg !21
  %168 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %167), !dbg !21
  %169 = extractelement <2 x half> %168, i32 0, !dbg !21
  %170 = extractelement <2 x half> %168, i32 1, !dbg !21
  %171 = fpext half %169 to float, !dbg !21
  %172 = fpext half %170 to float, !dbg !21
  %173 = insertelement <2 x i8> undef, i8 %93, i32 0, !dbg !21
  %174 = insertelement <2 x i8> %173, i8 %94, i32 1, !dbg !21
  %175 = bitcast <2 x i8> %174 to i16, !dbg !21
  %176 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %175), !dbg !21
  %177 = extractelement <2 x half> %176, i32 0, !dbg !21
  %178 = extractelement <2 x half> %176, i32 1, !dbg !21
  %179 = fpext half %177 to float, !dbg !21
  %180 = fpext half %178 to float, !dbg !21
  %181 = insertelement <2 x i8> undef, i8 %95, i32 0, !dbg !21
  %182 = insertelement <2 x i8> %181, i8 %96, i32 1, !dbg !21
  %183 = bitcast <2 x i8> %182 to i16, !dbg !21
  %184 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %183), !dbg !21
  %185 = extractelement <2 x half> %184, i32 0, !dbg !21
  %186 = extractelement <2 x half> %184, i32 1, !dbg !21
  %187 = fpext half %185 to float, !dbg !21
  %188 = fpext half %186 to float, !dbg !21
  %189 = insertelement <2 x i8> undef, i8 %97, i32 0, !dbg !21
  %190 = insertelement <2 x i8> %189, i8 %98, i32 1, !dbg !21
  %191 = bitcast <2 x i8> %190 to i16, !dbg !21
  %192 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %191), !dbg !21
  %193 = extractelement <2 x half> %192, i32 0, !dbg !21
  %194 = extractelement <2 x half> %192, i32 1, !dbg !21
  %195 = fpext half %193 to float, !dbg !21
  %196 = fpext half %194 to float, !dbg !21
  %197 = insertelement <2 x i8> undef, i8 %99, i32 0, !dbg !21
  %198 = insertelement <2 x i8> %197, i8 %100, i32 1, !dbg !21
  %199 = bitcast <2 x i8> %198 to i16, !dbg !21
  %200 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %199), !dbg !21
  %201 = extractelement <2 x half> %200, i32 0, !dbg !21
  %202 = extractelement <2 x half> %200, i32 1, !dbg !21
  %203 = fpext half %201 to float, !dbg !21
  %204 = fpext half %202 to float, !dbg !21
  %205 = insertelement <2 x i8> undef, i8 %101, i32 0, !dbg !21
  %206 = insertelement <2 x i8> %205, i8 %102, i32 1, !dbg !21
  %207 = bitcast <2 x i8> %206 to i16, !dbg !21
  %208 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %207), !dbg !21
  %209 = extractelement <2 x half> %208, i32 0, !dbg !21
  %210 = extractelement <2 x half> %208, i32 1, !dbg !21
  %211 = fpext half %209 to float, !dbg !21
  %212 = fpext half %210 to float, !dbg !21
  %213 = insertelement <2 x i8> undef, i8 %103, i32 0, !dbg !21
  %214 = insertelement <2 x i8> %213, i8 %104, i32 1, !dbg !21
  %215 = bitcast <2 x i8> %214 to i16, !dbg !21
  %216 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %215), !dbg !21
  %217 = extractelement <2 x half> %216, i32 0, !dbg !21
  %218 = extractelement <2 x half> %216, i32 1, !dbg !21
  %219 = fpext half %217 to float, !dbg !21
  %220 = fpext half %218 to float, !dbg !21
  %221 = insertelement <2 x i8> undef, i8 %105, i32 0, !dbg !21
  %222 = insertelement <2 x i8> %221, i8 %106, i32 1, !dbg !21
  %223 = bitcast <2 x i8> %222 to i16, !dbg !21
  %224 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %223), !dbg !21
  %225 = extractelement <2 x half> %224, i32 0, !dbg !21
  %226 = extractelement <2 x half> %224, i32 1, !dbg !21
  %227 = fpext half %225 to float, !dbg !21
  %228 = fpext half %226 to float, !dbg !21
  %229 = insertelement <2 x i8> undef, i8 %107, i32 0, !dbg !21
  %230 = insertelement <2 x i8> %229, i8 %108, i32 1, !dbg !21
  %231 = bitcast <2 x i8> %230 to i16, !dbg !21
  %232 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %231), !dbg !21
  %233 = extractelement <2 x half> %232, i32 0, !dbg !21
  %234 = extractelement <2 x half> %232, i32 1, !dbg !21
  %235 = fpext half %233 to float, !dbg !21
  %236 = fpext half %234 to float, !dbg !21
  %237 = getelementptr i8, ptr addrspace(1) %0, i64 %1, !dbg !20
  %238 = getelementptr i8, ptr addrspace(1) %237, i64 %53, !dbg !20
  %239 = getelementptr i8, ptr addrspace(1) %237, i64 %54, !dbg !20
  %240 = getelementptr i8, ptr addrspace(1) %238, i32 %45, !dbg !20
  %241 = getelementptr i8, ptr addrspace(1) %239, i32 %45, !dbg !20
  %242 = call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %240, i1 %49), !dbg !21
  %243 = extractvalue { i32, i32, i32, i32 } %242, 0, !dbg !21
  %244 = bitcast i32 %243 to <4 x i8>, !dbg !21
  %245 = extractvalue { i32, i32, i32, i32 } %242, 1, !dbg !21
  %246 = bitcast i32 %245 to <4 x i8>, !dbg !21
  %247 = extractvalue { i32, i32, i32, i32 } %242, 2, !dbg !21
  %248 = bitcast i32 %247 to <4 x i8>, !dbg !21
  %249 = extractvalue { i32, i32, i32, i32 } %242, 3, !dbg !21
  %250 = bitcast i32 %249 to <4 x i8>, !dbg !21
  %251 = extractelement <4 x i8> %244, i32 0, !dbg !21
  %252 = extractelement <4 x i8> %244, i32 1, !dbg !21
  %253 = extractelement <4 x i8> %244, i32 2, !dbg !21
  %254 = extractelement <4 x i8> %244, i32 3, !dbg !21
  %255 = extractelement <4 x i8> %246, i32 0, !dbg !21
  %256 = extractelement <4 x i8> %246, i32 1, !dbg !21
  %257 = extractelement <4 x i8> %246, i32 2, !dbg !21
  %258 = extractelement <4 x i8> %246, i32 3, !dbg !21
  %259 = extractelement <4 x i8> %248, i32 0, !dbg !21
  %260 = extractelement <4 x i8> %248, i32 1, !dbg !21
  %261 = extractelement <4 x i8> %248, i32 2, !dbg !21
  %262 = extractelement <4 x i8> %248, i32 3, !dbg !21
  %263 = extractelement <4 x i8> %250, i32 0, !dbg !21
  %264 = extractelement <4 x i8> %250, i32 1, !dbg !21
  %265 = extractelement <4 x i8> %250, i32 2, !dbg !21
  %266 = extractelement <4 x i8> %250, i32 3, !dbg !21
  %267 = call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %241, i1 %50), !dbg !21
  %268 = extractvalue { i32, i32, i32, i32 } %267, 0, !dbg !21
  %269 = bitcast i32 %268 to <4 x i8>, !dbg !21
  %270 = extractvalue { i32, i32, i32, i32 } %267, 1, !dbg !21
  %271 = bitcast i32 %270 to <4 x i8>, !dbg !21
  %272 = extractvalue { i32, i32, i32, i32 } %267, 2, !dbg !21
  %273 = bitcast i32 %272 to <4 x i8>, !dbg !21
  %274 = extractvalue { i32, i32, i32, i32 } %267, 3, !dbg !21
  %275 = bitcast i32 %274 to <4 x i8>, !dbg !21
  %276 = extractelement <4 x i8> %269, i32 0, !dbg !21
  %277 = extractelement <4 x i8> %269, i32 1, !dbg !21
  %278 = extractelement <4 x i8> %269, i32 2, !dbg !21
  %279 = extractelement <4 x i8> %269, i32 3, !dbg !21
  %280 = extractelement <4 x i8> %271, i32 0, !dbg !21
  %281 = extractelement <4 x i8> %271, i32 1, !dbg !21
  %282 = extractelement <4 x i8> %271, i32 2, !dbg !21
  %283 = extractelement <4 x i8> %271, i32 3, !dbg !21
  %284 = extractelement <4 x i8> %273, i32 0, !dbg !21
  %285 = extractelement <4 x i8> %273, i32 1, !dbg !21
  %286 = extractelement <4 x i8> %273, i32 2, !dbg !21
  %287 = extractelement <4 x i8> %273, i32 3, !dbg !21
  %288 = extractelement <4 x i8> %275, i32 0, !dbg !21
  %289 = extractelement <4 x i8> %275, i32 1, !dbg !21
  %290 = extractelement <4 x i8> %275, i32 2, !dbg !21
  %291 = extractelement <4 x i8> %275, i32 3, !dbg !21
  %292 = insertelement <2 x i8> undef, i8 %251, i32 0, !dbg !21
  %293 = insertelement <2 x i8> %292, i8 %252, i32 1, !dbg !21
  %294 = bitcast <2 x i8> %293 to i16, !dbg !21
  %295 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %294), !dbg !21
  %296 = extractelement <2 x half> %295, i32 0, !dbg !21
  %297 = extractelement <2 x half> %295, i32 1, !dbg !21
  %298 = fpext half %296 to float, !dbg !21
  %299 = fpext half %297 to float, !dbg !21
  %300 = insertelement <2 x i8> undef, i8 %253, i32 0, !dbg !21
  %301 = insertelement <2 x i8> %300, i8 %254, i32 1, !dbg !21
  %302 = bitcast <2 x i8> %301 to i16, !dbg !21
  %303 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %302), !dbg !21
  %304 = extractelement <2 x half> %303, i32 0, !dbg !21
  %305 = extractelement <2 x half> %303, i32 1, !dbg !21
  %306 = fpext half %304 to float, !dbg !21
  %307 = fpext half %305 to float, !dbg !21
  %308 = insertelement <2 x i8> undef, i8 %255, i32 0, !dbg !21
  %309 = insertelement <2 x i8> %308, i8 %256, i32 1, !dbg !21
  %310 = bitcast <2 x i8> %309 to i16, !dbg !21
  %311 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %310), !dbg !21
  %312 = extractelement <2 x half> %311, i32 0, !dbg !21
  %313 = extractelement <2 x half> %311, i32 1, !dbg !21
  %314 = fpext half %312 to float, !dbg !21
  %315 = fpext half %313 to float, !dbg !21
  %316 = insertelement <2 x i8> undef, i8 %257, i32 0, !dbg !21
  %317 = insertelement <2 x i8> %316, i8 %258, i32 1, !dbg !21
  %318 = bitcast <2 x i8> %317 to i16, !dbg !21
  %319 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %318), !dbg !21
  %320 = extractelement <2 x half> %319, i32 0, !dbg !21
  %321 = extractelement <2 x half> %319, i32 1, !dbg !21
  %322 = fpext half %320 to float, !dbg !21
  %323 = fpext half %321 to float, !dbg !21
  %324 = insertelement <2 x i8> undef, i8 %259, i32 0, !dbg !21
  %325 = insertelement <2 x i8> %324, i8 %260, i32 1, !dbg !21
  %326 = bitcast <2 x i8> %325 to i16, !dbg !21
  %327 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %326), !dbg !21
  %328 = extractelement <2 x half> %327, i32 0, !dbg !21
  %329 = extractelement <2 x half> %327, i32 1, !dbg !21
  %330 = fpext half %328 to float, !dbg !21
  %331 = fpext half %329 to float, !dbg !21
  %332 = insertelement <2 x i8> undef, i8 %261, i32 0, !dbg !21
  %333 = insertelement <2 x i8> %332, i8 %262, i32 1, !dbg !21
  %334 = bitcast <2 x i8> %333 to i16, !dbg !21
  %335 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %334), !dbg !21
  %336 = extractelement <2 x half> %335, i32 0, !dbg !21
  %337 = extractelement <2 x half> %335, i32 1, !dbg !21
  %338 = fpext half %336 to float, !dbg !21
  %339 = fpext half %337 to float, !dbg !21
  %340 = insertelement <2 x i8> undef, i8 %263, i32 0, !dbg !21
  %341 = insertelement <2 x i8> %340, i8 %264, i32 1, !dbg !21
  %342 = bitcast <2 x i8> %341 to i16, !dbg !21
  %343 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %342), !dbg !21
  %344 = extractelement <2 x half> %343, i32 0, !dbg !21
  %345 = extractelement <2 x half> %343, i32 1, !dbg !21
  %346 = fpext half %344 to float, !dbg !21
  %347 = fpext half %345 to float, !dbg !21
  %348 = insertelement <2 x i8> undef, i8 %265, i32 0, !dbg !21
  %349 = insertelement <2 x i8> %348, i8 %266, i32 1, !dbg !21
  %350 = bitcast <2 x i8> %349 to i16, !dbg !21
  %351 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %350), !dbg !21
  %352 = extractelement <2 x half> %351, i32 0, !dbg !21
  %353 = extractelement <2 x half> %351, i32 1, !dbg !21
  %354 = fpext half %352 to float, !dbg !21
  %355 = fpext half %353 to float, !dbg !21
  %356 = insertelement <2 x i8> undef, i8 %276, i32 0, !dbg !21
  %357 = insertelement <2 x i8> %356, i8 %277, i32 1, !dbg !21
  %358 = bitcast <2 x i8> %357 to i16, !dbg !21
  %359 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %358), !dbg !21
  %360 = extractelement <2 x half> %359, i32 0, !dbg !21
  %361 = extractelement <2 x half> %359, i32 1, !dbg !21
  %362 = fpext half %360 to float, !dbg !21
  %363 = fpext half %361 to float, !dbg !21
  %364 = insertelement <2 x i8> undef, i8 %278, i32 0, !dbg !21
  %365 = insertelement <2 x i8> %364, i8 %279, i32 1, !dbg !21
  %366 = bitcast <2 x i8> %365 to i16, !dbg !21
  %367 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %366), !dbg !21
  %368 = extractelement <2 x half> %367, i32 0, !dbg !21
  %369 = extractelement <2 x half> %367, i32 1, !dbg !21
  %370 = fpext half %368 to float, !dbg !21
  %371 = fpext half %369 to float, !dbg !21
  %372 = insertelement <2 x i8> undef, i8 %280, i32 0, !dbg !21
  %373 = insertelement <2 x i8> %372, i8 %281, i32 1, !dbg !21
  %374 = bitcast <2 x i8> %373 to i16, !dbg !21
  %375 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %374), !dbg !21
  %376 = extractelement <2 x half> %375, i32 0, !dbg !21
  %377 = extractelement <2 x half> %375, i32 1, !dbg !21
  %378 = fpext half %376 to float, !dbg !21
  %379 = fpext half %377 to float, !dbg !21
  %380 = insertelement <2 x i8> undef, i8 %282, i32 0, !dbg !21
  %381 = insertelement <2 x i8> %380, i8 %283, i32 1, !dbg !21
  %382 = bitcast <2 x i8> %381 to i16, !dbg !21
  %383 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %382), !dbg !21
  %384 = extractelement <2 x half> %383, i32 0, !dbg !21
  %385 = extractelement <2 x half> %383, i32 1, !dbg !21
  %386 = fpext half %384 to float, !dbg !21
  %387 = fpext half %385 to float, !dbg !21
  %388 = insertelement <2 x i8> undef, i8 %284, i32 0, !dbg !21
  %389 = insertelement <2 x i8> %388, i8 %285, i32 1, !dbg !21
  %390 = bitcast <2 x i8> %389 to i16, !dbg !21
  %391 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %390), !dbg !21
  %392 = extractelement <2 x half> %391, i32 0, !dbg !21
  %393 = extractelement <2 x half> %391, i32 1, !dbg !21
  %394 = fpext half %392 to float, !dbg !21
  %395 = fpext half %393 to float, !dbg !21
  %396 = insertelement <2 x i8> undef, i8 %286, i32 0, !dbg !21
  %397 = insertelement <2 x i8> %396, i8 %287, i32 1, !dbg !21
  %398 = bitcast <2 x i8> %397 to i16, !dbg !21
  %399 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %398), !dbg !21
  %400 = extractelement <2 x half> %399, i32 0, !dbg !21
  %401 = extractelement <2 x half> %399, i32 1, !dbg !21
  %402 = fpext half %400 to float, !dbg !21
  %403 = fpext half %401 to float, !dbg !21
  %404 = insertelement <2 x i8> undef, i8 %288, i32 0, !dbg !21
  %405 = insertelement <2 x i8> %404, i8 %289, i32 1, !dbg !21
  %406 = bitcast <2 x i8> %405 to i16, !dbg !21
  %407 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %406), !dbg !21
  %408 = extractelement <2 x half> %407, i32 0, !dbg !21
  %409 = extractelement <2 x half> %407, i32 1, !dbg !21
  %410 = fpext half %408 to float, !dbg !21
  %411 = fpext half %409 to float, !dbg !21
  %412 = insertelement <2 x i8> undef, i8 %290, i32 0, !dbg !21
  %413 = insertelement <2 x i8> %412, i8 %291, i32 1, !dbg !21
  %414 = bitcast <2 x i8> %413 to i16, !dbg !21
  %415 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %414), !dbg !21
  %416 = extractelement <2 x half> %415, i32 0, !dbg !21
  %417 = extractelement <2 x half> %415, i32 1, !dbg !21
  %418 = fpext half %416 to float, !dbg !21
  %419 = fpext half %417 to float, !dbg !21
  %420 = fadd float %115, %298, !dbg !22
  %421 = fadd float %116, %299, !dbg !22
  %422 = fadd float %123, %306, !dbg !22
  %423 = fadd float %124, %307, !dbg !22
  %424 = fadd float %131, %314, !dbg !22
  %425 = fadd float %132, %315, !dbg !22
  %426 = fadd float %139, %322, !dbg !22
  %427 = fadd float %140, %323, !dbg !22
  %428 = fadd float %147, %330, !dbg !22
  %429 = fadd float %148, %331, !dbg !22
  %430 = fadd float %155, %338, !dbg !22
  %431 = fadd float %156, %339, !dbg !22
  %432 = fadd float %163, %346, !dbg !22
  %433 = fadd float %164, %347, !dbg !22
  %434 = fadd float %171, %354, !dbg !22
  %435 = fadd float %172, %355, !dbg !22
  %436 = fadd float %179, %362, !dbg !22
  %437 = fadd float %180, %363, !dbg !22
  %438 = fadd float %187, %370, !dbg !22
  %439 = fadd float %188, %371, !dbg !22
  %440 = fadd float %195, %378, !dbg !22
  %441 = fadd float %196, %379, !dbg !22
  %442 = fadd float %203, %386, !dbg !22
  %443 = fadd float %204, %387, !dbg !22
  %444 = fadd float %211, %394, !dbg !22
  %445 = fadd float %212, %395, !dbg !22
  %446 = fadd float %219, %402, !dbg !22
  %447 = fadd float %220, %403, !dbg !22
  %448 = fadd float %227, %410, !dbg !22
  %449 = fadd float %228, %411, !dbg !22
  %450 = fadd float %235, %418, !dbg !22
  %451 = fadd float %236, %419, !dbg !22
  %452 = mul i64 %1, 2, !dbg !23
  %453 = getelementptr i8, ptr addrspace(1) %0, i64 %452, !dbg !20
  %454 = getelementptr i8, ptr addrspace(1) %453, i64 %53, !dbg !20
  %455 = getelementptr i8, ptr addrspace(1) %453, i64 %54, !dbg !20
  %456 = getelementptr i8, ptr addrspace(1) %454, i32 %45, !dbg !20
  %457 = getelementptr i8, ptr addrspace(1) %455, i32 %45, !dbg !20
  %458 = call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %456, i1 %49), !dbg !21
  %459 = extractvalue { i32, i32, i32, i32 } %458, 0, !dbg !21
  %460 = bitcast i32 %459 to <4 x i8>, !dbg !21
  %461 = extractvalue { i32, i32, i32, i32 } %458, 1, !dbg !21
  %462 = bitcast i32 %461 to <4 x i8>, !dbg !21
  %463 = extractvalue { i32, i32, i32, i32 } %458, 2, !dbg !21
  %464 = bitcast i32 %463 to <4 x i8>, !dbg !21
  %465 = extractvalue { i32, i32, i32, i32 } %458, 3, !dbg !21
  %466 = bitcast i32 %465 to <4 x i8>, !dbg !21
  %467 = extractelement <4 x i8> %460, i32 0, !dbg !21
  %468 = extractelement <4 x i8> %460, i32 1, !dbg !21
  %469 = extractelement <4 x i8> %460, i32 2, !dbg !21
  %470 = extractelement <4 x i8> %460, i32 3, !dbg !21
  %471 = extractelement <4 x i8> %462, i32 0, !dbg !21
  %472 = extractelement <4 x i8> %462, i32 1, !dbg !21
  %473 = extractelement <4 x i8> %462, i32 2, !dbg !21
  %474 = extractelement <4 x i8> %462, i32 3, !dbg !21
  %475 = extractelement <4 x i8> %464, i32 0, !dbg !21
  %476 = extractelement <4 x i8> %464, i32 1, !dbg !21
  %477 = extractelement <4 x i8> %464, i32 2, !dbg !21
  %478 = extractelement <4 x i8> %464, i32 3, !dbg !21
  %479 = extractelement <4 x i8> %466, i32 0, !dbg !21
  %480 = extractelement <4 x i8> %466, i32 1, !dbg !21
  %481 = extractelement <4 x i8> %466, i32 2, !dbg !21
  %482 = extractelement <4 x i8> %466, i32 3, !dbg !21
  %483 = call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %457, i1 %50), !dbg !21
  %484 = extractvalue { i32, i32, i32, i32 } %483, 0, !dbg !21
  %485 = bitcast i32 %484 to <4 x i8>, !dbg !21
  %486 = extractvalue { i32, i32, i32, i32 } %483, 1, !dbg !21
  %487 = bitcast i32 %486 to <4 x i8>, !dbg !21
  %488 = extractvalue { i32, i32, i32, i32 } %483, 2, !dbg !21
  %489 = bitcast i32 %488 to <4 x i8>, !dbg !21
  %490 = extractvalue { i32, i32, i32, i32 } %483, 3, !dbg !21
  %491 = bitcast i32 %490 to <4 x i8>, !dbg !21
  %492 = extractelement <4 x i8> %485, i32 0, !dbg !21
  %493 = extractelement <4 x i8> %485, i32 1, !dbg !21
  %494 = extractelement <4 x i8> %485, i32 2, !dbg !21
  %495 = extractelement <4 x i8> %485, i32 3, !dbg !21
  %496 = extractelement <4 x i8> %487, i32 0, !dbg !21
  %497 = extractelement <4 x i8> %487, i32 1, !dbg !21
  %498 = extractelement <4 x i8> %487, i32 2, !dbg !21
  %499 = extractelement <4 x i8> %487, i32 3, !dbg !21
  %500 = extractelement <4 x i8> %489, i32 0, !dbg !21
  %501 = extractelement <4 x i8> %489, i32 1, !dbg !21
  %502 = extractelement <4 x i8> %489, i32 2, !dbg !21
  %503 = extractelement <4 x i8> %489, i32 3, !dbg !21
  %504 = extractelement <4 x i8> %491, i32 0, !dbg !21
  %505 = extractelement <4 x i8> %491, i32 1, !dbg !21
  %506 = extractelement <4 x i8> %491, i32 2, !dbg !21
  %507 = extractelement <4 x i8> %491, i32 3, !dbg !21
  %508 = insertelement <2 x i8> undef, i8 %467, i32 0, !dbg !21
  %509 = insertelement <2 x i8> %508, i8 %468, i32 1, !dbg !21
  %510 = bitcast <2 x i8> %509 to i16, !dbg !21
  %511 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %510), !dbg !21
  %512 = extractelement <2 x half> %511, i32 0, !dbg !21
  %513 = extractelement <2 x half> %511, i32 1, !dbg !21
  %514 = fpext half %512 to float, !dbg !21
  %515 = fpext half %513 to float, !dbg !21
  %516 = insertelement <2 x i8> undef, i8 %469, i32 0, !dbg !21
  %517 = insertelement <2 x i8> %516, i8 %470, i32 1, !dbg !21
  %518 = bitcast <2 x i8> %517 to i16, !dbg !21
  %519 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %518), !dbg !21
  %520 = extractelement <2 x half> %519, i32 0, !dbg !21
  %521 = extractelement <2 x half> %519, i32 1, !dbg !21
  %522 = fpext half %520 to float, !dbg !21
  %523 = fpext half %521 to float, !dbg !21
  %524 = insertelement <2 x i8> undef, i8 %471, i32 0, !dbg !21
  %525 = insertelement <2 x i8> %524, i8 %472, i32 1, !dbg !21
  %526 = bitcast <2 x i8> %525 to i16, !dbg !21
  %527 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %526), !dbg !21
  %528 = extractelement <2 x half> %527, i32 0, !dbg !21
  %529 = extractelement <2 x half> %527, i32 1, !dbg !21
  %530 = fpext half %528 to float, !dbg !21
  %531 = fpext half %529 to float, !dbg !21
  %532 = insertelement <2 x i8> undef, i8 %473, i32 0, !dbg !21
  %533 = insertelement <2 x i8> %532, i8 %474, i32 1, !dbg !21
  %534 = bitcast <2 x i8> %533 to i16, !dbg !21
  %535 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %534), !dbg !21
  %536 = extractelement <2 x half> %535, i32 0, !dbg !21
  %537 = extractelement <2 x half> %535, i32 1, !dbg !21
  %538 = fpext half %536 to float, !dbg !21
  %539 = fpext half %537 to float, !dbg !21
  %540 = insertelement <2 x i8> undef, i8 %475, i32 0, !dbg !21
  %541 = insertelement <2 x i8> %540, i8 %476, i32 1, !dbg !21
  %542 = bitcast <2 x i8> %541 to i16, !dbg !21
  %543 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %542), !dbg !21
  %544 = extractelement <2 x half> %543, i32 0, !dbg !21
  %545 = extractelement <2 x half> %543, i32 1, !dbg !21
  %546 = fpext half %544 to float, !dbg !21
  %547 = fpext half %545 to float, !dbg !21
  %548 = insertelement <2 x i8> undef, i8 %477, i32 0, !dbg !21
  %549 = insertelement <2 x i8> %548, i8 %478, i32 1, !dbg !21
  %550 = bitcast <2 x i8> %549 to i16, !dbg !21
  %551 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %550), !dbg !21
  %552 = extractelement <2 x half> %551, i32 0, !dbg !21
  %553 = extractelement <2 x half> %551, i32 1, !dbg !21
  %554 = fpext half %552 to float, !dbg !21
  %555 = fpext half %553 to float, !dbg !21
  %556 = insertelement <2 x i8> undef, i8 %479, i32 0, !dbg !21
  %557 = insertelement <2 x i8> %556, i8 %480, i32 1, !dbg !21
  %558 = bitcast <2 x i8> %557 to i16, !dbg !21
  %559 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %558), !dbg !21
  %560 = extractelement <2 x half> %559, i32 0, !dbg !21
  %561 = extractelement <2 x half> %559, i32 1, !dbg !21
  %562 = fpext half %560 to float, !dbg !21
  %563 = fpext half %561 to float, !dbg !21
  %564 = insertelement <2 x i8> undef, i8 %481, i32 0, !dbg !21
  %565 = insertelement <2 x i8> %564, i8 %482, i32 1, !dbg !21
  %566 = bitcast <2 x i8> %565 to i16, !dbg !21
  %567 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %566), !dbg !21
  %568 = extractelement <2 x half> %567, i32 0, !dbg !21
  %569 = extractelement <2 x half> %567, i32 1, !dbg !21
  %570 = fpext half %568 to float, !dbg !21
  %571 = fpext half %569 to float, !dbg !21
  %572 = insertelement <2 x i8> undef, i8 %492, i32 0, !dbg !21
  %573 = insertelement <2 x i8> %572, i8 %493, i32 1, !dbg !21
  %574 = bitcast <2 x i8> %573 to i16, !dbg !21
  %575 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %574), !dbg !21
  %576 = extractelement <2 x half> %575, i32 0, !dbg !21
  %577 = extractelement <2 x half> %575, i32 1, !dbg !21
  %578 = fpext half %576 to float, !dbg !21
  %579 = fpext half %577 to float, !dbg !21
  %580 = insertelement <2 x i8> undef, i8 %494, i32 0, !dbg !21
  %581 = insertelement <2 x i8> %580, i8 %495, i32 1, !dbg !21
  %582 = bitcast <2 x i8> %581 to i16, !dbg !21
  %583 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %582), !dbg !21
  %584 = extractelement <2 x half> %583, i32 0, !dbg !21
  %585 = extractelement <2 x half> %583, i32 1, !dbg !21
  %586 = fpext half %584 to float, !dbg !21
  %587 = fpext half %585 to float, !dbg !21
  %588 = insertelement <2 x i8> undef, i8 %496, i32 0, !dbg !21
  %589 = insertelement <2 x i8> %588, i8 %497, i32 1, !dbg !21
  %590 = bitcast <2 x i8> %589 to i16, !dbg !21
  %591 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %590), !dbg !21
  %592 = extractelement <2 x half> %591, i32 0, !dbg !21
  %593 = extractelement <2 x half> %591, i32 1, !dbg !21
  %594 = fpext half %592 to float, !dbg !21
  %595 = fpext half %593 to float, !dbg !21
  %596 = insertelement <2 x i8> undef, i8 %498, i32 0, !dbg !21
  %597 = insertelement <2 x i8> %596, i8 %499, i32 1, !dbg !21
  %598 = bitcast <2 x i8> %597 to i16, !dbg !21
  %599 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %598), !dbg !21
  %600 = extractelement <2 x half> %599, i32 0, !dbg !21
  %601 = extractelement <2 x half> %599, i32 1, !dbg !21
  %602 = fpext half %600 to float, !dbg !21
  %603 = fpext half %601 to float, !dbg !21
  %604 = insertelement <2 x i8> undef, i8 %500, i32 0, !dbg !21
  %605 = insertelement <2 x i8> %604, i8 %501, i32 1, !dbg !21
  %606 = bitcast <2 x i8> %605 to i16, !dbg !21
  %607 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %606), !dbg !21
  %608 = extractelement <2 x half> %607, i32 0, !dbg !21
  %609 = extractelement <2 x half> %607, i32 1, !dbg !21
  %610 = fpext half %608 to float, !dbg !21
  %611 = fpext half %609 to float, !dbg !21
  %612 = insertelement <2 x i8> undef, i8 %502, i32 0, !dbg !21
  %613 = insertelement <2 x i8> %612, i8 %503, i32 1, !dbg !21
  %614 = bitcast <2 x i8> %613 to i16, !dbg !21
  %615 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %614), !dbg !21
  %616 = extractelement <2 x half> %615, i32 0, !dbg !21
  %617 = extractelement <2 x half> %615, i32 1, !dbg !21
  %618 = fpext half %616 to float, !dbg !21
  %619 = fpext half %617 to float, !dbg !21
  %620 = insertelement <2 x i8> undef, i8 %504, i32 0, !dbg !21
  %621 = insertelement <2 x i8> %620, i8 %505, i32 1, !dbg !21
  %622 = bitcast <2 x i8> %621 to i16, !dbg !21
  %623 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %622), !dbg !21
  %624 = extractelement <2 x half> %623, i32 0, !dbg !21
  %625 = extractelement <2 x half> %623, i32 1, !dbg !21
  %626 = fpext half %624 to float, !dbg !21
  %627 = fpext half %625 to float, !dbg !21
  %628 = insertelement <2 x i8> undef, i8 %506, i32 0, !dbg !21
  %629 = insertelement <2 x i8> %628, i8 %507, i32 1, !dbg !21
  %630 = bitcast <2 x i8> %629 to i16, !dbg !21
  %631 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %630), !dbg !21
  %632 = extractelement <2 x half> %631, i32 0, !dbg !21
  %633 = extractelement <2 x half> %631, i32 1, !dbg !21
  %634 = fpext half %632 to float, !dbg !21
  %635 = fpext half %633 to float, !dbg !21
  %636 = fadd float %420, %514, !dbg !22
  %637 = fadd float %421, %515, !dbg !22
  %638 = fadd float %422, %522, !dbg !22
  %639 = fadd float %423, %523, !dbg !22
  %640 = fadd float %424, %530, !dbg !22
  %641 = fadd float %425, %531, !dbg !22
  %642 = fadd float %426, %538, !dbg !22
  %643 = fadd float %427, %539, !dbg !22
  %644 = fadd float %428, %546, !dbg !22
  %645 = fadd float %429, %547, !dbg !22
  %646 = fadd float %430, %554, !dbg !22
  %647 = fadd float %431, %555, !dbg !22
  %648 = fadd float %432, %562, !dbg !22
  %649 = fadd float %433, %563, !dbg !22
  %650 = fadd float %434, %570, !dbg !22
  %651 = fadd float %435, %571, !dbg !22
  %652 = fadd float %436, %578, !dbg !22
  %653 = fadd float %437, %579, !dbg !22
  %654 = fadd float %438, %586, !dbg !22
  %655 = fadd float %439, %587, !dbg !22
  %656 = fadd float %440, %594, !dbg !22
  %657 = fadd float %441, %595, !dbg !22
  %658 = fadd float %442, %602, !dbg !22
  %659 = fadd float %443, %603, !dbg !22
  %660 = fadd float %444, %610, !dbg !22
  %661 = fadd float %445, %611, !dbg !22
  %662 = fadd float %446, %618, !dbg !22
  %663 = fadd float %447, %619, !dbg !22
  %664 = fadd float %448, %626, !dbg !22
  %665 = fadd float %449, %627, !dbg !22
  %666 = fadd float %450, %634, !dbg !22
  %667 = fadd float %451, %635, !dbg !22
  %668 = mul i64 %1, 3, !dbg !23
  %669 = getelementptr i8, ptr addrspace(1) %0, i64 %668, !dbg !20
  %670 = getelementptr i8, ptr addrspace(1) %669, i64 %53, !dbg !20
  %671 = getelementptr i8, ptr addrspace(1) %669, i64 %54, !dbg !20
  %672 = getelementptr i8, ptr addrspace(1) %670, i32 %45, !dbg !20
  %673 = getelementptr i8, ptr addrspace(1) %671, i32 %45, !dbg !20
  %674 = call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %672, i1 %49), !dbg !21
  %675 = extractvalue { i32, i32, i32, i32 } %674, 0, !dbg !21
  %676 = bitcast i32 %675 to <4 x i8>, !dbg !21
  %677 = extractvalue { i32, i32, i32, i32 } %674, 1, !dbg !21
  %678 = bitcast i32 %677 to <4 x i8>, !dbg !21
  %679 = extractvalue { i32, i32, i32, i32 } %674, 2, !dbg !21
  %680 = bitcast i32 %679 to <4 x i8>, !dbg !21
  %681 = extractvalue { i32, i32, i32, i32 } %674, 3, !dbg !21
  %682 = bitcast i32 %681 to <4 x i8>, !dbg !21
  %683 = extractelement <4 x i8> %676, i32 0, !dbg !21
  %684 = extractelement <4 x i8> %676, i32 1, !dbg !21
  %685 = extractelement <4 x i8> %676, i32 2, !dbg !21
  %686 = extractelement <4 x i8> %676, i32 3, !dbg !21
  %687 = extractelement <4 x i8> %678, i32 0, !dbg !21
  %688 = extractelement <4 x i8> %678, i32 1, !dbg !21
  %689 = extractelement <4 x i8> %678, i32 2, !dbg !21
  %690 = extractelement <4 x i8> %678, i32 3, !dbg !21
  %691 = extractelement <4 x i8> %680, i32 0, !dbg !21
  %692 = extractelement <4 x i8> %680, i32 1, !dbg !21
  %693 = extractelement <4 x i8> %680, i32 2, !dbg !21
  %694 = extractelement <4 x i8> %680, i32 3, !dbg !21
  %695 = extractelement <4 x i8> %682, i32 0, !dbg !21
  %696 = extractelement <4 x i8> %682, i32 1, !dbg !21
  %697 = extractelement <4 x i8> %682, i32 2, !dbg !21
  %698 = extractelement <4 x i8> %682, i32 3, !dbg !21
  %699 = call { i32, i32, i32, i32 } asm sideeffect "mov.u32 $0, $4;\0A\09mov.u32 $1, $5;\0A\09mov.u32 $2, $6;\0A\09mov.u32 $3, $7;\0A\09@$9 ld.global.v4.b32 { $0, $1, $2, $3 }, [ $8 + 0 ];", "=r,=r,=r,=r,r,r,r,r,l,b"(i32 0, i32 0, i32 0, i32 0, ptr addrspace(1) %673, i1 %50), !dbg !21
  %700 = extractvalue { i32, i32, i32, i32 } %699, 0, !dbg !21
  %701 = bitcast i32 %700 to <4 x i8>, !dbg !21
  %702 = extractvalue { i32, i32, i32, i32 } %699, 1, !dbg !21
  %703 = bitcast i32 %702 to <4 x i8>, !dbg !21
  %704 = extractvalue { i32, i32, i32, i32 } %699, 2, !dbg !21
  %705 = bitcast i32 %704 to <4 x i8>, !dbg !21
  %706 = extractvalue { i32, i32, i32, i32 } %699, 3, !dbg !21
  %707 = bitcast i32 %706 to <4 x i8>, !dbg !21
  %708 = extractelement <4 x i8> %701, i32 0, !dbg !21
  %709 = extractelement <4 x i8> %701, i32 1, !dbg !21
  %710 = extractelement <4 x i8> %701, i32 2, !dbg !21
  %711 = extractelement <4 x i8> %701, i32 3, !dbg !21
  %712 = extractelement <4 x i8> %703, i32 0, !dbg !21
  %713 = extractelement <4 x i8> %703, i32 1, !dbg !21
  %714 = extractelement <4 x i8> %703, i32 2, !dbg !21
  %715 = extractelement <4 x i8> %703, i32 3, !dbg !21
  %716 = extractelement <4 x i8> %705, i32 0, !dbg !21
  %717 = extractelement <4 x i8> %705, i32 1, !dbg !21
  %718 = extractelement <4 x i8> %705, i32 2, !dbg !21
  %719 = extractelement <4 x i8> %705, i32 3, !dbg !21
  %720 = extractelement <4 x i8> %707, i32 0, !dbg !21
  %721 = extractelement <4 x i8> %707, i32 1, !dbg !21
  %722 = extractelement <4 x i8> %707, i32 2, !dbg !21
  %723 = extractelement <4 x i8> %707, i32 3, !dbg !21
  %724 = insertelement <2 x i8> undef, i8 %683, i32 0, !dbg !21
  %725 = insertelement <2 x i8> %724, i8 %684, i32 1, !dbg !21
  %726 = bitcast <2 x i8> %725 to i16, !dbg !21
  %727 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %726), !dbg !21
  %728 = extractelement <2 x half> %727, i32 0, !dbg !21
  %729 = extractelement <2 x half> %727, i32 1, !dbg !21
  %730 = fpext half %728 to float, !dbg !21
  %731 = fpext half %729 to float, !dbg !21
  %732 = insertelement <2 x i8> undef, i8 %685, i32 0, !dbg !21
  %733 = insertelement <2 x i8> %732, i8 %686, i32 1, !dbg !21
  %734 = bitcast <2 x i8> %733 to i16, !dbg !21
  %735 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %734), !dbg !21
  %736 = extractelement <2 x half> %735, i32 0, !dbg !21
  %737 = extractelement <2 x half> %735, i32 1, !dbg !21
  %738 = fpext half %736 to float, !dbg !21
  %739 = fpext half %737 to float, !dbg !21
  %740 = insertelement <2 x i8> undef, i8 %687, i32 0, !dbg !21
  %741 = insertelement <2 x i8> %740, i8 %688, i32 1, !dbg !21
  %742 = bitcast <2 x i8> %741 to i16, !dbg !21
  %743 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %742), !dbg !21
  %744 = extractelement <2 x half> %743, i32 0, !dbg !21
  %745 = extractelement <2 x half> %743, i32 1, !dbg !21
  %746 = fpext half %744 to float, !dbg !21
  %747 = fpext half %745 to float, !dbg !21
  %748 = insertelement <2 x i8> undef, i8 %689, i32 0, !dbg !21
  %749 = insertelement <2 x i8> %748, i8 %690, i32 1, !dbg !21
  %750 = bitcast <2 x i8> %749 to i16, !dbg !21
  %751 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %750), !dbg !21
  %752 = extractelement <2 x half> %751, i32 0, !dbg !21
  %753 = extractelement <2 x half> %751, i32 1, !dbg !21
  %754 = fpext half %752 to float, !dbg !21
  %755 = fpext half %753 to float, !dbg !21
  %756 = insertelement <2 x i8> undef, i8 %691, i32 0, !dbg !21
  %757 = insertelement <2 x i8> %756, i8 %692, i32 1, !dbg !21
  %758 = bitcast <2 x i8> %757 to i16, !dbg !21
  %759 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %758), !dbg !21
  %760 = extractelement <2 x half> %759, i32 0, !dbg !21
  %761 = extractelement <2 x half> %759, i32 1, !dbg !21
  %762 = fpext half %760 to float, !dbg !21
  %763 = fpext half %761 to float, !dbg !21
  %764 = insertelement <2 x i8> undef, i8 %693, i32 0, !dbg !21
  %765 = insertelement <2 x i8> %764, i8 %694, i32 1, !dbg !21
  %766 = bitcast <2 x i8> %765 to i16, !dbg !21
  %767 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %766), !dbg !21
  %768 = extractelement <2 x half> %767, i32 0, !dbg !21
  %769 = extractelement <2 x half> %767, i32 1, !dbg !21
  %770 = fpext half %768 to float, !dbg !21
  %771 = fpext half %769 to float, !dbg !21
  %772 = insertelement <2 x i8> undef, i8 %695, i32 0, !dbg !21
  %773 = insertelement <2 x i8> %772, i8 %696, i32 1, !dbg !21
  %774 = bitcast <2 x i8> %773 to i16, !dbg !21
  %775 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %774), !dbg !21
  %776 = extractelement <2 x half> %775, i32 0, !dbg !21
  %777 = extractelement <2 x half> %775, i32 1, !dbg !21
  %778 = fpext half %776 to float, !dbg !21
  %779 = fpext half %777 to float, !dbg !21
  %780 = insertelement <2 x i8> undef, i8 %697, i32 0, !dbg !21
  %781 = insertelement <2 x i8> %780, i8 %698, i32 1, !dbg !21
  %782 = bitcast <2 x i8> %781 to i16, !dbg !21
  %783 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %782), !dbg !21
  %784 = extractelement <2 x half> %783, i32 0, !dbg !21
  %785 = extractelement <2 x half> %783, i32 1, !dbg !21
  %786 = fpext half %784 to float, !dbg !21
  %787 = fpext half %785 to float, !dbg !21
  %788 = insertelement <2 x i8> undef, i8 %708, i32 0, !dbg !21
  %789 = insertelement <2 x i8> %788, i8 %709, i32 1, !dbg !21
  %790 = bitcast <2 x i8> %789 to i16, !dbg !21
  %791 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %790), !dbg !21
  %792 = extractelement <2 x half> %791, i32 0, !dbg !21
  %793 = extractelement <2 x half> %791, i32 1, !dbg !21
  %794 = fpext half %792 to float, !dbg !21
  %795 = fpext half %793 to float, !dbg !21
  %796 = insertelement <2 x i8> undef, i8 %710, i32 0, !dbg !21
  %797 = insertelement <2 x i8> %796, i8 %711, i32 1, !dbg !21
  %798 = bitcast <2 x i8> %797 to i16, !dbg !21
  %799 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %798), !dbg !21
  %800 = extractelement <2 x half> %799, i32 0, !dbg !21
  %801 = extractelement <2 x half> %799, i32 1, !dbg !21
  %802 = fpext half %800 to float, !dbg !21
  %803 = fpext half %801 to float, !dbg !21
  %804 = insertelement <2 x i8> undef, i8 %712, i32 0, !dbg !21
  %805 = insertelement <2 x i8> %804, i8 %713, i32 1, !dbg !21
  %806 = bitcast <2 x i8> %805 to i16, !dbg !21
  %807 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %806), !dbg !21
  %808 = extractelement <2 x half> %807, i32 0, !dbg !21
  %809 = extractelement <2 x half> %807, i32 1, !dbg !21
  %810 = fpext half %808 to float, !dbg !21
  %811 = fpext half %809 to float, !dbg !21
  %812 = insertelement <2 x i8> undef, i8 %714, i32 0, !dbg !21
  %813 = insertelement <2 x i8> %812, i8 %715, i32 1, !dbg !21
  %814 = bitcast <2 x i8> %813 to i16, !dbg !21
  %815 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %814), !dbg !21
  %816 = extractelement <2 x half> %815, i32 0, !dbg !21
  %817 = extractelement <2 x half> %815, i32 1, !dbg !21
  %818 = fpext half %816 to float, !dbg !21
  %819 = fpext half %817 to float, !dbg !21
  %820 = insertelement <2 x i8> undef, i8 %716, i32 0, !dbg !21
  %821 = insertelement <2 x i8> %820, i8 %717, i32 1, !dbg !21
  %822 = bitcast <2 x i8> %821 to i16, !dbg !21
  %823 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %822), !dbg !21
  %824 = extractelement <2 x half> %823, i32 0, !dbg !21
  %825 = extractelement <2 x half> %823, i32 1, !dbg !21
  %826 = fpext half %824 to float, !dbg !21
  %827 = fpext half %825 to float, !dbg !21
  %828 = insertelement <2 x i8> undef, i8 %718, i32 0, !dbg !21
  %829 = insertelement <2 x i8> %828, i8 %719, i32 1, !dbg !21
  %830 = bitcast <2 x i8> %829 to i16, !dbg !21
  %831 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %830), !dbg !21
  %832 = extractelement <2 x half> %831, i32 0, !dbg !21
  %833 = extractelement <2 x half> %831, i32 1, !dbg !21
  %834 = fpext half %832 to float, !dbg !21
  %835 = fpext half %833 to float, !dbg !21
  %836 = insertelement <2 x i8> undef, i8 %720, i32 0, !dbg !21
  %837 = insertelement <2 x i8> %836, i8 %721, i32 1, !dbg !21
  %838 = bitcast <2 x i8> %837 to i16, !dbg !21
  %839 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %838), !dbg !21
  %840 = extractelement <2 x half> %839, i32 0, !dbg !21
  %841 = extractelement <2 x half> %839, i32 1, !dbg !21
  %842 = fpext half %840 to float, !dbg !21
  %843 = fpext half %841 to float, !dbg !21
  %844 = insertelement <2 x i8> undef, i8 %722, i32 0, !dbg !21
  %845 = insertelement <2 x i8> %844, i8 %723, i32 1, !dbg !21
  %846 = bitcast <2 x i8> %845 to i16, !dbg !21
  %847 = call <2 x half> asm "{ \0Acvt.rn.f16x2.e4m3x2 $0, $1; \0A}", "=r,h"(i16 %846), !dbg !21
  %848 = extractelement <2 x half> %847, i32 0, !dbg !21
  %849 = extractelement <2 x half> %847, i32 1, !dbg !21
  %850 = fpext half %848 to float, !dbg !21
  %851 = fpext half %849 to float, !dbg !21
  %852 = fadd float %636, %730, !dbg !22
  %853 = fadd float %637, %731, !dbg !22
  %854 = fadd float %638, %738, !dbg !22
  %855 = fadd float %639, %739, !dbg !22
  %856 = fadd float %640, %746, !dbg !22
  %857 = fadd float %641, %747, !dbg !22
  %858 = fadd float %642, %754, !dbg !22
  %859 = fadd float %643, %755, !dbg !22
  %860 = fadd float %644, %762, !dbg !22
  %861 = fadd float %645, %763, !dbg !22
  %862 = fadd float %646, %770, !dbg !22
  %863 = fadd float %647, %771, !dbg !22
  %864 = fadd float %648, %778, !dbg !22
  %865 = fadd float %649, %779, !dbg !22
  %866 = fadd float %650, %786, !dbg !22
  %867 = fadd float %651, %787, !dbg !22
  %868 = fadd float %652, %794, !dbg !22
  %869 = fadd float %653, %795, !dbg !22
  %870 = fadd float %654, %802, !dbg !22
  %871 = fadd float %655, %803, !dbg !22
  %872 = fadd float %656, %810, !dbg !22
  %873 = fadd float %657, %811, !dbg !22
  %874 = fadd float %658, %818, !dbg !22
  %875 = fadd float %659, %819, !dbg !22
  %876 = fadd float %660, %826, !dbg !22
  %877 = fadd float %661, %827, !dbg !22
  %878 = fadd float %662, %834, !dbg !22
  %879 = fadd float %663, %835, !dbg !22
  %880 = fadd float %664, %842, !dbg !22
  %881 = fadd float %665, %843, !dbg !22
  %882 = fadd float %666, %850, !dbg !22
  %883 = fadd float %667, %851, !dbg !22
  %884 = icmp slt i32 %45, %13, !dbg !24
  %885 = mul i64 %51, %4, !dbg !25
  %886 = mul i64 %52, %4, !dbg !25
  %887 = getelementptr i8, ptr addrspace(1) %3, i64 %885, !dbg !26
  %888 = getelementptr i8, ptr addrspace(1) %3, i64 %886, !dbg !26
  %889 = getelementptr i8, ptr addrspace(1) %887, i32 %45, !dbg !26
  %890 = getelementptr i8, ptr addrspace(1) %888, i32 %45, !dbg !26
  %891 = and i1 %46, %884, !dbg !27
  %892 = and i1 %47, %884, !dbg !27
  %893 = insertelement <1 x float> undef, float %852, i32 0, !dbg !28
  %894 = insertelement <1 x float> undef, float %853, i32 0, !dbg !28
  %895 = bitcast <1 x float> %893 to i32, !dbg !28
  %896 = bitcast <1 x float> %894 to i32, !dbg !28
  %897 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %895, i32 %896), !dbg !28
  %898 = extractelement <2 x i8> %897, i32 0, !dbg !28
  %899 = extractelement <2 x i8> %897, i32 1, !dbg !28
  %900 = insertelement <1 x float> undef, float %854, i32 0, !dbg !28
  %901 = insertelement <1 x float> undef, float %855, i32 0, !dbg !28
  %902 = bitcast <1 x float> %900 to i32, !dbg !28
  %903 = bitcast <1 x float> %901 to i32, !dbg !28
  %904 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %902, i32 %903), !dbg !28
  %905 = extractelement <2 x i8> %904, i32 0, !dbg !28
  %906 = extractelement <2 x i8> %904, i32 1, !dbg !28
  %907 = insertelement <1 x float> undef, float %856, i32 0, !dbg !28
  %908 = insertelement <1 x float> undef, float %857, i32 0, !dbg !28
  %909 = bitcast <1 x float> %907 to i32, !dbg !28
  %910 = bitcast <1 x float> %908 to i32, !dbg !28
  %911 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %909, i32 %910), !dbg !28
  %912 = extractelement <2 x i8> %911, i32 0, !dbg !28
  %913 = extractelement <2 x i8> %911, i32 1, !dbg !28
  %914 = insertelement <1 x float> undef, float %858, i32 0, !dbg !28
  %915 = insertelement <1 x float> undef, float %859, i32 0, !dbg !28
  %916 = bitcast <1 x float> %914 to i32, !dbg !28
  %917 = bitcast <1 x float> %915 to i32, !dbg !28
  %918 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %916, i32 %917), !dbg !28
  %919 = extractelement <2 x i8> %918, i32 0, !dbg !28
  %920 = extractelement <2 x i8> %918, i32 1, !dbg !28
  %921 = insertelement <1 x float> undef, float %860, i32 0, !dbg !28
  %922 = insertelement <1 x float> undef, float %861, i32 0, !dbg !28
  %923 = bitcast <1 x float> %921 to i32, !dbg !28
  %924 = bitcast <1 x float> %922 to i32, !dbg !28
  %925 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %923, i32 %924), !dbg !28
  %926 = extractelement <2 x i8> %925, i32 0, !dbg !28
  %927 = extractelement <2 x i8> %925, i32 1, !dbg !28
  %928 = insertelement <1 x float> undef, float %862, i32 0, !dbg !28
  %929 = insertelement <1 x float> undef, float %863, i32 0, !dbg !28
  %930 = bitcast <1 x float> %928 to i32, !dbg !28
  %931 = bitcast <1 x float> %929 to i32, !dbg !28
  %932 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %930, i32 %931), !dbg !28
  %933 = extractelement <2 x i8> %932, i32 0, !dbg !28
  %934 = extractelement <2 x i8> %932, i32 1, !dbg !28
  %935 = insertelement <1 x float> undef, float %864, i32 0, !dbg !28
  %936 = insertelement <1 x float> undef, float %865, i32 0, !dbg !28
  %937 = bitcast <1 x float> %935 to i32, !dbg !28
  %938 = bitcast <1 x float> %936 to i32, !dbg !28
  %939 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %937, i32 %938), !dbg !28
  %940 = extractelement <2 x i8> %939, i32 0, !dbg !28
  %941 = extractelement <2 x i8> %939, i32 1, !dbg !28
  %942 = insertelement <1 x float> undef, float %866, i32 0, !dbg !28
  %943 = insertelement <1 x float> undef, float %867, i32 0, !dbg !28
  %944 = bitcast <1 x float> %942 to i32, !dbg !28
  %945 = bitcast <1 x float> %943 to i32, !dbg !28
  %946 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %944, i32 %945), !dbg !28
  %947 = extractelement <2 x i8> %946, i32 0, !dbg !28
  %948 = extractelement <2 x i8> %946, i32 1, !dbg !28
  %949 = insertelement <1 x float> undef, float %868, i32 0, !dbg !28
  %950 = insertelement <1 x float> undef, float %869, i32 0, !dbg !28
  %951 = bitcast <1 x float> %949 to i32, !dbg !28
  %952 = bitcast <1 x float> %950 to i32, !dbg !28
  %953 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %951, i32 %952), !dbg !28
  %954 = extractelement <2 x i8> %953, i32 0, !dbg !28
  %955 = extractelement <2 x i8> %953, i32 1, !dbg !28
  %956 = insertelement <1 x float> undef, float %870, i32 0, !dbg !28
  %957 = insertelement <1 x float> undef, float %871, i32 0, !dbg !28
  %958 = bitcast <1 x float> %956 to i32, !dbg !28
  %959 = bitcast <1 x float> %957 to i32, !dbg !28
  %960 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %958, i32 %959), !dbg !28
  %961 = extractelement <2 x i8> %960, i32 0, !dbg !28
  %962 = extractelement <2 x i8> %960, i32 1, !dbg !28
  %963 = insertelement <1 x float> undef, float %872, i32 0, !dbg !28
  %964 = insertelement <1 x float> undef, float %873, i32 0, !dbg !28
  %965 = bitcast <1 x float> %963 to i32, !dbg !28
  %966 = bitcast <1 x float> %964 to i32, !dbg !28
  %967 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %965, i32 %966), !dbg !28
  %968 = extractelement <2 x i8> %967, i32 0, !dbg !28
  %969 = extractelement <2 x i8> %967, i32 1, !dbg !28
  %970 = insertelement <1 x float> undef, float %874, i32 0, !dbg !28
  %971 = insertelement <1 x float> undef, float %875, i32 0, !dbg !28
  %972 = bitcast <1 x float> %970 to i32, !dbg !28
  %973 = bitcast <1 x float> %971 to i32, !dbg !28
  %974 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %972, i32 %973), !dbg !28
  %975 = extractelement <2 x i8> %974, i32 0, !dbg !28
  %976 = extractelement <2 x i8> %974, i32 1, !dbg !28
  %977 = insertelement <1 x float> undef, float %876, i32 0, !dbg !28
  %978 = insertelement <1 x float> undef, float %877, i32 0, !dbg !28
  %979 = bitcast <1 x float> %977 to i32, !dbg !28
  %980 = bitcast <1 x float> %978 to i32, !dbg !28
  %981 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %979, i32 %980), !dbg !28
  %982 = extractelement <2 x i8> %981, i32 0, !dbg !28
  %983 = extractelement <2 x i8> %981, i32 1, !dbg !28
  %984 = insertelement <1 x float> undef, float %878, i32 0, !dbg !28
  %985 = insertelement <1 x float> undef, float %879, i32 0, !dbg !28
  %986 = bitcast <1 x float> %984 to i32, !dbg !28
  %987 = bitcast <1 x float> %985 to i32, !dbg !28
  %988 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %986, i32 %987), !dbg !28
  %989 = extractelement <2 x i8> %988, i32 0, !dbg !28
  %990 = extractelement <2 x i8> %988, i32 1, !dbg !28
  %991 = insertelement <1 x float> undef, float %880, i32 0, !dbg !28
  %992 = insertelement <1 x float> undef, float %881, i32 0, !dbg !28
  %993 = bitcast <1 x float> %991 to i32, !dbg !28
  %994 = bitcast <1 x float> %992 to i32, !dbg !28
  %995 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %993, i32 %994), !dbg !28
  %996 = extractelement <2 x i8> %995, i32 0, !dbg !28
  %997 = extractelement <2 x i8> %995, i32 1, !dbg !28
  %998 = insertelement <1 x float> undef, float %882, i32 0, !dbg !28
  %999 = insertelement <1 x float> undef, float %883, i32 0, !dbg !28
  %1000 = bitcast <1 x float> %998 to i32, !dbg !28
  %1001 = bitcast <1 x float> %999 to i32, !dbg !28
  %1002 = call <2 x i8> asm "cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \0A", "=h,r,r"(i32 %1000, i32 %1001), !dbg !28
  %1003 = extractelement <2 x i8> %1002, i32 0, !dbg !28
  %1004 = extractelement <2 x i8> %1002, i32 1, !dbg !28
  %1005 = insertelement <4 x i8> undef, i8 %898, i32 0, !dbg !28
  %1006 = insertelement <4 x i8> %1005, i8 %899, i32 1, !dbg !28
  %1007 = insertelement <4 x i8> %1006, i8 %905, i32 2, !dbg !28
  %1008 = insertelement <4 x i8> %1007, i8 %906, i32 3, !dbg !28
  %1009 = bitcast <4 x i8> %1008 to i32, !dbg !28
  %1010 = insertelement <4 x i8> undef, i8 %912, i32 0, !dbg !28
  %1011 = insertelement <4 x i8> %1010, i8 %913, i32 1, !dbg !28
  %1012 = insertelement <4 x i8> %1011, i8 %919, i32 2, !dbg !28
  %1013 = insertelement <4 x i8> %1012, i8 %920, i32 3, !dbg !28
  %1014 = bitcast <4 x i8> %1013 to i32, !dbg !28
  %1015 = insertelement <4 x i8> undef, i8 %926, i32 0, !dbg !28
  %1016 = insertelement <4 x i8> %1015, i8 %927, i32 1, !dbg !28
  %1017 = insertelement <4 x i8> %1016, i8 %933, i32 2, !dbg !28
  %1018 = insertelement <4 x i8> %1017, i8 %934, i32 3, !dbg !28
  %1019 = bitcast <4 x i8> %1018 to i32, !dbg !28
  %1020 = insertelement <4 x i8> undef, i8 %940, i32 0, !dbg !28
  %1021 = insertelement <4 x i8> %1020, i8 %941, i32 1, !dbg !28
  %1022 = insertelement <4 x i8> %1021, i8 %947, i32 2, !dbg !28
  %1023 = insertelement <4 x i8> %1022, i8 %948, i32 3, !dbg !28
  %1024 = bitcast <4 x i8> %1023 to i32, !dbg !28
  call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %1009, i32 %1014, i32 %1019, i32 %1024, ptr addrspace(1) %889, i1 %891), !dbg !28
  %1025 = insertelement <4 x i8> undef, i8 %954, i32 0, !dbg !28
  %1026 = insertelement <4 x i8> %1025, i8 %955, i32 1, !dbg !28
  %1027 = insertelement <4 x i8> %1026, i8 %961, i32 2, !dbg !28
  %1028 = insertelement <4 x i8> %1027, i8 %962, i32 3, !dbg !28
  %1029 = bitcast <4 x i8> %1028 to i32, !dbg !28
  %1030 = insertelement <4 x i8> undef, i8 %968, i32 0, !dbg !28
  %1031 = insertelement <4 x i8> %1030, i8 %969, i32 1, !dbg !28
  %1032 = insertelement <4 x i8> %1031, i8 %975, i32 2, !dbg !28
  %1033 = insertelement <4 x i8> %1032, i8 %976, i32 3, !dbg !28
  %1034 = bitcast <4 x i8> %1033 to i32, !dbg !28
  %1035 = insertelement <4 x i8> undef, i8 %982, i32 0, !dbg !28
  %1036 = insertelement <4 x i8> %1035, i8 %983, i32 1, !dbg !28
  %1037 = insertelement <4 x i8> %1036, i8 %989, i32 2, !dbg !28
  %1038 = insertelement <4 x i8> %1037, i8 %990, i32 3, !dbg !28
  %1039 = bitcast <4 x i8> %1038 to i32, !dbg !28
  %1040 = insertelement <4 x i8> undef, i8 %996, i32 0, !dbg !28
  %1041 = insertelement <4 x i8> %1040, i8 %997, i32 1, !dbg !28
  %1042 = insertelement <4 x i8> %1041, i8 %1003, i32 2, !dbg !28
  %1043 = insertelement <4 x i8> %1042, i8 %1004, i32 3, !dbg !28
  %1044 = bitcast <4 x i8> %1043 to i32, !dbg !28
  call void asm sideeffect "@$5 st.global.v4.b32 [ $4 + 0 ], { $0, $1, $2, $3 };", "r,r,r,r,l,b"(i32 %1029, i32 %1034, i32 %1039, i32 %1044, ptr addrspace(1) %890, i1 %892), !dbg !28
  ret void, !dbg !29
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 65535) i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { "nvvm.reqntid"="128" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "triton", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "reduce.py", directory: "/Volumes/case_sensitive_workspace/triton/python/triton_kernels/triton_kernels")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "_reduce_forward", linkageName: "_reduce_forward", scope: !1, file: !1, line: 144, type: !4, scopeLine: 144, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DISubroutineType(cc: DW_CC_normal, types: !5)
!5 = !{null, !6, !8, !8, !6, !8, !9, !9, !9, !9, !9, !9, !9, !9, !9, !6, !6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer", baseType: !7, size: 64, dwarfAddressSpace: 1)
!7 = !DIBasicType(name: "unknown_type", encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 151, column: 14, scope: !3)
!11 = !DILocation(line: 152, column: 14, scope: !3)
!12 = !DILocation(line: 156, column: 15, scope: !3)
!13 = !DILocation(line: 156, column: 35, scope: !3)
!14 = !DILocation(line: 157, column: 17, scope: !3)
!15 = !DILocation(line: 157, column: 39, scope: !3)
!16 = !DILocation(line: 165, column: 20, scope: !3)
!17 = !DILocation(line: 166, column: 18, scope: !3)
!18 = !DILocation(line: 173, column: 16, scope: !3)
!19 = !DILocation(line: 181, column: 38, scope: !3)
!20 = !DILocation(line: 181, column: 18, scope: !3)
!21 = !DILocation(line: 182, column: 13, scope: !3)
!22 = !DILocation(line: 199, column: 13, scope: !3)
!23 = !DILocation(line: 181, column: 22, scope: !3)
!24 = !DILocation(line: 206, column: 18, scope: !3)
!25 = !DILocation(line: 220, column: 18, scope: !3)
!26 = !DILocation(line: 220, column: 14, scope: !3)
!27 = !DILocation(line: 221, column: 30, scope: !3)
!28 = !DILocation(line: 221, column: 5, scope: !3)
!29 = !DILocation(line: 144, column: 1, scope: !3)
