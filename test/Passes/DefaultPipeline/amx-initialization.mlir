
// RUN: LIBXSMM_TARGET=spr tpp-opt --default-pipeline %s | FileCheck %s --check-prefix=CHECK-AMX-BF16



// CHECK-AMX-BF16-LABEL:   llvm.func @entry
// CHECK-AMX-BF16:         llvm.call_intrinsic "llvm.x86.tileloadd64.internal"{{.*}} -> !llvm.x86_amx
// CHECK-AMX-BF16:         llvm.call_intrinsic "llvm.x86.tileloadd64.internal"{{.*}} -> !llvm.x86_amx
// CHECK-AMX-BF16:         llvm.call_intrinsic "llvm.x86.tilezero.internal"{{.*}} -> !llvm.x86_amx
// CHECK-AMX-BF16:         llvm.call_intrinsic "llvm.x86.tdpbf16ps.internal"{{.*}} -> !llvm.x86_amx
func.func @entry(%arg0: memref<16x32xbf16>,
             %arg1: memref<16x32xbf16>,
             %arg2: memref<16x16xf32>) {
  %0 = arith.constant 0 : index
  %1 = amx.tile_load %arg0[%0, %0] : memref<16x32xbf16>  into !amx.tile<16x32xbf16>
  %2 = amx.tile_load %arg1[%0, %0] : memref<16x32xbf16>  into !amx.tile<16x32xbf16>
  %3 = amx.tile_zero : !amx.tile<16x16xf32>
  %4 = amx.tile_mulf %1, %2, %3 : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
  amx.tile_store %arg2[%0, %0], %4 : memref<16x16xf32>, !amx.tile<16x16xf32>
  return
}
