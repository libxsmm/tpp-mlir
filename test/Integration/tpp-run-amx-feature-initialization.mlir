// RUN: not --crash tpp-run %s -e entry -entry-point-result=void -mattr=amx-bf16 2>&1 | FileCheck %s --check-prefix=CHECK-AMX-BF16
// RUN: not --crash env LIBXSMM_TARGET=spr tpp-run %s -e entry -entry-point-result=void -mattr=amx-bf16 2>&1 | FileCheck %s --check-prefix=CHECK-AMX-BF16-SETUP

//Tests for unsuccessfull compilation implying AMX pipeline was not initialized
// CHECK-AMX-BF16: error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast

//Tests for successfull compilation implying AMX pipeline was initialized properly.
// CHECK-AMX-BF16-SETUP-NOT: error: LLVM Translation failed for operation: builtin.unrealized_conversion_cast
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