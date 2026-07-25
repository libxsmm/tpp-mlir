// RUN: tpp-opt %s | tpp-opt | FileCheck %s

// FP8 support: libxsmm names E5M2 as BF8 and E4M3 as HF8. FP8 uses a VNNI
// blocking factor of 4 (vnni_4 unary transform).

// CHECK-LABEL: @xsmm_fp8_dialect
func.func @xsmm_fp8_dialect(%arg0: memref<2x2xf8E5M2>, %arg1: memref<2x2xf8E5M2>,
                            %arg2: memref<2x2xf8E5M2>, %arg3: memref<2x2xf8E4M3FN>,
                            %arg4: memref<2x2xf8E4M3FN>, %arg5: memref<2x2xf8E4M3FN>) {
  %d = arith.constant 0 : i64

  // CHECK: xsmm.gemm(data_type = bf8
  xsmm.gemm (data_type = bf8, %d, %arg0, %arg1, %arg2)
    : (i64, memref<2x2xf8E5M2>, memref<2x2xf8E5M2>, memref<2x2xf8E5M2>) -> ()

  // CHECK: xsmm.gemm(data_type = hf8
  xsmm.gemm (data_type = hf8, %d, %arg3, %arg4, %arg5)
    : (i64, memref<2x2xf8E4M3FN>, memref<2x2xf8E4M3FN>, memref<2x2xf8E4M3FN>) -> ()

  // CHECK: xsmm.gemm.dispatch [2, 2, 2, 2, 2, 2] flags = (vnni_b) data_type = bf8
  %0 = xsmm.gemm.dispatch [2, 2, 2, 2, 2, 2] flags = (vnni_b) data_type = bf8

  // CHECK: xsmm.gemm.dispatch [2, 2, 2, 2, 2, 2] flags = (vnni_b) data_type = hf8
  %1 = xsmm.gemm.dispatch [2, 2, 2, 2, 2, 2] flags = (vnni_b) data_type = hf8

  // CHECK: xsmm.unary.dispatch vnni_4 [32, 32, 512, 32] flags = (none) data_type = bf8
  %2 = xsmm.unary.dispatch vnni_4 [32, 32, 512, 32] flags = (none) data_type = bf8

  // CHECK: xsmm.unary.dispatch vnni_4 [32, 32, 512, 32] flags = (none) data_type = hf8
  %3 = xsmm.unary.dispatch vnni_4 [32, 32, 512, 32] flags = (none) data_type = hf8

  return
}
