// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

func.func @entry(%arg0: memref<64x4x4xbf16>, %arg1: memref<64x2x4x2xbf16>, %arg2: memref<4x4xbf16>, %arg3: memref<4xbf16>) {
  %c16_i64 = arith.constant 16 : i64
  %func = xsmm.fused_brgemm.dispatch [4, 4, 4, 4, 4, 4, 8, 8][add, relu]
    flags = (vnni_b) binary_flags = (bcast_col_in0) unary_flags = (none) data_type = bf16
  xsmm.fused_brgemm(data_type = bf16, %func, %arg0, %arg1, %arg2, %arg3, %c16_i64) : (i64, memref<64x4x4xbf16>, memref<64x2x4x2xbf16>, memref<4x4xbf16>, memref<4xbf16>, i64) -> ()

  %threshold = arith.constant 0.0 : bf16
  %outVal = arith.constant 66.0 : bf16
  %trueOut = memref.alloc(): memref<4x4xbf16>
  linalg.fill ins(%outVal : bf16) outs(%trueOut : memref<4x4xbf16>)
  check.expect_almost_eq(%trueOut, %arg2, %threshold): memref<4x4xbf16>, memref<4x4xbf16>, bf16

  return
}
