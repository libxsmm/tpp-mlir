// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

func.func @entry(%arg0: memref<64x4x4xf32>, %arg1: memref<64x4x4xf32>, %arg2: memref<4x4xf32>, %arg3: memref<4xf32>) {
  %c16_i64 = arith.constant 16 : i64
  %func = xsmm.fused_brgemm.dispatch [4, 4, 4, 4, 4, 4, 8, 8][add, relu]
    flags = (none) binary_flags = (bcast_col_in0) unary_flags = (none) data_type = f32
  xsmm.fused_brgemm(data_type = f32, %func, %arg0, %arg1, %arg2, %arg3, %c16_i64)
    : (i64, memref<64x4x4xf32>, memref<64x4x4xf32>, memref<4x4xf32>, memref<4xf32>, i64) -> ()

  %threshold = arith.constant 0.0 : f32
  %outVal = arith.constant 66.0 : f32
  %trueOut = memref.alloc(): memref<4x4xf32>
  linalg.fill ins(%outVal : f32) outs(%trueOut : memref<4x4xf32>)
  check.expect_almost_eq(%trueOut, %arg2, %threshold): memref<4x4xf32>, memref<4x4xf32>, f32
  return
}
