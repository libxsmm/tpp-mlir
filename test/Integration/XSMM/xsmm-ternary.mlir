// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) -> memref<3x3xf32> {
  %c2_i64 = arith.constant 2 : i64
  %0 = xsmm.brgemm.dispatch [3, 3, 4, 4, 3, 3, 12, 12] flags = (none) data_type = f32
  xsmm.brgemm(data_type = f32, %0, %arg0, %arg1, %arg2, %c2_i64)
    : (i64, memref<2x3x4xf32>, memref<2x4x3xf32>, memref<3x3xf32>, i64) -> ()

  return %arg2 : memref<3x3xf32>
}

// CHECK: ( 9, 9, 9 )
// CHECK: ( 9, 9, 9 )
// CHECK: ( 9, 9, 9 )
