// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<3x3xf32>) -> memref<3x3xf32> {
  %cst = arith.constant 5.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<3x3xf32>)
  %0 = xsmm.unary.dispatch zero [3, 3, 3, 3] flags = (none) data_type = f32
  xsmm.unary zero(data_type = f32, %0, %arg0, %arg0) : (i64, memref<3x3xf32>, memref<3x3xf32>) -> ()

  return %arg0 : memref<3x3xf32>
}

// CHECK: ( 0, 0, 0 )
// CHECK: ( 0, 0, 0 )
// CHECK: ( 0, 0, 0 )
