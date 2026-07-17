// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @entry(%arg0: memref<3x3xbf16>) -> memref<3x3xbf16> {
  %cst = arith.constant 5.0 : bf16
  linalg.fill ins(%cst : bf16) outs(%arg0 : memref<3x3xbf16>)
  %0 = xsmm.unary.dispatch zero [3, 3, 3, 3] flags = (none) data_type = bf16
  xsmm.unary zero(data_type = bf16, %0, %arg0, %arg0) : (i64, memref<3x3xbf16>, memref<3x3xbf16>) -> ()

  return %arg0 : memref<3x3xbf16>
}

// CHECK: ( 0, 0, 0 )
// CHECK: ( 0, 0, 0 )
// CHECK: ( 0, 0, 0 )
