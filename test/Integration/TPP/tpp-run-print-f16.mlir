// RUN: tpp-run %s -e entry -entry-point-result=void -print | FileCheck %s

func.func @entry(%arg0: memref<8x8xf16>) -> memref<8x8xf16> {
  %cst = arith.constant 9.0 : f16
  linalg.fill ins(%cst : f16) outs(%arg0 : memref<8x8xf16>)
  return %arg0 : memref<8x8xf16>
}

// CHECK-COUNT-8: ( 9, 9, 9, 9, 9, 9, 9, 9 )
