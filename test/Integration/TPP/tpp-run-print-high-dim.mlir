// RUN: tpp-run %s -e entry -entry-point-result=void -print | FileCheck %s

func.func @entry(%arg0: memref<2x4x8xf32>) -> memref<2x4x8xf32> {
  %cst = arith.constant 9.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<2x4x8xf32>)
  return %arg0 : memref<2x4x8xf32>
}

// CHECK-COUNT-8: ( 9, 9, 9, 9, 9, 9, 9, 9 )
