// RUN: tpp-run %s -e entry -entry-point-result=void -print | FileCheck %s

func.func @entry(%arg0: memref<8x8xi32>) -> memref<8x8xi32> {
  %cst = arith.constant 9 : i32
  linalg.fill ins(%cst : i32) outs(%arg0 : memref<8x8xi32>)
  return %arg0 : memref<8x8xi32>
}

// CHECK-COUNT-8: ( 9, 9, 9, 9, 9, 9, 9, 9 )
