// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @entry() -> memref<1x1xf32> {
  %c1 = arith.constant 1.0 : f32
  %c0 = arith.constant 0 : index

  %0 = arith.addf %c1, %c1 : f32
  %out = memref.alloc() : memref<1x1xf32>
  memref.store %0, %out[%c0, %c0] : memref<1x1xf32>

  return %out : memref<1x1xf32>
}

// CHECK: ( 2 )
