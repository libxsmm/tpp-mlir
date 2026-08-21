// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// FP8 GEMM with a VNNI blocking factor of 4. libxsmm names E5M2 as BF8.
// Weights are in VNNI4 layout: [K/4][N][4]. K = 4, N = 6.
// Inputs and the accumulator are auto-initialized to 1.0, so each output
// element is: sum_{k=0..3}(1 * 1) + 1 = 5.

memref.global "private" constant @wt : memref<1x6x4xf8E5M2> = dense<1.0> alignment = 64 

func.func @entry(%arg0: memref<6x4xf8E5M2>, %arg1: memref<6x6xf8E5M2>) -> memref<6x6xf8E5M2> {
  %0 = memref.get_global @wt : memref<1x6x4xf8E5M2>
  %1 = xsmm.gemm.dispatch [6, 6, 4, 4, 6, 6] flags = (vnni_b) data_type = bf8
  xsmm.gemm(data_type = bf8, %1, %arg0, %0, %arg1)
    : (i64, memref<6x4xf8E5M2>, memref<1x6x4xf8E5M2>, memref<6x6xf8E5M2>) -> ()

  return %arg1: memref<6x6xf8E5M2>
}

// CHECK-COUNT-6: ( 5, 5, 5, 5, 5, 5 )
