// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// FP8 GEMM with a VNNI blocking factor of 4. libxsmm names E4M3 as HF8.
// Weights are in VNNI4 layout: [K/4][N][4]. K = 4, N = 6.
// Inputs and the accumulator are auto-initialized to 1.0, so each output
// element is: sum_{k=0..3}(1 * 1) + 1 = 5.

memref.global "private" constant @wt : memref<1x6x4xf8E4M3FN> = dense<1.0> alignment = 64 

func.func @entry(%arg0: memref<6x4xf8E4M3FN>, %arg1: memref<6x6xf8E4M3FN>) -> memref<6x6xf8E4M3FN> {
  %0 = memref.get_global @wt : memref<1x6x4xf8E4M3FN>
  %1 = xsmm.gemm.dispatch [6, 6, 4, 4, 6, 6] flags = (vnni_b) data_type = hf8
  xsmm.gemm(data_type = hf8, %1, %arg0, %0, %arg1)
    : (i64, memref<6x4xf8E4M3FN>, memref<1x6x4xf8E4M3FN>, memref<6x6xf8E4M3FN>) -> ()

  return %arg1: memref<6x6xf8E4M3FN>
}

// CHECK-COUNT-6: ( 5, 5, 5, 5, 5, 5 )
