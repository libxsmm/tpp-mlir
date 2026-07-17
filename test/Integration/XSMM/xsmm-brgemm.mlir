// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

memref.global "private" constant @__constant_bias : memref<2x16x64xf32> = dense<1.0> {alignment = 64 : i64}

func.func @entry(%arg0: memref<2x32x16xf32>, %arg1: memref<64x32xf32>) -> memref<64x32xf32> {
  %0 = memref.get_global @__constant_bias : memref<2x16x64xf32>
  // [32x16] * [16x64] -> [32x64]
  // m = 32, n = 64, k = 16
  // lda = 16, ldb = 64, ldc = 64
  // stride_a = 512, stride_b = 1024
  %1 = xsmm.brgemm.dispatch [32, 64, 16, 16, 64, 64, 512, 1024] flags = (none) data_type = f32
  %c2 = arith.constant 2 : i64
  xsmm.brgemm(data_type = f32, %1, %arg0, %0, %arg1, %c2)
    : (i64, memref<2x32x16xf32>, memref<2x16x64xf32>, memref<64x32xf32>, i64) -> ()
  return %arg1 : memref<64x32xf32>
}

// CHECK-COUNT-64: ( 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33 )
