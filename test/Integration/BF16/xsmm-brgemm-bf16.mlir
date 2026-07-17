// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

memref.global "private" constant @__constant_bias : memref<2x3x6x2xbf16> = dense<3.0> {alignment = 64 : i64}

func.func @entry(%arg0: memref<2x6x6xbf16>, %arg1: memref<6x6xbf16>) -> memref<6x6xbf16> {
  %0 = memref.get_global @__constant_bias : memref<2x3x6x2xbf16>
  // [m, n, k, lda, ldb, ldc, stride_a, stride_b]
  // m = 6 n = 6 k = 6
  // lda = 6, ldb = 6, ldc = 6
  %1 = xsmm.brgemm.dispatch [6, 6, 6, 6, 6, 6, 36, 36] flags = (vnni_b) data_type = bf16
  %c2 = arith.constant 2 : i64
  xsmm.brgemm(data_type = bf16, %1, %arg0, %0, %arg1, %c2)
    : (i64, memref<2x6x6xbf16>, memref<2x3x6x2xbf16>, memref<6x6xbf16>, i64) -> ()

  return %arg1: memref<6x6xbf16>
}

// CHECK-COUNT-6: ( 37, 37, 37, 37, 37, 37 )
