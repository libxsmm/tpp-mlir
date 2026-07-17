// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

memref.global "private" constant @__constant_bias : memref<3x6x2xbf16> = dense<3.0> {alignment = 64 : i64}

func.func @entry(%arg0: memref<6x6xbf16>, %arg1: memref<6x6xbf16>) -> memref<6x6xbf16> {
  %0 = memref.get_global @__constant_bias : memref<3x6x2xbf16>
  // ldb = (6 * 2) / 2 = 6
  %1 = xsmm.gemm.dispatch [6, 6, 6, 6, 6, 6] flags = (vnni_b) data_type = bf16
  xsmm.gemm(data_type = bf16, %1, %arg0, %0, %arg1)
    : (i64, memref<6x6xbf16>, memref<3x6x2xbf16>, memref<6x6xbf16>) -> ()

  return %arg1: memref<6x6xbf16>
}

// CHECK-COUNT-6: ( 19, 19, 19, 19, 19, 19 )
