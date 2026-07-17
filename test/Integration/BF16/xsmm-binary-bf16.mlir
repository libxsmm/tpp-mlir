// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

memref.global "private" constant @__constant_bias : memref<3x3xbf16> = dense<1.0> {alignment = 64 : i64}

func.func @entry(%arg0: memref<3x3xbf16>, %arg1: memref<3x3xbf16>) -> memref<3x3xbf16> {
  %0 = memref.get_global @__constant_bias : memref<3x3xbf16>
  %1 = xsmm.binary.dispatch add [3, 3, 3, 3, 3] flags = (none) data_type = bf16
  xsmm.binary add(data_type = bf16, %1, %arg0, %0, %arg1) : (i64, memref<3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>) -> ()

  return %arg1 : memref<3x3xbf16>
}

// CHECK: ( 2, 2, 2 )
// CHECK: ( 2, 2, 2 )
// CHECK: ( 2, 2, 2 )
