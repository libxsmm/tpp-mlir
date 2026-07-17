// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 -splat-to-random -init-type=identity 2>&1 | \
// RUN: FileCheck %s

func.func @entry(%arg0: tensor<2x2xi32>) {
  %0 = arith.constant dense<2.0> : tensor<1x1xbf16>
  %1 = arith.constant dense<0.0> : tensor<4x4xf32>
  %2 = arith.constant dense<1.0> : tensor<4x4xf32>
  return
}

// Integer identity of the argument goes into a global variable.
// CHECK: memref.global "private" @__wrapper_0 : memref<2x2xi32> = dense<{{.}}[1, 0], [0, 1]{{.}}> {alignment = 128 : i64}
// CHECK-LABEL: @_entry
// Dense non-zero identity (changes 2.0 to 1.0)
// CHECK: arith.constant dense<1.000000e+00> : tensor<1x1xbf16>
// Dense zero identity does not change (remains 0.0)
// CHECK: arith.constant dense<0.000000e+00> : tensor<4x4xf32>
// Dense non-zero identity (4x4 identity matrix)
// CHECK: arith.constant dense<{{.}}[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
// CHECK-SAME:                      [0.000000e+00, 1.000000e+00, 0.000000e+00, 0.000000e+00],
// CHECK-SAME:                      [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00],
// CHECK-SAME:                      [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]{{.}}> : tensor<4x4xf32>
