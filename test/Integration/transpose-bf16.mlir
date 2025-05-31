// RUN: tpp-run --vector-to-XSMM %s -e entry -entry-point-result=void -print --seed 123 2>&1 | FileCheck %s
// RUN: tpp-run --linalg-to-loops %s -e entry -entry-point-result=void -print --seed 123 2>&1 | FileCheck %s
// RUN: tpp-opt --default-tpp-passes="vector-to-xsmm" %s  -mlir-print-ir-after=vectorization-pass  2>&1  | FileCheck %s --check-prefix=VECTOR
// TODO(ROLF): -mlir-print-ir-after=vectorization-pass is an antipattern in tests, IMHO.
// RUN: tpp-run --vector-to-XSMM %s -e entry -entry-point-result=void -print-mlir=mid  2>&1 | FileCheck %s --check-prefix=XSMM

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @entry(%arg0 : tensor<4x4xbf16>, %arg1 : tensor<2x4x2xbf16>)-> tensor<2x4x2xbf16> {
    %expand_shape = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape[2, 2, 4]
      : tensor<4x4xbf16>
      into tensor<2x2x4xbf16>
    %retval = linalg.transpose ins(%expand_shape : tensor<2x2x4xbf16>)
      outs(%arg1 : tensor<2x4x2xbf16>) permutation = [0, 2, 1]
    return %retval: tensor<2x4x2xbf16>
  }
}

// VECTOR: vector.transfer_read
// VECTOR: vector.transpose
// VECTOR: vector.transfer_write

// XSMM: call @xsmm_unary_dispatch
// XSMM: call @xsmm_unary_invoke

// CHECK:	( 0, 0.000375748 )
// CHECK:	( 0.129883, 0.298828 )
// CHECK:	( 0.151367, 0.0981445 )
// CHECK:	( 0.0106201, 0.0112305 )
// CHECK:	( 0, 0.128906 )
// CHECK:	( 0, 0.0148315 )
// CHECK:	( 0, 0 )
// CHECK:	( 0.0500488, 0 )
