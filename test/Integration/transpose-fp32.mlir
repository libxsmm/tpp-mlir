// RUN: tpp-run  --vector-to-XSMM %s -e entry -entry-point-result=void -print --seed 123 2>&1 | FileCheck %s
// RUN: tpp-run --linalg-to-loops %s -e entry -entry-point-result=void -print --seed 123 2>&1 | FileCheck %s
// RUN: tpp-opt --default-tpp-passes="vector-to-xsmm" %s  -mlir-print-ir-after=vectorization-pass  2>&1  | FileCheck %s --check-prefix=VECTOR
// TODO(ROLF): -mlir-print-ir-after=vectorization-pass is an antipattern in tests, IMHO.
// RUN: tpp-run  --vector-to-XSMM %s -e entry -entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s --check-prefix=XSMM

module {
  func.func @entry(%arg0: tensor<3x5xf32>, %arg1: tensor<5x3xf32>)->tensor<5x3xf32> {
    %out = linalg.transpose ins(%arg0 : tensor<3x5xf32>) outs(%arg1 : tensor<5x3xf32>) permutation = [1, 0]
    return %out: tensor<5x3xf32>
  }
}


// VECTOR: vector.transfer_read
// VECTOR: vector.transpose
// VECTOR: vector.transfer_write

// XSMM: call @xsmm_unary_dispatch
// XSMM: call @xsmm_unary_invoke

// CHECK: ( 0, 0.298506, 0 )
// CHECK: ( 0.130352, 0.0983867, 0.0499509 )
// CHECK: ( 0.151291, 0.011257, 0.129232 )
// CHECK: ( 0.0106365, 0, 0.0148569 )
// CHECK: ( 0.000375301, 0, 0 )

