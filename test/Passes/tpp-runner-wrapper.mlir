// RUN: tpp-opt %s -tpp-runner-wrapper -split-input-file | FileCheck %s

func.func @entry(%arg0: tensor<8x8xf16>,
                 %arg1: tensor<8x8xf16>,
                 %arg2: tensor<8x8xf16>) -> tensor<8x8xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xf16>, tensor<8x8xf16>)
                     outs(%arg2 : tensor<8x8xf16>) -> tensor<8x8xf16>
  return %0 : tensor<8x8xf16>
}

// CHECK-LABEL: func.func @_entry
// CHECK: linalg.matmul
// CHECK-LABEL: func.func @entry
// CHECK: memref.get_global @__wrapper_0
// CHECK: bufferization.to_tensor
// CHECK: memref.get_global @__wrapper_1
// CHECK: bufferization.to_tensor
// CHECK: memref.get_global @__wrapper_2
// CHECK: bufferization.to_tensor
// CHECK: call @_entry
