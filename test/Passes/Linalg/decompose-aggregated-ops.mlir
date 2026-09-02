// RUN: tpp-opt %s -decompose-aggregated-ops | FileCheck %s

// CHECK-LABEL: softmax
func.func @softmax(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // CHECK-NOT: linalg.softmax
  // CHECK-COUNT-4: linalg.generic
  %softmax = linalg.softmax dimension(3)
    ins(%arg0: tensor<2x2x2x2xf32>) outs(%arg1: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %softmax : tensor<2x2x2x2xf32>
}
