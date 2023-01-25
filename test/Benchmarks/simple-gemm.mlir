// RUN: tpp-opt %s -default-tpp-passes | \
// RUN: tpp-run -n 2000\
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s
//

func.func @entry(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D : tensor<4x4xf32>
}

// Output
// CHECK-COUNT-4: ( 9, 9, 9, 9 )
// Stats
// CHECK: ( {{[0-9]+}}{{.?}}{{[0-9e-]+}}, {{[0-9]+}}{{.?}}{{[0-9e-]+}} )
