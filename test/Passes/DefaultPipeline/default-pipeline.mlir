// RUN: tpp-opt %s -default-pipeline | FileCheck %s

func.func @matmul(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D : tensor<4x4xf32>
}

// CHECK: llvm.func @xsmm_gemm_invoke
// CHECK: llvm.func @xsmm_gemm_dispatch
// CHECK: llvm.func @matmul(%[[ARG0:.+]]: !llvm.ptr,
// CHECK:   llvm.call @xsmm_gemm_dispatch
// CHECK:   llvm.call @xsmm_gemm_invoke
// CHECK:   llvm.return
