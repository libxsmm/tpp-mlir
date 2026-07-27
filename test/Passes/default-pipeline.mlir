// RUN: tpp-opt %s -default-pipeline | FileCheck %s --check-prefix=XSMM
// RUN: tpp-opt %s -default-pipeline -linalg-to-loops | FileCheck %s --check-prefix=LOOPS
// RUN: tpp-opt %s -default-pipeline -nano-kernels | FileCheck %s --check-prefix=NANO

func.func @matmul(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D : tensor<4x4xf32>
}

// XSMM: llvm.func @xsmm_gemm_invoke
// XSMM: llvm.func @xsmm_gemm_dispatch
// XSMM: llvm.func @matmul(%[[ARG0:.+]]: !llvm.ptr,
// XSMM:   llvm.call @xsmm_gemm_dispatch
// XSMM:   llvm.call @xsmm_gemm_invoke
// XSMM:   llvm.return

// LOOPS: %[[A:[0-9]+]] = llvm.load {{.*}} : !llvm.ptr -> f32
// LOOPS: %[[B:[0-9]+]] = llvm.load {{.*}} : !llvm.ptr -> f32
// LOOPS: %[[C:[0-9]+]] = llvm.load {{.*}} : !llvm.ptr -> f32
// LOOPS: %[[acc:[0-9]+]] = llvm.fmul %[[A]], %[[B]] : f32
// LOOPS: llvm.fadd %[[C]], %[[acc]] : f32

// NANO: llvm.load {{.*}} : !llvm.ptr -> vector<32xf32>
// NANO: llvm.load {{.*}} : !llvm.ptr -> vector<32xf32>
// NANO: llvm.load {{.*}} : !llvm.ptr -> vector<16xf32>
// NANO-COUNT-16: llvm.fmul
// NANO-COUNT-4: llvm.fadd
// NANO: llvm.store {{.*}} : vector<16xf32>, !llvm.ptr