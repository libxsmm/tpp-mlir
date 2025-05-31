// RUN: tpp-opt %s -default-pipeline | FileCheck %s
// RUN: tpp-sched --bundle=default-pipeline %s | FileCheck %s

module {
  ml_program.global private mutable @unused_global(dense<0> : tensor<i64>) : tensor<i64>
  func.func private @unused_func(!llvm.ptr, i64)

  func.func @entry(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>)
                      outs(%arg2 : tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// CHECK: module
// CHECK-NOT: @unused_func
// CHECK: @entry
// CHECK:   llvm.return
