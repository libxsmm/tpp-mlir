// RUN: tpp-opt %s --gemm-k-cache-blocking="k-cache-blocking=32" --split-input-file | FileCheck %s
// RUN: tpp-opt %s --gemm-k-cache-blocking="k-cache-blocking=0" --split-input-file | FileCheck -check-prefix=DISABLED %s

// Cache-block the batch-reduce (K) dimension of a tensor brgemm by 32.
module {
  func.func @brgemm_k_cache_blocking(%arg0: tensor<128x256x512xf32>, %arg1: tensor<128x512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<128x256x512xf32>, tensor<128x512x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}

// CHECK-LABEL: func.func @brgemm_k_cache_blocking
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// The batch-reduce (K) dimension is tiled by 32, threading the accumulator.
// CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C128]] step %[[C32]] iter_args(%[[ACC:.+]] = %arg2) -> (tensor<256x256xf32>)
// CHECK:   tensor.extract_slice %arg0[%[[I]], 0, 0] [32, 256, 512]
// CHECK:   tensor.extract_slice %arg1[%[[I]], 0, 0] [32, 512, 256]
// CHECK:   %[[MM:.+]] = linalg.batch_reduce_matmul ins({{.*}} : tensor<32x256x512xf32>, tensor<32x512x256xf32>) outs({{.*}} : tensor<256x256xf32>)
// CHECK:   scf.yield

// A tile size of 0 disables the pass; the op is left unchanged.
// DISABLED-LABEL: func.func @brgemm_k_cache_blocking
// DISABLED-NOT: scf.for
// DISABLED: linalg.batch_reduce_matmul ins(%arg0, %arg1

// -----

// The tile must strictly divide the batch-reduce extent; otherwise no tiling.
module {
  func.func @brgemm_no_divide(%arg0: tensor<48x256x512xf32>, %arg1: tensor<48x512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<48x256x512xf32>, tensor<48x512x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}

// 32 does not divide the batch-reduce extent 48, so the op is unchanged.
// CHECK-LABEL: func.func @brgemm_no_divide
// CHECK-NOT: scf.for
