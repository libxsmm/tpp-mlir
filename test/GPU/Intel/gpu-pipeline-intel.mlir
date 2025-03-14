// RUN: tpp-opt %s -gpu-pipeline=gpu=intel \
// RUN:  -gpu-block-tile=128,128 -gpu-thread-tile=32,32 -k-tile=32 -stages=1 \
// RUN:  -split-input-file | \
// RUN: FileCheck %s

func.func @linalg_matmul(%arg0: tensor<128x1024xf16>,
                 %arg1: tensor<1024x1024xf16>,
                 %arg2: tensor<128x1024xf16>) -> tensor<128x1024xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x1024xf16>, tensor<1024x1024xf16>)
                     outs(%arg2 : tensor<128x1024xf16>) -> tensor<128x1024xf16>
  return %0 : tensor<128x1024xf16>
}

// CHECK: module attributes {gpu.container_module}
// CHECK-LABEL: func.func @linalg_matmul(
// CHECK-SAME:  %[[arg0:.+]]: memref<128x1024xf16>, %[[arg1:.+]]: memref<1024x1024xf16>, %[[arg2:.+]]: memref<128x1024xf16>
// CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[c4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[c8:.+]] = arith.constant 8 : index
// CHECK:       gpu.launch_func  @linalg_matmul_kernel::@linalg_matmul_kernel blocks in (%[[c8]], %[[c1]], %[[c1]]) threads in (%[[c4]], %[[c4]], %[[c1]])  args(%[[arg2]] : memref<128x1024xf16>, %[[arg0]] : memref<128x1024xf16>, %[[arg1]] : memref<1024x1024xf16>
//
// CHECK-LABEL: gpu.func @linalg_matmul_kernel(
// CHECK-SAME:  %[[C:.+]]: memref<128x1024xf16>, %[[A:.+]]: memref<128x1024xf16>, %[[B:.+]]: memref<1024x1024xf16>
// CHECK-DAG:   %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[c32:.+]] = arith.constant 32 : index
// CHECK-DAG:   %[[c1024:.+]] = arith.constant 1024 : index
// CHECK-COUNT-8: xegpu.load_nd
// CHECK-COUNT-8: arith.extf
// CHECK-COUNT-2: xegpu.prefetch_nd
// CHECK:         %[[out:.+]]:14 = scf.for %[[iv:.+]] = %[[c0]] to %[[c1024]] step %[[c32]]
// CHECK-SAME:    {
// CHECK-COUNT-4:   xegpu.load_nd
// CHECK-COUNT-4:   xegpu.update_nd_offset
// CHECK-COUNT-2:   xegpu.prefetch_nd
// CHECK-COUNT-2:   xegpu.update_nd_offset
// CHECK-COUNT-12:  vector.extract_strided_slice
// CHECK-COUNT-16:  xegpu.dpas
// CHECK:           scf.yield
// CHECK:         }
// CHECK-COUNT-8: arith.truncf
// CHECK-COUNT-8: xegpu.store_nd
// CHECK:         gpu.return
