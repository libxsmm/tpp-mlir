// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

func.func @matmul(%arg0: tensor<2048x2048xbf16>, %arg1: tensor<2048x2048xbf16>, %arg2: tensor<2048x2048xbf16>)
    -> tensor<2048x2048xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<2048x2048xbf16>, tensor<2048x2048xbf16>)
                     outs(%arg2: tensor<2048x2048xbf16>)
    -> tensor<2048x2048xbf16>
  return %0 : tensor<2048x2048xbf16>
}

// CHECK-LABEL: @matmul(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<2048x2048xbf16>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<2048x2048xbf16>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]+]]: memref<2048x2048xbf16>
// CHECK: memref.subview %[[ARG0]]
// CHECK: call @xsmm_unary_invoke
// CHECK: memref.subview %[[ARG1]]
// CHECK: call @xsmm_unary_invoke
// CHECK: memref.subview %[[ARG2]]
// CHECK: call @xsmm_intel_amx_tile_config_invoke
// CHECK: call @xsmm_brgemm_invoke
// CHECK: call @xsmm_intel_amx_tile_config_invoke

// -----

func.func @batch_matmul(%arg0: tensor<8x2048x2048xbf16>, %arg1: tensor<8x2048x2048xbf16>, %arg2: tensor<8x2048x2048xbf16>)
    -> tensor<8x2048x2048xbf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1: tensor<8x2048x2048xbf16>, tensor<8x2048x2048xbf16>)
                           outs(%arg2: tensor<8x2048x2048xbf16>)
    -> tensor<8x2048x2048xbf16>
  return %0 : tensor<8x2048x2048xbf16>
}

// CHECK-LABEL: @batch_matmul(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]+]]: memref<8x2048x2048xbf16>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]+]]: memref<8x2048x2048xbf16>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]+]]: memref<8x2048x2048xbf16>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK: scf.parallel{{.*}}= (%[[C0]]) to (%[[C8]])
// CHECK: %[[BATCH_SUBVIEW:.+]] = memref.subview %[[ARG2]]
// CHECK: memref.subview %[[ARG0]]
// CHECK: call @xsmm_unary_invoke
// CHECK: memref.subview %[[ARG1]]
// CHECK: call @xsmm_unary_invoke
// CHECK: memref.subview %[[BATCH_SUBVIEW]]
// CHECK: call @xsmm_intel_amx_tile_config_invoke
// CHECK: call @xsmm_brgemm_invoke
// CHECK: call @xsmm_intel_amx_tile_config_invoke
