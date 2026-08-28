// RUN: tpp-opt %s -pack-matmul="block-factors=32,32,32" -split-input-file | FileCheck %s

func.func @block_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_linalg_matmul(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF0]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF1:.*]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF1]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF2]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%[[PACK2]] : tensor<4x4x32x32xf32>)
// CHECK: %[[OUT:.+]] = linalg.unpack %[[VAL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x4x32x32xf32> -> tensor<128x128xf32>
// CHECK: return %[[OUT]] : tensor<128x128xf32>
// CHECK: }

// -----

func.func @block_dims_equal_to_factors(
  %arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%arg2: tensor<32x32xf32>)
    -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_dims_equal_to_factors(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<32x32xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<32x32xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<1x1x32x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF0]] : tensor<32x32xf32> -> tensor<1x1x32x32xf32>
// CHECK: %[[BUF1:.*]] = tensor.empty() : tensor<1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF1]] : tensor<32x32xf32> -> tensor<1x1x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<1x1x32x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF2]] : tensor<32x32xf32> -> tensor<1x1x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x1x32x32xf32>)
// CHECK: %[[OUT:.+]] = linalg.unpack %[[VAL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<1x1x32x32xf32> -> tensor<32x32xf32>
// CHECK: return %[[OUT]] : tensor<32x32xf32>
// CHECK: }

// -----

// Adapt tile sizes to small dimensions.
// Assume that there is separate cost function that controls
// if packing should take place at all.
func.func @block_small_dims_matmul(
  %arg0: tensor<5x6xf32>, %arg1: tensor<6x5xf32>, %arg2: tensor<5x5xf32>)
    -> tensor<5x5xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<5x6xf32>, tensor<6x5xf32>)
                     outs(%arg2: tensor<5x5xf32>)
    -> tensor<5x5xf32>
  return %0 : tensor<5x5xf32>
}

// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_small_dims_matmul(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<5x6xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<6x5xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<5x5xf32>) -> tensor<5x5xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<1x1x5x6xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [5, 6] into %[[BUF0]] : tensor<5x6xf32> -> tensor<1x1x5x6xf32>
// CHECK: %[[BUF1:.*]] = tensor.empty() : tensor<1x1x6x5xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [6, 5] into %[[BUF1]] : tensor<6x5xf32> -> tensor<1x1x6x5xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<1x1x5x5xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [5, 5] into %[[BUF2]] : tensor<5x5xf32> -> tensor<1x1x5x5xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x1x5x6xf32>, tensor<1x1x6x5xf32>) outs(%[[PACK2]] : tensor<1x1x5x5xf32>)
// CHECK: %[[OUT:.+]] = linalg.unpack %[[VAL]] inner_dims_pos = [0, 1] inner_tiles = [5, 5] into %[[ARG2]] : tensor<1x1x5x5xf32> -> tensor<5x5xf32>
// CHECK: return %[[OUT]] : tensor<5x5xf32>
// CHECK: }

// -----

func.func @block_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg2: tensor<128x128xf32>) -> tensor<128x128xf32>
  %1 = linalg.matmul ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                      outs(%0: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CHECK-LABEL: func @block_linalg_matmul(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32)
// CHECK-SAME:  outs(%[[ARG2]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK: %[[EMPTY_ARG0:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK_ARG0:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:  outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[EMPTY_ARG0]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[EMPTY_ARG1:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK_ARG1:.+]] = linalg.pack %[[ARG1]]
// CHECK-SAME:  outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[EMPTY_ARG1]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[EMPTY_FILL:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK_FILL:.+]] = linalg.pack %[[FILL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[EMPTY_FILL]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK: %{{.+}} = linalg.unpack %[[RES]] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[FILL]] : tensor<4x4x32x32xf32> -> tensor<128x128xf32>

// -----

#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @block_linalg_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%arg2: tensor<128x128xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %1 = arith.mulf %a, %b : f32
      %2 = arith.addf %c, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func @block_linalg_matmul(
// CHECK-SAME:    %[[ARG0:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG1:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[ARG2:[0-9a-z]+]]: tensor<128x128xf32>) -> tensor<128x128xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF0]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF1:.*]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF1]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUF2]] : tensor<128x128xf32> -> tensor<4x4x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP5]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%[[PACK2]] : tensor<4x4x32x32xf32>)
// CHECK: %[[OUT:.+]] = linalg.unpack %[[VAL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x4x32x32xf32> -> tensor<128x128xf32>
// CHECK: return %[[OUT]] : tensor<128x128xf32>

// -----

func.func @batch_matmul_rewrite(%arg0: tensor<512x64x128xf32>, %arg1: tensor<512x128x64xf32>) -> tensor<512x64x64xf32> {
  %0 = tensor.empty() : tensor<512x64x64xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<512x64x128xf32>, tensor<512x128x64xf32>)
                           outs(%0 : tensor<512x64x64xf32>) -> tensor<512x64x64xf32>
  return %1 : tensor<512x64x64xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d6, d5)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>
// CHECK-LABEL: batch_matmul_rewrite
// CHECK-SAME: %[[ARG0:.+]]: tensor<512x64x128xf32>, %[[ARG1:.+]]: tensor<512x128x64xf32>
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<512x64x64xf32>
// CHECK: %[[ARG0_PACK_OUT:.+]] = tensor.empty() : tensor<512x2x4x32x32xf32>
// CHECK: %[[ARG0_PACK:.+]] = linalg.pack %[[ARG0]]
// CHECK-SAME:  outer_dims_perm = [0, 1, 2] inner_dims_pos = [1, 2] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[ARG0_PACK_OUT]] : tensor<512x64x128xf32> -> tensor<512x2x4x32x32xf32>
// CHECK: %[[ARG1_PACK_OUT:.+]] = tensor.empty() : tensor<512x2x4x32x32xf32>
// CHECK: %[[ARG1_PACK:.+]] = linalg.pack %[[ARG1]]
// CHECK-SAME:  outer_dims_perm = [0, 2, 1] inner_dims_pos = [1, 2] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[ARG1_PACK_OUT]] : tensor<512x128x64xf32> -> tensor<512x2x4x32x32xf32>
// CHECK: %[[OUT_PACK_OUT:.+]] = tensor.empty() : tensor<512x2x2x32x32xf32>
// CHECK: %[[OUT_PACK:.+]] = linalg.pack %[[OUT]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[OUT_PACK_OUT]] : tensor<512x64x64xf32> -> tensor<512x2x2x32x32xf32>
// CHECK: %[[GEN:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[GEN]]
// CHECK-SAME:  inner_dims_pos = [1, 2] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[OUT]] : tensor<512x2x2x32x32xf32> -> tensor<512x64x64xf32>
