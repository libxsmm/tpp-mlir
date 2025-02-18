// RUN: tpp-opt %s -propagate-pack-and-unpack -simplify-pack -canonicalize -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_with_relu(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_1 = linalg.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_1 : tensor<4x8x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<4x8x32x32xf32>
  %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%unpack : tensor<128x256xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maximumf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<128x256xf32>
  return %4 : tensor<128x256xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @matmul_with_relu(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>)
// CHECK-SAME:  outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[VAL1:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]]],
// CHECK-SMAE:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  outs(%[[VAL]] : tensor<4x8x32x32xf32>)
// CHECK: %[[OUT:.+]] = linalg.unpack %[[VAL1]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
// CHECK: return %[[OUT]] : tensor<128x256xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_with_add(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_1 = linalg.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_1 : tensor<4x8x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<4x8x32x32xf32>
  %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%unpack : tensor<128x256xf32>) outs(%arg2 : tensor<128x256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
  } -> tensor<128x256xf32>
  return %4 : tensor<128x256xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @matmul_with_add(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>)
// CHECK-SAME:  outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[BUFF2_2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2_2:.+]] = linalg.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2_2]] : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[VAL]] : tensor<4x8x32x32xf32>)
// CHECK-SAME:  outs(%[[PACK2_2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[OUT:.+]] = linalg.unpack %[[VAL1]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
// CHECK: return %[[OUT]] : tensor<128x256xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv_with_relu(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = linalg.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = linalg.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
    %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%unpack : tensor<1x56x56x64xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maximumf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @conv_with_relu(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[VAL1:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  outs(%[[VAL]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv_with_add(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = linalg.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = linalg.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%unpack : tensor<1x56x56x64xf32>) outs(%arg2 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %in, %out : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @conv_with_add(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>)
// CHECK-SAME:  outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF2_2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2_2:.+]] = linalg.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2_2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[VAL]] : tensor<1x2x56x56x32xf32>)
// CHECK-SAME:  outs(%[[PACK2_2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d3)>

func.func @conv_with_add_bcast(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>, %arg3: tensor<64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = linalg.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = linalg.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%unpack, %arg3 : tensor<1x56x56x64xf32>, tensor<64xf32>) outs(%arg2 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.addf %in, %in_2 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
// CHECK-LABEL: func.func @conv_with_add_bcast(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9]*]]: tensor<64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF3:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK3:.+]] = linalg.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF3]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[EXPAND:.+]] = tensor.expand_shape %[[ARG3]] {{\[}}[0, 1]] output_shape [2, 32] : tensor<64xf32> into tensor<2x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[VAL]], %[[EXPAND]] : tensor<1x2x56x56x32xf32>, tensor<2x32xf32>)
// CHECK-SAME:  outs(%[[PACK3]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>

func.func @conv_with_add_bcast2(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>, %arg3: tensor<56x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = linalg.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = linalg.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%unpack, %arg3 : tensor<1x56x56x64xf32>, tensor<56x64xf32>) outs(%arg2 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.addf %in, %in_2 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3, d4)>
// CHECK-LABEL: func.func @conv_with_add_bcast2(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9]*]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG3:[a-zA-Z0-9]*]]: tensor<56x64xf32>) -> tensor<1x56x56x64xf32>
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>)
// CHECK-SAME:  outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[BUFF4:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK4:.+]] = linalg.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF4]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF3:.+]] = tensor.empty() : tensor<2x56x32xf32>
// CHECK: %[[PACK3:.+]] = linalg.pack %[[ARG3]] outer_dims_perm = [1, 0] inner_dims_pos = [1] inner_tiles = [32] into %[[BUFF3]] : tensor<56x64xf32> -> tensor<2x56x32xf32>
// CHECK: %[[VAL1:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[VAL]], %[[PACK3]] : tensor<1x2x56x56x32xf32>, tensor<2x56x32xf32>)
// CHECK-SAME:  outs(%[[PACK4]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[VAL1]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[ARG2]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv_with_pad(%arg0: tensor<1x56x56x64xf32>, %arg1: tensor<1x1x64x64xf32>, %arg2: tensor<1x56x56x64xf32>) -> tensor<1x58x58x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %1 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %1 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = linalg.pack %arg2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = linalg.unpack %3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %arg2 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%unpack : tensor<1x56x56x64xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maximumf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  %padded = tensor.pad %4 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst : f32
  } : tensor<1x56x56x64xf32> to tensor<1x58x58x64xf32>
  return %padded : tensor<1x58x58x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @conv_with_pad(
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x1x64x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x56x56x64xf32>) -> tensor<1x58x58x64xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF0]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %[[BUFF1]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[BUFF2]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]
// CHECK-SAME:  ins(%[[PACK0]], %[[PACK1]] : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>)
// CHECK-SAME:  outs(%[[PACK2]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[VAL1:.+]] = linalg.generic {indexing_maps = [#[[MAP3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%[[VAL]] : tensor<1x2x56x56x32xf32>)
// CHECK: %[[PADDED:.+]] = tensor.pad %[[VAL1]] low[0, 0, 1, 1, 0] high[0, 0, 1, 1, 0]
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<1x58x58x64xf32>
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[PADDED]] outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %[[OUT]] : tensor<1x2x58x58x32xf32> -> tensor<1x58x58x64xf32>
// CHECK: return %[[UNPACK]] : tensor<1x58x58x64xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

func.func @fill(%arg0: f32, %arg1: tensor<1x56x56x64xf32>, %arg2: tensor<1x1x64x64xf32>) -> tensor<1x56x56x64xf32> {
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%arg0 : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack = linalg.pack %arg1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %2 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %3 = tensor.empty() : tensor<2x2x1x1x32x32xf32>
  %pack_0 = linalg.pack %arg2 outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %3 : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
  %4 = tensor.empty() : tensor<1x2x56x56x32xf32>
  %pack_1 = linalg.pack %1 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %4 : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.mulf %in, %in_2 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
  } -> tensor<1x2x56x56x32xf32>
  %unpack = linalg.unpack %5 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %0 : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>
  return %unpack : tensor<1x56x56x64xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func.func @fill
// CHECK-SAME: %[[ARG0:.+]]: f32,
// CHECK-SAME: %[[ARG1:.+]]: tensor<1x56x56x64xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<1x1x64x64xf32>
// CHECK: %[[RES:.+]] = tensor.empty() : tensor<1x56x56x64xf32>
// CHECK: %[[EMPTY_ARG1:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACK_ARG1:.+]] = linalg.pack %[[ARG1]]
// CHECK-SAME: outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME: into %[[EMPTY_ARG1]] : tensor<1x56x56x64xf32> -> tensor<1x2x56x56x32xf32>
// CHECK: %[[EMPTY_ARG2:.+]] = tensor.empty() : tensor<2x2x1x1x32x32xf32>
// CHECK: %[[PACK_ARG2:.+]] = linalg.pack %[[ARG2]]
// CHECK-SAME: outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32]
// CHECK-SAME: into %[[EMPTY_ARG2]] : tensor<1x1x64x64xf32> -> tensor<2x2x1x1x32x32xf32>
// CHECK: %[[EMPTY_FILL:.+]] = tensor.empty() : tensor<1x2x56x56x32xf32>
// CHECK: %[[PACKED_FILL:.+]] = linalg.fill ins(%[[ARG0]] : f32)
// CHECK-SAME: outs(%[[EMPTY_FILL]] : tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32>
// CHECK: %[[GEN:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: ins(%[[PACK_ARG1]], %[[PACK_ARG2]]
// CHECK-SAME: outs(%[[PACKED_FILL]]
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[GEN]]
// CHECK-SAME: outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32]
// CHECK-SAME: into %[[RES]] : tensor<1x2x56x56x32xf32> -> tensor<1x56x56x64xf32>

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @matmul_with_relu_and_bias(%arg0: tensor<256x512xf32>, %arg1: tensor<512x1024xf32>, %arg2: tensor<256x1024xf32>, %arg3: tensor<1024xf32>) -> tensor<256x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3 : tensor<1024xf32>) outs(%arg2 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<256x1024xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<256x512xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<32x16x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<512x1024xf32> -> tensor<32x16x32x32xf32>
  %3 = tensor.empty() : tensor<8x32x32x32xf32>
  %pack_1 = linalg.pack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %3 : tensor<256x1024xf32> -> tensor<8x32x32x32xf32>
  %4 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<8x16x32x32xf32>, tensor<32x16x32x32xf32>) outs(%pack_1 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.mulf %in, %in_2 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
  } -> tensor<8x32x32x32xf32>
  %unpack = linalg.unpack %4 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<8x32x32x32xf32> -> tensor<256x1024xf32>
  %5 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%unpack : tensor<256x1024xf32>) {
    ^bb0(%out: f32):
      %6 = arith.maximumf %out, %cst : f32
      linalg.yield %6 : f32
  } -> tensor<256x1024xf32>
  return %5 : tensor<256x1024xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @matmul_with_relu_and_bias
// CHECK-SAME:  %[[ARG0:.+]]: tensor<256x512xf32>, %[[ARG1:.+]]: tensor<512x1024xf32>,
// CHECK-SAME:  %[[ARG2]]: tensor<256x1024xf32>, %[[ARG3:.+]]: tensor<1024xf32>) -> tensor<256x1024xf32> {
// CHECK: %[[BCAST:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[ARG3]]
// CHECK-SAME:  outs(%[[ARG2]]
// CHECK: %[[PACK:.+]] = linalg.pack %[[ARG0:.+]] inner_dims_pos = [0, 1]
// CHECK-SAME:  inner_tiles = [32, 32] into %{{.+}} : tensor<256x512xf32> -> tensor<8x16x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1]
// CHECK-SAME:  inner_tiles = [32, 32] into %{{.+}} : tensor<512x1024xf32> -> tensor<32x16x32x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[BCAST]] inner_dims_pos = [0, 1]
// CHECK-SAME:  inner_tiles = [32, 32] into %{{.+}} : tensor<256x1024xf32> -> tensor<8x32x32x32xf32>
// CHECK: %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[PACK]], %[[PACK1]]
// CHECK-SAME:  outs(%[[PACK2]]
// CHECK: %[[RELU:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP5]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:  outs(%[[MATMUL]]
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[RELU]] inner_dims_pos = [0, 1]
// CHECK-SAME:  inner_tiles = [32, 32] into %[[ARG2]] : tensor<8x32x32x32xf32> -> tensor<256x1024xf32>


// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func.func @simple_4_layers_mlp_destination_passing
// 3 packs for the first layer, 2 packs for all the others.
// CHECK-COUNT-9: linalg.pack
func.func @simple_4_layers_mlp_destination_passing(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x1024xf32>, %arg4: tensor<1024xf32>, %arg5: tensor<1024x2048xf32>, %arg6: tensor<2048xf32>, %arg7: tensor<2048x1024xf32>, %arg8: tensor<1024xf32>, %arg9: tensor<128x1024xf32>, %arg10: tensor<128x2048xf32>, %arg11: tensor<128x1024xf32>, %arg12: tensor<128x512xf32>) -> tensor<128x1024xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%arg12 : tensor<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<128x512xf32>
  %1 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %2 = tensor.empty() : tensor<16x8x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<256x512xf32> -> tensor<16x8x32x32xf32>
  %3 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack_1 = linalg.pack %0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %3 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %4 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x8x32x32xf32>, tensor<16x8x32x32xf32>) outs(%pack_1 : tensor<4x16x32x32xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %24 = arith.mulf %in, %in_14 : f32
      %25 = arith.addf %out, %24 : f32
      linalg.yield %25 : f32
  } -> tensor<4x16x32x32xf32>
  %unpack = linalg.unpack %4 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<4x16x32x32xf32> -> tensor<128x512xf32>
  %5 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%unpack : tensor<128x512xf32>) {
    ^bb0(%out: f32):
      %24 = arith.maximumf %out, %cst : f32
      linalg.yield %24 : f32
  } -> tensor<128x512xf32>
  %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg4 : tensor<1024xf32>) outs(%arg11 : tensor<128x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<128x1024xf32>
  %7 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack_2 = linalg.pack %5 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %7 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %8 = tensor.empty() : tensor<32x16x32x32xf32>
  %pack_3 = linalg.pack %arg3 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %8 : tensor<512x1024xf32> -> tensor<32x16x32x32xf32>
  %9 = tensor.empty() : tensor<4x32x32x32xf32>
  %pack_4 = linalg.pack %6 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %9 : tensor<128x1024xf32> -> tensor<4x32x32x32xf32>
  %10 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_2, %pack_3 : tensor<4x16x32x32xf32>, tensor<32x16x32x32xf32>) outs(%pack_4 : tensor<4x32x32x32xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %24 = arith.mulf %in, %in_14 : f32
      %25 = arith.addf %out, %24 : f32
      linalg.yield %25 : f32
  } -> tensor<4x32x32x32xf32>
  %unpack_5 = linalg.unpack %10 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %6 : tensor<4x32x32x32xf32> -> tensor<128x1024xf32>
  %11 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%unpack_5 : tensor<128x1024xf32>) {
    ^bb0(%out: f32):
      %24 = arith.maximumf %out, %cst : f32
      linalg.yield %24 : f32
  } -> tensor<128x1024xf32>
  %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6 : tensor<2048xf32>) outs(%arg10 : tensor<128x2048xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<128x2048xf32>
  %13 = tensor.empty() : tensor<4x32x32x32xf32>
  %pack_6 = linalg.pack %11 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %13 : tensor<128x1024xf32> -> tensor<4x32x32x32xf32>
  %14 = tensor.empty() : tensor<64x32x32x32xf32>
  %pack_7 = linalg.pack %arg5 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %14 : tensor<1024x2048xf32> -> tensor<64x32x32x32xf32>
  %15 = tensor.empty() : tensor<4x64x32x32xf32>
  %pack_8 = linalg.pack %12 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %15 : tensor<128x2048xf32> -> tensor<4x64x32x32xf32>
  %16 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_6, %pack_7 : tensor<4x32x32x32xf32>, tensor<64x32x32x32xf32>) outs(%pack_8 : tensor<4x64x32x32xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %24 = arith.mulf %in, %in_14 : f32
      %25 = arith.addf %out, %24 : f32
      linalg.yield %25 : f32
  } -> tensor<4x64x32x32xf32>
  %unpack_9 = linalg.unpack %16 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %12 : tensor<4x64x32x32xf32> -> tensor<128x2048xf32>
  %17 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%unpack_9 : tensor<128x2048xf32>) {
    ^bb0(%out: f32):
      %24 = arith.maximumf %out, %cst : f32
      linalg.yield %24 : f32
  } -> tensor<128x2048xf32>
  %18 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg8 : tensor<1024xf32>) outs(%arg9 : tensor<128x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<128x1024xf32>
  %19 = tensor.empty() : tensor<4x64x32x32xf32>
  %pack_10 = linalg.pack %17 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %19 : tensor<128x2048xf32> -> tensor<4x64x32x32xf32>
  %20 = tensor.empty() : tensor<32x64x32x32xf32>
  %pack_11 = linalg.pack %arg7 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %20 : tensor<2048x1024xf32> -> tensor<32x64x32x32xf32>
  %21 = tensor.empty() : tensor<4x32x32x32xf32>
  %pack_12 = linalg.pack %18 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %21 : tensor<128x1024xf32> -> tensor<4x32x32x32xf32>
  %22 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_10, %pack_11 : tensor<4x64x32x32xf32>, tensor<32x64x32x32xf32>) outs(%pack_12 : tensor<4x32x32x32xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %24 = arith.mulf %in, %in_14 : f32
      %25 = arith.addf %out, %24 : f32
      linalg.yield %25 : f32
  } -> tensor<4x32x32x32xf32>
  %unpack_13 = linalg.unpack %22 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %18 : tensor<4x32x32x32xf32> -> tensor<128x1024xf32>
  %23 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%unpack_13 : tensor<128x1024xf32>) {
    ^bb0(%out: f32):
      %24 = arith.maximumf %out, %cst : f32
      linalg.yield %24 : f32
  } -> tensor<128x1024xf32>
  // CHECK: %[[UNPACK:.+]] = linalg.unpack %{{.+}} inner_dims_pos = [0, 1]
  // CHECK-SAME:  inner_tiles = [32, 32] into %{{.+}} : tensor<4x32x32x32xf32> -> tensor<128x1024xf32>
  // CHECK-NEXT: return %[[UNPACK]] : tensor<128x1024xf32>
  return %23 : tensor<128x1024xf32>
}
