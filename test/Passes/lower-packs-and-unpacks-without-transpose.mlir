// RUN: tpp-opt %s -lower-packs-unpacks-without-transpose -canonicalize -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d2, d1 * 32 + d3)>
func.func @single_packed_arg(%arg0: tensor<128x512xf32>, %arg1: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%pack : tensor<4x16x32x32xf32>) outs(%arg1 : tensor<128x512xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<128x512xf32>
  return %1 : tensor<128x512xf32>
}
// CHECK-LABEL: func.func @single_packed_arg(
// CHECK-SAME: %[[ARG0:.+]]: tensor<128x512xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<128x512xf32>)
// CHECK-SAME: -> tensor<128x512xf32>
  // CHECK: %[[EXP0:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [4, 32, 16, 32] : tensor<128x512xf32> into tensor<4x32x16x32xf32>
  // CHECK: %[[RES:.+]] = linalg.generic {{.*}} ins(%[[EXP0]] : tensor<4x32x16x32xf32>) outs(%[[ARG1]] : tensor<128x512xf32>)
  // CHECK-NOT: tensor.collapse_shape
  // CHECK: return %[[RES]]

// -----

// NB: obtained from a M=128, N=256, K=512 linalg.matmul by -pack-matmul
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
func.func @revert_all_packing(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_1 = linalg.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_1 : tensor<4x8x32x32xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %4 = arith.mulf %in, %in_2 : f32
    %5 = arith.addf %out, %4 : f32
    linalg.yield %5 : f32
  } -> tensor<4x8x32x32xf32>
  %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  return %unpack : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @revert_all_packing(
// CHECK-SAME: %[[ARG0:.+]]: tensor<128x512xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<512x256xf32>,
// CHECK-SAME: %[[ARG2:.+]]: tensor<128x256xf32>)
// CHECK-SAME: -> tensor<128x256xf32>
  // CHECK: %[[EXP0:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [4, 32, 16, 32] : tensor<128x512xf32> into tensor<4x32x16x32xf32>
  // CHECK: %[[EXP1:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [16, 32, 8, 32] : tensor<512x256xf32> into tensor<16x32x8x32xf32>
  // CHECK: %[[EXP2:.+]] = tensor.expand_shape %[[ARG2]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [4, 32, 8, 32] : tensor<128x256xf32> into tensor<4x32x8x32xf32>
  // CHECK: %[[RES:.+]] = linalg.generic {{.*}} ins(%[[EXP0]], %[[EXP1]] : tensor<4x32x16x32xf32>, tensor<16x32x8x32xf32>) outs(%[[EXP2]] : tensor<4x32x8x32xf32>)
  // CHECK: %[[COL:.+]] = tensor.collapse_shape %[[RES]] {{\[}}[0, 1], [2, 3]{{\]}} : tensor<4x32x8x32xf32> into tensor<128x256xf32>
  // CHECK: return %[[COL]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
func.func @only_keep_constant_packed_non_prepacked(%arg0: tensor<128x512xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant dense<1.000000e-03> : tensor<512x256xf32>
  %cst_empty = tensor.empty() : tensor<8x16x32x32xf32>
  %cst_packed = linalg.pack %cst outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %cst_empty : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = linalg.pack %arg0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %cst_packed : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_0 : tensor<4x8x32x32xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %3 = arith.mulf %in, %in_1 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<4x8x32x32xf32>
  // NB: unpack's outer_dims_perm should match those of corresponding pack - in case of elision it should match with an identity perm
  %unpack = linalg.unpack %2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  return %unpack : tensor<128x256xf32>
}
// CHECK: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d2, d5)>
// CHECK: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d1, d4)>
// CHECK-LABEL: func.func @only_keep_constant_packed_non_prepacked(
// CHECK-SAME: %[[ARG0:.+]]: tensor<128x512xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<128x256xf32>)
// CHECK-SAME: -> tensor<128x256xf32>
  // NB: even if the following is the case, this does not mean the layout will be preserved in general
  // CHECK: %[[CST:.*]] = arith.constant dense<1.000000e-03> : tensor<8x16x32x32xf32>
  // CHECK: %[[EXP0:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [4, 32, 16, 32] : tensor<128x512xf32> into tensor<4x32x16x32xf32>
  // CHECK: %[[EXP1:.+]] = tensor.expand_shape %[[ARG1]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [4, 32, 8, 32] : tensor<128x256xf32> into tensor<4x32x8x32xf32>
  // CHECK: %[[RES:.+]] = linalg.generic {{.*}} ins(%[[EXP0]], %[[CST]] : tensor<4x32x16x32xf32>, tensor<8x16x32x32xf32>) outs(%[[EXP1]] : tensor<4x32x8x32xf32>)
  // CHECK: %[[COL:.+]] = tensor.collapse_shape %[[RES]] {{\[}}[0, 1], [2, 3]{{\]}} : tensor<4x32x8x32xf32> into tensor<128x256xf32>
  // CHECK: return %[[COL]]


// -----

// NB: obtained from a M=?, N=256, K=512 linalg.matmul by -pack-matmul
#map = affine_map<()[s0] -> (s0 ceildiv 32)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @revert_packing_with_leading_dim_dynamic(%arg0: tensor<?x512xf32>, %arg1: tensor<?x256xf32>) -> tensor<?x256xf32> {
    %cst = arith.constant dense<1.000000e-03> : tensor<8x16x32x32xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x512xf32>
    %0 = affine.apply #map()[%dim]
    %1 = tensor.empty(%0) : tensor<?x16x32x32xf32>
    %pack = linalg.pack %arg0 padding_value(%cst_0 : f32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<?x512xf32> -> tensor<?x16x32x32xf32>
    %dim_1 = tensor.dim %arg1, %c0 : tensor<?x256xf32>
    %2 = affine.apply #map()[%dim_1]
    %3 = tensor.empty(%2) : tensor<?x8x32x32xf32>
    %pack_2 = linalg.pack %arg1 padding_value(%cst_0 : f32) inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %3 : tensor<?x256xf32> -> tensor<?x8x32x32xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %cst : tensor<?x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_2 : tensor<?x8x32x32xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %5 = arith.mulf %in, %in_3 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<?x8x32x32xf32>
    %unpack = linalg.unpack %4 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : tensor<?x8x32x32xf32> -> tensor<?x256xf32>
    return %unpack : tensor<?x256xf32>
  }
}
// CHECK-LABEL: func.func @revert_packing_with_leading_dim_dynamic(
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x512xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x256xf32>)
// CHECK-SAME: -> tensor<?x256xf32>
//  func.func @revert_packing_with_one_dim_dynamic(%arg0: tensor<?x512xf32>, %arg1: tensor<?x256xf32>) -> tensor<?x256xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e-03> : tensor<8x16x32x32xf32>
  // CHECK: %[[M:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[M_DUP:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  // CHECK: %[[M_ROUNDED_UP:.*]] = affine.apply {{.*}}()[%[[M_DUP]], %[[M]]]
  // CHECK: %[[ARG0_PADDED:.*]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[M_ROUNDED_UP]], 0]
  // CHECK: %[[M_PADDED:.*]] = tensor.dim %[[ARG0_PADDED]], %[[C0]]
  // CHECK: %[[NUM_CHUNKS_PADDED_M:.*]] = arith.divsi %[[M_PADDED]], %[[C32]]
  // CHECK: %[[EXP0:.+]] = tensor.expand_shape %[[ARG0_PADDED]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [%[[NUM_CHUNKS_PADDED_M]], 32, 16, 32] : tensor<?x512xf32> into tensor<?x32x16x32xf32>
  // CHECK: %[[M_ARG1:.*]] = tensor.dim %[[ARG1]], %[[C0]]
  // CHECK: %[[M_ARG1_DUP:.*]] = tensor.dim %[[ARG1]], %[[C0]]
  // CHECK: %[[M_ARG1_ROUNDED_UP:.*]] = affine.apply {{.*}}()[%[[M_ARG1_DUP]], %[[M_ARG1]]]
  // CHECK: %[[ARG1_PADDED:.*]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[M_ARG1_ROUNDED_UP]], 0]
  // CHECK: %[[M_ARG1_PADDED:.*]] = tensor.dim %[[ARG1_PADDED]], %[[C0]]
  // CHECK: %[[NUM_CHUNKS_PADDED_M_ARG1:.*]] = arith.divsi %[[M_ARG1_PADDED]], %[[C32]]
  // CHECK: %[[EXP1:.+]] = tensor.expand_shape %[[ARG1_PADDED]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [%[[NUM_CHUNKS_PADDED_M_ARG1]], 32, 8, 32] : tensor<?x256xf32> into tensor<?x32x8x32xf32>
  // CHECK: %[[RES:.+]] = linalg.generic {{.*}} ins(%[[EXP0]], %[[CST]] : tensor<?x32x16x32xf32>, tensor<8x16x32x32xf32>) outs(%[[EXP1]] : tensor<?x32x8x32xf32>)
  // CHECK: %[[COL:.+]] = tensor.collapse_shape %[[RES]] {{\[}}[0, 1], [2, 3]{{\]}} : tensor<?x32x8x32xf32> into tensor<?x256xf32>
  // CHECK: %[[M_DUP2:.*]] = tensor.dim %[[ARG1]], %[[C0]]
  // CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[COL]][0, 0] [%[[M_DUP2]], 256] [1, 1] : tensor<?x256xf32> to tensor<?x256xf32>
  // CHECK: %[[COPY:.+]] = linalg.copy ins(%[[SLICE]] : tensor<?x256xf32>) outs(%[[ARG1]]
  // CHECK: return %[[COPY]]
