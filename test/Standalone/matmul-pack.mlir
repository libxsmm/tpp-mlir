// RUN: standalone-opt -split-input-file -pack-matmul="block-factors=32,32,32" --canonicalize %s | FileCheck %s

func.func @matmul(%arg0: tensor<128x512xf32>, 
                  %arg1: tensor<512x256xf32>, 
                  %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x512xf32>, tensor<512x256xf32>) outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<128x512xf32>, 
// CHECK-SAME:  %[[ARG1:.+]]: tensor<512x256xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK: %[[BUFF0:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// CHECK: %[[PACK0:.+]] = linalgx.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF0]] : (tensor<128x512xf32> tensor<4x16x32x32xf32>) -> tensor<4x16x32x32xf32>
// CHECK: %[[BUFF1:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalgx.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF1]] : (tensor<512x256xf32> tensor<8x16x32x32xf32>) -> tensor<8x16x32x32xf32>
// CHECK: %[[BUFF2:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: %[[PACK2:.+]] = linalgx.pack %[[ARG2]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[BUFF2]] : (tensor<128x256xf32> tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%[[PACK2]] : tensor<4x8x32x32xf32>)
// CHECK: %[[OUT:.+]] = linalgx.unpack %[[VAL]] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %[[ARG2]] : (tensor<4x8x32x32xf32> tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK: return %[[OUT]] : tensor<128x256xf32>
// CHECK: }

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul(%arg0: tensor<128x256xf32>,
                  %arg1: tensor<256x256xf32>,
                  %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x256xf32>, tensor<256x256xf32>) outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  %1 = linalg.generic {indexing_maps = [#map0], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<128x256xf32>) {
    ^bb0(%arg3: f32):
      %20 = mathx.relu %arg3 : f32
      linalg.yield %20 : f32
  } -> tensor<128x256xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0: tensor<128x256xf32>) outs(%1: tensor<128x256xf32>) {
    ^bb0(%arg6 : f32, %arg7 : f32):
      %23 = arith.addf %arg6, %arg7 : f32
      linalg.yield %23 : f32
  } -> tensor<128x256xf32>
  return %3 : tensor<128x256xf32>
}
