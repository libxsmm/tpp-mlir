// RUN: tpp-opt -pack-conv2DNchwFchw -split-input-file %s | FileCheck %s

func.func @conv_2d_nchw_fchw(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

// CHECK: func.func @conv_2d_nchw_fchw(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<14x512x28x28xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<1024x512x1x1xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
// CHECK: %[[BUF0:.+]] = tensor.empty() : tensor<14x16x28x28x32xf32>
// CHECK: %[[PACK0:.+]] = linalg.pack %[[ARG0]] inner_dims_pos = [1] inner_tiles = [32] into %[[BUF0]] : tensor<14x512x28x28xf32> -> tensor<14x16x28x28x32xf32>
// CHECK: %[[BUF1:.+]] = tensor.empty() : tensor<32x16x1x1x32x32xf32>
// CHECK: %[[PACK1:.+]] = linalg.pack %[[ARG1]] inner_dims_pos = [1, 0] inner_tiles = [32, 32] into %[[BUF1]] : tensor<1024x512x1x1xf32> -> tensor<32x16x1x1x32x32xf32>
// CHECK: %[[BUF2:.+]] = tensor.empty() : tensor<14x32x28x28x32xf32>
// CHECK: %[[PACK2:.+]] = linalg.pack %[[ARG2]] inner_dims_pos = [1] inner_tiles = [32] into %[[BUF2]] : tensor<14x1024x28x28xf32> -> tensor<14x32x28x28x32xf32>
// CHECK: %[[VAL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]} ins(%[[PACK0]], %[[PACK1]] : tensor<14x16x28x28x32xf32>, tensor<32x16x1x1x32x32xf32>) outs(%[[PACK2]] : tensor<14x32x28x28x32xf32>)
// CHECK: %[[OUT:.+]] = linalg.unpack %[[VAL]] inner_dims_pos = [1] inner_tiles = [32] into %[[ARG2]] : tensor<14x32x28x28x32xf32> -> tensor<14x1024x28x28xf32>
// CHECK: return %[[OUT]] : tensor<14x1024x28x28xf32>
