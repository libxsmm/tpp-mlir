// RUN: tpp-opt %s \
// RUN: -convert-linalg-to-tpp -bufferize | FileCheck %s

#map0 = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @mlp_1layer(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>, %arg2: tensor<1x512xf32>, %output: tensor<128x512xf32>) -> tensor<128x512xf32> {
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<1x512xf32>) outs(%output : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x512xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) -> tensor<128x512xf32>
  %c0 = arith.constant 0.0 : f32
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maxf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x512xf32>
  return %3 : tensor<128x512xf32>
}
// CHECK: func.func @mlp_1layer(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<1x512xf32>,
// CHECK-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
// CHECK: tpp.identity ins(%[[ARG2]] : memref<1x512xf32>) outs(%[[ARG3:.+]] : memref<128x512xf32>)
// CHECK: tpp.gemm ins(%[[ARG0]] : memref<128x256xf32>, %[[ARG1]] : memref<256x512xf32>, %[[ARG3]] : memref<128x512xf32>) outs(%[[ARG3]] : memref<128x512xf32>)
// CHECK: tpp.relu ins(%[[ARG3]] : memref<128x512xf32>) outs(%[[ARG3]] : memref<128x512xf32>)
// CHECK: return
