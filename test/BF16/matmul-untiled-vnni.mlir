// RUN: tpp-opt %s --pack-vnni | FileCheck %s

#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @mlp2(%arg0: tensor<32x64x4x4xbf16>, %arg1: tensor<128x64x4x4xbf16>, %arg2: tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16> {
  %0 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x64x4x4xbf16>, tensor<128x64x4x4xbf16>) outs(%arg2 : tensor<32x128x4x4xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %4 = arith.mulf %in, %in_0 : bf16
    %5 = arith.addf %out, %4 : bf16
    linalg.yield %5 : bf16
  } -> tensor<32x128x4x4xbf16>
  return %0 : tensor<32x128x4x4xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// CHECK: func.func @mlp2(
// CHECK: %[[ARG0:.*]]: tensor<32x64x4x4xbf16>,
// CHECK: %[[ARG1:.*]]: tensor<128x64x4x4xbf16>,
// CHECK: %[[ARG2:.*]]: tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16> {
// CHECK:  %[[PACKBUF:.*]] = tensor.empty() : tensor<128x64x2x4x2xbf16>
// CHECK:  linalg.generic
// CHECK:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK:  iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]

