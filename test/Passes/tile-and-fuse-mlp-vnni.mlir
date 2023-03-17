// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=1,1" | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

func.func @mlp(%arg0: tensor<8x48x32x32xbf16>, %arg1: tensor<48x48x16x32x2xbf16>, %arg2: tensor<1536xbf16>, %arg3: tensor<8x48x32x32xbf16>) -> tensor<8x48x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  // CHECK: scf.forall
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x48x32x32xbf16>, tensor<48x48x16x32x2xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_0 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
  } -> tensor<8x48x32x32xbf16>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] : tensor<1536xbf16> into tensor<48x32xbf16>
  %1 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %expanded : tensor<8x48x32x32xbf16>, tensor<48x32xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %3 = arith.addf %in, %in_0 : bf16
      linalg.yield %3 : bf16
  } -> tensor<8x48x32x32xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<8x48x32x32xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) attrs =  {__internal_linalg_transform__ = "fusion"} {
    ^bb0(%in: bf16, %out: bf16):
      %3 = arith.maxf %in, %cst : bf16
      linalg.yield %3 : bf16
  } -> tensor<8x48x32x32xbf16>
  return %2 : tensor<8x48x32x32xbf16>
}
