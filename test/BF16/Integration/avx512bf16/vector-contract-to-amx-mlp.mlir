// RUN: tpp-run -e entry --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run %s -e entry --entry-point-result=void --vector-to-kernels --registerBlocking=32,32,32 -print  --splat-to-random --init-type normal  -seed 123  > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2


#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @entry(%arg0: tensor<1x2x32x32xbf16>, %arg1: tensor<2x2x16x32x2xbf16>, %arg2: tensor<2x32xbf16>, %arg3: tensor<1x2x32x32xbf16>, %arg4: tensor<2x2x16x32x2xbf16>, %arg5: tensor<2x32xbf16>, %arg6: tensor<1x2x32x32xbf16>) -> tensor<1x2x32x32xbf16> {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [1, 2, 32, 16, 2] : tensor<1x2x32x32xbf16> into tensor<1x2x32x16x2xbf16>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%expanded, %arg1 : tensor<1x2x32x16x2xbf16>, tensor<2x2x16x32x2xbf16>) outs(%arg3 : tensor<1x2x32x32xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_1 : bf16
      %5 = arith.addf %out, %4 : bf16
      linalg.yield %5 : bf16
    } -> tensor<1x2x32x32xbf16>
    %1 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<2x32xbf16>) outs(%0 : tensor<1x2x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4 = arith.addf %in, %out : bf16
      linalg.yield %4 : bf16
    } -> tensor<1x2x32x32xbf16>
    %expanded_0 = tensor.expand_shape %1 [[0], [1], [2], [3, 4]] output_shape [1, 2, 32, 16, 2] : tensor<1x2x32x32xbf16> into tensor<1x2x32x16x2xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%expanded_0, %arg4 : tensor<1x2x32x16x2xbf16>, tensor<2x2x16x32x2xbf16>) outs(%arg6 : tensor<1x2x32x32xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_1 : bf16
      %5 = arith.addf %out, %4 : bf16
      linalg.yield %5 : bf16
    } -> tensor<1x2x32x32xbf16>
    %3 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg5 : tensor<2x32xbf16>) outs(%2 : tensor<1x2x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4 = arith.addf %in, %out : bf16
      linalg.yield %4 : bf16
    } -> tensor<1x2x32x32xbf16>
    return %3 : tensor<1x2x32x32xbf16>
  }
}
