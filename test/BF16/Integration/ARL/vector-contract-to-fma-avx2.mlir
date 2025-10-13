// RUN: tpp-run -e gemm_splat --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e gemm_splat --entry-point-result=void -print --disable-vnni-packing --vector-to-kernels --registerBlocking=4,16,1  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2

func.func @gemm_splat(%arg0: tensor<8x32x32x32xbf16>, %arg1: tensor<32x32x32x32xbf16>, %arg2: tensor<8x32x32x32xbf16>) -> tensor<8x32x32x32xbf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%arg2 : tensor<8x32x32x32xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %1 = arith.mulf %in, %in_0 : bf16
    %2 = arith.addf %out, %1 : bf16
    linalg.yield %2 : bf16
  } -> tensor<8x32x32x32xbf16>
  return %0 : tensor<8x32x32x32xbf16>
}

// -----

// RUN: tpp-run -e mlp_splat --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e mlp_splat --entry-point-result=void -print --disable-vnni-packing --vector-to-kernels --registerBlocking=4,16,1  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @mlp_splat(%arg0: tensor<8x32x32x32xbf16>, %arg1: tensor<32x32x32x32xbf16>, %arg2: tensor<32x32xbf16>, %arg3: tensor<8x32x32x32xbf16>) -> tensor<8x32x32x32xbf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%arg3 : tensor<8x32x32x32xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %3 = arith.mulf %in, %in_0 : bf16
    %4 = arith.addf %out, %3 : bf16
    linalg.yield %4 : bf16
  } -> tensor<8x32x32x32xbf16>
  %1 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<32x32xbf16>) outs(%0 : tensor<8x32x32x32xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %3 = arith.addf %in, %out : bf16
    linalg.yield %3 : bf16
  } -> tensor<8x32x32x32xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %2 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<8x32x32x32xbf16>) {
  ^bb0(%out: bf16):
    %3 = arith.maximumf %out, %cst : bf16
    linalg.yield %3 : bf16
  } -> tensor<8x32x32x32xbf16>
  return %2 : tensor<8x32x32x32xbf16>
}

