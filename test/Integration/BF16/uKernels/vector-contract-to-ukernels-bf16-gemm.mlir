// RUN: tpp-run -e gemm --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e gemm --entry-point-result=void -print --nano-kernels --gemm-unroll=1,16,1 --registerBlocking=8,32,2  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2
#map = affine_map<(d0,  d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0,  d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0,  d1, d2, d3, d4) -> (d2, d3)>

func.func @gemm(%arg0: tensor<16x128x64x2xbf16>, %arg1: tensor<16x64x128x2xbf16>, %arg2: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x128x64x2xbf16>, tensor<16x64x128x2xbf16>) outs(%arg2 : tensor<128x128xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %1 = arith.mulf %in, %in_0 : bf16
    %2 = arith.addf %out, %1 : bf16
    linalg.yield %2 : bf16
  } -> tensor<128x128xbf16>
  return %0 : tensor<128x128xbf16>
}
