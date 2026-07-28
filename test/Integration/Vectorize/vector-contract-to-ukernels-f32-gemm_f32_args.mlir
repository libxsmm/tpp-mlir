// RUN: tpp-run -e gemm_f32_args --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e gemm_f32_args --entry-point-result=void  --nano-kernels --registerBlocking=8,32,1 --gemm-unroll=1,16,1 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @gemm_f32_args(%arg0: tensor<8x4x32x32xf32>, %arg1: tensor<4x4x32x32xf32>, %arg2: tensor<8x4x32x32xf32>) -> tensor<8x4x32x32xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%arg2 : tensor<8x4x32x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<8x4x32x32xf32>
  return %0 : tensor<8x4x32x32xf32>
}
