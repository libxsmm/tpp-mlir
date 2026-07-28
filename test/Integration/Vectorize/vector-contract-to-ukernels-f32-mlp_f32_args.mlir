// RUN: tpp-run -e mlp_f32_args --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e mlp_f32_args --entry-point-result=void  --nano-kernels --registerBlocking=8,32,1 --gemm-unroll=1,16,1 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

#mlp_map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#mlp_map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#mlp_map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#mlp_map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#mlp_map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @mlp_f32_args(%arg0: tensor<8x4x32x32xf32>, %arg1: tensor<4x4x32x32xf32>, %arg2: tensor<4x32xf32>, %arg3: tensor<8x4x32x32xf32>) -> tensor<8x4x32x32xf32> {
  %0 = linalg.generic {indexing_maps = [#mlp_map, #mlp_map1, #mlp_map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%arg3 : tensor<8x4x32x32xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %3 = arith.mulf %in, %in_0 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<8x4x32x32xf32>
  %1 = linalg.generic {indexing_maps = [#mlp_map3, #mlp_map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<4x32xf32>) outs(%0 : tensor<8x4x32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.addf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<8x4x32x32xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.generic {indexing_maps = [#mlp_map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<8x4x32x32xf32>) {
  ^bb0(%out: f32):
    %3 = arith.maximumf %out, %cst : f32
    linalg.yield %3 : f32
  } -> tensor<8x4x32x32xf32>
  return %2 : tensor<8x4x32x32xf32>
}
