// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 1612185600

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @entry(%arg0: tensor<8x32x32x32xf32>, %arg1: tensor<32x32x32x32xf32>, %arg2: tensor<32x32xf32>, %arg3: tensor<8x32x32x32xf32>, %arg4: tensor<32x32x32x32xf32>, %arg5: tensor<32x32xf32>, %arg6: tensor<8x32x32x32xf32>, %arg7: tensor<32x32x32x32xf32>, %arg8: tensor<32x32xf32>, %arg9: tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32> attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"reg_gemm_unroll" = [1, 16, 1]>>} {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg3 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.mulf %in, %in_2 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<8x32x32x32xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<32x32xf32>) outs(%0 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<8x32x32x32xf32>) {
    ^bb0(%out: f32):
      %9 = arith.maximumf %out, %cst : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%2, %arg4 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg6 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.mulf %in, %in_2 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<8x32x32x32xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg5 : tensor<32x32xf32>) outs(%3 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %5 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4 : tensor<8x32x32x32xf32>) {
    ^bb0(%out: f32):
      %9 = arith.maximumf %out, %cst_0 : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%5, %arg7 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg9 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.mulf %in, %in_2 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<8x32x32x32xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg8 : tensor<32x32xf32>) outs(%6 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %8 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%7 : tensor<8x32x32x32xf32>) {
    ^bb0(%out: f32):
      %9 = arith.maximumf %out, %cst_1 : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    return %8 : tensor<8x32x32x32xf32>
  }
}
