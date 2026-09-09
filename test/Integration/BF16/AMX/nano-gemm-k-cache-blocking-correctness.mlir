// End-to-end correctness of nano (AMX) K-cache-blocking. The reduction (K) has
// 8 batch-reduce blocks; blocking by 4 accumulates into a stationary bf16 C in
// place.

// RUN: tpp-run %s -e entry --entry-point-result=void -print --splat-to-random --init-type normal -seed 123 > %t.1
// RUN: tpp-run %s -e entry --entry-point-result=void --nano-kernels --registerBlocking=32,32,32 --gemm-unroll=16,16,16 --k-cache-blocking=4 -print --splat-to-random --init-type normal -seed 123 > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @entry(%arg0: tensor<2x8x32x32xbf16>, %arg1: tensor<2x8x16x32x2xbf16>, %arg2: tensor<2x2x32x32xbf16>) -> tensor<2x2x32x32xbf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x2x32x32xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x2x32x32xf32>) -> tensor<2x2x32x32xf32>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [2, 8, 32, 16, 2] : tensor<2x8x32x32xbf16> into tensor<2x8x32x16x2xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%expanded, %arg1 : tensor<2x8x32x16x2xbf16>, tensor<2x8x16x32x2xbf16>) outs(%1 : tensor<2x2x32x32xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %4 = arith.extf %in : bf16 to f32
      %5 = arith.extf %in_0 : bf16 to f32
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<2x2x32x32xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<2x2x32x32xf32>) outs(%arg2 : tensor<2x2x32x32xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %4 = arith.truncf %in : f32 to bf16
      linalg.yield %4 : bf16
    } -> tensor<2x2x32x32xbf16>
    return %3 : tensor<2x2x32x32xbf16>
  }
}
