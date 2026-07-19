// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 67108864

#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d4)>

func.func @entry(%arg0: tensor<64x8x32x32xf32>, %arg1: tensor<64x32x8x64xf32>, %out_b: tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32> {
  %cst_1 = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst_1 : f32) outs(%out_b : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %11 = linalg.generic {
    "__Softmax_times_V__",
    indexing_maps = [#map4, #map9, #map10],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<64x8x32x32xf32>, tensor<64x32x8x64xf32>) outs(%fill : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x8x64xf32>
  return %11 : tensor<64x32x8x64xf32>
}
