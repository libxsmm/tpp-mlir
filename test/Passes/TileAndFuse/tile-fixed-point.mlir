// RUN: tpp-opt %s -tile-consumer-and-fuse-producers | FileCheck %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="num-iters=1" | FileCheck %s -check-prefix=ONE
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="num-iters=2" | FileCheck %s -check-prefix=TWO

#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>

func.func @fixed_point_fusion(%arg0: tensor<64x8x32x32xf32>, %arg1: tensor<64x32x8x64xf32>,
                              %arg2: tensor<8x64x512xf32>) -> tensor<64x32x512xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<64x32x8x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %2 = linalg.generic {
    indexing_maps = [#map4, #map9, #map10],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<64x8x32x32xf32>, tensor<64x32x8x64xf32>) outs(%1 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x8x64xf32>
  %3 = tensor.empty() : tensor<64x32x512xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<64x32x512xf32>) -> tensor<64x32x512xf32>
  %5 = linalg.generic {
    indexing_maps = [#map4, #map12, #map11],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]}
    ins(%2, %arg2 : tensor<64x32x8x64xf32>, tensor<8x64x512xf32>) outs(%4 : tensor<64x32x512xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x512xf32>
  return %5 : tensor<64x32x512xf32>
}

// CHECK-LABEL: fixed_point_fusion
// CHECK: %{{.+}} = scf.forall (%{{.+}}) in (64)
// CHECK: %{{.+}} = scf.forall (%{{.+}}) in (8)

// ONE-LABEL: fixed_point_fusion
// ONE-COUNT-1: %{{.+}} = scf.forall

// TWO-LABEL: fixed_point_fusion
// TWO-COUNT-2: %{{.+}} = scf.forall
