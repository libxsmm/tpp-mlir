// RUN: tpp-opt %s -tile-consumer-and-fuse-producers | FileCheck %s

#map = affine_map<(d0, d1, d2, d5, d3, d4) -> (d0, d1, d2, d5)>
#map1 = affine_map<(d0, d1, d2, d5, d3, d4) -> (d2, d5, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d5, d3, d4) -> (d0, d1, d3, d4)>

#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>

#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d1, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
// CHECK-DAG: #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP7:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: mha_tensorflow
func.func @mha_tensorflow(%arg1: tensor<64x32x8x64xf32>,
                           %arg0: tensor<64x32x8x64xf32>,
                           %arg2: tensor<64x32x8x64xf32>) -> tensor<64x32x512xf32> {
  %cst_3 = arith.constant dense<4.0> : tensor<8x64x8x64xf32>
  %cst_4 = arith.constant dense<2.0> : tensor<8x64x8x64xf32>
  %cst_5 = arith.constant dense<1.0> : tensor<8x64x8x64xf32>
  %cst_6 = arith.constant dense<3.0> : tensor<8x64x512xf32>

  %cst_1 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<64x32x8x64xf32>

  // CHECK: %{{.+}} = scf.forall (%{{.+}}) in (64)
  // CHECK: %{{.+}} = scf.forall (%{{.+}}) in (8)
  %fill_1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %3 = linalg.generic {
    "__projection_Q__",
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel"]}
    ins(%arg0, %cst_4 : tensor<64x32x8x64xf32>, tensor<8x64x8x64xf32>) outs(%fill_1 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x8x64xf32>
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "reduction", "reduction", "parallel"]

  %fill_2 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %5 = linalg.generic {
    "__projection_K__",
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel"]}
    ins(%arg1, %cst_5 : tensor<64x32x8x64xf32>, tensor<8x64x8x64xf32>) outs(%fill_2 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x8x64xf32>
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "reduction", "reduction", "parallel"]

  %6 = tensor.empty() : tensor<64x8x32x32xf32>
  %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<64x8x32x32xf32>) -> tensor<64x8x32x32xf32>
  %8 = linalg.generic {
    "__Q_times_K__",
    indexing_maps = [#map4, #map5, #map6],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%5, %3 : tensor<64x32x8x64xf32>, tensor<64x32x8x64xf32>) outs(%7 : tensor<64x8x32x32xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x8x32x32xf32>
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME: indexing_maps =  [#[[MAP3]], #[[MAP4]], #[[MAP5]]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]

  %fill = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %2 = linalg.generic {
    "__projection_V__",
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel"]}
    ins(%arg2, %cst_3 : tensor<64x32x8x64xf32>, tensor<8x64x8x64xf32>) outs(%fill : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x8x64xf32>
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction", "parallel"]

  %fill_4 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64x32x8x64xf32>) -> tensor<64x32x8x64xf32>
  %11 = linalg.generic {
    "__Softmax_times_V__",
    indexing_maps = [#map4, #map9, #map10],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%8, %2 : tensor<64x8x32x32xf32>, tensor<64x32x8x64xf32>) outs(%fill_4 : tensor<64x32x8x64xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x8x64xf32>
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP3]], #[[MAP6]], #[[MAP7]]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]

  %result = tensor.empty() : tensor<64x32x512xf32>
  %fill_r = linalg.fill ins(%cst_1 : f32) outs(%result : tensor<64x32x512xf32>) -> tensor<64x32x512xf32>
  %12 = linalg.generic {
    "__projection_Wo__",
    indexing_maps = [#map4, #map12, #map11],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel"]}
    ins(%11, %cst_6 : tensor<64x32x8x64xf32>, tensor<8x64x512xf32>) outs(%fill_r : tensor<64x32x512xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %23 = arith.mulf %in, %in_8 : f32
      %24 = arith.addf %out, %23 : f32
      linalg.yield %24 : f32
  } -> tensor<64x32x512xf32>
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction", "parallel"]

  return %12: tensor<64x32x512xf32>
}
