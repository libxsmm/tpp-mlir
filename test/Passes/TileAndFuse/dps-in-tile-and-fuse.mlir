// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -cse | FileCheck %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -cse -bufferize | FileCheck -check-prefix=ALLOC %s
// ALLOC-NOT: memref.alloc

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

// This test check that destination-passing-style is respected by fusion.
// %arg3 should not be used in the scf.forall region.

func.func @dps_test(%arg0: tensor<8x48x32x16x2xbf16>,
               %arg1: tensor<48x48x16x32x2xbf16>,
               %arg2: tensor<1536xbf16>,
               %arg3: tensor<8x48x32x32xbf16>) -> tensor<8x48x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x48x32x16x2xbf16>, tensor<48x48x16x32x2xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x48x32x32xbf16>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] output_shape [48, 32] : tensor<1536xbf16> into tensor<48x32xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %expanded : tensor<8x48x32x32xbf16>, tensor<48x32xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x48x32x32xbf16>
  %3 = linalg.generic {__internal_linalg_transform__ = "fusion", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<8x48x32x32xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x48x32x32xbf16>
  return %3 : tensor<8x48x32x32xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @dps_test
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x48x32x16x2xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<48x48x16x32x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1536xbf16>,
// CHECK-SAME:  %[[ARG3:.+]]: tensor<8x48x32x32xbf16>
// CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) in (8, 48)
// CHECK-SAME:  shared_outs(%[[ARG6:.+]] = %[[ARG3]])
// CHECK: %[[SLICE_ARG0:.+]] = tensor.extract_slice
// CHECK-SAME:  %[[ARG0]][%[[I]], 0, 0, 0, 0] [1, 48, 32, 16, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<8x48x32x16x2xbf16> to tensor<48x32x16x2xbf16>
// CHECK: %[[SLICE_ARG1:.+]] = tensor.extract_slice
// CHECK-SAME:  %[[ARG1:.+]][%[[J]], 0, 0, 0, 0] [1, 48, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<48x48x16x32x2xbf16> to tensor<48x16x32x2xbf16>
// CHECK: %[[SLICE_INIT:.+]] = tensor.extract_slice
// CHECK-SAME:  %[[ARG6]][%[[I]], %[[J]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<8x48x32x32xbf16> to tensor<32x32xbf16>
// CHECK: %[[MUL:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[SLICE_ARG0]], %[[SLICE_ARG1]]
// CHECK-SAME:  outs(%[[SLICE_INIT]]
// CHECK: %[[SLICE_EXPAND:.+]] = tensor.extract_slice
// CHECK-SAME:  %{{.+}}[%[[J]], 0] [1, 32] [1, 1]
// CHECK-SAME:  : tensor<48x32xbf16> to tensor<32xbf16>
// CHECK: %[[ADD:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[MUL]], %[[SLICE_EXPAND]]
// CHECK-SAME:  outs(%[[SLICE_INIT]]
// CHECK: %[[RELU:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[ADD]]
// CHECK-SAME:  outs(%[[SLICE_INIT]]
