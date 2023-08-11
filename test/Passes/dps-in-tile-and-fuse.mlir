// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -cse | FileCheck %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -cse -bufferize | FileCheck %s -check-prefix=ALLOC
// ALLOC-COUNT-1: memref.alloc

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

// This test check that destination-passing-style is respected by fusion.
// %arg3 should not be used in the scf.forall region.

func.func @dps_test(%arg0: tensor<8x48x32x32xbf16>, 
               %arg1: tensor<48x48x16x32x2xbf16>, 
               %arg2: tensor<1536xbf16>, 
               %arg3: tensor<8x48x32x32xbf16>) -> tensor<8x48x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x48x32x32xbf16>, tensor<48x48x16x32x2xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x48x32x32xbf16>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] : tensor<1536xbf16> into tensor<48x32xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %expanded : tensor<8x48x32x32xbf16>, tensor<48x32xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x48x32x32xbf16>
  %3 = linalg.generic {__internal_linalg_transform__ = "fusion", indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<8x48x32x32xbf16>) outs(%arg3 : tensor<8x48x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maxf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x48x32x32xbf16> 
  return %3 : tensor<8x48x32x32xbf16>
}

// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: func.func @dps_test
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x48x32x32xbf16>, 
// CHECK-SAME:  %[[ARG1:.+]]: tensor<48x48x16x32x2xbf16>, 
// CHECK-SAME:  %[[ARG2:.+]]: tensor<1536xbf16>, 
// CHECK-SAME:  %[[ARG3:.+]]: tensor<8x48x32x32xbf16>
// CHECK: %[[C48:.+]] = arith.constant 48 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: scf.forall (%[[I:.+]], %[[J:.+]]) in (%[[C8]], %[[C48]]) 
// CHECK-SAME:  shared_outs(%[[ARG6:.+]] = %[[ARG3]])
// CHECK: %[[SLICE_ARG0:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[ARG0]][%[[I]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<8x48x32x32xbf16> to tensor<48x32x32xbf16>
// CHECK: %[[SLICE_ARG1:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[ARG1:.+]][%[[J]], 0, 0, 0, 0] [1, 48, 16, 32, 2] [1, 1, 1, 1, 1] 
// CHECK-SAME:  : tensor<48x48x16x32x2xbf16> to tensor<48x16x32x2xbf16>
// CHECK: %[[SLICE_INIT:.+]] = tensor.extract_slice 
// CHECK-SAME:  %[[ARG6]][%[[I]], %[[J]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] 
// CHECK-SAME:  : tensor<8x48x32x32xbf16> to tensor<32x32xbf16>
// CHECK: %[[MUL:.+]] = tpp.brgemm
// CHECK-SAME: (%[[SLICE_ARG0]] : tensor<48x32x32xbf16>, %[[SLICE_ARG1]] : tensor<48x16x32x2xbf16>,
// CHECK-SAME: %[[SLICE_INIT]] : tensor<32x32xbf16>)
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
