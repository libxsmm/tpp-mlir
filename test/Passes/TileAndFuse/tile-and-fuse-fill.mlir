// RUN: tpp-opt %s -tile-consumer-and-fuse-producers | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @fuse_fill(%arg0: tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32> {
  %cst = arith.constant dense<1.000000e+00> : tensor<32x32x32x32xf32>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<8x32x32x32xf32>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<8x32x32x32xf32>
  %1 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %cst : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%1 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.mulf %in, %in_2 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
  } -> tensor<8x32x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_0 : tensor<8x32x32x32xf32>) outs(%2 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.addf %in, %out : f32
      linalg.yield %11 : f32
  } -> tensor<8x32x32x32xf32>
  %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%3 : tensor<8x32x32x32xf32>) {
    ^bb0(%out: f32):
      %11 = arith.maximumf %out, %cst_1 : f32
      linalg.yield %11 : f32
  } -> tensor<8x32x32x32xf32>
  %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%4, %cst : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%1 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %11 = arith.mulf %in, %in_2 : f32
      %12 = arith.addf %out, %11 : f32
      linalg.yield %12 : f32
  } -> tensor<8x32x32x32xf32>
  %6 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_0 : tensor<8x32x32x32xf32>) outs(%5 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %11 = arith.addf %in, %out : f32
      linalg.yield %11 : f32
  } -> tensor<8x32x32x32xf32>
  %7 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%6 : tensor<8x32x32x32xf32>) {
    ^bb0(%out: f32):
      %11 = arith.maximumf %out, %cst_1 : f32
      linalg.yield %11 : f32
  } -> tensor<8x32x32x32xf32>
  return %7 : tensor<8x32x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @fuse_fill
// CHECK-SAME: %[[ARG0:.+]]: tensor<8x32x32x32xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<8x32x32x32xf32>
// CHECK: %[[LAYER:.+]] = scf.forall (%[[ARG1:.+]], %[[ARG2:.+]]) in (8, 32) shared_outs(%[[ARG3:.+]] = %[[EMPTY]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG1]], %[[ARG2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<8x32x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[SLICE]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[SLICE_1:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG1]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<8x32x32x32xf32> to tensor<32x32x32xf32>
// CHECK: %[[GEMM:.+]] = linalg.batch_reduce_matmul ins(%[[SLICE_1]], %{{.+}} : tensor<32x32x32xf32>, tensor<32x32x32xf32>)
// CHECK-SAME:  outs(%[[FILL]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[ADD:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]
// CHECK: ^bb0
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]]], iterator_types = ["parallel", "parallel"]
// CHECK: ^bb0
// CHECK: arith.maximumf
// CHECK: linalg.yield

// CHECK: %{{.+}} = scf.forall (%[[ARG1:.+]], %[[ARG2:.+]]) in (8, 32) shared_outs(%[[ARG3:.+]] = %[[EMPTY]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG1]], %[[ARG2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<8x32x32x32xf32> to tensor<32x32xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[SLICE]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[SLICE_1:.+]] = tensor.extract_slice %[[LAYER]][%[[ARG1]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<8x32x32x32xf32> to tensor<32x32x32xf32>
// CHECK: %[[GEMM:.+]] = linalg.batch_reduce_matmul ins(%[[SLICE_1]], %{{.+}} : tensor<32x32x32xf32>, tensor<32x32x32xf32>)
// CHECK-SAME:  outs(%[[FILL]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[ADD:.+]] = linalg.generic
// CHECK-SAME:   indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]
// CHECK: ^bb0
// CHECK: arith.addf
// CHECK: linalg.yield
// CHECK: %{{.+}} = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]]], iterator_types = ["parallel", "parallel"]
// CHECK: ^bb0
// CHECK: arith.maximumf
// CHECK: linalg.yield
