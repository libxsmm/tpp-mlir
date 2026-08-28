// RUN: tpp-opt %s -element-wise-fusion -tile-consumer-and-fuse-producers="use-for-all=false" | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

func.func @mlp(%arg0: tensor<32x64x4x4xbf16>, %arg1: tensor<128x64x4x4xbf16>, %arg2: tensor<512xbf16>, %arg3: tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x64x4x4xbf16>, tensor<128x64x4x4xbf16>) outs(%arg3 : tensor<32x128x4x4xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_0 : bf16
      %5 = arith.addf %out, %4 : bf16
      linalg.yield %5 : bf16
  } -> tensor<32x128x4x4xbf16>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] output_shape [128, 4] : tensor<512xbf16> into tensor<128x4xbf16>
  %1 = tensor.empty() : tensor<32x128x4x4xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %expanded : tensor<32x128x4x4xbf16>, tensor<128x4xbf16>) outs(%1 : tensor<32x128x4x4xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %4 = arith.addf %in, %in_0 : bf16
      linalg.yield %4 : bf16
  } -> tensor<32x128x4x4xbf16>
  %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<32x128x4x4xbf16>) outs(%1 : tensor<32x128x4x4xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %4 = arith.maximumf %in, %cst : bf16
      linalg.yield %4 : bf16
  } -> tensor<32x128x4x4xbf16>
  return %3 : tensor<32x128x4x4xbf16>
}

// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1) -> (d1)>

// CHECK: func.func @mlp(
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %{{.+}} = scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK: %{{.+}} = scf.for %[[J:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK: %{{.+}} = linalg.batch_reduce_matmul
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK: ^bb0(
// CHECK-NEXT:  %{{.+}} = arith.addf
// CHECK-NEXT:  %{{.+}} = arith.maximumf
