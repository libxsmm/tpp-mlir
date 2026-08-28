//RUN: tpp-opt %s --split-input-file --convert-linalg-to-inplace | FileCheck %s

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @add_inplace(%arg0: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<1024xbf16>
    %e = tensor.empty() : tensor<256x1024xbf16>
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map1],
      iterator_types = ["parallel", "parallel"]}
      ins(%cst_1, %arg0 : tensor<1024xbf16>, tensor<256x1024xbf16>)
      outs(%e : tensor<256x1024xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %res = arith.addf %in, %in_1 : bf16
      linalg.yield %res : bf16
    } -> tensor<256x1024xbf16>
    return %0 : tensor<256x1024xbf16>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @add_inplace(
// CHECK-SAME: %[[ARG0:.*]]: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<1024xbf16>
// CHECK:  %[[RES:.*]] = linalg.generic {indexing_maps = [#[[MAP]], #[[MAP1]]],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[CST]] : tensor<1024xbf16>) outs(%[[ARG0]] : tensor<256x1024xbf16>) {
// CHECK:   ^bb0(%[[in:.*]]: bf16, %[[out:.*]]: bf16):
// CHECK:     %[[ADD:.*]] = arith.addf %[[in]], %[[out]] : bf16
// CHECK:     linalg.yield %[[ADD]] : bf16
// CHECK:   } -> tensor<256x1024xbf16>
// CHECK: return %[[RES]] : tensor<256x1024xbf16>
// CHECK: }

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @generic_eltwise_unary_to_inplace(%arg0: tensor<8x4xf32>, %arg1: f32) -> tensor<8x4xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<8x4xf32>) outs(%e : tensor<8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %res = arith.maximumf %in, %arg1 : f32
    linalg.yield %res : f32
  } -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @generic_eltwise_unary_to_inplace(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8x4xf32>,
// CHECK-SAME: %[[ARG1:.*]]: f32
// CHECK:  linalg.generic{{.*}}indexing_maps = [#[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: outs(%[[ARG0]] :{{.*}})
// CHECK:   ^bb0(%[[out:.*]]: f32):
// CHECK:     %[[RES:.*]] = arith.maximumf %[[out]], %[[ARG1]]
// CHECK:     linalg.yield %[[RES]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @generic_eltwise_unary_dynamic_to_inplace(
    %arg0: tensor<?x?xf32>, %arg1: f32) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.mulf %in, %arg1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @generic_eltwise_unary_dynamic_to_inplace(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?xf32>,
// CHECK-SAME: %[[ARG1:.*]]: f32
// CHECK:  linalg.generic{{.*}}indexing_maps = [#[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: outs(%[[ARG0]] :{{.*}})
// CHECK:   ^bb0(%[[out:.*]]: f32):
// CHECK:     %[[RES:.*]] = arith.mulf %[[out]], %[[ARG1]]
// CHECK:     linalg.yield %[[RES]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @no_inplace_generic_used_output(%arg0: tensor<8x4xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<8x4xf32>)
    outs(%arg1 : tensor<8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %res = arith.mulf %in, %out : f32
    linalg.yield %res : f32
  } -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_inplace_generic_used_output(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8x4xf32>,
// CHECK-SAME: %[[ARG1:.*]]: tensor<8x4xf32>
// CHECK:  linalg.generic
// CHECK-SAME: ins(%[[ARG0]] :{{.*}}) outs(%[[ARG1]] :{{.*}})

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @no_inplace_generic_eltwise_binary(%arg0: tensor<8x4xf32>, %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<8x4xf32>, tensor<8x4xf32>)
    outs(%e : tensor<8x4xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %res = arith.mulf %in, %in_1 : f32
    linalg.yield %res : f32
  } -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_inplace_generic_eltwise_binary(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8x4xf32>,
// CHECK-SAME: %[[ARG1:.*]]: tensor<8x4xf32>
// CHECK:  %[[EMPTY:.+]] = tensor.empty
// CHECK:  linalg.generic
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] :{{.*}}) outs(%[[EMPTY]] :{{.*}})

// -----

#map = affine_map<(d0, d1) -> (0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @no_inplace_generic_mismatched_input_output(
    %arg0: tensor<1x1xf32>) -> tensor<6x9xf32>  {
  %e = tensor.empty() : tensor<6x9xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0 : tensor<1x1xf32>) outs(%e : tensor<6x9xf32>) {
  ^bb0(%a: f32, %b: f32):
    linalg.yield %a: f32
  } -> tensor<6x9xf32>
  return %0 : tensor<6x9xf32>
}

// CHECK-LABEL: func.func @no_inplace_generic_mismatched_input_output(
// CHECK-SAME: %[[ARG0:.*]]: tensor<1x1xf32>
// CHECK:  %[[EMPTY:.+]] = tensor.empty
// CHECK:  linalg.generic
// CHECK-SAME: ins(%[[ARG0]] :{{.*}}) outs(%[[EMPTY]] :{{.*}})

// -----

#map = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @no_inplace_generic_mismatched_maps(
    %arg0: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %cst = arith.constant 2.0 : f32
  %0 = tensor.empty() : tensor<8x4xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<8x4xf32>) outs(%0 : tensor<8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.mulf %in, %cst : f32
    linalg.yield %2 : f32
  } -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @no_inplace_generic_mismatched_maps(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8x4xf32>
// CHECK:  %[[EMPTY:.+]] = tensor.empty
// CHECK:  linalg.generic
// CHECK-SAME: ins(%[[ARG0]] :{{.*}}) outs(%[[EMPTY]] :{{.*}})

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @no_inplace_generic_transposed_maps(
    %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %cst = arith.constant 2.0 : f32
  %0 = tensor.empty() : tensor<8x8xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<8x8xf32>) outs(%0 : tensor<8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.mulf %in, %cst : f32
    linalg.yield %2 : f32
  } -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @no_inplace_generic_transposed_maps(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8x8xf32>
// CHECK:  %[[EMPTY:.+]] = tensor.empty
// CHECK:  linalg.generic
// CHECK-SAME: ins(%[[ARG0]] :{{.*}}) outs(%[[EMPTY]] :{{.*}})
