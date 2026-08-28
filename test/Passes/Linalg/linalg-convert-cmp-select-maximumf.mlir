// RUN: tpp-opt %s --linalg-convert-compare-select-to-maximumf-pass --split-input-file | FileCheck %s

func.func @compare_select_to_max(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xf32>{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<256x1024xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : tensor<256x1024xf32>) outs(%0 : tensor<256x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %15 = arith.cmpf ugt, %in, %cst : f32
      %16 = arith.select %15, %in, %cst : f32
      linalg.yield %16 : f32
    } -> tensor<256x1024xf32>

  return %1 : tensor<256x1024xf32>
}

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK: func.func @compare_select_to_max(
// CHECK-SAME: %[[ARG0:.+]]: tensor<256x1024xf32>
// CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<256x1024xf32>
// CHECK: %[[temp1:.*]] = linalg.generic {indexing_maps = [#map, #map]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%[[ARG0]] : tensor<256x1024xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<256x1024xf32>) {
// CHECK:    ^bb0(%[[in:.*]]: f32, %[[out:.*]]: f32):
// CHECK:      %[[temp2:.*]] = arith.maximumf %[[in]], %[[cst]] : f32
// CHECK:      linalg.yield %[[temp2]] : f32
// CHECK:    } -> tensor<256x1024xf32>
// CHECK:    return %[[temp1]] : tensor<256x1024xf32>


// -----

func.func @compare_select_to_max_inplace(%arg0: tensor<256x1024xf32>) -> tensor<256x1024xf32>{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg0 : tensor<256x1024xf32>) {
    ^bb0(%out: f32):
      %15 = arith.cmpf ugt, %out, %cst : f32
      %16 = arith.select %15, %out, %cst : f32
      linalg.yield %16 : f32
    } -> tensor<256x1024xf32>

  return %0 : tensor<256x1024xf32>
}

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK: func.func @compare_select_to_max_inplace(
// CHECK-SAME: %[[ARG0:.+]]: tensor<256x1024xf32>
// CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[temp1:.*]] = linalg.generic {indexing_maps = [#map]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: outs(%[[ARG0]] : tensor<256x1024xf32>) {
// CHECK:    ^bb0(%[[out:.*]]: f32):
// CHECK:      %[[temp2:.*]] = arith.maximumf %[[out]], %[[cst]] : f32
// CHECK:      linalg.yield %[[temp2]] : f32
// CHECK:    } -> tensor<256x1024xf32>
// CHECK:    return %[[temp1]] : tensor<256x1024xf32>


// -----

func.func @non_zero_compare() -> tensor<256x1024xf32>{
%cst_5 = arith.constant 1.000000e+00 : f32
%5 = tensor.empty() : tensor<256x1024xf32>
%2 = tensor.empty() : tensor<256x1024xf32>
%6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %15 = arith.cmpf ugt, %in, %cst_5 : f32
    %16 = arith.select %15, %in, %cst_5 : f32
    linalg.yield %16 : f32
  } -> tensor<256x1024xf32>

  return %6: tensor<256x1024xf32>
}

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK: func.func @non_zero_compare()
// CHECK: -> tensor<256x1024xf32> {
// CHECK-DAG: %[[cst:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[temp0:.*]] = tensor.empty() : tensor<256x1024xf32>
// CHECK: %[[temp1:.*]] = tensor.empty() : tensor<256x1024xf32>
// CHECK:%[[temp2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[temp0]] : tensor<256x1024xf32>) outs(%[[temp1]] : tensor<256x1024xf32>) {
// CHECK: ^bb0(%[[in:.*]], %[[out:.*]]: f32):
// CHECK-NOT:  %[[temp2:.*]] = arith.maximumf %[[out]], %[[cst]] : f32
// CHECK-NOT:   linalg.yield %[[temp2]] : f32

// -----

func.func @non_compare_select() -> tensor<256x1024xf32>{
%cst_5 = arith.constant 0.000000e+00 : f32
%5 = tensor.empty() : tensor<256x1024xf32>
%2 = tensor.empty() : tensor<256x1024xf32>
%6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<256x1024xf32>) outs(%2 : tensor<256x1024xf32>) {
  ^bb0(%in: f32, %out: f32):
    %15 = arith.cmpf ugt, %in, %cst_5 : f32
    %temp = arith.cmpf ult, %in, %cst_5 : f32
    %16 = arith.select %temp, %in, %cst_5 : f32
    linalg.yield %16 : f32
  } -> tensor<256x1024xf32>

return %6: tensor<256x1024xf32>
}

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK: func.func @non_compare_select()
// CHECK: -> tensor<256x1024xf32> {
// CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[temp0:.*]] = tensor.empty() : tensor<256x1024xf32>
// CHECK: %[[temp1:.*]] = tensor.empty() : tensor<256x1024xf32>
// CHECK:%[[temp2:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[temp0]] : tensor<256x1024xf32>) outs(%[[temp1]] : tensor<256x1024xf32>) {
// CHECK: ^bb0(%[[in:.*]], %[[out:.*]]: f32):
// CHECK-NOT:  %[[temp2:.*]] = arith.maximumf %[[out]], %[[cst]] : f32
// CHECK-NOT:   linalg.yield %[[temp2]] : f32
