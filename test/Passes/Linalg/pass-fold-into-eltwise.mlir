// RUN: tpp-opt %s -fold-into-eltwise -split-input-file | FileCheck %s

func.func @broadcast_into_add_outer_dim(%arg0: tensor<8xf32>,
    %arg1: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %e = tensor.empty() : tensor<16x8xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<8xf32>) outs(%e : tensor<16x8xf32>) dimensions = [0]
  %1 = linalg.add ins(%0, %arg1 : tensor<16x8xf32>, tensor<16x8xf32>)
                  outs(%e : tensor<16x8xf32>) -> tensor<16x8xf32>
  return %1 : tensor<16x8xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @broadcast_into_add_outer_dim(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME: ins(%[[ARG0]],{{.*}})
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

func.func @broadcast_into_add_inner_dim(%arg0: tensor<8xf32>,
    %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<8xf32>) outs(%e : tensor<8x4xf32>) dimensions = [1]
  %1 = linalg.add ins(%0, %arg1 : tensor<8x4xf32>, tensor<8x4xf32>)
                  outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @broadcast_into_add_inner_dim(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME: ins(%[[ARG0]],{{.*}})
// CHECK: arith.addf
// CHECK: linalg.yield

// -----

func.func @broadcast_into_mul(%arg0: tensor<8xf32>,
    %arg1: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<8xf32>) outs(%e : tensor<8x4xf32>) dimensions = [1]
  %1 = linalg.mul ins(%0, %arg1 : tensor<8x4xf32>, tensor<8x4xf32>)
                  outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @broadcast_into_mul(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME: ins(%[[ARG0]],{{.*}})
// CHECK: arith.mulf
// CHECK: linalg.yield

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @broadcast_into_generic(%arg0: tensor<4xf32>,
    %arg1: tensor<4x2x8xf32>) -> tensor<4x2x8xf32> {
  %e = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<4xf32>) outs(%e : tensor<4x8xf32>) dimensions = [1]
  %1 = linalg.generic {indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%0 : tensor<4x8xf32>) outs(%arg1 : tensor<4x2x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<4x2x8xf32>
  return %1 : tensor<4x2x8xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @broadcast_into_generic(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<4xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP1]]]
// CHECK-SAME: ins(%[[ARG0]] :{{.*}})

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @broadcast_into_generic_transposed(%arg0: tensor<8xf32>,
    %arg1: tensor<4x2x8xf32>) -> tensor<4x2x8xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<8xf32>) outs(%e : tensor<8x4xf32>) dimensions = [1]
  %1 = linalg.generic {indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%0 : tensor<8x4xf32>) outs(%arg1 : tensor<4x2x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<4x2x8xf32>
  return %1 : tensor<4x2x8xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: @broadcast_into_generic_transposed(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP1]]]
// CHECK-SAME: ins(%[[ARG0]] :{{.*}})

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @broadcast_into_generic_multidim(%arg0: tensor<6x2xf32>,
    %arg1: tensor<2x4x6x8xf32>) -> tensor<2x4x6x8xf32> {
  %e = tensor.empty() : tensor<6x4x2xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<6x2xf32>) outs(%e : tensor<6x4x2xf32>) dimensions = [1]
  %1 = linalg.generic {indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%0 : tensor<6x4x2xf32>) outs(%arg1 : tensor<2x4x6x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<2x4x6x8xf32>
  return %1 : tensor<2x4x6x8xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @broadcast_into_generic_multidim(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<6x2xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP1]]]
// CHECK-SAME: ins(%[[ARG0]] :{{.*}})

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @broadcast_into_generic_multiple_operands(%arg0: tensor<4xf32>,
    %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %e = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<4xf32>) outs(%e : tensor<4x8xf32>) dimensions = [1]
  %1 = linalg.generic {indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1, %0 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%arg2 : tensor<4x8xf32>) {
    ^bb0(%in: f32, %in1: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      %2 = arith.addf %1, %in1 : f32
      linalg.yield %2 : f32
    } -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @broadcast_into_generic_multiple_operands(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<4xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: tensor<4x8xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<4x8xf32>
// CHECK-NOT: linalg.broadcast
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP]]]
// CHECK-SAME: ins(%[[ARG1]], %[[ARG0]] :{{.*}})

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @no_fold_non_eltwise(%arg0: tensor<16xf32>,
    %arg1: tensor<32x64xf32>,
    %arg2: tensor<16x64xf32>) -> tensor<16x64xf32> {
  %e = tensor.empty() : tensor<16x32xf32>
  %0 = linalg.broadcast ins(%arg0 : tensor<16xf32>) outs(%e : tensor<16x32xf32>) dimensions = [1]
  %1 = linalg.matmul ins(%0, %arg1 : tensor<16x32xf32>, tensor<32x64xf32>)
                     outs(%arg2 : tensor<16x64xf32>) -> tensor<16x64xf32>
  return %1 : tensor<16x64xf32>
}

// CHECK-LABEL: @no_fold_non_eltwise(
// CHECK: linalg.broadcast
// CHECK: linalg.matmul

// -----

func.func @no_broadcast_fold_non_tensor(%arg0: memref<8xf32>,
    %arg1: memref<8x4xf32>,
    %arg2: memref<8x4xf32>) {
  linalg.broadcast ins(%arg0 : memref<8xf32>) outs(%arg2 : memref<8x4xf32>) dimensions = [1]
  linalg.add ins(%arg2, %arg1 : memref<8x4xf32>, memref<8x4xf32>)
             outs(%arg2 : memref<8x4xf32>)
  return
}

// CHECK-LABEL: @no_broadcast_fold_non_tensor(
// CHECK: linalg.broadcast
// CHECK: linalg.add

// -----

func.func @fill_lhs_into_max_float(%arg0: tensor<8x4xf32>,
    %cst: f32) -> tensor<8x4xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.fill ins(%cst : f32)
    outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  %1 = linalg.max ins(%0, %arg0 : tensor<8x4xf32>, tensor<8x4xf32>)
    outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @fill_lhs_into_max_float(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x4xf32>
// CHECK-SAME:  %[[CST:.+]]: f32
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: ins(%[[ARG0]] :{{.*}})
// CHECK: arith.maximumf %{{.+}}, %[[CST]]
// CHECK: linalg.yield

// -----

func.func @fill_rhs_into_max_float(%arg0: tensor<8x4xf32>,
    %cst: f32) -> tensor<8x4xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.fill ins(%cst : f32)
    outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  %1 = linalg.max ins(%arg0, %0 : tensor<8x4xf32>, tensor<8x4xf32>)
    outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %1 : tensor<8x4xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @fill_rhs_into_max_float(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x4xf32>
// CHECK-SAME:  %[[CST:.+]]: f32
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: ins(%[[ARG0]] :{{.*}})
// CHECK: arith.maximumf %{{.+}}, %[[CST]]
// CHECK: linalg.yield

// -----

func.func @fill_into_max_int(%arg0: tensor<8x4xi32>,
    %cst: i32) -> tensor<8x4xi32> {
  %e = tensor.empty() : tensor<8x4xi32>
  %0 = linalg.fill ins(%cst : i32)
    outs(%e : tensor<8x4xi32>) -> tensor<8x4xi32>
  %1 = linalg.max ins(%arg0, %0 : tensor<8x4xi32>, tensor<8x4xi32>)
    outs(%e : tensor<8x4xi32>) -> tensor<8x4xi32>
  return %1 : tensor<8x4xi32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @fill_into_max_int(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x4xi32>
// CHECK-SAME:  %[[CST:.+]]: i32
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME: ins(%[[ARG0]] :{{.*}})
// CHECK: arith.maxsi %{{.+}}, %[[CST]]
// CHECK: linalg.yield

// -----

func.func @double_fill_into_max(%cst: f32,
    %cst1: f32) -> tensor<8x4xf32> {
  %e = tensor.empty() : tensor<8x4xf32>
  %0 = linalg.fill ins(%cst : f32)
    outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  %1 = linalg.fill ins(%cst1 : f32)
    outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  %2 = linalg.max ins(%0, %1 : tensor<8x4xf32>, tensor<8x4xf32>)
    outs(%e : tensor<8x4xf32>) -> tensor<8x4xf32>
  return %2 : tensor<8x4xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: @double_fill_into_max(
// CHECK-SAME:  %[[CST:.+]]: f32,
// CHECK-SAME:  %[[CST1:.+]]: f32
// CHECK-NOT: linalg.fill
// CHECK: linalg.generic{{.*}}indexing_maps = [#[[MAP]]]
// CHECK: arith.maximumf %[[CST]], %[[CST1]]
// CHECK: linalg.yield

// -----

func.func @no_fill_fold_non_tensor(%arg0: memref<8x4xf32>,
    %arg1: memref<8x4xf32>, %cst: f32) {
  linalg.fill ins(%cst : f32) outs(%arg1 : memref<8x4xf32>)
  linalg.max ins(%arg0, %arg1 : memref<8x4xf32>, memref<8x4xf32>) outs(%arg1 : memref<8x4xf32>)
  return
}

// CHECK-LABEL: @no_fill_fold_non_tensor(
// CHECK: linalg.fill
// CHECK: linalg.max
