// RUN: tpp-opt %s -convert-vector-to-x86 -split-input-file | FileCheck %s

!vecA = vector<1x1xf32>
!vecB = vector<1x64xf32>
!vecC = vector<1x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_outer_product_to_fma(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @matmul_outer_product_to_fma
// CHECK-COUNT-1: vector.shape_cast{{.*}}to vector<1xf32>
// CHECK-COUNT-2: vector.shape_cast{{.*}}to vector<64xf32>
// CHECK: vector.broadcast{{.*}}vector<1xf32> to vector<64xf32>
// CHECK: vector.fma{{.*}}vector<64xf32>
// CHECK: vector.shape_cast{{.*}}vector<64xf32> to vector<1x64xf32>

// -----

!vecA = vector<1x1x1xf32>
!vecB = vector<1x1x64xf32>
!vecC = vector<1x1x64xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_to_fma(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @batch_matmul_to_fma
// CHECK: vector.broadcast
// CHECK: vector.fma{{.*}}vector<64xf32>

// -----

!vecA = vector<1x1x1xf32>
!vecB = vector<1x1x64xf32>
!vecC = vector<1x64xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @brgemm_to_fma(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @brgemm_to_fma
// CHECK: vector.broadcast
// CHECK: vector.fma{{.*}}vector<64xf32>

// -----

!vecA = vector<1x1xf32>
!vecB = vector<1x64xf32>
!vecC = vector<1x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @negative_invalid_kind(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<mul>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_invalid_kind
// CHECK-NOT: vector.fma
// CHECK: vector.contract

// -----

!vecA = vector<1x1x1xf32>
!vecB = vector<1x1x64xf32>
!vecC = vector<1x1x1x64xf32>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
func.func @negative_multiple_parallel_dims(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// Currently, this case does not lower due to simple matching rules.
// Multiple parallel dimensions are expected to be removed much earlier.
// TODO: Evaluate contraction canonicalization.

// CHECK-LABEL: @negative_multiple_parallel_dims
// CHECK-NOT: vector.fma
// CHECK: vector.contract

// -----

!vecA = vector<1x1xf32>
!vecB = vector<64x1xf32>
!vecC = vector<1x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @negative_matmul_transposed_b(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// Currently, this case does not lower due to simple matching rules.

// CHECK-LABEL: @negative_matmul_transposed_b
// CHECK-NOT: vector.fma
// CHECK: vector.contract

// -----

!vecA = vector<1x2xf32>
!vecB = vector<2x64xf32>
!vecC = vector<1x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @negative_matmul_not_outer_product(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_matmul_not_outer_product
// CHECK-NOT: vector.fma
// CHECK: vector.contract

// -----

!vecA = vector<3x1x1xf32>
!vecB = vector<3x1x64xf32>
!vecC = vector<3x1x64xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @negative_non_unit_batch_dim(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// Batch dimension should've been simplified earlier.

// CHECK-LABEL: @negative_non_unit_batch_dim
// CHECK-NOT: vector.fma
// CHECK: vector.contract

// -----

!vecA = vector<3x1x1xf32>
!vecB = vector<3x1x64xf32>
!vecC = vector<1x64xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @negative_non_unit_batch_reduce_dim(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// Batch-reduce dimension should've been simplified earlier.

// CHECK-LABEL: @negative_non_unit_batch_reduce_dim
// CHECK-NOT: vector.fma
// CHECK: vector.contract
