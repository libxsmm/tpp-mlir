// RUN: tpp-opt %s -register-unroll -split-input-file | FileCheck %s

!vecA = vector<4x2xf32>
!vecB = vector<2x64xf32>
!vecC = vector<4x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @vector_matmul(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_gemm_unroll", [2, 32, 1]>
  >>}
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// Validate that all dimension are unrolled.

// CHECK-LABEL: @vector_matmul
// CHECK-COUNT-8: vector.contract{{.*}}: vector<2x1xf32>, vector<1x32xf32> into vector<2x32xf32>

// -----

!vecA = vector<3x4x2xf32>
!vecB = vector<3x2x64xf32>
!vecC = vector<3x4x64xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @vector_batch_matmul(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_gemm_unroll", [4, 64, 1]>
  >>}
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// Validate that batch dimension is unrolled by 1.

// CHECK-LABEL: @vector_batch_matmul
// CHECK-COUNT-6: vector.contract{{.*}}: vector<1x4x1xf32>, vector<1x1x64xf32> into vector<1x4x64xf32>

// -----

!vecA = vector<3x4x2xf32>
!vecB = vector<3x2x64xf32>
!vecC = vector<4x64xf32>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @vector_batch_reduce_matmul(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_gemm_unroll", [4, 64, 2]>
  >>}
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// Validate that batch reduce dimension is unrolled by 1.

// CHECK-LABEL: @vector_batch_reduce_matmul
// CHECK-COUNT-3: vector.contract{{.*}}: vector<1x4x2xf32>, vector<1x2x64xf32> into vector<4x64xf32>

// -----

!vecA = vector<3x2x1x2xbf16>
!vecB = vector<3x1x64x2xbf16>
!vecC = vector<2x64xbf16>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
func.func @vector_brgemm_vnni(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_gemm_unroll", [1, 64, 1]>
  >>}
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// Validate that VNNI dimension remains intact and other dims are unrolled.

// CHECK-LABEL: @vector_brgemm_vnni
// CHECK-COUNT-6: vector.contract{{.*}}: vector<1x1x1x2xbf16>, vector<1x1x64x2xbf16> into vector<1x64xbf16>

// -----

!matA = tensor<256x512xf32>
!matB = tensor<512x128xf32>
!matC = tensor<256x128xf32>
!vecA = vector<4x1xf32>
!vecB = vector<1x64xf32>
!vecC = vector<4x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @propagate_unroll_to_contract_reads_writes(
  %arg0: !matA, %arg1: !matB, %arg2: !matC, %idx: index) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_gemm_unroll", [2, 32, 1]>
  >>}
{
  %cst = arith.constant 0.0 : f32
  %lhs = vector.transfer_read %arg0[%idx, %idx], %cst
    {in_bounds = [true, true]} : !matA, !vecA
  %rhs = vector.transfer_read %arg1[%idx, %idx], %cst
    {in_bounds = [true, true]} : !matB, !vecB
  %acc = vector.transfer_read %arg2[%idx, %idx], %cst
    {in_bounds = [true, true]} : !matC, !vecC
  %res = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %lhs, %rhs, %acc
    : !vecA, !vecB into !vecC
  %out = vector.transfer_write %res, %arg2[%idx, %idx]
    {in_bounds = [true, true]} : !vecC, !matC
  return %out : !matC
}

// Validate that unroll shape is correctly propagated to contraction's read
// producers and write consumer, and that they are unrolled too.

// CHECK-LABEL: @propagate_unroll_to_contract_reads_writes
// CHECK-SAME: %[[arg0:.+]]: tensor<256x512xf32>,
// CHECK-SAME: %[[arg1:.+]]: tensor<512x128xf32>,
// CHECK-SAME: %[[arg2:.+]]: tensor<256x128xf32>,
// CHECK-COUNT-2: vector.transfer_read %[[arg0]]{{.*}}vector<2x1xf32>
// CHECK-COUNT-2: vector.transfer_read %[[arg1]]{{.*}}vector<1x32xf32>
// CHECK-COUNT-4: vector.transfer_read %[[arg2]]{{.*}}vector<2x32xf32>
// CHECK-COUNT-4: vector.contract{{.*}}: vector<2x1xf32>, vector<1x32xf32> into vector<2x32xf32>
// CHECK-COUNT-4: vector.transfer_write{{.*}}vector<2x32xf32>

// -----

!vecA = vector<4x2xf32>
!vecB = vector<2x64xf32>
!vecC = vector<4x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @negative_invalid_unroll_opt(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_gemm_unroll", [2]>
  >>}
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_invalid_unroll_opt
// CHECK-NOT: vector.extract_strided_slice
// CHECK-COUNT-1: vector.contract
// CHECK-NOT: vector.contract
// CHECK-NOT: vector.insert_strided_slice

// -----

!vecA = vector<4x2xf32>
!vecB = vector<2x64xf32>
!vecC = vector<4x64xf32>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @negative_invalid_kind(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_gemm_unroll", [2, 32, 1]>
  >>}
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
// CHECK-COUNT-1: vector.contract{{.*}}: vector<4x2xf32>, vector<2x64xf32> into vector<4x64xf32>

// -----

!vecA = vector<3x4x2xf32>
!vecB = vector<5x2x64xf32>
!vecC = vector<3x5x4x64xf32>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
func.func @negative_multiple_parallel_dims(
  %arg0: !vecA, %arg1: !vecB, %arg2: !vecC) -> !vecC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_gemm_unroll", [2, 32, 1]>
  >>}
{
  %0 = vector.contract {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"],
    kind = #vector.kind<add>}
    %arg0, %arg1, %arg2
    : !vecA, !vecB into !vecC
  return %0 : !vecC
}

// CHECK-LABEL: @negative_multiple_parallel_dims
// CHECK:      vector.contract
// CHECK-SAME: vector<3x4x2xf32>, vector<5x2x64xf32> into vector<3x5x4x64xf32>
