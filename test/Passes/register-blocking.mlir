// RUN: tpp-opt %s -register-blocking -split-input-file | FileCheck %s

!matA = tensor<256x512xf32>
!matB = tensor<512x128xf32>
!matC = tensor<256x128xf32>
func.func @matmul(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  %0 = linalg.matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  return %0 : !matC
}

// CHECK-LABEL: @matmul
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
// Matrix C block loops for M and N dims
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C256]] step %[[C2]]
// CHECK:   scf.for %{{.+}} = %[[C0]] to %[[C128]] step %[[C32]]
// Reduction loop for K dim
// CHECK:      scf.for %{{.+}} = %[[C0]] to %[[C512]] step %[[C1]]
// CHECK:        linalg.matmul{{.*}}ins({{.*}}: tensor<2x1xf32>, tensor<1x32xf32>)
// CHECK-SAME:   -> tensor<2x32xf32>

// -----

!matA = tensor<16x256x512xf32>
!matB = tensor<16x512x128xf32>
!matC = tensor<16x256x128xf32>
func.func @batch_matmul(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 4]>
  >>}
{
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  return %0 : !matC
}

// Check that batch dimension is tiled by 1.
// Loops order should be as follows:
//   - batch dimension
//   - parallel dimensions
//   - reduction dimension

// CHECK-LABEL: @batch_matmul
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// Fallback - unit step batch dim loop
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C16]] step %[[C1]]
// CHECK-COUNT-3: scf.for
// CHECK:      linalg.batch_matmul
// CHECK-SAME:   ins({{.*}}: tensor<1x2x4xf32>, tensor<1x4x32xf32>)
// CHECK-SAME:   -> tensor<1x2x32xf32>

// -----

!matA = tensor<16x256x512xf32>
!matB = tensor<16x512x128xf32>
!matC = tensor<256x128xf32>
func.func @batch_reduce_matmul(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 4]>
  >>}
{
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  return %0 : !matC
}

// CHECK-LABEL: @batch_reduce_matmul
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// Matrix C block loops for M and N dims
// CHECK-COUNT-2: scf.for
// Fallback - unit step batch reduction dim loop
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C16]] step %[[C1]]
// Reduction loop for K dim
// CHECK-COUNT-1: scf.for
// CHECK:      linalg.batch_reduce_matmul
// CHECK-SAME:   ins({{.*}}: tensor<1x2x4xf32>, tensor<1x4x32xf32>)
// CHECK-SAME:   -> tensor<2x32xf32>

// -----

!matA = tensor<5x256x16x2xbf16>
!matB = tensor<5x16x128x2xbf16>
!matC = tensor<256x128xbf16>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
func.func @brgemm_vnni(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [4, 32, 1]>
  >>}
{
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) {
  ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
    %1 = arith.mulf %in, %in_1 : bf16
    %2 = arith.addf %out, %1 : bf16
    linalg.yield %2 : bf16
  } -> !matC
  return %0 : !matC
}

// VNNI dimension must remain untiled.
// To achieve that, generic's iterator loops, affine maps, and operands' shapes
// must be correctly matched.

// CHECK-LABEL: @brgemm_vnni
// CHECK-COUNT-4: scf.for
// CHECK:      linalg.generic
// CHECK-SAME:   ins({{.*}}: tensor<1x4x1x2xbf16>, tensor<1x1x32x2xbf16>)
// CHECK-SAME:   outs({{.*}}: tensor<4x32xbf16>)

// -----

!matA = tensor<512x256xf32>
!matB = tensor<512x128xf32>
!matC = tensor<256x128xf32>
func.func @matmul_transpose_a(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  %0 = linalg.matmul
    indexing_maps = [affine_map<(m, n, k) -> (k, m)>, // transpose
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (m, n)>]
    ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  return %0 : !matC
}

// CHECK-LABEL: @matmul_transpose_a
// CHECK-COUNT-3: scf.for
// CHECK:      linalg.matmul
// CHECK-SAME:   ins({{.*}}: tensor<1x2xf32>, tensor<1x32xf32>)
// CHECK-SAME:   outs({{.*}}: tensor<2x32xf32>)

// -----

!matA = tensor<256x512xf32>
!matB = tensor<512x128xf32>
!matC = tensor<256x128xf32>
func.func @peel_remainder_block(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [6, 32, 1]>
  >>}
{
  %0 = linalg.matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  return %0 : !matC
}

// Check that the last block is peeled when a dimension is not perfectly
// divisible by a tile sizes.
// This ensures there are no dynamic shapes after tiling.

// CHECK-LABEL: @peel_remainder_block(
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[C252:.+]] = arith.constant 252 : index
// Matrix C block loops for M and N dims [full tiles]
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C252]] step %[[C6]]
// CHECK:           scf.for %{{.+}} = %[[C0]] to %[[C128]] step %[[C32]]
// Reduction loop for K dim
// CHECK:      scf.for
// CHECK:        linalg.matmul
// CHECK-SAME:     ins({{.*}}: tensor<6x1xf32>, tensor<1x32xf32>)
// CHECK-SAME:     outs({{.*}}: tensor<6x32xf32>)
// Tail block loops [remainder tile]
// Peeled M dim block offsets and shapes are statically know and don't require a loop.
// Thus, only loops for the N and the K dims are present here.
// CHECK-COUNT-2: scf.for
// CHECK:           linalg.matmul
// CHECK-SAME:        ins({{.*}}: tensor<4x1xf32>, tensor<1x32xf32>)
// CHECK-SAME:        outs({{.*}}: tensor<4x32xf32>)

// -----

!matA = tensor<3x5x256x16x2xbf16>
!matB = tensor<3x5x16x128x2xbf16>
!matC = tensor<256x128xbf16>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5, d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d3)>
func.func @tile_multiple_reduction_dims_vnni(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [8, 32, 4]>
  >>}
{
  %0 = linalg.contract
    indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  return %0 : !matC
}

// Validate that:
//   - parallel loops are tiled as specified
//   - K-dim is tiled as specified
//   - VNNI dim remains untiled
//   - other reduction dimensions are tiled by one

// CHECK-LABEL: @tile_multiple_reduction_dims_vnni
// CHECK-COUNT-5: scf.for
// CHECK:      linalg.contract
// CHECK-SAME:   ins({{.*}}: tensor<1x1x8x4x2xbf16>, tensor<1x1x4x32x2xbf16>)
// CHECK-SAME:   outs({{.*}}: tensor<8x32xbf16>)

// -----

!matA = tensor<4x48xf32>
!matB = tensor<48x128xf32>
!matC = tensor<4x128xf32>
func.func @tile_small_dims(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [6, 32, 64]>
  >>}
{
  %0 = linalg.matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  return %0 : !matC
}

// Validate that partial tiling is still performed when dimensions
// smaller than tiles sizes are present.
// Expects to tile N dim, and leave M and K dims untiled.

// CHECK-LABEL: @tile_small_dims
// CHECK-COUNT-1: scf.for
// CHECK:      linalg.matmul
// CHECK-SAME:   ins({{.*}}: tensor<4x48xf32>, tensor<48x32xf32>)
// CHECK-SAME:   outs({{.*}}: tensor<4x32xf32>)

// -----

!matA = tensor<4x32x64x16xf32>
!matB = tensor<4x32x16x32xf32>
!matC = tensor<4x4x64x32xf32>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
func.func @negative_tile_multiple_parallel_dims(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.mulf %in, %in_1 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> !matC
  return %0 : !matC
}

// CHECK-LABEL: @negative_tile_multiple_parallel_dims
// CHECK-NOT: scf.for
// CHECK:     linalg.generic

// -----

!matA = tensor<256x512xf32>
!matB = tensor<512x128xf32>
!matC = tensor<256x128xf32>
func.func @negative_invalid_blocking_opt(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2]>
  >>}
{
  %0 = linalg.matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  return %0 : !matC
}

// CHECK-LABEL: @negative_invalid_blocking_opt
// CHECK-NOT: scf.for
// CHECK:     linalg.matmul

// -----

!matA = memref<256x512xf32>
!matB = memref<512x128xf32>
!matC = memref<256x128xf32>
func.func @negative_tile_memref(
  %arg0: !matA, %arg1: !matB, %arg2: !matC)
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  linalg.matmul ins(%arg0, %arg1 : !matA, !matB) outs(%arg2 : !matC)
  return
}

// CHECK-LABEL: @negative_tile_memref
// CHECK-NOT: scf.for
// CHECK:     linalg.matmul

// -----

!matA = memref<256x?xf32>
!matB = memref<?x128xf32>
!matC = memref<256x128xf32>
func.func @negative_tile_dynamic_shapes(
  %arg0: !matA, %arg1: !matB, %arg2: !matC)
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  linalg.matmul ins(%arg0, %arg1 : !matA, !matB) outs(%arg2 : !matC)
  return
}

// CHECK-LABEL: @negative_tile_dynamic_shapes
// CHECK-NOT: scf.for
// CHECK:     linalg.matmul

// -----

!matA = tensor<256x512x3xf32>
!matB = tensor<512x128x3xf32>
!matC = tensor<256x128xf32>
func.func @negative_tile_non_projected_permutation(
  %arg0: !matA, %arg1: !matB, %arg2: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  %0 = linalg.generic {
    indexing_maps = [affine_map<(m, n, k) -> (m, k, 1)>,
                     affine_map<(m, n, k) -> (k, n, 1)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.mulf %in, %in_1 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> !matC
  return %0 : !matC
}

// CHECK-LABEL: @negative_tile_non_projected_permutation
// CHECK-NOT: scf.for
// CHECK:     linalg.generic

// -----

!matA = tensor<256x512xf32>
!matB = tensor<512x128xf32>
!matC = tensor<256x128xf32>
func.func @fuse_eltwise_consumer_and_initializator(
  %arg0: !matA, %arg1: !matB, %bias: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  %c0 = arith.constant 0.0 : f32
  %e = tensor.empty() : !matC
  %acc = linalg.fill ins(%c0 : f32) outs(%e : !matC) -> !matC
  %0 = linalg.matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%acc : !matC) -> !matC
  %1 = linalg.add ins(%0, %bias : !matC, !matC)
    outs(%e : !matC) -> !matC
  return %1 : !matC
}

// CHECK-LABEL: @fuse_eltwise_consumer_and_initializator
// CHECK-COUNT-2: scf.for
// CHECK: linalg.fill{{.*}}-> tensor<2x32xf32>
// CHECK: scf.for
// CHECK:   linalg.matmul{{.*}}-> tensor<2x32xf32>
// CHECK:   scf.yield
// CHECK: linalg.add{{.*}}-> tensor<2x32xf32>

// -----

!mat = tensor<256x256xf32>
func.func @fuse_only_eltwise_consumers(
  %arg0: !mat, %arg1: !mat, %arg2: !mat, %arg3: !mat) -> !mat
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  %e = tensor.empty() : !mat
  %acc = linalg.sub ins(%arg0, %arg1 : !mat, !mat)
    outs(%e : !mat) -> !mat
  %0 = linalg.matmul ins(%arg0, %arg1 : !mat, !mat)
    outs(%acc : !mat) -> !mat
  %1 = linalg.add ins(%0, %arg3 : !mat, !mat)
    outs(%e : !mat) -> !mat
  %2 = linalg.matmul ins(%1, %arg1 : !mat, !mat)
    outs(%arg2 : !mat) -> !mat
  %3 = linalg.mul ins(%2, %arg3 : !mat, !mat)
    outs(%e : !mat) -> !mat
  return %3 : !mat
}

// CHECK-LABEL: @fuse_only_eltwise_consumers
// Unfused eltwise producer
// CHECK: linalg.sub{{.*}}-> tensor<256x256xf32>
// Tiled and fused first matmul
// CHECK-COUNT-2: scf.for
// CHECK: scf.for
// CHECK:   linalg.matmul{{.*}}-> tensor<2x32xf32>
// CHECK:   scf.yield
// CHECK: linalg.add{{.*}}-> tensor<2x32xf32>
// Tiled and fused second matmul
// CHECK-COUNT-2: scf.for
// CHECK: scf.for
// CHECK:   linalg.matmul{{.*}}-> tensor<2x32xf32>
// CHECK:   scf.yield
// CHECK: linalg.mul{{.*}}-> tensor<2x32xf32>

// -----

!matA = tensor<256x512xf32>
!matB = tensor<512x128xf32>
!matC = tensor<256x128xf32>
func.func @negative_fuse_multi_use_consumer(
  %arg0: !matA, %arg1: !matB, %arg2: !matC, %bias: !matC) -> !matC
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  %0 = linalg.matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  %1 = linalg.add ins(%0, %bias : !matC, !matC)
    outs(%0 : !matC) -> !matC
  return %1 : !matC
}

// Do not fuse consumers with multiple uses (even if it has only one user)
// to avoid introducing recomputations.

// CHECK-LABEL: @negative_fuse_multi_use_consumer
// CHECK-COUNT-3: scf.for
// CHECK:   linalg.matmul{{.*}}-> tensor<2x32xf32>
// CHECK:   scf.yield
// CHECK: linalg.add{{.*}}-> tensor<256x128xf32>

// -----

!matA = tensor<256x512xf32>
!matB = tensor<512x128xf32>
!matC = tensor<256x128xf32>
!matD = tensor<128x256xf32>
func.func @negative_fuse_transposed_consumer(
  %arg0: !matA, %arg1: !matB, %arg2: !matC, %bias: !matC) -> !matD
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>
  >>}
{
  %e = tensor.empty() : !matD
  %0 = linalg.matmul ins(%arg0, %arg1 : !matA, !matB)
    outs(%arg2 : !matC) -> !matC
  %1 = linalg.generic {
    indexing_maps = [affine_map<(m, n) -> (m, n)>,
                     affine_map<(m, n) -> (m, n)>,
                     affine_map<(m, n) -> (n, m)>], // transpose
    iterator_types = ["parallel", "parallel"]}
    ins(%0, %bias : !matC, !matC)
    outs(%e : !matD) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %2 = arith.addf %in, %in_1 : f32
    linalg.yield %2 : f32
  } -> !matD
  return %1 : !matD
}

// CHECK-LABEL: @negative_fuse_transposed_consumer
// CHECK-COUNT-3: scf.for
// CHECK:   linalg.matmul{{.*}}-> tensor<2x32xf32>
// CHECK:   scf.yield
// CHECK: linalg.generic{{.*}}outs({{.*}}: tensor<128x256xf32>)
