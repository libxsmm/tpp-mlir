// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void --sfc-order=false --def-parallel --parallel-task-grid=2,8 --print-mlir=late 2>&1 | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @entry(%arg0: tensor<8x32x32x32xf32>, %arg1: tensor<32x32x32x32xf32>, %arg2: tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg2 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<8x32x32x32xf32>
    return %0 : tensor<8x32x32x32xf32>
  }
}

// CHECK: func.func @_entry(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<8x32x32x32xf32>) {
// CHECK: %[[c2:.*]] = arith.constant 2 : index
// CHECK: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK: %[[c0:.*]] = arith.constant 0 : index
// CHECK: %[[c8:.*]] = arith.constant 8 : index
// CHECK: %[[c32:.*]] = arith.constant 32 : index
// CHECK: %[[c1:.*]] = arith.constant 1 : index
// CHECK: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK: %[[c1024_i64:.*]] = arith.constant 1024 : i64
// CHECK: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK: %[[temp0:.*]] = call @xsmm_brgemm_dispatch(%[[c1_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c0_i64]])
// CHECK:    omp.parallel {
// CHECK:      omp.wsloop {
// CHECK:        omp.loop_nest (%[[ARG3:.*]], %[[ARG4:.*]]) : index = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c8]]) collapse(2) {
// CHECK:          memref.alloca_scope  {
// CHECK:            scf.for %[[ARG5:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:	       %[[temp1:.*]] = arith.addi %[[ARG5]], %[[ARG3]] : index
// CHECK:              scf.for %[[ARG6:.*]] = %[[c0]] to %[[c8]] step %[[c1]] {
// CHECK:                %[[temp2:.*]] = arith.addi %[[ARG6]], %[[ARG4]] : index

