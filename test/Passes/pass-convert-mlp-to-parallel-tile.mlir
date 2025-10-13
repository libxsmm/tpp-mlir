// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void --def-parallel --parallel-task-grid=2,16  --print-mlir=late 2>&1 | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @entry(%arg0: tensor<8x32x32x32xf32>, %arg1: tensor<32x32x32x32xf32>, %arg2: tensor<32x32xf32>, %arg3: tensor<8x32x32x32xf32>, %arg4: tensor<32x32x32x32xf32>, %arg5: tensor<32x32xf32>, %arg6: tensor<8x32x32x32xf32>, %arg7: tensor<32x32x32x32xf32>, %arg8: tensor<32x32xf32>, %arg9: tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg3 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.mulf %in, %in_2 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<8x32x32x32xf32>
    %1 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<32x32xf32>) outs(%0 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<8x32x32x32xf32>) {
    ^bb0(%out: f32):
      %9 = arith.maximumf %out, %cst : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%2, %arg4 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg6 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.mulf %in, %in_2 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<8x32x32x32xf32>
    %4 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg5 : tensor<32x32xf32>) outs(%3 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %5 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4 : tensor<8x32x32x32xf32>) {
    ^bb0(%out: f32):
      %9 = arith.maximumf %out, %cst_0 : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%5, %arg7 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%arg9 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %9 = arith.mulf %in, %in_2 : f32
      %10 = arith.addf %out, %9 : f32
      linalg.yield %10 : f32
    } -> tensor<8x32x32x32xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg8 : tensor<32x32xf32>) outs(%6 : tensor<8x32x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %8 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%7 : tensor<8x32x32x32xf32>) {
    ^bb0(%out: f32):
      %9 = arith.maximumf %out, %cst_1 : f32
      linalg.yield %9 : f32
    } -> tensor<8x32x32x32xf32>
    return %8 : tensor<8x32x32x32xf32>
  }
}


//CHECK: func.func @_entry(%[[ARG0:.*]]: memref<8x32x32x32xf32>, %[[ARG1:.*]]: memref<32x32x32x32xf32>, %[[ARG2:.*]]: memref<32x32xf32>, %[[ARG3:.*]]: memref<8x32x32x32xf32>, %[[ARG4:.*]]: memref<32x32x32x32xf32>, %[[ARG5:.*]]: memref<32x32xf32>, %[[ARG6:.*]]: memref<8x32x32x32xf32>, %[[ARG7:.*]]: memref<32x32x32x32xf32>, %[[ARG8:.*]]: memref<32x32xf32>, %[[ARG9:.*]]: memref<8x32x32x32xf32>) {
//CHECK: %[[c16:.*]] = arith.constant 16 : index
//CHECK: %[[c2:.*]] = arith.constant 2 : index
//CHECK: %[[c32_i64:.*]] = arith.constant 32 : i64
//CHECK: %[[c0:.*]] = arith.constant 0 : index
//CHECK: %[[c8:.*]] = arith.constant 8 : index
//CHECK: %[[c32:.*]] = arith.constant 32 : index
//CHECK: %[[c1:.*]] = arith.constant 1 : index
//CHECK: %[[c1_i64:.*]] = arith.constant 1 : i64
//CHECK: %[[c1024_i64:.*]] = arith.constant 1024 : i64
//CHECK: %[[c0_i64:.*]] = arith.constant 0 : i64
//CHECK: %[[c5_i64:.*]] = arith.constant 5 : i64
//CHECK: %[[c4_i64:.*]] = arith.constant 4 : i64
//CHECK: %[[temp0:.*]] = call @xsmm_fused_brgemm_dispatch(%[[c1_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c32_i64]], %[[c1024_i64]], %[[c1024_i64]], %[[c0_i64]], %[[c0_i64]], %[[c5_i64]], %[[c4_i64]], %[[c1_i64]])
//CHECK:  omp.parallel {
//CHECK:      omp.wsloop {
//CHECK:        omp.loop_nest (%[[ARG10:.*]], %[[ARG11:.*]]) : index = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c16]]) collapse(2) {
//CHECK:          memref.alloca_scope  {
//CHECK:            scf.for %[[ARG12:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
//CHECK:             %[[temp1:.*]] = arith.addi %[[ARG12]], %[[ARG10]] : index
//CHECK:             scf.for %[[ARG13:.*]] = %[[c0]] to %[[c16]] step %[[c1]] {
//CHECK:                %[[temp2:.*]] = arith.addi %[[ARG13]], %[[ARG11]] : index
//CHECK:  omp.parallel {
//CHECK:      omp.wsloop {
//CHECK:        omp.loop_nest (%[[ARG10:.*]], %[[ARG11:.*]]) : index = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c16]]) collapse(2) {
//CHECK:          memref.alloca_scope  {
//CHECK:            scf.for %[[ARG12:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
//CHECK:             %[[temp1:.*]] = arith.addi %[[ARG12]], %[[ARG10]] : index
//CHECK:             scf.for %[[ARG13:.*]] = %[[c0]] to %[[c16]] step %[[c1]] {
//CHECK:                %[[temp2:.*]] = arith.addi %[[ARG13]], %[[ARG11]] : index
//CHECK:  omp.parallel {
//CHECK:      omp.wsloop {
//CHECK:        omp.loop_nest (%[[ARG10:.*]], %[[ARG11:.*]]) : index = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c16]]) collapse(2) {
//CHECK:          memref.alloca_scope  {
//CHECK:            scf.for %[[ARG12:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
//CHECK:             %[[temp1:.*]] = arith.addi %[[ARG12]], %[[ARG10]] : index
//CHECK:             scf.for %[[ARG13:.*]] = %[[c0]] to %[[c16]] step %[[c1]] {
//CHECK:                %[[temp2:.*]] = arith.addi %[[ARG13]], %[[ARG11]] : index

