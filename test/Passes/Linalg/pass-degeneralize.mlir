// RUN: tpp-opt %s -linalg-generalize-named-ops -linalg-degeneralize-generic-ops -split-input-file | FileCheck %s

func.func @degeneralize(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = tensor.empty() : tensor<3x3xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%arg1 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %4 = linalg.matmul ins(%1, %2 : tensor<3x3xf32>, tensor<3x3xf32>)
                     outs(%3 : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %4 : tensor<3x3xf32>
}

// CHECK-LABEL: degeneralize
// CHECK-SAME: %[[ARG0:.+]]: tensor<3x3xf32>, %[[ARG1:.+]]: tensor<3x3xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK: %[[FILL_ARG0:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[ARG0]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %[[FILL_ARG1:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[ARG1]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %[[OUT:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %{{.+}} = linalg.matmul ins(%[[FILL_ARG0]], %[[FILL_ARG1]] : tensor<3x3xf32>, tensor<3x3xf32>)
// CHECK-SAME:  outs(%[[OUT]] : tensor<3x3xf32>) -> tensor<3x3xf32>

// -----

func.func @degeneralize(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) -> memref<3x3xf32> {
  %alloc = memref.alloc() : memref<3x3xf32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<3x3xf32>)
  linalg.fill ins(%cst : f32) outs(%arg1 : memref<3x3xf32>)
  linalg.fill ins(%cst : f32) outs(%alloc : memref<3x3xf32>)
  linalg.matmul ins(%arg0, %arg1 : memref<3x3xf32>, memref<3x3xf32>)
                outs(%alloc : memref<3x3xf32>)
  return %alloc : memref<3x3xf32>
}

// CHECK-LABEL: degeneralize
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3x3xf32>
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ARG0]] : memref<3x3xf32>)
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ARG1]] : memref<3x3xf32>)
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<3x3xf32>)
// CHECK: linalg.matmul ins(%[[ARG0]], %[[ARG1]] : memref<3x3xf32>, memref<3x3xf32>)
// CHECK-SAME:  outs(%[[ALLOC]] : memref<3x3xf32>)

// -----

func.func @degeneralize(%arg0: memref<3x3x3xf32>, %arg1: memref<3x3x3xf32>) -> memref<3x3xf32> {
  %alloc = memref.alloc() : memref<3x3xf32>
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%alloc : memref<3x3xf32>)
  linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<3x3x3xf32>, memref<3x3x3xf32>)
                             outs(%alloc : memref<3x3xf32>)
  return %alloc : memref<3x3xf32>
}

// CHECK-LABEL: degeneralize
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3x3xf32>, %[[ARG1:.+]]: memref<3x3x3xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<3x3xf32>
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC]] : memref<3x3xf32>)
// CHECK: linalg.batch_reduce_matmul ins(%[[ARG0]], %[[ARG1]] : memref<3x3x3xf32>, memref<3x3x3xf32>)
// CHECK-SAME:  outs(%[[ALLOC]] : memref<3x3xf32>)

// -----

func.func @degeneralize(%arg0: tensor<3x3x3xf32>, %arg1: tensor<3x3x3xf32>) -> tensor<3x3xf32> {
  %0 = tensor.empty() : tensor<3x3xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %2 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<3x3x3xf32>, tensor<3x3x3xf32>)
                                  outs(%1 : tensor<3x3xf32>) -> tensor<3x3xf32>
  return %2 : tensor<3x3xf32>
}

// CHECK-LABEL: degeneralize
// CHECK-SAME:  %[[ARG0:.+]]: tensor<3x3x3xf32>, %[[ARG1:.+]]: tensor<3x3x3xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: %{{.+}} = linalg.batch_reduce_matmul ins(%[[ARG0]], %[[ARG1]] : tensor<3x3x3xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:  outs(%[[FILL]] : tensor<3x3xf32>) -> tensor<3x3xf32>

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @transpose_degeneralize(%arg0 : tensor<128x262144xf32>, %arg1: tensor<262144x128xf32>) 
    -> tensor<262144x128xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1], 
    iterator_types = ["parallel", "parallel"]} 
  ins(%arg0 : tensor<128x262144xf32>) outs(%arg1 : tensor<262144x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<262144x128xf32>
  return %0 : tensor<262144x128xf32>
}

// CHECK-LABEL: transpose_degeneralize
// CHECK-SAME: %[[ARG0:.+]]: tensor<128x262144xf32>, %[[ARG1:.+]]: tensor<262144x128xf32>
// CHECK: %[[T:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<128x262144xf32>) outs(%[[ARG1]] : tensor<262144x128xf32>) 
// CHECK-SAME:  permutation = [1, 0]
// CHECK: return %[[T]] : tensor<262144x128xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @transpose_degeneralize_1(%arg0 : tensor<1x2x3x4xf32>, %arg1 : tensor<1x3x2x4xf32>)
    -> tensor<1x3x2x4xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  ins(%arg0 : tensor<1x2x3x4xf32>) outs(%arg1 : tensor<1x3x2x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x3x2x4xf32>
  return %0 : tensor<1x3x2x4xf32>
}

// CHECK-LABEL: transpose_degeneralize_1
// CHECK-SAME: %[[ARG0:.+]]: tensor<1x2x3x4xf32>, %[[ARG1:.+]]: tensor<1x3x2x4xf32>
// CHECK: %[[T:.+]] = linalg.transpose ins(%[[ARG0]] : tensor<1x2x3x4xf32>) outs(%[[ARG1]] : tensor<1x3x2x4xf32>) 
// CHECK-SAME:  permutation = [0, 2, 1, 3]
// CHECK: return %[[T]] : tensor<1x3x2x4xf32>

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @transpose_degeneralize_memref(%arg0 : memref<128x262144xf32>, %arg1: memref<262144x128xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg0 : memref<128x262144xf32>) outs(%arg1 : memref<262144x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

// CHECK-LABEL: transpose_degeneralize_memref
// CHECK-SAME: %[[ARG0:.+]]: memref<128x262144xf32>, %[[ARG1:.+]]: memref<262144x128xf32>
// CHECK: linalg.transpose ins(%[[ARG0]] : memref<128x262144xf32>) outs(%[[ARG1]] : memref<262144x128xf32>) 
// CHECK-SAME:  permutation = [1, 0]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @transpose_degeneralize_copy(%arg0 : tensor<128x262144xf32>, %arg1: tensor<128x262144xf32>) 
    -> tensor<128x262144xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]}
  ins(%arg0 : tensor<128x262144xf32>) outs(%arg1 : tensor<128x262144xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<128x262144xf32>
  return %0 : tensor<128x262144xf32>
}

// CHECK-LABEL: transpose_degeneralize_copy
// CHECK-NOT: linalg.transpose
// CHECK: linalg.generic
