// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

func.func @fill_op(%arg0: memref<32x32xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[DIS]], %[[CST]], %[[ARG0]])

// -----

func.func @fill_op(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @fill_op(%arg0: memref<32x32xf32, strided<[32, 2], offset: ?>>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32, strided<[32, 2], offset: ?>>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @fill_op(%arg0: memref<32x32xf32>, %cst: f32) {
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32x32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @fill_op(%arg0: memref<32x32xbf16>) {
  %cst = arith.constant 0.0 : bf16
  linalg.fill ins(%cst : bf16) outs(%arg0 : memref<32x32xbf16>)
  return
}

// CHECK-LABEL: fill_op
// CHECK-SAME: %[[ARG0:.+]]: memref<32x32xbf16>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : bf16
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = bf16
// CHECK: xsmm.unary zero(data_type = bf16, %[[DIS]], %[[CST]], %[[ARG0]])

// -----

func.func @fill_op(%arg0: memref<32xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<32xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK: linalg.fill
// CHECK-NOT: xsmm.unary

// -----

func.func @transpose_op(%arg0: memref<3x5xf32>, %arg1: memref<5x3xf32>) {
  linalg.transpose ins(%arg0: memref<3x5xf32>) outs(%arg1: memref<5x3xf32>) permutation = [1, 0]
  return
}

// CHECK-LABEL: transpose_op
// CHECK-SAME: %[[ARG0:.+]]: memref<3x5xf32>, %[[ARG1:.+]]: memref<5x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch transpose [3, 5, 5, 3] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

func.func @transpose_op(%arg0: memref<5x3x5xf32>, %arg1: memref<5x5x3xf32>) {
  linalg.transpose ins(%arg0: memref<5x3x5xf32>) outs(%arg1: memref<5x5x3xf32>) permutation = [0, 2, 1]
  return
}

// CHECK-LABEL: transpose_op
// CHECK-NOT: xsmm.unary transpose
// CHECK: linalg.transpose

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu(%arg0: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<4x3xf32>) outs(%arg0 : memref<4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maxf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu
// CHECK-SAME: %[[ARG0:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 3, 3] flags = (none) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG0]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

func.func @relu_1(%arg0: memref<1x3xf32>, %arg1: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map1, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<1x3xf32>) outs(%arg1 : memref<4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maxf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_1
// CHECK-SAME: %[[ARG0:.+]]: memref<1x3xf32>, %[[ARG1:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 3, 3] flags = (bcast_col) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @relu_2(%arg0: memref<4x1xf32>, %arg1: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map1, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<4x1xf32>) outs(%arg1 : memref<4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maxf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_2
// CHECK-SAME: %[[ARG0:.+]]: memref<4x1xf32>, %[[ARG1:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 1, 3] flags = (bcast_row) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @relu_3(%arg0: f32, %arg1: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map1, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : f32) outs(%arg1 : memref<4x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maxf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_3
// CHECK-SAME: %[[ARG0:.+]]: f32, %[[ARG1:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 1, 3] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]])

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

func.func @relu_4(%arg0: f32, %arg1: memref<4x3xf32, strided<[?, ?], offset: 0>>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map1, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : f32) outs(%arg1 : memref<4x3xf32, strided<[?, ?], offset: 0>>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maxf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_4
// CHECK-NOT: xsmm.unary relu
// CHECK: linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu_5(%arg1: memref<4x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg1 : memref<4x3xf32>) {
    ^bb0(%in: f32):
      %0 = arith.maxf %in, %cst : f32
      linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: relu_5
// CHECK-SAME: %[[ARG1:.+]]: memref<4x3xf32>
// CHECK: %[[DIS:.+]] = xsmm.unary.dispatch relu [4, 3, 3, 3] flags = (none) data_type = f32
// CHECK: xsmm.unary relu(data_type = f32, %[[DIS]], %[[ARG1]], %[[ARG1]])
