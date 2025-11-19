// RUN: tpp-opt %s -default-tpp-passes="linalg-to-loops" -split-input-file | FileCheck %s
// RUN: tpp-sched --bundle=default-tpp-passes --linalg-to-loops %s --split-input-file | FileCheck %s

// CHECK-NOT: func.func private @xsmm_
// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @matmul(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       arith.mulf
  // CHECK:       arith.addf
  %D = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>

  return %D : tensor<4x4xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-NOT: func.func private @xsmm_
// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// CHECK-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       scf.for
  // CHECK:         scf.for
  // CHECK:           scf.for
  // CHECK:             arith.mulf
  // CHECK:             arith.addf
  %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<4x8x32x32xf32>

  return %1 :  tensor<4x8x32x32xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-NOT: func.func private @xsmm_
// CHECK: func.func @mlp(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// CHECK-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
func.func @mlp(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
  %arg2: tensor<512xf32>,  %output: tensor<128x512xf32>) -> tensor<128x512xf32> {

  // Identity
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     memref.load
  // CHECK:     memref.store
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%output : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x512xf32>

  // Matmul
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       arith.mulf
  // CHECK:       arith.addf
  %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  // Relu
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     arith.maximumf
  %c0 = arith.constant 0.0 : f32
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maximumf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x512xf32>

  return %3 : tensor<128x512xf32>
}

// -----

// CHECK-LABEL: softmax
func.func @softmax(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // CHECK-NOT: linalg.softmax
  %softmax = linalg.softmax dimension(3)
    ins(%arg0: tensor<2x2x2x2xf32>) outs(%arg1: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %softmax : tensor<2x2x2x2xf32>
}
