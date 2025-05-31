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

// 1x1 Conv2D shapes
!conv1x1_input_tensor_t  = tensor<1x7x7x2048xf32> // N,H,W,Ic
!conv1x1_filter_tensor_t = tensor<1x1x2048x512xf32> // H,W,Ic,Oc
!conv1x1_output_tensor_t = tensor<1x7x7x512xf32> // N,H,W,Oc

// CHECK-NOT: func.func private @xsmm_
// CHECK-LABEL: @conv2d_1x1(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1(
      %arg0 : !conv1x1_input_tensor_t) -> !conv1x1_output_tensor_t {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_9 = arith.constant dense<0.000000e+00> : !conv1x1_output_tensor_t

  // Conv2D weights
  %cst = arith.constant dense<0.00332225906> : !conv1x1_filter_tensor_t

  // 1x1 Conv2D
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       memref.store
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       scf.for
  // CHECK:         arith.mulf
  // CHECK:         arith.addf
  %0 = tensor.empty() : !conv1x1_output_tensor_t
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : !conv1x1_output_tensor_t) -> !conv1x1_output_tensor_t
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
              ins(%arg0, %cst : !conv1x1_input_tensor_t, !conv1x1_filter_tensor_t)
              outs(%1 : !conv1x1_output_tensor_t) -> !conv1x1_output_tensor_t

  // CHECK: return {{.*}} : memref<1x7x7x512xf32>
  return %2 : !conv1x1_output_tensor_t
}

// -----

#map = affine_map<(d0, d1) -> (d0 + d1)>

// CHECK-NOT: func.func private @xsmm_
// CHECK-LABEL: @conv2d_1x1_decomposed(
// CHECK-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1_decomposed(
      %arg0 : tensor<1x7x7x2048xf32>) -> tensor<1x7x7x512xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index

  // Conv2D weights
  %cst = arith.constant dense<0.00332225906> : tensor<2048x512xf32>

  // 1x1 Conv2D
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       memref.store
  // CHECK: scf.for
  // CHECK:   scf.for
  // CHECK:     scf.for
  // CHECK:       scf.for
  // CHECK:         arith.mulf
  // CHECK:         arith.addf
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x7x7x512xf32>
  %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x7x7x512xf32>) -> tensor<1x7x7x512xf32>
  %2 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %1) -> (tensor<1x7x7x512xf32>) {
    %3 = scf.for %arg3 = %c0 to %c7 step %c1 iter_args(%arg4 = %arg2) -> (tensor<1x7x7x512xf32>) {
      %4 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %arg4) -> (tensor<1x7x7x512xf32>) {
        %5 = scf.for %arg7 = %c0 to %c1 step %c1 iter_args(%arg8 = %arg6) -> (tensor<1x7x7x512xf32>) {
          %6 = affine.apply #map(%arg3, %arg5)
          %extracted_slice = tensor.extract_slice %arg0[%arg1, %6, %arg7, 0] [1, 1, 7, 2048] [1, 1, 1, 1] : tensor<1x7x7x2048xf32> to tensor<7x2048xf32>
          %extracted_slice_1 = tensor.extract_slice %arg8[%arg1, %arg3, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : tensor<1x7x7x512xf32> to tensor<7x512xf32>
          %7 = linalg.matmul ins(%extracted_slice, %cst : tensor<7x2048xf32>, tensor<2048x512xf32>) outs(%extracted_slice_1 : tensor<7x512xf32>) -> tensor<7x512xf32>
          %inserted_slice = tensor.insert_slice %7 into %arg8[%arg1, %arg3, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : tensor<7x512xf32> into tensor<1x7x7x512xf32>
          scf.yield %inserted_slice : tensor<1x7x7x512xf32>
        }
        scf.yield %5 : tensor<1x7x7x512xf32>
      }
      scf.yield %4 : tensor<1x7x7x512xf32>
    }
    scf.yield %3 : tensor<1x7x7x512xf32>
  }

  // CHECK: return {{.*}} : memref<1x7x7x512xf32>
  return %2 : tensor<1x7x7x512xf32>
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
