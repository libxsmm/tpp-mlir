// RUN: tpp-run %s -e entry -entry-point-result=void \
// RUN:  -print-mlir=mid -seed 123 -splat-to-random 2>&1 | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @entry(%arg0: tensor<8x8xf32>, %output: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %c0 = arith.constant 0.0 : f32
  %out_shape = tensor.empty() : tensor<8x8xf32>
  %zero_init = linalg.fill ins(%c0 : f32) outs(%output : tensor<8x8xf32>) -> tensor<8x8xf32>

  %weights0 = arith.constant dense<0.01> : tensor<8x8xf32>
  %weights1 = arith.constant dense<0.02> : tensor<8x8xf32>
  %bias0 = arith.constant dense<0.1> : tensor<8xf32>
  %bias1 = arith.constant dense<0.2> : tensor<8xf32>

  %0 = linalg.matmul ins(%arg0, %weights0 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%zero_init : tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = linalg.generic {indexing_maps = [#map1, #map0, #map1], iterator_types = ["parallel", "parallel"]}
    ins(%0, %bias0 : tensor<8x8xf32>, tensor<8xf32>)
    outs(%out_shape : tensor<8x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %add = arith.addf %in, %in_0 : f32
      linalg.yield %add : f32
  } -> tensor<8x8xf32>
  %2 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
    ins(%1 : tensor<8x8xf32>)
    outs(%out_shape : tensor<8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.maximumf %in, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<8x8xf32>

  %3 = linalg.matmul ins(%2, %weights1 : tensor<8x8xf32>, tensor<8x8xf32>) outs(%zero_init : tensor<8x8xf32>) -> tensor<8x8xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map0, #map1], iterator_types = ["parallel", "parallel"]}
    ins(%3, %bias1 : tensor<8x8xf32>, tensor<8xf32>)
    outs(%out_shape : tensor<8x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %add = arith.addf %in, %in_0 : f32
      linalg.yield %add : f32
  } -> tensor<8x8xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
    ins(%4 : tensor<8x8xf32>)
    outs(%out_shape : tensor<8x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %16 = arith.maximumf %in, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<8x8xf32>
  return %5 : tensor<8x8xf32>
}

// Ensure that each weight and bias gets their own global buffer.
// CHECK-DAG: memref.global "private" constant @__constant_1x1x8x8xf32 : memref<1x1x8x8xf32>
// CHECK-DAG: memref.global "private" constant @__constant_1x1x8x8xf32_0 : memref<1x1x8x8xf32>
// CHECK-DAG: memref.global "private" constant @__constant_8xf32 : memref<8xf32>
// CHECK-DAG: memref.global "private" constant @__constant_8xf32_1 : memref<8xf32>

// Randomized input.
// CHECK-DAG: memref.global "private" @__wrapper_0 : memref<8x8xf32>
// CHECK-LABEL: @entry
// CHECK: %[[input:.+]] = memref.get_global @__wrapper_0
// CHECK: call @_entry(%[[input]]
