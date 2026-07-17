// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void \
// RUN:  -print-mlir=mid -seed=123 -splat-to-random 2>&1 | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// 1 layer MLP with compile-time weights and bias values.
func.func @entry(%arg0: tensor<4x8x8x8xbf16>, %output: tensor<4x8x8x8xbf16>) -> tensor<4x8x8x8xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %out_shape = tensor.empty() : tensor<4x8x8x8xbf16>
  %zero_init = linalg.fill ins(%cst : bf16) outs(%out_shape : tensor<4x8x8x8xbf16>) -> tensor<4x8x8x8xbf16>

  %weights = arith.constant dense<0.01> : tensor<8x8x8x8xbf16>
  %bias = arith.constant dense<0.4> : tensor<4x8x8x8xbf16>

  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %weights : tensor<4x8x8x8xbf16>, tensor<8x8x8x8xbf16>) outs(%zero_init : tensor<4x8x8x8xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<4x8x8x8xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %bias : tensor<4x8x8x8xbf16>, tensor<4x8x8x8xbf16>) outs(%out_shape : tensor<4x8x8x8xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<4x8x8x8xbf16>
  %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<4x8x8x8xbf16>) outs(%output : tensor<4x8x8x8xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<4x8x8x8xbf16>

  return %3 : tensor<4x8x8x8xbf16>
}

// Random initialization of splat tensors ensures that bias shape does not get folded
// due to compile time packing.
// CHECK-NOT: memref.global "private" constant @__constant_{{.*}}: memref<8x8xbf16>
// CHECK-DAG: memref.global "private" constant @__constant_{{.*}}: memref<4x8x8x8xbf16>
// CHECK-DAG: memref.global "private" constant @__constant_{{.*}}: memref<8x8x4x8x2xbf16>
// CHECK: xsmm_brgemm_invoke
// CHECK: xsmm_binary_invoke
// CHECK: xsmm_unary_invoke
