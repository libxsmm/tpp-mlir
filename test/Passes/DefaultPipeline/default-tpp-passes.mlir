// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s
// RUN: tpp-sched --bundle=default-tpp-passes %s --split-input-file | FileCheck %s

// CHECK: func.func @matmul(
// CHECK-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @matmul(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: call @xsmm_gemm_dispatch
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index
  // CHECK-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr

  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index
  // CHECK-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr

  // CHECK: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index
  // CHECK-NEXT: %[[cast_ptr2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[cast_ptr2]] : i64 to !llvm.ptr

  // CHECK: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]], %[[llvm_ptr2]], %[[C0]]
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>

  return %D : tensor<4x4xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: func.func @blocked_matmul(
// CHECK-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// CHECK-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// CHECK-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK: call @xsmm_brgemm_dispatch
  // CHECK: scf.parallel
  // CHECK:   %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // CHECK:   %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // CHECK:   %[[ptr2:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // CHECK:   call @xsmm_brgemm_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}, %[[ptr2]], %{{.+}}
  %1 = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%arg2 : tensor<4x8x32x32xf32>) {
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

// CHECK-LABEL: func.func @mlp(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// CHECK-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
func.func @mlp(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
  %arg2: tensor<512xf32>,  %output: tensor<128x512xf32>) -> tensor<128x512xf32> {

  // Identity:
  // CHECK: %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x512xf32>
  // CHECK: call @xsmm_unary_dispatch
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr

  // CHECK: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[alloc]]
  // CHECK-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr
  // CHECK: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]]

  // CHECK-DAG: call @xsmm_brgemm_dispatch
  // CHECK-DAG: call @xsmm_unary_dispatch
  // CHECK: scf.parallel
  %outShape = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%outShape : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x512xf32>

  // Matmul
  // CHECK: %[[ptr2:.+]] = memref.extract_aligned_pointer_as_index %{{.+}} : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> index
  // CHECK: %[[ptr2_cast:.+]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK: %[[llvm_ptr2:.+]] = llvm.inttoptr %[[ptr2_cast]] : i64 to !llvm.ptr

  // CHECK: %[[ptr3:.+]] = memref.extract_aligned_pointer_as_index %{{.+}} : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> index
  // CHECK: %[[ptr3_cast:.+]] = arith.index_cast %[[ptr3]] : index to i64
  // CHECK: %[[llvm_ptr3:.+]] = llvm.inttoptr %[[ptr3_cast]] : i64 to !llvm.ptr

  // CHECK: %[[ptr4:.+]] = memref.extract_aligned_pointer_as_index %{{.+}} : memref<32x32xf32, strided<[512, 1], offset: ?>> -> index
  // CHECK: %[[ptr4_cast:.+]] = arith.index_cast %[[ptr4]] : index to i64
  // CHECK: %[[llvm_ptr4:.+]] = llvm.inttoptr %[[ptr4_cast]] : i64 to !llvm.ptr

  // CHECK: call @xsmm_brgemm_invoke({{.*}}%[[llvm_ptr2]], %{{.+}}, %[[llvm_ptr3]], %{{.+}}, %[[llvm_ptr4]], %{{.+}}
  %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  // Relu
  // CHECK-NEXT: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr4]], %{{.+}}, %[[llvm_ptr4]], %{{.+}}
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
  // CHECK-COUNT-4: linalg.generic
  %softmax = linalg.softmax dimension(3)
    ins(%arg0: tensor<2x2x2x2xf32>) outs(%arg1: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %softmax : tensor<2x2x2x2xf32>
}

// CHECK-LABEL: batch_matmul_rewrite
func.func @batch_matmul_rewrite(%arg0: tensor<512x32x64xf32>, %arg1: tensor<512x64x32xf32>) -> tensor<512x32x32xf32> {
  %0 = tensor.empty() : tensor<512x32x32xf32>
  // CHECK-DAG: %[[C0_i:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1_i:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[C512_i:.+]] = arith.constant 512 : index
  // CHECK: %{{.+}} = call @xsmm_brgemm_dispatch
  // CHECK: scf.parallel{{.*}}(%[[C0_i]]) to (%[[C512_i]]) step (%[[C1_i]])
  // CHECK: xsmm_brgemm_invoke
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<512x32x64xf32>, tensor<512x64x32xf32>)
                           outs(%0 : tensor<512x32x32xf32>) -> tensor<512x32x32xf32>
  return %1 : tensor<512x32x32xf32>
}

// -----

func.func @linalg_copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  linalg.copy ins(%arg0 : memref<2x2xf32>) outs(%arg1 : memref<2x2xf32>)
  return
}

// CHECK-LABEL: linalg_copy
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2xf32>, %[[ARG1:.+]]: memref<2x2xf32>
// CHECK: xsmm_unary_dispatch
// CHECK: %[[PTR_0:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
// CHECK: %[[PTR_1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
// CHECK: xsmm_unary_invoke
