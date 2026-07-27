// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s --check-prefix=XSMM
// RUN: tpp-opt %s -default-tpp-passes="linalg-to-loops" -split-input-file | FileCheck %s --check-prefix=LOOP

// XSMM: func.func @matmul(
// XSMM-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// XSMM-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// XSMM-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
// LOOP-NOT: func.func private @xsmm_
// LOOP: func.func @matmul(
// LOOP-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// LOOP-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// LOOP-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @matmul(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // LOOP: scf.for
  // LOOP:   scf.for
  // LOOP:     scf.for
  // LOOP:       arith.mulf
  // LOOP:       arith.addf
  // XSMM: %[[C0:.+]] = arith.constant 0 : index
  // XSMM: call @xsmm_gemm_dispatch
  // XSMM: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index
  // XSMM-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr

  // XSMM: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index
  // XSMM-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr

  // XSMM: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index
  // XSMM-NEXT: %[[cast_ptr2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[cast_ptr2]] : i64 to !llvm.ptr

  // XSMM: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]], %[[llvm_ptr2]], %[[C0]]
  %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>

  return %D : tensor<4x4xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// XSMM-LABEL: func.func @blocked_matmul(
// XSMM-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// XSMM-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// XSMM-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>)
// LOOP-NOT: func.func private @xsmm_
// LOOP-LABEL: func.func @blocked_matmul(
// LOOP-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// LOOP-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// LOOP-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>)
func.func @blocked_matmul(%arg0: tensor<4x16x32x32xf32>, %arg1: tensor<8x16x32x32xf32>, %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // XSMM: call @xsmm_brgemm_dispatch
  // XSMM: scf.parallel
  // XSMM:   %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   %[[ptr2:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   call @xsmm_brgemm_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}, %[[ptr2]], %{{.+}}
  // LOOP: scf.for
  // LOOP:   scf.for
  // LOOP:     scf.for
  // LOOP:       scf.for
  // LOOP:         scf.for
  // LOOP:           scf.for
  // LOOP:             arith.mulf
  // LOOP:             arith.addf
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

// XSMM-LABEL: func.func @mlp(
// XSMM-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// XSMM-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// XSMM-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// XSMM-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
// LOOP-NOT: func.func private @xsmm_
// LOOP: func.func @mlp(
// LOOP-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// LOOP-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// LOOP-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// LOOP-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
func.func @mlp(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>,
  %arg2: tensor<512xf32>,  %output: tensor<128x512xf32>) -> tensor<128x512xf32> {

  // Identity
  // LOOP: scf.for
  // LOOP:   scf.for
  // LOOP:     memref.load
  // LOOP:     memref.store
  // Matmul
  // LOOP: scf.for
  // LOOP:   scf.for
  // LOOP:     scf.for
  // LOOP:       arith.mulf
  // LOOP:       arith.addf
  // Relu
  // LOOP: scf.for
  // LOOP:   scf.for
  // LOOP:     arith.maximumf

  // Identity:
  // XSMM: %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x512xf32>
  // XSMM: call @xsmm_unary_dispatch
  // XSMM: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // XSMM-NEXT: %[[cast_ptr0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[cast_ptr0]] : i64 to !llvm.ptr

  // XSMM: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[alloc]]
  // XSMM-NEXT: %[[cast_ptr1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[cast_ptr1]] : i64 to !llvm.ptr
  // XSMM: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]]

  // XSMM-DAG: call @xsmm_brgemm_dispatch
  // XSMM-DAG: call @xsmm_unary_dispatch
  // XSMM: scf.parallel
  %outShape = tensor.empty() : tensor<128x512xf32>
  %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<512xf32>) outs(%outShape : tensor<128x512xf32>) {
    ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
  } -> tensor<128x512xf32>

  // Matmul
  // XSMM: %[[ptr2:.+]] = memref.extract_aligned_pointer_as_index %{{.+}} : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> index
  // XSMM: %[[ptr2_cast:.+]] = arith.index_cast %[[ptr2]] : index to i64
  // XSMM: %[[llvm_ptr2:.+]] = llvm.inttoptr %[[ptr2_cast]] : i64 to !llvm.ptr

  // XSMM: %[[ptr3:.+]] = memref.extract_aligned_pointer_as_index %{{.+}} : memref<8x32x32xf32, strided<[1024, 32, 1], offset: ?>> -> index
  // XSMM: %[[ptr3_cast:.+]] = arith.index_cast %[[ptr3]] : index to i64
  // XSMM: %[[llvm_ptr3:.+]] = llvm.inttoptr %[[ptr3_cast]] : i64 to !llvm.ptr

  // XSMM: %[[ptr4:.+]] = memref.extract_aligned_pointer_as_index %{{.+}} : memref<32x32xf32, strided<[512, 1], offset: ?>> -> index
  // XSMM: %[[ptr4_cast:.+]] = arith.index_cast %[[ptr4]] : index to i64
  // XSMM: %[[llvm_ptr4:.+]] = llvm.inttoptr %[[ptr4_cast]] : i64 to !llvm.ptr

  // XSMM: call @xsmm_brgemm_invoke({{.*}}%[[llvm_ptr2]], %{{.+}}, %[[llvm_ptr3]], %{{.+}}, %[[llvm_ptr4]], %{{.+}}
  %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>) outs(%1 : tensor<128x512xf32>) attrs =  {iterator_ranges = [128, 512, 256]} {
    ^bb0(%arg9: f32, %arg10: f32, %arg11: f32):
      %16 = arith.mulf %arg9, %arg10 : f32
      %17 = arith.addf %arg11, %16 : f32
      linalg.yield %17 : f32
  } -> tensor<128x512xf32>

  // Relu
  // XSMM-NEXT: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr4]], %{{.+}}, %[[llvm_ptr4]], %{{.+}}
  %c0 = arith.constant 0.0 : f32
  %3 = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} outs(%2 : tensor<128x512xf32>) {
    ^bb0(%arg9: f32):
      %16 = arith.maximumf %arg9, %c0 : f32
      linalg.yield %16 : f32
  } -> tensor<128x512xf32>

  return %3 : tensor<128x512xf32>
}

// -----

// XSMM-LABEL: softmax
// LOOP-LABEL: softmax
func.func @softmax(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // XSMM-NOT: linalg.softmax
  // XSMM-COUNT-4: linalg.generic
  // LOOP-NOT: linalg.softmax
  %softmax = linalg.softmax dimension(3)
    ins(%arg0: tensor<2x2x2x2xf32>) outs(%arg1: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  return %softmax : tensor<2x2x2x2xf32>
}

// XSMM-LABEL: batch_matmul_rewrite
func.func @batch_matmul_rewrite(%arg0: tensor<512x32x64xf32>, %arg1: tensor<512x64x32xf32>) -> tensor<512x32x32xf32> {
  %0 = tensor.empty() : tensor<512x32x32xf32>
  // XSMM-DAG: %[[C0_i:.+]] = arith.constant 0 : index
  // XSMM-DAG: %[[C1_i:.+]] = arith.constant 1 : index
  // XSMM-DAG: %[[C512_i:.+]] = arith.constant 512 : index
  // XSMM: %{{.+}} = call @xsmm_brgemm_dispatch
  // XSMM: scf.parallel{{.*}}(%[[C0_i]]) to (%[[C512_i]]) step (%[[C1_i]])
  // XSMM: xsmm_brgemm_invoke
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<512x32x64xf32>, tensor<512x64x32xf32>)
                           outs(%0 : tensor<512x32x32xf32>) -> tensor<512x32x32xf32>
  return %1 : tensor<512x32x32xf32>
}

// -----

func.func @linalg_copy(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  linalg.copy ins(%arg0 : memref<2x2xf32>) outs(%arg1 : memref<2x2xf32>)
  return
}

// XSMM-LABEL: linalg_copy
// XSMM-SAME: %[[ARG0:.+]]: memref<2x2xf32>, %[[ARG1:.+]]: memref<2x2xf32>
// XSMM: xsmm_unary_dispatch
// XSMM: %[[PTR_0:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
// XSMM: %[[PTR_1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
// XSMM: xsmm_unary_invoke

// -----


func.func @fill_op(%arg0: memref<3x3xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<3x3xf32>)
  return
}

// XSMM-LABEL: fill_op
// XSMM-SAME: %[[ARG0:.+]]: memref<3x3xf32>
// XSMM-DAG: %[[C0:.+]] = arith.constant 0 : index
// XSMM-DAG: %[[C2:.+]] = arith.constant 2 : i64
// XSMM-DAG: %[[C1:.+]] = arith.constant 1 : i64
// XSMM-DAG: %[[C3:.+]] = arith.constant 3 : i64
// XSMM-DAG: %[[C8:.+]] = arith.constant 8 : i64
// XSMM-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// XSMM: %[[DIS:.+]] = call @xsmm_unary_dispatch(%[[C2]], %[[C1]], %[[C3]], %[[C3]], %[[C1]], %[[C3]], %[[C8]])
// XSMM: %[[PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<3x3xf32> -> index
// XSMM: %[[PTR_TO_INT:.+]] = arith.index_cast %[[PTR]] : index to i64
// XSMM: %[[LLVM_PTR:.+]] = llvm.inttoptr %[[PTR_TO_INT]] : i64 to !llvm.ptr
// XSMM: call @xsmm_unary_scalar_invoke(%[[C1]], %[[DIS]], %[[CST]], %[[LLVM_PTR]], %[[C0]])

// -----

func.func @fill_op_i32(%arg0: memref<3x3xi32>) {
  %cst = arith.constant 0 : i32
  linalg.fill ins(%cst : i32) outs(%arg0 : memref<3x3xi32>)
  return
}

// XSMM-LABEL: fill_op_i32
// XSMM-NOT: xsmm
// XSMM: linalg.fill

// -----

func.func @gemm_with_zero(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<3x3xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %mul = linalg.matmul ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>)
                       outs(%fill: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %mul : tensor<3x3xf32>
}

// XSMM-LABEL: gemm_with_zero
// XSMM-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>
// XSMM-DAG: %[[C0:.+]] = arith.constant 0 : index
// XSMM-DAG: %[[C1:.+]] = arith.constant 1 : i64
// XSMM-DAG: %[[C3:.+]] = arith.constant 3 : i64
// XSMM-DAG: %[[C4:.+]] = arith.constant 4 : i64
// XSMM-NOT: xsmm_unary_dispatch
// XSMM: %[[DIS:.+]] = call @xsmm_gemm_dispatch(%[[C1]], %[[C3]], %[[C3]], %[[C3]], %[[C3]], %[[C3]], %[[C3]], %[[C4]])
// XSMM: %[[INT_PTR_ARG0:.+]] = memref.extract_aligned_pointer_as_index
// XSMM: %[[CAST_ARG0:.+]] = arith.index_cast %[[INT_PTR_ARG0]] : index to i64
// XSMM: %[[LLVM_PTR_ARG0:.+]] = llvm.inttoptr %[[CAST_ARG0]] : i64 to !llvm.ptr
// XSMM: %[[INT_PTR_ARG1:.+]] = memref.extract_aligned_pointer_as_index
// XSMM: %[[CAST_ARG1:.+]] = arith.index_cast %[[INT_PTR_ARG1]] : index to i64
// XSMM: %[[LLVM_PTR_ARG1:.+]] = llvm.inttoptr %[[CAST_ARG1]] : i64 to !llvm.ptr
// XSMM: %[[INT_PTR_ALLOC:.+]] = memref.extract_aligned_pointer_as_index
// XSMM: %[[CAST_ALLOC:.+]] = arith.index_cast %[[INT_PTR_ALLOC]] : index to i64
// XSMM: %[[LLVM_PTR_ALLOC:.+]] = llvm.inttoptr %[[CAST_ALLOC]] : i64 to !llvm.ptr
// XSMM: call @xsmm_gemm_invoke(%[[C1]], %[[DIS]], %[[LLVM_PTR_ARG0]], %[[C0]], %[[LLVM_PTR_ARG1]], %[[C0]], %[[LLVM_PTR_ALLOC]], %[[C0]])

// -----

// XSMM: func.func @add(
// XSMM-SAME:  %[[ARG0:.+]]: memref<3x3xf32>,
// XSMM-SAME:  %[[ARG1:.+]]: memref<3x3xf32>
func.func @add(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  // XSMM: %[[C0:.+]] = arith.constant 0 : index
  // XSMM: call @xsmm_binary_dispatch
  // XSMM: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // XSMM-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr
  // XSMM: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // XSMM-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr
  // XSMM: call @xsmm_binary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
  linalg.add ins(%arg0, %arg1: memref<3x3xf32>, memref<3x3xf32>)
             outs(%arg1: memref<3x3xf32>)
  return
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// XSMM: func.func @add_mapping(
func.func @add_mapping(%arg0: memref<1x10x10xf32>, %arg1: memref<1x10x10xf32>) {
  // XSMM: memref.subview
  // XSMM-NOT: scf.parallel
  // XSMM: call @xsmm_binary_dispatch
  // XSMM: %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM: %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM: call @xsmm_binary_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}
  %subview = memref.subview %arg0[0, 0, 0] [1, 10, 10] [1, 1, 1] : memref<1x10x10xf32> to memref<10x10xf32>
  %subview_0 = memref.subview %arg1[0, 0, 0] [1, 10, 10] [1, 1, 1] : memref<1x10x10xf32> to memref<10x10xf32>
  linalg.add ins(%subview, %subview_0 : memref<10x10xf32>, memref<10x10xf32>)
             outs(%subview_0 : memref<10x10xf32>)
  return
}

// -----

#map = affine_map<(d0, d1)[s0] -> (d0 * 10 + d1 + s0)>

// XSMM-LABEL: @add_mapping_parallel
func.func @add_mapping_parallel(%arg0: memref<10x10x10xf32>, %arg1: memref<10x10x10xf32>) {
  // XSMM: call @xsmm_binary_dispatch
  // XSMM: scf.parallel
  // XSMM: %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM: %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM: call @xsmm_binary_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  scf.parallel (%arg2) = (%c0) to (%c10) step (%c1) {
    %subview = memref.subview %arg0[%arg2, 0, 0] [1, 10, 10] [1, 1, 1] : memref<10x10x10xf32> to memref<10x10xf32, #map>
    %subview_0 = memref.subview %arg1[%arg2, 0, 0] [1, 10, 10] [1, 1, 1] : memref<10x10x10xf32> to memref<10x10xf32, #map>
    linalg.add ins(%subview, %subview_0 : memref<10x10xf32, #map>, memref<10x10xf32, #map>)
               outs(%subview_0 : memref<10x10xf32, #map>)
    scf.reduce
  }

  return
}

// -----

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// XSMM: func.func @identity(
// XSMM-SAME:  %[[ARG0:.+]]: memref<3x3xf32>,
// XSMM-SAME:  %[[ARG1:.+]]: f32
func.func @identity(%arg0: memref<3x3xf32>, %arg1: f32) {
  // XSMM: linalg.fill ins(%[[ARG1]] : f32) outs(%[[ARG0]] : memref<3x3xf32>)
  linalg.generic {
    indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]}
    ins(%arg1: f32) outs(%arg0: memref<3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + d1 + s0)>

// XSMM-LABEL: @identity_mapping
func.func @identity_mapping(%arg0: memref<64xf32>) -> memref<12x56x56x64xf32> {
  // XSMM: call @xsmm_unary_dispatch
  // XSMM: scf.parallel
  // XSMM: %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM: %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   call @xsmm_unary_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c1 = arith.constant 1 : index
  %c56 = arith.constant 56 : index
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<12x56x56x64xf32>
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c12, %c56) step (%c1, %c1) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 56, 64] [1, 1, 1, 1]
      : memref<12x56x56x64xf32> to memref<56x64xf32, #map>
    linalg.generic {
      indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]}
      ins(%arg0: memref<64xf32>) outs(%subview: memref<56x64xf32, #map>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    }
    scf.reduce
  }

  return %alloc : memref<12x56x56x64xf32>
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// XSMM: func.func @relu(
// XSMM-SAME:  %[[ARG0:.+]]: memref<3x3xf32>
func.func @relu(%arg0: memref<3x3xf32>) {
  // XSMM: %[[C0:.*]] = arith.constant 0 : index
  // XSMM: call @xsmm_unary_dispatch
  // XSMM: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // XSMM-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr
  // XSMM: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr0]], %[[C0]]
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
    ins(%arg0: memref<3x3xf32>) outs(%arg0: memref<3x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maximumf %in, %c0 : f32
      linalg.yield %2 : f32
  }
  return
}

// -----

#map = affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>

// XSMM-LABEL: @relu_3d(
// XSMM-SAME: %[[arg:.*]]: memref<64x32x32xf32>) {
func.func @relu_3d(%arg0: memref<64x32x32xf32>) -> memref<64x32x32xf32> {
  // XSMM: call @xsmm_unary_dispatch
  // XSMM: scf.parallel
  // XSMM: %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   call @xsmm_unary_invoke({{.*}}%[[ptr0]], %{{.+}}
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0_f32 = arith.constant 0.0 : f32
  scf.parallel (%arg1) = (%c0) to (%c64) step (%c1) {
    %subview = memref.subview %arg0[%arg1, 0, 0] [1, 32, 32] [1, 1, 1] : memref<64x32x32xf32> to memref<32x32xf32, #map>
    linalg.generic {
    indexing_maps = [#map0, #map0], iterator_types = ["parallel", "parallel"]}
    ins(%subview: memref<32x32xf32, #map>) outs(%subview: memref<32x32xf32, #map>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maximumf %in, %c0_f32 : f32
      linalg.yield %2 : f32
    }
    scf.reduce
  }

  return %arg0 : memref<64x32x32xf32>
}

// -----

// XSMM: func.func @brgemm(
// XSMM-SAME:  %[[ARG0:.+]]: memref<2x3x4xf32>,
// XSMM-SAME:  %[[ARG1:.+]]: memref<2x4x3xf32>,
// XSMM-SAME:  %[[ARG2:.+]]: memref<3x3xf32>
func.func @brgemm(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) {
  // XSMM: %[[C0:.*]] = arith.constant 0 : index
  // XSMM: call @xsmm_brgemm_dispatch
  // XSMM: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // XSMM-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr

  // XSMM: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // XSMM-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr

  // XSMM: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // XSMM-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr

  // XSMM: call @xsmm_brgemm_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]], %[[llvm_ptr2]], %[[C0]]
  linalg.batch_reduce_matmul ins(%arg0, %arg1: memref<2x3x4xf32>, memref<2x4x3xf32>)
                             outs(%arg2: memref<3x3xf32>)

  return
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

// XSMM-LABEL: func.func @brgemm_bf16
// XSMM-SAME:  %[[ARG0:.+]]: memref<64x4x4xbf16>,
// XSMM-SAME:  %[[ARG1:.+]]: memref<64x2x4x2xbf16>,
// XSMM-SAME:  %[[ARG2:.+]]: memref<4x4xbf16>
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @brgemm_bf16(%arg0: memref<64x4x4xbf16>, %arg1: memref<64x2x4x2xbf16>,
                                %arg2: memref<4x4xbf16>) {
    // XSMM: %[[C0:.*]] = arith.constant 0 : index
    // XSMM: call @xsmm_brgemm_dispatch
    // XSMM: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
    // XSMM-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
    // XSMM-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr

    // XSMM: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
    // XSMM-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
    // XSMM-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr

    // XSMM: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
    // XSMM-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64
    // XSMM-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr

    // XSMM: call @xsmm_brgemm_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]], %[[llvm_ptr2]], %[[C0]]
    %expanded = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [64, 4, 2, 2]
      : memref<64x4x4xbf16> into memref<64x4x2x2xbf16>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
      ins(%expanded, %arg1 : memref<64x4x2x2xbf16>, memref<64x2x4x2xbf16>)
      outs(%arg2 : memref<4x4xbf16>) {
        ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
          %5 = arith.mulf %in, %in_5 : bf16
          %6 = arith.addf %out, %5 : bf16
          linalg.yield %6 : bf16
    }
    return
  }
}

// -----

// XSMM: func.func @gemm(
// XSMM-SAME:  %[[ARG0:.+]]: memref<4x8xf32>,
// XSMM-SAME:  %[[ARG1:.+]]: memref<8x4xf32>,
// XSMM-SAME:  %[[ARG2:.+]]: memref<4x4xf32>)
func.func @gemm(%A: memref<4x8xf32>,
          %B: memref<8x4xf32>, %C: memref<4x4xf32>) {
  // XSMM: %[[C0:.*]] = arith.constant 0 : index
  // XSMM: call @xsmm_gemm_dispatch
  // XSMM: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // XSMM-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64

  // XSMM-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr

  // XSMM: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // XSMM-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr

  // XSMM: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // XSMM-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr
  // XSMM: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]], %[[llvm_ptr2]], %[[C0]]
  linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>)
                outs(%C : memref<4x4xf32>)

  return
}

// -----

// XSMM-LABEL: func.func @blocked_matmul(
// XSMM-SAME: %[[ARG0:.+]]: memref<4x16x32x32xf32>,
// XSMM-SAME: %[[ARG1:.+]]: memref<8x16x32x32xf32>,
// XSMM-SAME: %[[ARG2:.+]]: memref<4x8x32x32xf32>)
func.func @blocked_matmul(%arg0: memref<4x16x32x32xf32>, %arg1: memref<8x16x32x32xf32>, %arg2: memref<4x8x32x32xf32>) {
  // XSMM: call @xsmm_brgemm_dispatch
  // XSMM: scf.parallel
  // XSMM:   %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   %[[ptr2:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   call @xsmm_brgemm_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}, %[[ptr2]], %{{.+}}
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c4, %c8) step (%c1, %c1) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<4x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1] : memref<8x16x32x32xf32> to memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<4x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.batch_reduce_matmul ins(%subview, %subview_0 :
                                   memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>,
                                   memref<16x32x32xf32, strided<[1024, 32, 1], offset: ?>>)
                               outs(%subview_1 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    scf.reduce
  }

  return
}

// -----

// Conv2D weights
memref.global "private" constant @__constant_2048x512xf32 : memref<2048x512xf32> = dense<0.00332225906> {alignment = 128 : i64}

// XSMM-LABEL: @conv2d_1x1(
// XSMM-SAME: %[[arg:.*]]: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
func.func @conv2d_1x1(%arg0: memref<1x7x7x2048xf32>) -> memref<1x7x7x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c7 = arith.constant 7 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = memref.get_global @__constant_2048x512xf32 : memref<2048x512xf32>

  // 1x1 Conv2D
  // XSMM: call @xsmm_gemm_dispatch
  // XSMM: scf.for
  // XSMM:   %[[ptr0:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   %[[ptr1:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   %[[ptr2:.*]] = llvm.inttoptr %{{.+}} : i64 to !llvm.ptr
  // XSMM:   call @xsmm_gemm_invoke({{.*}}%[[ptr0]], %{{.+}}, %[[ptr1]], %{{.+}}, %[[ptr2]], %{{.+}}
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<1x7x7x512xf32>
  linalg.fill ins(%cst : f32) outs(%alloc : memref<1x7x7x512xf32>)
  scf.for %arg1 = %c0 to %c7 step %c1 {
    %subview = memref.subview %arg0[0, %arg1, 0, 0] [1, 1, 7, 2048] [1, 1, 1, 1] : memref<1x7x7x2048xf32> to memref<7x2048xf32, strided<[2048, 1], offset: ?>>
    %subview_0 = memref.subview %alloc[0, %arg1, 0, 0] [1, 1, 7, 512] [1, 1, 1, 1] : memref<1x7x7x512xf32> to memref<7x512xf32, strided<[512, 1], offset: ?>>
    linalg.matmul ins(%subview, %0 : memref<7x2048xf32, strided<[2048, 1], offset: ?>>,
                                     memref<2048x512xf32>)
                  outs(%subview_0 : memref<7x512xf32, strided<[512, 1], offset: ?>>)
  }

  // XSMM: return {{.*}} : memref<1x7x7x512xf32>
  return %alloc : memref<1x7x7x512xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

// XSMM: func.func @mlp(
// XSMM-SAME:  %[[ARG0:.+]]: memref<128x256xf32>,
// XSMM-SAME:  %[[ARG1:.+]]: memref<256x512xf32>,
// XSMM-SAME:  %[[ARG2:.+]]: memref<512xf32>,
// XSMM-SAME:  %[[ARG3:.+]]: memref<128x512xf32>)
func.func @mlp(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32>,
  %arg2: memref<512xf32>,  %arg3: memref<128x512xf32>) {

  // XSMM: %[[C0:.*]] = arith.constant 0 : index

  // Identity
  // XSMM: call @xsmm_unary_dispatch
  // XSMM: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // XSMM-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr

  // XSMM: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG3]]
  // XSMM-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr

  // XSMM: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr0]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
  linalg.generic {
    indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
    ins(%arg2: memref<512xf32>) outs(%arg3: memref<128x512xf32>) {
    ^bb0(%in : f32, %out: f32):
      linalg.yield %in : f32
  }

  // Matmul
  // XSMM: call @xsmm_gemm_dispatch
  // XSMM: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
  // XSMM-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr

  // XSMM: %[[ptr3:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // XSMM-NEXT: %[[ptr_cast3:.*]] = arith.index_cast %[[ptr3]] : index to i64
  // XSMM-NEXT: %[[llvm_ptr3:.*]] = llvm.inttoptr %[[ptr_cast3]] : i64 to !llvm.ptr

  // XSMM: call @xsmm_gemm_invoke({{.*}}%[[llvm_ptr2]], %[[C0]], %[[llvm_ptr3]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
  linalg.matmul ins(%arg0, %arg1 : memref<128x256xf32>, memref<256x512xf32>)
                outs(%arg3 : memref<128x512xf32>)

  // Relu
  // XSMM: call @xsmm_unary_dispatch
  // XSMM: call @xsmm_unary_invoke({{.*}}%[[llvm_ptr1]], %[[C0]], %[[llvm_ptr1]], %[[C0]]
  %c0 = arith.constant 0.0 : f32
  linalg.generic {
    indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
    ins(%arg3: memref<128x512xf32>) outs(%arg3: memref<128x512xf32>) {
    ^bb0(%in: f32, %out: f32):
      %2 = arith.maximumf %in, %c0 : f32
      linalg.yield %2 : f32
  }

  return
}
