// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 2, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-LABEL: func.func @matmul_tensor(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x1024xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<512x2048x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<128x2048xbf16>)
func.func @matmul_tensor(%arg0: tensor<128x1024xbf16>,
                  %arg1: tensor<512x2048x2xbf16>,
                  %arg2: tensor<128x2048xbf16>) -> tensor<128x2048xbf16> {
  // CHECK-DAG: %[[of:.*]] = arith.constant 0 : index
  // CHECK-DAG: call @xsmm_brgemm_dispatch
  // CHECK-DAG: %[[expand_shape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0], [1, 2]] output_shape [128, 512, 2]
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[expand_shape]]
  // CHECK: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr

  // CHECK-DAG: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr

  // CHECK-DAG: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr

  // CHECK: call @xsmm_brgemm_invoke({{.*}}, {{.*}}, %[[llvm_ptr0]], %[[of]], %[[llvm_ptr1]], %[[of]], %[[llvm_ptr2]], %[[of]], {{.*}})
  %result = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<128x1024xbf16>, tensor<512x2048x2xbf16>)
    outs(%arg2 : tensor<128x2048xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
  } -> tensor<128x2048xbf16>

  return %result : tensor<128x2048xbf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 2, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-LABEL: func.func @matmul_memref(
// CHECK-SAME:  %[[ARG0:.+]]: memref<128x1024xbf16>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<512x2048x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<128x2048xbf16>)
func.func @matmul_memref(%arg0: memref<128x1024xbf16>,
                  %arg1: memref<512x2048x2xbf16>,
                  %arg2: memref<128x2048xbf16>) -> memref<128x2048xbf16> {
  // CHECK-DAG: %[[of:.*]] = arith.constant 0 : index
  // CHECK-DAG: call @xsmm_brgemm_dispatch
  // CHECK-DAG:  %[[expand_shape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0], [1, 2]] output_shape [128, 512, 2] 
  // CHECK: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[expand_shape]]
  // CHECK-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr

  // CHECK-DAG: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr

  // CHECK-DAG: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr

  // CHECK: call @xsmm_brgemm_invoke({{.*}}, {{.*}}, %[[llvm_ptr0]], %[[of]], %[[llvm_ptr1]], %[[of]], %[[llvm_ptr2]], %[[of]], {{.*}})
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : memref<128x1024xbf16>, memref<512x2048x2xbf16>)
    outs(%arg2 : memref<128x2048xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
  }

  return %arg2 : memref<128x2048xbf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3 floordiv 2, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

// CHECK-LABEL: func.func @brgemm_static_tensor(
// CHECK-SAME: %[[ARG0:.+]]: memref<4x256x512xbf16>,
// CHECK-SAME: %[[ARG1:.+]]: memref<4x512x1024xbf16>,
// CHECK-SAME: %[[ARG2:.+]]: memref<256x1024xbf16>)
func.func @brgemm_static_tensor(%arg0: tensor<4x256x512xbf16>, %arg1: tensor<4x512x1024xbf16>, %arg2: tensor<256x1024xbf16>) -> tensor<256x1024xbf16> {
  %0 = tensor.empty() : tensor<4x256x1024x2xbf16>
  %1 = tensor.pack %arg1 inner_dims_pos = [1] inner_tiles = [2] into %0 : tensor<4x512x1024xbf16> -> tensor<4x256x1024x2xbf16>
  // CHECK-DAG: %[[alloc:.*]] = memref.alloc{{.*}}: memref<4x256x1024x2xbf16>
  // CHECK-DAG: %[[of:.*]] = arith.constant 0 : index
  // CHECK-DAG: call @xsmm_brgemm_dispatch

  // CHECK-DAG: %[[expand_shape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0], [1], [2, 3]] output_shape [4, 256, 256, 2]
  // CHECK-DAG: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[expand_shape]]
  // CHECK-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr

  // CHECK-DAG: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[alloc]]
  // CHECK-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr

  // CHECK-DAG: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr

  // CHECK: call @xsmm_brgemm_invoke({{.*}}, {{.*}}, %[[llvm_ptr0]], %[[of]], %[[llvm_ptr1]], %[[of]], %[[llvm_ptr2]], %[[of]], {{.*}})

  %2 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %1 : tensor<4x256x512xbf16>, tensor<4x256x1024x2xbf16>)
    outs(%arg2 : tensor<256x1024xbf16>) {
      ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
        %5 = arith.mulf %in, %in_5 : bf16
        %6 = arith.addf %out, %5 : bf16
        linalg.yield %6 : bf16
  } -> tensor<256x1024xbf16>

  return %2 : tensor<256x1024xbf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3 floordiv 2, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

// CHECK-LABEL: func.func @brgemm_static_memref(
// CHECK-SAME: %[[ARG0:.+]]: memref<4x256x512xbf16>,
// CHECK-SAME: %[[ARG1:.+]]: memref<4x256x1024x2xbf16>,
// CHECK-SAME: %[[ARG2:.+]]: memref<256x1024xbf16>)
func.func @brgemm_static_memref(%arg0: memref<4x256x512xbf16>, %arg1: memref<4x256x1024x2xbf16>, %arg2: memref<256x1024xbf16>) -> memref<256x1024xbf16> {
  // CHECK-DAG: %[[of:.*]] = arith.constant 0 : index
  // CHECK-DAG: call @xsmm_brgemm_dispatch
  
  // CHECK-DAG: %[[expand_shape:.*]] = memref.expand_shape %[[ARG0]] {{\[}}[0], [1], [2, 3]] output_shape [4, 256, 256, 2]
  // CHECK-NEXT: %[[ptr0:.*]] = memref.extract_aligned_pointer_as_index %[[expand_shape]]
  // CHECK-NEXT: %[[ptr_cast0:.*]] = arith.index_cast %[[ptr0]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr0:.*]] = llvm.inttoptr %[[ptr_cast0]] : i64 to !llvm.ptr

  // CHECK-DAG: %[[ptr1:.*]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
  // CHECK-NEXT: %[[ptr_cast1:.*]] = arith.index_cast %[[ptr1]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr1:.*]] = llvm.inttoptr %[[ptr_cast1]] : i64 to !llvm.ptr

  // CHECK-DAG: %[[ptr2:.*]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
  // CHECK-NEXT: %[[ptr_cast2:.*]] = arith.index_cast %[[ptr2]] : index to i64
  // CHECK-NEXT: %[[llvm_ptr2:.*]] = llvm.inttoptr %[[ptr_cast2]] : i64 to !llvm.ptr

  // CHECK: call @xsmm_brgemm_invoke({{.*}}, {{.*}}, %[[llvm_ptr0]], %[[of]], %[[llvm_ptr1]], %[[of]], %[[llvm_ptr2]], %[[of]], {{.*}})
   linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : memref<4x256x512xbf16>, memref<4x256x1024x2xbf16>)
    outs(%arg2 : memref<256x1024xbf16>) {
      ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
        %5 = arith.mulf %in, %in_5 : bf16
        %6 = arith.addf %out, %5 : bf16
        linalg.yield %6 : bf16
  }

  return %arg2 : memref<256x1024xbf16>
}


