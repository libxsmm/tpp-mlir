// RUN: tpp-opt %s -tpp-mapping | FileCheck %s

!A_tensor_t = tensor<256x512xbf16>
!B_tensor_t = tensor<512x1024xbf16>
!C_tensor_t = tensor<256x1024xbf16>

func.func @matmul_static(
    %A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t) -> !C_tensor_t {
   %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                           outs(%C: !C_tensor_t) -> !C_tensor_t
   return %matmul : !C_tensor_t
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4 floordiv 2, d3, d1)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

// CHECK-LABEL: matmul_static
// CHECK-SAME: %[[ARG0:.+]]: tensor<256x512xbf16>, %[[ARG1:.+]]: tensor<512x1024xbf16>, %[[ARG2:.+]]: tensor<256x1024xbf16>
// CHECK: %[[EMPTY_0:.+]] =  tensor.empty() : tensor<8x16x32x32xbf16>
// CHECK: %[[PACK:.+]] = tensor.pack %[[ARG0]] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
// CHECK-SAME:  into %[[EMPTY_0]] : tensor<256x512xbf16> -> tensor<8x16x32x32xbf16>
// CHECK: %[[EMPTY_1:.+]] = tensor.empty() : tensor<32x16x32x32xbf16>
// CHECK: %[[PACK_0:.+]] = tensor.pack %[[ARG1]] outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32]
// CHECK-SAME:  into %{{.+}} : tensor<512x1024xbf16> -> tensor<32x16x32x32xbf16>
// CHECK: %[[EMPTY_2:.+]] = tensor.empty() : tensor<32x16x16x32x2xbf16>
// CHECK: %[[PACK_1:.+]] = tensor.pack %[[PACK_0]] inner_dims_pos = [2] inner_tiles = [2] into %[[EMPTY_2]]
// CHECK-SAME:  : tensor<32x16x32x32xbf16> -> tensor<32x16x16x32x2xbf16>
// CHECK: %{{.+}} = scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (8, 32) shared_outs(%[[ARG5:.+]] = %[[ARG2]])
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[APPLY_1:.+]] = affine.apply #[[MAP]](%[[ARG4]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[PACK]][%[[ARG3]], 0, 0, 0] [1, 16, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<8x16x32x32xbf16> to tensor<16x32x32xbf16>
// CHECK: %[[SLICE_2:.+]] = tensor.extract_slice %[[PACK_1]][%[[ARG4]], 0, 0, 0, 0] [1, 16, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<32x16x16x32x2xbf16> to tensor<16x16x32x2xbf16>
// CHECK: %[[SLICE_3:.+]] = tensor.extract_slice %[[ARG5]][%[[APPLY]], %[[APPLY_1]]] [32, 32] [1, 1]
// CHECK-SAME:  : tensor<256x1024xbf16> to tensor<32x32xbf16>
// CHECK: %[[GEMM:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]],
// CHECK-SAME:  iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]
