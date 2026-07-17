// RUN: tpp-opt -pack-vnni -split-input-file %s | FileCheck %s

func.func @brgemm(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>,
                  %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<32x4x4xbf16>, tensor<32x4x4xbf16>)
                                  outs(%arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0: tensor<4x4xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

// CHECK-LABEL: brgemm
// CHECK-SAME:  %[[ARG0:.+]]: tensor<32x4x4xbf16>, %[[ARG1:.+]]: tensor<32x4x4xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<4x4xbf16>
// CHECK: %[[VNNI_A:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1], [2, 3]] output_shape{{.*}}: tensor<32x4x4xbf16> into tensor<32x4x{{2|1}}x{{2|4}}xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<32x{{2|1}}x4x{{2|4}}xbf16>
// CHECK: %[[PACK:.+]] = linalg.pack %[[ARG1]]
// CHECK-SAME:  inner_dims_pos = [1] inner_tiles = [{{2|4}}] into %[[EMPTY]]
// CHECK-SAME:  : tensor<32x4x4xbf16> -> tensor<32x{{2|1}}x4x{{2|4}}xbf16>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%[[VNNI_A]], %[[PACK]]
// CHECK-SAME: outs(%[[ARG2]]

// -----

// VNNI packing assumes an already packed matmul in the following format:
// [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb][jb]
func.func @matmul(%arg0: tensor<128x128xbf16>, %arg1: tensor<128x128xbf16>,
                  %arg2: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xbf16>, tensor<128x128xbf16>)
                     outs(%arg2: tensor<128x128xbf16>) -> tensor<128x128xbf16>
  return %0 : tensor<128x128xbf16>
}

// CHECK-LABEL: matmul
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

func.func @prepacked_matmul(%pack: tensor<4x4x32x32xbf16>, %pack_0: tensor<4x4x32x32xbf16>,
                           %pack_1: tensor<4x4x32x32xbf16>) -> tensor<4x4x32x32xbf16> {
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%pack, %pack_0 : tensor<4x4x32x32xbf16>, tensor<4x4x32x32xbf16>)
    outs(%pack_1 : tensor<4x4x32x32xbf16>) {
    ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
      %4 = arith.mulf %in, %in_2 : bf16
      %5 = arith.addf %out, %4 : bf16
      linalg.yield %5 : bf16
  } -> tensor<4x4x32x32xbf16>
  return %1 : tensor<4x4x32x32xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d5, d6)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d5, d4, d6)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4)>

// CHECK-LABEL: prepacked_matmul
// CHECK-SAME:  %[[ARG0:.+]]: tensor<4x4x32x32xbf16>, %[[ARG1:.+]]: tensor<4x4x32x32xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<4x4x32x32xbf16>
// CHECK: %[[VNNI_A:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1], [2], [3, 4]]
// CHECK-SAME: output_shape{{.*}}: tensor<4x4x32x32xbf16> into tensor<4x4x32x{{16|8}}x{{2|4}}xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<4x4x{{16|8}}x32x{{2|4}}xbf16>
// CHECK: %[[PACK:.+]] = linalg.pack %[[ARG1]] inner_dims_pos = [2] inner_tiles = [{{2|4}}] into %[[EMPTY]]
// CHECK-SAME:  : tensor<4x4x32x32xbf16> -> tensor<4x4x{{16|8}}x32x{{2|4}}xbf16>
// CHECK: {{.+}} = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%[[VNNI_A]], %[[PACK]]
// CHECK-SAME: outs(%[[ARG2]]

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>

func.func @already_packed_matmul(%arg0: tensor<4x4x32x16x2xbf16>, %arg1: tensor<4x4x16x32x2xbf16>,
                                 %arg2: tensor<4x4x32x32xbf16>) -> tensor<4x4x32x32xbf16> {
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x4x32x16x2xbf16>, tensor<4x4x16x32x2xbf16>)
    outs(%arg2 : tensor<4x4x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
  } -> tensor<4x4x32x32xbf16>
  return %1 : tensor<4x4x32x32xbf16>
}

// CHECK-LABEL: already_packed_matmul
// CHECK-NOT: expand_shape
// CHECK-NOT: linalg.pack
// CHECK: linalg.generic

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3 floordiv 2, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

func.func @no_pack_invalid_reduction_map(%arg0: tensor<3x16x8xbf16>,
  %arg1: tensor<3x4x16x2xbf16>, %arg2: tensor<16x16xbf16>) -> tensor<16x16xbf16> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : tensor<3x16x8xbf16>, tensor<3x4x16x2xbf16>)
    outs(%arg2 : tensor<16x16xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
    } -> tensor<16x16xbf16>
  return %0 : tensor<16x16xbf16>
}

// CHECK: no_pack_invalid_reduction_map
// CHECK-NOT: expand_shape
// CHECK-NOT: linalg.pack
// CHECK: linalg.generic

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d2, d4, d6)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2 floordiv 2, d6 floordiv 2, d5, d3, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d4, d5)>

func.func @no_pack_not_contraction(%arg0: tensor<4x4x32x32xbf16>, %arg1: tensor<4x2x16x32x2x2xbf16>,
                                 %arg2: tensor<4x4x32x32xbf16>) -> tensor<4x4x32x32xbf16> {
  %1 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x4x32x32xbf16>, tensor<4x2x16x32x2x2xbf16>)
    outs(%arg2 : tensor<4x4x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_0 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
  } -> tensor<4x4x32x32xbf16>
  return %1 : tensor<4x4x32x32xbf16>
}

// CHECK: no_pack_not_contraction
// CHECK-NOT: expand_shape
// CHECK-NOT: linalg.pack
// CHECK: linalg.generic
