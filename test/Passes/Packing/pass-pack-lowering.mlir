// RUN: tpp-opt %s -lower-packs-unpacks -split-input-file | FileCheck %s

func.func @matmul_pack(%arg0: tensor<1024x512xbf16>, %arg1: tensor<16x32x32x32xbf16>) -> tensor<16x32x32x32xbf16> {
  %pack = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : tensor<1024x512xbf16> -> tensor<16x32x32x32xbf16>
  return %pack : tensor<16x32x32x32xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: matmul_pack
// CHECK-SAME: %[[ARG0:.+]]: tensor<1024x512xbf16>, %[[ARG1:.+]]: tensor<16x32x32x32xbf16>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (16, 32) shared_outs(%[[ARG4:.+]] = %[[ARG1]])
// CHECK: %[[APPLY_ONE:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[APPLY_TWO:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[APPLY_ONE]], %[[APPLY_TWO]]] [32, 32] [1, 1]
// CHECK-SAME:  : tensor<1024x512xbf16> to tensor<32x32xbf16>
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice %[[SLICE]] into %[[ARG4]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<32x32xbf16> into tensor<16x32x32x32xbf16>

// -----

func.func @matmul_unpack(%arg0: tensor<16x16x32x32xbf16>, %arg1: tensor<512x512xbf16>) -> tensor<512x512xbf16> {
  %unpack = linalg.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg1 : tensor<16x16x32x32xbf16> -> tensor<512x512xbf16>
  return %unpack : tensor<512x512xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 floordiv 32)>

// CHECK-LABEL: matmul_unpack
// CHECK-SAME: %[[ARG0:.+]]: tensor<16x16x32x32xbf16>, %[[ARG1:.+]]: tensor<512x512xbf16>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) = (0, 0) to (512, 512) step (32, 32) shared_outs(%[[ARG4:.+]] = %[[ARG1]])
// CHECK: %[[APPLY_ONE:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[APPLY_TWO:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[APPLY_ONE]], %[[APPLY_TWO]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<16x16x32x32xbf16> to tensor<32x32xbf16>
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice %[[SLICE]] into %[[ARG4]][%[[ARG2]], %[[ARG3]]] [32, 32] [1, 1]
// CHECK-SAME:  : tensor<32x32xbf16> into tensor<512x512xbf16>

// -----

func.func @pack_fusion(%arg0: tensor<1024x512xbf16>, %arg1: tensor<16x32x16x32x2xbf16>) -> tensor<16x32x16x32x2xbf16> {
  %1 = tensor.empty() : tensor<16x32x32x32xbf16>
  %pack_0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1
    : tensor<1024x512xbf16> -> tensor<16x32x32x32xbf16>
  %pack_1 = linalg.pack %pack_0 inner_dims_pos = [2] inner_tiles = [2] into %arg1
    : tensor<16x32x32x32xbf16> -> tensor<16x32x16x32x2xbf16>
  return %pack_1 : tensor<16x32x16x32x2xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: pack_fusion
// CHECK-SAME: %[[ARG0:.+]]: tensor<1024x512xbf16>, %[[ARG1:.+]]: tensor<16x32x16x32x2xbf16>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (16, 32) shared_outs(%[[ARG4:.+]] = %[[ARG1]])
// CHECK: %[[APPLY_ONE:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[APPLY_TWO:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[APPLY_ONE]], %[[APPLY_TWO]]] [32, 32] [1, 1]
// CHECK-SAME:  : tensor<1024x512xbf16> to tensor<32x32xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16x32x2xbf16>
// CHECK: %[[EXP:.+]] = tensor.expand_shape %[[SLICE]] {{\[}}[0, 1], [2]] output_shape [16, 2, 32] : tensor<32x32xbf16> into tensor<16x2x32xbf16>
// CHECK: %[[TRN:.+]] = linalg.transpose ins(%[[EXP]] : tensor<16x2x32xbf16>) outs(%[[EMPTY]] : tensor<16x32x2xbf16>)
// CHECK-SAME:  permutation = [0, 2, 1]
// CHECK: scf.forall.in_parallel
// CHECK: tensor.parallel_insert_slice %[[TRN]] into %[[ARG4]][%[[ARG2]], %[[ARG3]], 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<16x32x2xbf16> into tensor<16x32x16x32x2xbf16>

// -----

func.func @expect_to_fuse_first_and_second(%arg0: tensor<1024x512xbf16>, %arg1: tensor<16x32x16x32x2xbf16>,
  %arg2: tensor<8x32x16x32x2x2xbf16>) -> tensor<8x32x16x32x2x2xbf16> {
  %1 = tensor.empty() : tensor<16x32x32x32xbf16>
  %pack_0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1
    : tensor<1024x512xbf16> -> tensor<16x32x32x32xbf16>
  %pack_1 = linalg.pack %pack_0 inner_dims_pos = [2] inner_tiles = [2] into %arg1
    : tensor<16x32x32x32xbf16> -> tensor<16x32x16x32x2xbf16>
  %pack_2 = linalg.pack %pack_1 inner_dims_pos = [0] inner_tiles = [2] into %arg2
    : tensor<16x32x16x32x2xbf16> -> tensor<8x32x16x32x2x2xbf16>
  return %pack_2 : tensor<8x32x16x32x2x2xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: expect_to_fuse_first_and_second
// CHECK-SAME: %[[ARG0:.+]]: tensor<1024x512xbf16>, %[[ARG1:.+]]: tensor<16x32x16x32x2xbf16>, %[[ARG2:.+]]: tensor<8x32x16x32x2x2xbf16>
// CHECK: %[[LOOP:.+]] = scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (16, 32) shared_outs(%[[ARG5:.+]] = %[[ARG1]])
// CHECK: %[[APPLY_1:.+]] = affine.apply #[[MAP]](%[[ARG4]])
// CHECK: %[[APPLY_2:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[APPLY_1]], %[[APPLY_2]]] [32, 32] [1, 1] : tensor<1024x512xbf16> to tensor<32x32xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16x32x2xbf16>
// CHECK: %[[EXP:.+]] = tensor.expand_shape %[[SLICE]] {{\[}}[0, 1], [2]] output_shape [16, 2, 32] : tensor<32x32xbf16> into tensor<16x2x32xbf16>
// CHECK: %[[TRAN:.+]] = linalg.transpose ins(%[[EXP]] : tensor<16x2x32xbf16>) outs(%[[EMPTY]] : tensor<16x32x2xbf16>)
// CHECK-SAME:  permutation = [0, 2, 1]
// CHECK: tensor.parallel_insert_slice %[[TRAN]] into %[[ARG5]][%[[ARG3]], %[[ARG4]], 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<16x32x2xbf16> into tensor<16x32x16x32x2xbf16>
// CHECK: %[[EXP_2:.+]] = tensor.expand_shape %[[LOOP]] {{\[}}[0, 1], [2], [3], [4], [5]] output_shape [8, 2, 32, 16, 32, 2] : tensor<16x32x16x32x2xbf16> into tensor<8x2x32x16x32x2xbf16>
// CHECK: %{{.+}} = linalg.transpose ins(%[[EXP_2]] : tensor<8x2x32x16x32x2xbf16>) outs(%[[ARG2]] : tensor<8x32x16x32x2x2xbf16>)
// CHECK-SAME:  permutation = [0, 2, 3, 4, 5, 1]

// -----

func.func @expect_not_to_fuse(%arg0: tensor<1024x512xbf16>, %arg1: tensor<8x32x32x32x2xbf16>) -> tensor<8x32x32x32x2xbf16> {
  %1 = tensor.empty() : tensor<16x32x32x32xbf16>
  %pack_0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1
    : tensor<1024x512xbf16> -> tensor<16x32x32x32xbf16>
  %pack_1 = linalg.pack %pack_0 inner_dims_pos = [0] inner_tiles = [2] into %arg1
    : tensor<16x32x32x32xbf16> -> tensor<8x32x32x32x2xbf16>
  return %pack_1 : tensor<8x32x32x32x2xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: expect_not_to_fuse
// CHECK-SAME: %[[ARG0:.+]]: tensor<1024x512xbf16>, %[[ARG1:.+]]: tensor<8x32x32x32x2xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<16x32x32x32xbf16>
// CHECK: %[[LOOP:.+]] = scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (16, 32) shared_outs(%[[ARG4:.+]] = %[[EMPTY]])
// CHECK: %[[APPLY_1:.+]] = affine.apply #[[MAP]](%[[ARG3]])
// CHECK: %[[APPLY_2:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[APPLY_1]], %[[APPLY_2]]] [32, 32] [1, 1] : tensor<1024x512xbf16> to tensor<32x32xbf16>
// CHECK: tensor.parallel_insert_slice %[[SLICE]] into %[[ARG4]][%[[ARG2]], %[[ARG3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : tensor<32x32xbf16> into tensor<16x32x32x32xbf16>
// CHECK: %[[EXP_1:.+]] = tensor.expand_shape %[[LOOP]] {{\[}}[0, 1], [2], [3], [4]] output_shape [8, 2, 32, 32, 32] : tensor<16x32x32x32xbf16> into tensor<8x2x32x32x32xbf16>
// CHECK: %{{.+}} = linalg.transpose ins(%[[EXP_1]] : tensor<8x2x32x32x32xbf16>) outs(%[[ARG1]] : tensor<8x32x32x32x2xbf16>)
// CHECK-SAME:  permutation = [0, 2, 3, 4, 1]

// -----

func.func @pack_fusion_outer_only(%arg0: tensor<1024x512xbf16>, %arg1: tensor<16x16x32x32x2xbf16>) -> tensor<16x16x32x32x2xbf16> {
  %1 = tensor.empty() : tensor<16x32x32x32xbf16>
  %pack_0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1
    : tensor<1024x512xbf16> -> tensor<16x32x32x32xbf16>
  %pack_1 = linalg.pack %pack_0 inner_dims_pos = [1] inner_tiles = [2] into %arg1
    : tensor<16x32x32x32xbf16> -> tensor<16x16x32x32x2xbf16>
  return %pack_1 : tensor<16x16x32x32x2xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 32)>

// CHECK-LABEL: pack_fusion_outer_only
// CHECK-SAME: %[[ARG0:.+]]: tensor<1024x512xbf16>, %[[ARG1:.+]]: tensor<16x16x32x32x2xbf16>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]]) in (16) shared_outs(%[[ARG3:.+]] = %[[ARG1]])
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][0, %[[APPLY]]] [1024, 32] [1, 1] : tensor<1024x512xbf16> to tensor<1024x32xbf16>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x32x32x32xbf16>
// CHECK: %[[EXP:.+]] = tensor.expand_shape %[[SLICE]] {{\[}}[0, 1], [2, 3]] output_shape [32, 32, 1, 32] : tensor<1024x32xbf16> into tensor<32x32x1x32xbf16>
// CHECK: %[[TRAN:.+]] = linalg.transpose ins(%[[EXP]] : tensor<32x32x1x32xbf16>) outs(%[[EMPTY]] : tensor<1x32x32x32xbf16>)
// CHECK-SAME:  permutation = [2, 0, 1, 3]
// CHECK: %[[SLICE_1:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG2]], 0, 0, 0, 0] [1, 16, 32, 32, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<16x16x32x32x2xbf16> to tensor<1x16x32x32x2xbf16>
// CHECK: %[[EXP_2:.+]] = tensor.expand_shape %[[TRAN]] {{\[}}[0], [1, 2], [3], [4]] output_shape [1, 16, 2, 32, 32] : tensor<1x32x32x32xbf16> into tensor<1x16x2x32x32xbf16>
// CHECK: %[[TRAN_2:.+]] = linalg.transpose ins(%[[EXP_2]] : tensor<1x16x2x32x32xbf16>) outs(%[[SLICE_1]] : tensor<1x16x32x32x2xbf16>)
// CHECK-SAME:  permutation = [0, 1, 3, 4, 2]
// CHECK: tensor.parallel_insert_slice %[[TRAN_2]] into %[[ARG3]][%[[ARG2]], 0, 0, 0, 0] [1, 16, 32, 32, 2] [1, 1, 1, 1, 1]
// CHECK-SAME:  : tensor<1x16x32x32x2xbf16> into tensor<16x16x32x32x2xbf16>

// -----

func.func @vnni_packing(%arg0: tensor<16x16xbf16>, %arg1: tensor<8x16x2xbf16>) -> tensor<8x16x2xbf16> {
  %pack = linalg.pack %arg0 inner_dims_pos = [0] inner_tiles = [2] into %arg1 : tensor<16x16xbf16> -> tensor<8x16x2xbf16>
  return %pack : tensor<8x16x2xbf16>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL: vnni_packing
// CHECK-SAME: %[[ARG0:.+]]: tensor<16x16xbf16>, %[[ARG1:.+]]: tensor<8x16x2xbf16>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]]) in (8) shared_outs(%[[ARG3:.+]] = %[[ARG1]])
// CHECK: %[[APPLY:.+]] = affine.apply #[[MAP]](%[[ARG2]])
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[APPLY]], 0] [2, 16] [1, 1]
// CHECK-SAME:  : tensor<16x16xbf16> to tensor<2x16xbf16>
// CHECK: %[[SLICE_0:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG2]], 0, 0] [1, 16, 2] [1, 1, 1]
// CHECK-SAME:  : tensor<8x16x2xbf16> to tensor<1x16x2xbf16>
// CHECK: %[[EXP:.+]] = tensor.expand_shape %extracted_slice {{\[}}[0, 1], [2]]
// CHECK-SAME:  : tensor<2x16xbf16> into tensor<1x2x16xbf16>
// CHECK: %[[TRAN:.+]] = linalg.transpose ins(%[[EXP]] : tensor<1x2x16xbf16>)
// CHECK-SAME:  outs(%[[SLICE_0]] : tensor<1x16x2xbf16>) permutation = [0, 2, 1]
