// RUN: tpp-opt -pack-vnni -split-input-file %s | FileCheck %s

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @brgemm_vnni_2(%arg0: tensor<5x32x64xbf16>, %arg1: tensor<5x64x32xbf16>,
                    %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>{
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<5x32x64xbf16>, tensor<5x64x32xbf16>)
                                    outs(%arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0: tensor<32x32xbf16>
  }
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

// CHECK-LABEL: @brgemm_vnni_2(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<5x32x64xbf16>, %[[ARG1:.+]]: tensor<5x64x32xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<32x32xbf16>
// CHECK: %[[VNNI_A:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1], [2, 3]]
// CHECK-SAME: output_shape{{.*}}: tensor<5x32x64xbf16> into tensor<5x32x32x2xbf16>
// CHECK: %[[PACK:.+]] = linalg.pack %[[ARG1]]
// CHECK-SAME:  inner_dims_pos = [1] inner_tiles = [2]
// CHECK-SAME:  : tensor<5x64x32xbf16> -> tensor<5x32x32x2xbf16>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%[[VNNI_A]], %[[PACK]]
// CHECK-SAME: outs(%[[ARG2]]

// -----

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 4 : i32>>
} {
  func.func @brgemm_vnni_4(%arg0: tensor<5x32x64xbf16>, %arg1: tensor<5x64x32xbf16>,
                    %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>{
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<5x32x64xbf16>, tensor<5x64x32xbf16>)
                                    outs(%arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0: tensor<32x32xbf16>
  }
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

// CHECK-LABEL: @brgemm_vnni_4(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<5x32x64xbf16>, %[[ARG1:.+]]: tensor<5x64x32xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<32x32xbf16>
// CHECK: %[[VNNI_A:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1], [2, 3]]
// CHECK-SAME: output_shape{{.*}}: tensor<5x32x64xbf16> into tensor<5x32x16x4xbf16>
// CHECK: %[[PACK:.+]] = linalg.pack %[[ARG1]]
// CHECK-SAME:  inner_dims_pos = [1] inner_tiles = [4]
// CHECK-SAME:  : tensor<5x64x32xbf16> -> tensor<5x16x32x4xbf16>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%[[VNNI_A]], %[[PACK]]
// CHECK-SAME: outs(%[[ARG2]]

// -----

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 0 : i32>>
} {
  func.func @invalid_vnni_factor_0(%arg0: tensor<5x32x64xbf16>, %arg1: tensor<5x64x32xbf16>,
                    %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>{
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<5x32x64xbf16>, tensor<5x64x32xbf16>)
                                    outs(%arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0: tensor<32x32xbf16>
  }
}

// CHECK-LABEL: @invalid_vnni_factor_0(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<5x32x64xbf16>, %[[ARG1:.+]]: tensor<5x64x32xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<32x32xbf16>
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_reduce_matmul

// -----

// Blocking factor is expected to be divisible by 2.
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 5 : i32>>
} {
  func.func @invalid_vnni_factor_5(%arg0: tensor<5x32x64xbf16>, %arg1: tensor<5x64x32xbf16>,
                    %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>{
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1: tensor<5x32x64xbf16>, tensor<5x64x32xbf16>)
                                    outs(%arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0: tensor<32x32xbf16>
  }
}

// CHECK-LABEL: @invalid_vnni_factor_5(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<5x32x64xbf16>, %[[ARG1:.+]]: tensor<5x64x32xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<32x32xbf16>
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_reduce_matmul
