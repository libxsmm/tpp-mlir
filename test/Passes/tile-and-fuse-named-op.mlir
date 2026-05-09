// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=2,2 use-for-all=false" -cse | FileCheck %s

// CHECK: func.func @matmul_sequence_fusion_expect_no_fusion
func.func @matmul_sequence_fusion_expect_no_fusion(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  return %2 : tensor<32x32xf32>
}

// CHECK-COUNT-2: scf.for
// CHECK: linalg.matmul
// CHECK-COUNT-2: scf.for
// CHECK: linalg.matmul
// CHECK-COUNT-2: scf.for
// CHECK: linalg.matmul

// -----

func.func @matmul_eletwise_matmul_and_relu(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
   %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %3 = linalg.max ins(%0, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %3 : tensor<32x32xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @matmul_eletwise_matmul_and_relu
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// CHECK: linalg.matmul
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME:  {indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:  outs({{.+}} : tensor<2x2xf32>)
// CHECK: arith.maximumf
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }

// -----

func.func @matmul_eletwise_blk_matmul(%arg0: tensor<4x4x32x32xf32>, %arg1: tensor<4x4x32x32xf32>, %arg2: tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32> {
    %0 = tensor.empty() : tensor<4x4x32x32xf32>
    %transposed = linalg.transpose ins(%arg1 : tensor<4x4x32x32xf32>) outs(%0 : tensor<4x4x32x32xf32>) permutation = [0, 1, 3, 2]
    %1 = linalg.mmt4d ins(%arg0, %transposed : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%arg2 : tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : tensor<4x4x32x32xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32>
    %4 = linalg.max ins(%1, %3 : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>) outs(%arg2 : tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32>
    return %4 : tensor<4x4x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @matmul_eletwise_blk_matmul(
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C4]] step %[[C2]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C4]] step %[[C2]]
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.addf
// CHECK: linalg.fill
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: ^bb0(
// CHECK-NEXT: arith.maximumf
// CHECK: scf.yield %{{.+}} : tensor<4x4x32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<4x4x32x32xf32>
// CHECK-NEXT: }

// -----

func.func @matmul_sequence_fusion_with_relu(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  %3 = tensor.empty() : tensor<32x32xf32>
  %4 = linalg.fill ins(%c0 : f32) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %5 = linalg.max ins(%2, %4 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %5 : tensor<32x32xf32>
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @matmul_sequence_fusion_with_relu
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-COUNT-2: linalg.matmul
// CHECK: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// CHECK-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// CHECK: linalg.matmul
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  outs({{.+}} : tensor<2x2xf32>)
// CHECK-NEXT: ^bb0(
// CHECK-NEXT: arith.maximumf
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }
// CHECK: scf.yield %{{.+}} : tensor<32x32xf32>
// CHECK-NEXT: }

// -----

func.func @matmul_chain_multi_use_into_relu(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
   %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %3 = linalg.max ins(%0, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %3 : tensor<32x32xf32>
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK:   func.func @matmul_chain_multi_use_into_relu
// CHECK-SAME:                                               %[[VAL_0:.*]]: tensor<32x64xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: tensor<64x32xf32>,
// CHECK-SAME:                                               %[[VAL_2:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CHECK:           %[[VAL_7:.*]] = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<32x32xf32>) {
// CHECK:             %[[VAL_10:.*]] = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}) -> (tensor<32x32xf32>) {
// CHECK-COUNT-3:       %{{.+}} = tensor.extract_slice
// CHECK-COUNT-1:       %[[VAL_16:.*]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x64xf32>, tensor<64x2xf32>) outs(%{{.+}} : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK-NEXT:               %{{.+}} = linalg.generic
// CHECK-NEXT:               ^bb0
// CHECK:                 %[[VAL_27:.*]] = arith.maximumf %{{.+}}, %{{.+}} : f32
// CHECK:                 linalg.yield %[[VAL_27]] : f32
// CHECK:               } -> tensor<2x2xf32>
// CHECK:             }
// CHECK:           } {parallel = "root"}
// CHECK:           return %{{.+}} : tensor<32x32xf32>
// CHECK:         }

// -----

func.func @negative_matmul_chain_multi_user(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
   %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %1 = tensor.empty() : tensor<32x32xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %3 = linalg.max ins(%0, %2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = linalg.add ins(%3, %0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %4 : tensor<32x32xf32>
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @negative_matmul_chain_multi_user(
// CHECK-SAME:                                                %[[VAL_0:.*]]: tensor<32x64xf32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: tensor<64x32xf32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = scf.for %[[VAL_8:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_9:.*]] = %[[VAL_2]]) -> (tensor<32x32xf32>) {
// CHECK:             %[[VAL_10:.*]] = scf.for %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_12:.*]] = %[[VAL_9]]) -> (tensor<32x32xf32>) {
// CHECK:               %[[VAL_13:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_8]], 0] [2, 64] [1, 1] : tensor<32x64xf32> to tensor<2x64xf32>
// CHECK:               %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_1]][0, %[[VAL_11]]] [64, 2] [1, 1] : tensor<64x32xf32> to tensor<64x2xf32>
// CHECK:               %[[VAL_15:.*]] = tensor.extract_slice %[[VAL_12]]{{\[}}%[[VAL_8]], %[[VAL_11]]] [2, 2] [1, 1] : tensor<32x32xf32> to tensor<2x2xf32>
// CHECK:               %[[VAL_16:.*]] = linalg.matmul ins(%[[VAL_13]], %[[VAL_14]] : tensor<2x64xf32>, tensor<64x2xf32>) outs(%[[VAL_15]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:               %[[VAL_17:.*]] = tensor.insert_slice %[[VAL_16]] into %[[VAL_12]]{{\[}}%[[VAL_8]], %[[VAL_11]]] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<32x32xf32>
// CHECK:               scf.yield %[[VAL_17]] : tensor<32x32xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_10]] : tensor<32x32xf32>
// CHECK:           } {parallel = "root"}
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<32x32xf32>
// CHECK:           %[[VAL_19:.*]] = linalg.fill ins(%[[VAL_3]] : f32) outs(%[[VAL_18]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:           %[[VAL_20:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_7]], %[[VAL_19]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_7]] : tensor<32x32xf32>) {
// CHECK:           ^bb0(%[[VAL_21:.*]]: f32, %[[VAL_22:.*]]: f32, %[[VAL_23:.*]]: f32):
// CHECK:             %[[VAL_24:.*]] = arith.maximumf %[[VAL_21]], %[[VAL_22]] : f32
// CHECK:             linalg.yield %[[VAL_24]] : f32
// CHECK:           } -> tensor<32x32xf32>
// CHECK:           %[[VAL_25:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_20]], %[[VAL_7]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_20]] : tensor<32x32xf32>) {
// CHECK:           ^bb0(%[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32, %[[VAL_28:.*]]: f32):
// CHECK:             %[[VAL_29:.*]] = arith.addf %[[VAL_26]], %[[VAL_27]] : f32
// CHECK:             linalg.yield %[[VAL_29]] : f32
// CHECK:           } -> tensor<32x32xf32>
// CHECK:           return %[[VAL_25]] : tensor<32x32xf32>
// CHECK:         }
