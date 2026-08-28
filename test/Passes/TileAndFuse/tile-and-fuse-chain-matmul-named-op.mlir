// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=2,2 use-for-all=false" -canonicalize | FileCheck -check-prefix=CONF1 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=2,0 use-for-all=false" -canonicalize | FileCheck -check-prefix=CONF2 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=0,2 use-for-all=false" -canonicalize | FileCheck -check-prefix=CONF3 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=0,0 use-for-all=false" -canonicalize | FileCheck -check-prefix=CONF4 %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_sequence_fusion(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  %3 = tensor.empty() : tensor<32x32xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %5 = linalg.max ins(%2, %4 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %5 : tensor<32x32xf32>
}

// CONF1: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CONF1-LABEL:   func.func @matmul_sequence_fusion(
// CONF1-SAME:                                      %[[VAL_0:.*]]: tensor<32x64xf32>, %[[VAL_1:.*]]: tensor<64x32xf32>, %[[VAL_2:.*]]: tensor<32x32xf32>,
// CONF1-SAME:                                      %[[VAL_3:.*]]: tensor<32x64xf32>, %[[VAL_4:.*]]: tensor<32x64xf32>,
// CONF1-SAME:                                      %[[VAL_5:.*]]: tensor<64x32xf32>,
// CONF1-SAME:                                      %[[VAL_6:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CONF1:           %[[VAL_7:.*]] = arith.constant 64 : index
// CONF1:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f32
// CONF1:           %[[VAL_9:.*]] = arith.constant 0 : index
// CONF1:           %[[VAL_10:.*]] = arith.constant 32 : index
// CONF1:           %[[VAL_11:.*]] = arith.constant 2 : index
// CONF1:           %[[VAL_12:.*]] = scf.for %[[VAL_13:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_14:.*]] = %[[VAL_2]]) -> (tensor<32x32xf32>) {
// CONF1:             %[[VAL_15:.*]] = scf.for %[[VAL_16:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_17:.*]] = %[[VAL_14]]) -> (tensor<32x32xf32>) {
// CONF1:               %[[VAL_18:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_13]], 0] [2, 64] [1, 1] : tensor<32x64xf32> to tensor<2x64xf32>
// CONF1:               %[[VAL_19:.*]] = tensor.extract_slice %[[VAL_1]][0, %[[VAL_16]]] [64, 2] [1, 1] : tensor<64x32xf32> to tensor<64x2xf32>
// CONF1:               %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_17]]{{\[}}%[[VAL_13]], %[[VAL_16]]] [2, 2] [1, 1] : tensor<32x32xf32> to tensor<2x2xf32>
// CONF1:               %[[VAL_21:.*]] = linalg.matmul ins(%[[VAL_18]], %[[VAL_19]] : tensor<2x64xf32>, tensor<64x2xf32>) outs(%[[VAL_20]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CONF1:               %[[VAL_22:.*]] = tensor.insert_slice %[[VAL_21]] into %[[VAL_17]]{{\[}}%[[VAL_13]], %[[VAL_16]]] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<32x32xf32>
// CONF1:               scf.yield %[[VAL_22]] : tensor<32x32xf32>
// CONF1:             }
// CONF1:             scf.yield %[[VAL_15]] : tensor<32x32xf32>
// CONF1:           } {parallel = "root"}
// CONF1:           %[[VAL_23:.*]] = scf.for %[[VAL_24:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_25:.*]] = %[[VAL_4]]) -> (tensor<32x64xf32>) {
// CONF1:             %[[VAL_26:.*]] = scf.for %[[VAL_27:.*]] = %[[VAL_9]] to %[[VAL_7]] step %[[VAL_11]] iter_args(%[[VAL_28:.*]] = %[[VAL_25]]) -> (tensor<32x64xf32>) {
// CONF1:               %[[VAL_29:.*]] = tensor.extract_slice %[[VAL_12]]{{\[}}%[[VAL_24]], 0] [2, 32] [1, 1] : tensor<32x32xf32> to tensor<2x32xf32>
// CONF1:               %[[VAL_30:.*]] = tensor.extract_slice %[[VAL_3]][0, %[[VAL_27]]] [32, 2] [1, 1] : tensor<32x64xf32> to tensor<32x2xf32>
// CONF1:               %[[VAL_31:.*]] = tensor.extract_slice %[[VAL_28]]{{\[}}%[[VAL_24]], %[[VAL_27]]] [2, 2] [1, 1] : tensor<32x64xf32> to tensor<2x2xf32>
// CONF1:               %[[VAL_32:.*]] = linalg.matmul ins(%[[VAL_29]], %[[VAL_30]] : tensor<2x32xf32>, tensor<32x2xf32>) outs(%[[VAL_31]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CONF1:               %[[VAL_33:.*]] = tensor.insert_slice %[[VAL_32]] into %[[VAL_28]]{{\[}}%[[VAL_24]], %[[VAL_27]]] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<32x64xf32>
// CONF1:               scf.yield %[[VAL_33]] : tensor<32x64xf32>
// CONF1:             }
// CONF1:             scf.yield %[[VAL_26]] : tensor<32x64xf32>
// CONF1:           } {parallel = "root"}
// CONF1:           %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
// CONF1:           %[[VAL_34:.*]] = scf.for %[[VAL_35:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_36:.*]] = %[[VAL_2]]) -> (tensor<32x32xf32>) {
// CONF1:             %[[VAL_37:.*]] = scf.for %[[VAL_38:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_39:.*]] = %[[VAL_36]]) -> (tensor<32x32xf32>) {
// CONF1:               %[[VAL_40:.*]] = tensor.extract_slice %[[VAL_23]]{{\[}}%[[VAL_35]], 0] [2, 64] [1, 1] : tensor<32x64xf32> to tensor<2x64xf32>
// CONF1:               %[[VAL_41:.*]] = tensor.extract_slice %[[VAL_5]][0, %[[VAL_38]]] [64, 2] [1, 1] : tensor<64x32xf32> to tensor<64x2xf32>
// CONF1:               %[[VAL_42:.*]] = tensor.extract_slice %[[VAL_6]]{{\[}}%[[VAL_35]], %[[VAL_38]]] [2, 2] [1, 1] : tensor<32x32xf32> to tensor<2x2xf32>
// CONF1:               %[[VAL_43:.*]] = linalg.matmul ins(%[[VAL_40]], %[[VAL_41]] : tensor<2x64xf32>, tensor<64x2xf32>) outs(%[[VAL_42]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CONF1:               %[[VAL_44:.*]] = tensor.extract_slice %[[EMPTY]]{{\[}}%[[VAL_35]], %[[VAL_38]]] [2, 2] [1, 1] : tensor<32x32xf32> to tensor<2x2xf32>
// CONF1:               %[[VAL_45:.*]] = linalg.fill ins(%[[VAL_8]] : f32) outs(%[[VAL_44]] : tensor<2x2xf32>) -> tensor<2x2xf32>
// CONF1:               %[[VAL_46:.*]] = tensor.extract_slice %[[VAL_39]]{{\[}}%[[VAL_35]], %[[VAL_38]]] [2, 2] [1, 1] : tensor<32x32xf32> to tensor<2x2xf32>
// CONF1:               %[[VAL_47:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_43]], %[[VAL_45]] : tensor<2x2xf32>, tensor<2x2xf32>) outs(%[[VAL_46]] : tensor<2x2xf32>) {
// CONF1:               ^bb0(%[[VAL_48:.*]]: f32, %[[VAL_49:.*]]: f32, %[[VAL_50:.*]]: f32):
// CONF1:                 %[[VAL_51:.*]] = arith.maximumf %[[VAL_48]], %[[VAL_49]] : f32
// CONF1:                 linalg.yield %[[VAL_51]] : f32
// CONF1:               } -> tensor<2x2xf32>
// CONF1:               %[[VAL_52:.*]] = tensor.insert_slice %[[VAL_47]] into %[[VAL_39]]{{\[}}%[[VAL_35]], %[[VAL_38]]] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<32x32xf32>
// CONF1:               scf.yield %[[VAL_52]] : tensor<32x32xf32>
// CONF1:             }
// CONF1:             scf.yield %[[VAL_37]] : tensor<32x32xf32>
// CONF1:           } {parallel = "root"}
// CONF1:           return %[[VAL_34]] : tensor<32x32xf32>
// CONF1:         }


// CONF2: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF2-LABEL:   func.func @matmul_sequence_fusion(
// CONF2-SAME:                                      %[[VAL_0:.*]]: tensor<32x64xf32>, %[[VAL_1:.*]]: tensor<64x32xf32>, %[[VAL_2:.*]]: tensor<32x32xf32>,
// CONF2-SAME:                                      %[[VAL_3:.*]]: tensor<32x64xf32>, %[[VAL_4:.*]]: tensor<32x64xf32>, %[[VAL_5:.*]]: tensor<64x32xf32>,
// CONF2-SAME:                                      %[[VAL_6:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CONF2:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
// CONF2:           %[[VAL_8:.*]] = arith.constant 0 : index
// CONF2:           %[[VAL_9:.*]] = arith.constant 32 : index
// CONF2:           %[[VAL_10:.*]] = arith.constant 2 : index
// CONF2:           %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
// CONF2:           %[[VAL_11:.*]] = scf.for %[[VAL_12:.*]] = %[[VAL_8]] to %[[VAL_9]] step %[[VAL_10]] iter_args(%[[VAL_13:.*]] = %[[VAL_2]]) -> (tensor<32x32xf32>) {
// CONF2:             %[[VAL_14:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_12]], 0] [2, 64] [1, 1] : tensor<32x64xf32> to tensor<2x64xf32>
// CONF2:             %[[VAL_15:.*]] = tensor.extract_slice %[[VAL_13]]{{\[}}%[[VAL_12]], 0] [2, 32] [1, 1] : tensor<32x32xf32> to tensor<2x32xf32>
// CONF2:             %[[VAL_16:.*]] = linalg.matmul ins(%[[VAL_14]], %[[VAL_1]] : tensor<2x64xf32>, tensor<64x32xf32>) outs(%[[VAL_15]] : tensor<2x32xf32>) -> tensor<2x32xf32>
// CONF2:             %[[VAL_17:.*]] = tensor.extract_slice %[[VAL_4]]{{\[}}%[[VAL_12]], 0] [2, 64] [1, 1] : tensor<32x64xf32> to tensor<2x64xf32>
// CONF2:             %[[VAL_18:.*]] = linalg.matmul ins(%[[VAL_16]], %[[VAL_3]] : tensor<2x32xf32>, tensor<32x64xf32>) outs(%[[VAL_17]] : tensor<2x64xf32>) -> tensor<2x64xf32>
// CONF2:             %[[VAL_19:.*]] = tensor.extract_slice %[[VAL_6]]{{\[}}%[[VAL_12]], 0] [2, 32] [1, 1] : tensor<32x32xf32> to tensor<2x32xf32>
// CONF2:             %[[VAL_20:.*]] = linalg.matmul ins(%[[VAL_18]], %[[VAL_5]] : tensor<2x64xf32>, tensor<64x32xf32>) outs(%[[VAL_19]] : tensor<2x32xf32>) -> tensor<2x32xf32>
// CONF2:             %[[VAL_21:.*]] = tensor.extract_slice %[[EMPTY]]{{\[}}%[[VAL_12]], 0] [2, 32] [1, 1] : tensor<32x32xf32> to tensor<2x32xf32>
// CONF2:             %[[VAL_22:.*]] = linalg.fill ins(%[[VAL_7]] : f32) outs(%[[VAL_21]] : tensor<2x32xf32>) -> tensor<2x32xf32>
// CONF2:             %[[VAL_23:.*]] = tensor.extract_slice %[[VAL_13]]{{\[}}%[[VAL_12]], 0] [2, 32] [1, 1] : tensor<32x32xf32> to tensor<2x32xf32>
// CONF2:             %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_20]], %[[VAL_22]] : tensor<2x32xf32>, tensor<2x32xf32>) outs(%[[VAL_23]] : tensor<2x32xf32>) {
// CONF2:             ^bb0(%[[VAL_25:.*]]: f32, %[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
// CONF2:               %[[VAL_28:.*]] = arith.maximumf %[[VAL_25]], %[[VAL_26]] : f32
// CONF2:               linalg.yield %[[VAL_28]] : f32
// CONF2:             } -> tensor<2x32xf32>
// CONF2:             %[[VAL_29:.*]] = tensor.insert_slice %[[VAL_24]] into %[[VAL_13]]{{\[}}%[[VAL_12]], 0] [2, 32] [1, 1] : tensor<2x32xf32> into tensor<32x32xf32>
// CONF2:             scf.yield %[[VAL_29]] : tensor<32x32xf32>
// CONF2:           } {parallel = "root"}
// CONF2:           return %[[VAL_11]] : tensor<32x32xf32>
// CONF2:         }


// CONF3: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF3-LABEL:   func.func @matmul_sequence_fusion(
// CONF3-SAME:                                      %[[VAL_0:.*]]: tensor<32x64xf32>, %[[VAL_1:.*]]: tensor<64x32xf32>, %[[VAL_2:.*]]: tensor<32x32xf32>,
// CONF3-SAME:                                      %[[VAL_3:.*]]: tensor<32x64xf32>, %[[VAL_4:.*]]: tensor<32x64xf32>, %[[VAL_5:.*]]: tensor<64x32xf32>,
// CONF3-SAME:                                      %[[VAL_6:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CONF3:           %[[VAL_7:.*]] = arith.constant 64 : index
// CONF3:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f32
// CONF3:           %[[VAL_9:.*]] = arith.constant 0 : index
// CONF3:           %[[VAL_10:.*]] = arith.constant 32 : index
// CONF3:           %[[VAL_11:.*]] = arith.constant 2 : index
// CONF3:           %[[VAL_12:.*]] = scf.for %[[VAL_13:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_14:.*]] = %[[VAL_2]]) -> (tensor<32x32xf32>) {
// CONF3:             %[[VAL_15:.*]] = tensor.extract_slice %[[VAL_1]][0, %[[VAL_13]]] [64, 2] [1, 1] : tensor<64x32xf32> to tensor<64x2xf32>
// CONF3:             %[[VAL_16:.*]] = tensor.extract_slice %[[VAL_14]][0, %[[VAL_13]]] [32, 2] [1, 1] : tensor<32x32xf32> to tensor<32x2xf32>
// CONF3:             %[[VAL_17:.*]] = linalg.matmul ins(%[[VAL_0]], %[[VAL_15]] : tensor<32x64xf32>, tensor<64x2xf32>) outs(%[[VAL_16]] : tensor<32x2xf32>) -> tensor<32x2xf32>
// CONF3:             %[[VAL_18:.*]] = tensor.insert_slice %[[VAL_17]] into %[[VAL_14]][0, %[[VAL_13]]] [32, 2] [1, 1] : tensor<32x2xf32> into tensor<32x32xf32>
// CONF3:             scf.yield %[[VAL_18]] : tensor<32x32xf32>
// CONF3:           } {parallel = "root"}
// CONF3:           %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_9]] to %[[VAL_7]] step %[[VAL_11]] iter_args(%[[VAL_21:.*]] = %[[VAL_4]]) -> (tensor<32x64xf32>) {
// CONF3:             %[[VAL_22:.*]] = tensor.extract_slice %[[VAL_3]][0, %[[VAL_20]]] [32, 2] [1, 1] : tensor<32x64xf32> to tensor<32x2xf32>
// CONF3:             %[[VAL_23:.*]] = tensor.extract_slice %[[VAL_21]][0, %[[VAL_20]]] [32, 2] [1, 1] : tensor<32x64xf32> to tensor<32x2xf32>
// CONF3:             %[[VAL_24:.*]] = linalg.matmul ins(%[[VAL_12]], %[[VAL_22]] : tensor<32x32xf32>, tensor<32x2xf32>) outs(%[[VAL_23]] : tensor<32x2xf32>) -> tensor<32x2xf32>
// CONF3:             %[[VAL_25:.*]] = tensor.insert_slice %[[VAL_24]] into %[[VAL_21]][0, %[[VAL_20]]] [32, 2] [1, 1] : tensor<32x2xf32> into tensor<32x64xf32>
// CONF3:             scf.yield %[[VAL_25]] : tensor<32x64xf32>
// CONF3:           } {parallel = "root"}
// CONF3:           %[[EMPTY:.*]] = tensor.empty() : tensor<32x32xf32>
// CONF3:           %[[VAL_26:.*]] = scf.for %[[VAL_27:.*]] = %[[VAL_9]] to %[[VAL_10]] step %[[VAL_11]] iter_args(%[[VAL_28:.*]] = %[[VAL_2]]) -> (tensor<32x32xf32>) {
// CONF3:             %[[VAL_29:.*]] = tensor.extract_slice %[[VAL_5]][0, %[[VAL_27]]] [64, 2] [1, 1] : tensor<64x32xf32> to tensor<64x2xf32>
// CONF3:             %[[VAL_30:.*]] = tensor.extract_slice %[[VAL_6]][0, %[[VAL_27]]] [32, 2] [1, 1] : tensor<32x32xf32> to tensor<32x2xf32>
// CONF3:             %[[VAL_31:.*]] = linalg.matmul ins(%[[VAL_19]], %[[VAL_29]] : tensor<32x64xf32>, tensor<64x2xf32>) outs(%[[VAL_30]] : tensor<32x2xf32>) -> tensor<32x2xf32>
// CONF3:             %[[VAL_32:.*]] = tensor.extract_slice %[[EMPTY]][0, %[[VAL_27]]] [32, 2] [1, 1] : tensor<32x32xf32> to tensor<32x2xf32>
// CONF3:             %[[VAL_33:.*]] = linalg.fill ins(%[[VAL_8]] : f32) outs(%[[VAL_32]] : tensor<32x2xf32>) -> tensor<32x2xf32>
// CONF3:             %[[VAL_34:.*]] = tensor.extract_slice %[[VAL_28]][0, %[[VAL_27]]] [32, 2] [1, 1] : tensor<32x32xf32> to tensor<32x2xf32>
// CONF3:             %[[VAL_35:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_31]], %[[VAL_33]] : tensor<32x2xf32>, tensor<32x2xf32>) outs(%[[VAL_34]] : tensor<32x2xf32>) {
// CONF3:             ^bb0(%[[VAL_36:.*]]: f32, %[[VAL_37:.*]]: f32, %[[VAL_38:.*]]: f32):
// CONF3:               %[[VAL_39:.*]] = arith.maximumf %[[VAL_36]], %[[VAL_37]] : f32
// CONF3:               linalg.yield %[[VAL_39]] : f32
// CONF3:             } -> tensor<32x2xf32>
// CONF3:             %[[VAL_40:.*]] = tensor.insert_slice %[[VAL_35]] into %[[VAL_28]][0, %[[VAL_27]]] [32, 2] [1, 1] : tensor<32x2xf32> into tensor<32x32xf32>
// CONF3:             scf.yield %[[VAL_40]] : tensor<32x32xf32>
// CONF3:           } {parallel = "root"}
// CONF3:           return %[[VAL_26]] : tensor<32x32xf32>
// CONF3:         }


// CONF4: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF4-LABEL:   func.func @matmul_sequence_fusion(
// CONF4-SAME:                                      %[[VAL_0:.*]]: tensor<32x64xf32>, %[[VAL_1:.*]]: tensor<64x32xf32>, %[[VAL_2:.*]]: tensor<32x32xf32>,
// CONF4-SAME:                                      %[[VAL_3:.*]]: tensor<32x64xf32>, %[[VAL_4:.*]]: tensor<32x64xf32>, %[[VAL_5:.*]]: tensor<64x32xf32>,
// CONF4-SAME:                                      %[[VAL_6:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CONF4:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
// CONF4:           %[[VAL_8:.*]] = linalg.matmul ins(%[[VAL_0]], %[[VAL_1]] : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[VAL_2]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CONF4:           %[[VAL_9:.*]] = linalg.matmul ins(%[[VAL_8]], %[[VAL_3]] : tensor<32x32xf32>, tensor<32x64xf32>) outs(%[[VAL_4]] : tensor<32x64xf32>) -> tensor<32x64xf32>
// CONF4:           %[[VAL_10:.*]] = linalg.matmul ins(%[[VAL_9]], %[[VAL_5]] : tensor<32x64xf32>, tensor<64x32xf32>) outs(%[[VAL_6]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CONF4:           %[[VAL_11:.*]] = tensor.empty() : tensor<32x32xf32>
// CONF4:           %[[VAL_12:.*]] = linalg.fill ins(%[[VAL_7]] : f32) outs(%[[VAL_11]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CONF4:           %[[VAL_13:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_10]], %[[VAL_12]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[VAL_2]] : tensor<32x32xf32>) {
// CONF4:           ^bb0(%[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32, %[[VAL_16:.*]]: f32):
// CONF4:             %[[VAL_17:.*]] = arith.maximumf %[[VAL_14]], %[[VAL_15]] : f32
// CONF4:             linalg.yield %[[VAL_17]] : f32
// CONF4:           } -> tensor<32x32xf32>
// CONF4:           return %[[VAL_13]] : tensor<32x32xf32>
// CONF4:         }