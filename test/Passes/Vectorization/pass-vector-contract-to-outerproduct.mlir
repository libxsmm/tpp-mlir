// RUN: tpp-opt %s  --vector-contract-to-outerproduct --split-input-file | FileCheck %s


#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @matmul_k_one(%arg0: tensor<4x1xf32>, %arg1: tensor<1x64xf32>, %arg2: tensor<4x64xf32>) -> tensor<4x64xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x1xf32>, vector<4x1xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x1xf32>, vector<1x64xf32> into vector<4x64xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
    return %4 : tensor<4x64xf32>
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:   func.func @matmul_k_one(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<4x1xf32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: tensor<1x64xf32>,
// CHECK-SAME:                            %[[VAL_2:.*]]: tensor<4x64xf32>) -> tensor<4x64xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]]], %[[VAL_5]] {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
// CHECK:           %[[VAL_7:.*]] = scf.for %[[VAL_8:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_3]] iter_args(%[[VAL_9:.*]] = %[[VAL_6]]) -> (vector<4x64xf32>) {
// CHECK:             %[[VAL_10:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_8]]], %[[VAL_5]] {in_bounds = [true], permutation_map = #[[$ATTR_0]]} : tensor<4x1xf32>, vector<4xf32>
// CHECK:             %[[VAL_11:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_8]], %[[VAL_4]]], %[[VAL_5]] {in_bounds = [true]} : tensor<1x64xf32>, vector<64xf32>
// CHECK:             %[[VAL_12:.*]] = vector.outerproduct %[[VAL_10]], %[[VAL_11]], %[[VAL_9]] {kind = #{{.*}}<add>} : vector<4xf32>, vector<64xf32>
// CHECK:             scf.yield %[[VAL_12]] : vector<4x64xf32>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = vector.transfer_write %[[VAL_7]], %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]]] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
// CHECK:           return %[[VAL_13]] : tensor<4x64xf32>
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @matmul_k_one_bf16(%arg0: tensor<4x1xbf16>, %arg1: tensor<1x64xbf16>, %arg2: tensor<4x64xbf16>) -> tensor<4x64xbf16> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x1xbf16>, vector<4x1xbf16>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xbf16>, vector<1x64xbf16>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x64xbf16>, vector<4x64xbf16>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x1xbf16>, vector<1x64xbf16> into vector<4x64xbf16>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xbf16>, tensor<4x64xbf16>
    return %4 : tensor<4x64xbf16>
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:   func.func @matmul_k_one_bf16(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<4x1xbf16>,
// CHECK-SAME:                            %[[VAL_1:.*]]: tensor<1x64xbf16>,
// CHECK-SAME:                            %[[VAL_2:.*]]: tensor<4x64xbf16>) -> tensor<4x64xbf16> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]]], %[[VAL_5]] {in_bounds = [true, true]} : tensor<4x64xbf16>, vector<4x64xbf16>
// CHECK:           %[[VAL_7:.*]] = scf.for %[[VAL_8:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_3]] iter_args(%[[VAL_9:.*]] = %[[VAL_6]]) -> (vector<4x64xbf16>) {
// CHECK:             %[[VAL_10:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_8]]], %[[VAL_5]] {in_bounds = [true], permutation_map = #[[$ATTR_0]]} : tensor<4x1xbf16>, vector<4xbf16>
// CHECK:             %[[VAL_11:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_8]], %[[VAL_4]]], %[[VAL_5]] {in_bounds = [true]} : tensor<1x64xbf16>, vector<64xbf16>
// CHECK:             %[[VAL_12:.*]] = vector.outerproduct %[[VAL_10]], %[[VAL_11]], %[[VAL_9]] {kind = #{{.*}}<add>} : vector<4xbf16>, vector<64xbf16>
// CHECK:             scf.yield %[[VAL_12]] : vector<4x64xbf16>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = vector.transfer_write %[[VAL_7]], %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]]] {in_bounds = [true, true]} : vector<4x64xbf16>, tensor<4x64xbf16>
// CHECK:           return %[[VAL_13]] : tensor<4x64xbf16>
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @matmul_k_one_memref(%arg0: memref<4x1xf32>, %arg1: memref<1x64xf32>, %arg2: memref<4x64xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x1xf32>, vector<4x1xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32>, vector<1x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32>, vector<4x64xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x1xf32>, vector<1x64xf32> into vector<4x64xf32>
    vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32>
    return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:   func.func @matmul_k_one_memref(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<4x1xf32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: memref<1x64xf32>,
// CHECK-SAME:                            %[[VAL_2:.*]]: memref<4x64xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]]], %[[VAL_5]] {in_bounds = [true, true]} : memref<4x64xf32>, vector<4x64xf32>
// CHECK:           %[[VAL_7:.*]] = scf.for %[[VAL_8:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_3]] iter_args(%[[VAL_9:.*]] = %[[VAL_6]]) -> (vector<4x64xf32>) {
// CHECK:             %[[VAL_10:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_8]]], %[[VAL_5]] {in_bounds = [true], permutation_map = #[[$ATTR_0]]} : memref<4x1xf32>, vector<4xf32>
// CHECK:             %[[VAL_11:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_8]], %[[VAL_4]]], %[[VAL_5]] {in_bounds = [true]} : memref<1x64xf32>, vector<64xf32>
// CHECK:             %[[VAL_12:.*]] = vector.outerproduct %[[VAL_10]], %[[VAL_11]], %[[VAL_9]] {kind = #{{.*}}<add>} : vector<4xf32>, vector<64xf32>
// CHECK:             scf.yield %[[VAL_12]] : vector<4x64xf32>
// CHECK:           }
// CHECK:           vector.transfer_write %[[VAL_7]], %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]]] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32>
// CHECK:           return
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @matmul_any_k(%arg0: tensor<4x2xf32>, %arg1: tensor<2x64xf32>, %arg2: tensor<4x64xf32>) -> tensor<4x64xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x2xf32>, vector<4x2xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<2x64xf32>, vector<2x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x2xf32>, vector<2x64xf32> into vector<4x64xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
    return %4 : tensor<4x64xf32>
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL:   func.func @matmul_any_k(
// CHECK-SAME:                      %[[VAL_0:.*]]: tensor<4x2xf32>,
// CHECK-SAME:                      %[[VAL_1:.*]]: tensor<2x64xf32>,
// CHECK-SAME:                      %[[VAL_2:.*]]: tensor<4x64xf32>) -> tensor<4x64xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_5]], %[[VAL_5]]], %[[VAL_6]] {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
// CHECK:           %[[VAL_8:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_10:.*]] = %[[VAL_7]]) -> (vector<4x64xf32>) {
// CHECK:             %[[VAL_11:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_5]], %[[VAL_9]]], %[[VAL_6]] {in_bounds = [true], permutation_map = #[[$ATTR_0]]} : tensor<4x2xf32>, vector<4xf32>
// CHECK:             %[[VAL_12:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_5]]], %[[VAL_6]] {in_bounds = [true]} : tensor<2x64xf32>, vector<64xf32>
// CHECK:             %[[VAL_13:.*]] = vector.outerproduct %[[VAL_11]], %[[VAL_12]], %[[VAL_10]] {kind = #{{.*}}<add>} : vector<4xf32>, vector<64xf32>
// CHECK:             scf.yield %[[VAL_13]] : vector<4x64xf32>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = vector.transfer_write %[[VAL_8]], %[[VAL_2]]{{\[}}%[[VAL_5]], %[[VAL_5]]] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
// CHECK:           return %[[VAL_14]] : tensor<4x64xf32>
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @batch_reduce_matmul_tiled(%arg0: memref<16x32x16x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<16x32x16x32xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg3, %arg4) in (16, 32) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
      scf.for %arg5 = %c0 to %c16 step %c4 {  // m-loop
        scf.for %arg6 = %c0 to %c32 step %c16 { //n-loop
          scf.for %arg7 = %c0 to %c32 step %c1 { //BRGEMM-loop
            scf.for %arg8 = %c0 to %c32 step %c8 { //k-loop
              %subview_2 = memref.subview %subview[%arg7, %arg5, %arg8] [1, 4, 4] [1, 1, 1] : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[512, 32, 1], offset: ?>>
              %subview_3 = memref.subview %subview_0[%arg7, %arg8, %arg6] [1, 4, 2] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
              %subview_4 = memref.subview %subview_1[%arg5, %arg6] [4, 2] [1, 1] : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
              %0 = vector.transfer_read %subview_2[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x4xf32, strided<[512, 32, 1], offset: ?>>, vector<1x4x4xf32>
              %1 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x4x2xf32>
              %2 = vector.transfer_read %subview_4[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x2xf32, strided<[32, 1], offset: ?>>, vector<4x2xf32>
              %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<1x4x4xf32>, vector<1x4x2xf32> into vector<4x2xf32>
              vector.transfer_write %3, %subview_4[%c0, %c0] {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32, strided<[32, 1], offset: ?>>
            }
          }
        }
      }
    }
    return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-LABEL:   func.func @batch_reduce_matmul_tiled(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<16x32x16x32xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<32x32x32x32xf32>,
// CHECK-SAME:                     %[[VAL_2:.*]]: memref<16x32x16x32xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           scf.forall (%[[VAL_10:.*]], %[[VAL_11:.*]]) in (16, 32) {
// CHECK:             %[[VAL_12:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_10]], 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CHECK:             %[[VAL_13:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_11]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             %[[VAL_14:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_9]] to %[[VAL_8]] step %[[VAL_7]] {
// CHECK:               scf.for %[[VAL_16:.*]] = %[[VAL_9]] to %[[VAL_6]] step %[[VAL_8]] {
// CHECK:                 scf.for %[[VAL_17:.*]] = %[[VAL_9]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:                   scf.for %[[VAL_18:.*]] = %[[VAL_9]] to %[[VAL_6]] step %[[VAL_4]] {
// CHECK:                     %[[VAL_19:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_17]], %[[VAL_15]], %[[VAL_18]]] [1, 4, 4] [1, 1, 1] : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>> to memref<1x4x4xf32, strided<[512, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_20:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_17]], %[[VAL_18]], %[[VAL_16]]] [1, 4, 2] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_21:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_15]], %[[VAL_16]]] [4, 2] [1, 1] : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:                     %[[VAL_22:.*]] = vector.transfer_read %[[VAL_21]]{{\[}}%[[VAL_9]], %[[VAL_9]]], %[[VAL_3]] {in_bounds = [true, true]} : memref<4x2xf32, strided<[32, 1], offset: ?>>, vector<4x2xf32>
// CHECK:                     %[[VAL_23:.*]] = scf.for %[[VAL_24:.*]] = %[[VAL_9]] to %[[VAL_7]] step %[[VAL_5]] iter_args(%[[VAL_25:.*]] = %[[VAL_22]]) -> (vector<4x2xf32>) {
// CHECK:                       %[[VAL_26:.*]] = vector.transfer_read %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_9]], %[[VAL_24]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #[[$ATTR_0]]} : memref<1x4x4xf32, strided<[512, 32, 1], offset: ?>>, vector<4xf32>
// CHECK:                       %[[VAL_27:.*]] = vector.transfer_read %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_24]], %[[VAL_9]]], %[[VAL_3]] {in_bounds = [true], permutation_map = #[[$ATTR_1]]} : memref<1x4x2xf32, strided<[1024, 32, 1], offset: ?>>, vector<2xf32>
// CHECK:                       %[[VAL_28:.*]] = vector.outerproduct %[[VAL_26]], %[[VAL_27]], %[[VAL_25]] {kind = #{{.*}}<add>} : vector<4xf32>, vector<2xf32>
// CHECK:                       scf.yield %[[VAL_28]] : vector<4x2xf32>
// CHECK:                     }
// CHECK:                     vector.transfer_write %[[VAL_23]], %[[VAL_21]]{{\[}}%[[VAL_9]], %[[VAL_9]]] {in_bounds = [true, true]} : vector<4x2xf32>, memref<4x2xf32, strided<[32, 1], offset: ?>>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<4x4xf32>, %arg1: tensor<8x64xf32>,
      %arg2: tensor<8x64xf32>) -> tensor<8x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0],
      %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x2xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0],
      %cst {in_bounds = [true, true]} : tensor<8x64xf32>, vector<2x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0],
      %cst {in_bounds = [true, true]} : tensor<8x64xf32>, vector<4x64xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>} %0, %1, %2 : vector<4x2xf32>, vector<2x64xf32>
      into vector<4x64xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]}
      : vector<4x64xf32>, tensor<8x64xf32>
    return %4 : tensor<8x64xf32>
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL:   func.func @entry(
// CHECK-SAME:                     %[[VAL_0:.*]]: tensor<4x4xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: tensor<8x64xf32>,
// CHECK-SAME:                     %[[VAL_2:.*]]: tensor<8x64xf32>) -> tensor<8x64xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_5]], %[[VAL_5]]], %[[VAL_6]] {in_bounds = [true, true]} : tensor<8x64xf32>, vector<4x64xf32>
// CHECK:           %[[VAL_8:.*]] = scf.for %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_10:.*]] = %[[VAL_7]]) -> (vector<4x64xf32>) {
// CHECK:             %[[VAL_11:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_5]], %[[VAL_9]]], %[[VAL_6]] {in_bounds = [true], permutation_map = #[[$ATTR_0]]} : tensor<4x4xf32>, vector<4xf32>
// CHECK:             %[[VAL_12:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_5]]], %[[VAL_6]] {in_bounds = [true]} : tensor<8x64xf32>, vector<64xf32>
// CHECK:             %[[VAL_13:.*]] = vector.outerproduct %[[VAL_11]], %[[VAL_12]], %[[VAL_10]] {kind = #{{.*}}<add>} : vector<4xf32>, vector<64xf32>
// CHECK:             scf.yield %[[VAL_13]] : vector<4x64xf32>
// CHECK:           }
// CHECK:           %[[VAL_14:.*]] = vector.transfer_write %[[VAL_8]], %[[VAL_2]]{{\[}}%[[VAL_5]], %[[VAL_5]]] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<8x64xf32>
// CHECK:           return %[[VAL_14]] : tensor<8x64xf32>
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @gemm_non_zero_offset(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c4, %c0], %cst {in_bounds = [true, true]} : tensor<8x8xf32>, vector<4x4xf32>
    %1 = vector.transfer_read %arg1[%c0, %c4], %cst {in_bounds = [true, true]} : tensor<8x8xf32>, vector<4x4xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x8xf32>, vector<4x4xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
    %4 = vector.transfer_write %3, %arg2[%c4, %c4] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<8x8xf32>
    return %4 : tensor<8x8xf32>
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL:   func.func @gemm_non_zero_offset(
// CHECK-SAME:                                    %[[VAL_0:.*]]: tensor<8x8xf32>, %[[VAL_1:.*]]: tensor<8x8xf32>,
// CHECK-SAME:                                    %[[VAL_2:.*]]: tensor<8x8xf32>) -> tensor<8x8xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_4]], %[[VAL_3]]], %[[VAL_5]] {in_bounds = [true, true]} : tensor<8x8xf32>, vector<4x4xf32>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_4]]], %[[VAL_5]] {in_bounds = [true, true]} : tensor<8x8xf32>, vector<4x4xf32>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_5]] {in_bounds = [true, true]} : tensor<8x8xf32>, vector<4x4xf32>
// CHECK:           %[[VAL_9:.*]] = vector.contract {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #{{.*}}<add>} %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
// CHECK:           %[[VAL_10:.*]] = vector.transfer_write %[[VAL_9]], %[[VAL_2]]{{\[}}%[[VAL_4]], %[[VAL_4]]] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<8x8xf32>
// CHECK:           return %[[VAL_10]] : tensor<8x8xf32>
// CHECK:         }

// -----
