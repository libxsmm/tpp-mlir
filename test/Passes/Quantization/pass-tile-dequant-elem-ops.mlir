// RUN: tpp-opt --tile-dequant-elem-ops %s | FileCheck %s -check-prefix=IR

func.func @entry(%arg0: memref<32x32xi32>, %arg1: memref<64xf32>, %arg2: memref<2x2x16x32x4xi8>, %arg3: memref<64xf32>, %arg4: memref<2x2x32x32xf32>) {
  %cst = arith.constant dense<[0, 1, 1, 0]> : vector<4xi16>
  %cst_0 = arith.constant dense<[0, 0, 1, 1]> : vector<4xi16>
  %expand_shape = memref.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 1, 32, 1] : memref<64xf32> into memref<2x1x32x1xf32>
  %expand_shape_1 = memref.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [2, 1, 32, 1] : memref<64xf32> into memref<2x1x32x1xf32>
  scf.forall (%arg5) in (4) {
    %0 = vector.extract %cst_0[%arg5] : i16 from vector<4xi16>
    %1 = vector.extract %cst[%arg5] : i16 from vector<4xi16>
    %2 = arith.index_cast %0 : i16 to index
    %3 = arith.index_cast %1 : i16 to index
    %subview = memref.subview %expand_shape[%2, 0, 0, 0] [1, 1, 32, 1] [1, 1, 1, 1] : memref<2x1x32x1xf32> to memref<32xf32, strided<[1], offset: ?>>
    %subview_2 = memref.subview %expand_shape_1[%2, 0, 0, 0] [1, 1, 32, 1] [1, 1, 1, 1] : memref<2x1x32x1xf32> to memref<32xf32, strided<[1], offset: ?>>
    %subview_3 = memref.subview %arg4[%2, %3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %subview, %subview_2 : memref<32x32xi32>, memref<32xf32, strided<[1], offset: ?>>, memref<32xf32, strided<[1], offset: ?>>) outs(%subview_3 : memref<32x32xf32, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: i32, %in_4: f32, %in_5: f32, %out: f32):
      %4 = arith.mulf %in_4, %in_5 : f32
      %5 = arith.sitofp %in : i32 to f32
      %6 = arith.mulf %5, %4 : f32
      linalg.yield %6 : f32
    }
  }
  return
}

// IR-LABEL:   func.func @entry(
// IR-SAME:                     %[[ARG0:.*]]: memref<32x32xi32>,
// IR-SAME:                     %[[ARG1:.*]]: memref<64xf32>, %[[ARG2:.*]]: memref<2x2x16x32x4xi8>,
// IR-SAME:                     %[[ARG3:.*]]: memref<64xf32>,
// IR-SAME:                     %[[ARG4:.*]]: memref<2x2x32x32xf32>) {
// IR:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// IR:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// IR:           %[[CONSTANT_2:.*]] = arith.constant 16 : index
// IR:           %[[CONSTANT_3:.*]] = arith.constant 32 : index
// IR:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
// IR:           %[[CONSTANT_5:.*]] = arith.constant 0 : index
// IR:           %[[CONSTANT_6:.*]] = arith.constant dense<[0, 1, 1, 0]> : vector<4xi16>
// IR:           %[[CONSTANT_7:.*]] = arith.constant dense<[0, 0, 1, 1]> : vector<4xi16>
// IR:           %[[EXPAND_SHAPE_0:.*]] = memref.expand_shape %[[ARG1]] {{\[\[}}0, 1, 2, 3]] output_shape [2, 1, 32, 1] : memref<64xf32> into memref<2x1x32x1xf32>
// IR:           %[[EXPAND_SHAPE_1:.*]] = memref.expand_shape %[[ARG3]] {{\[\[}}0, 1, 2, 3]] output_shape [2, 1, 32, 1] : memref<64xf32> into memref<2x1x32x1xf32>
// IR:           scf.forall (%[[VAL_0:.*]]) in (4) {
// IR:             %[[EXTRACT_0:.*]] = vector.extract %[[CONSTANT_7]]{{\[}}%[[VAL_0]]] : i16 from vector<4xi16>
// IR:             %[[EXTRACT_1:.*]] = vector.extract %[[CONSTANT_6]]{{\[}}%[[VAL_0]]] : i16 from vector<4xi16>
// IR:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i16 to index
// IR:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_1]] : i16 to index
// IR:             %[[SUBVIEW_0:.*]] = memref.subview %[[EXPAND_SHAPE_0]]{{\[}}%[[INDEX_CAST_0]], 0, 0, 0] [1, 1, 32, 1] [1, 1, 1, 1] : memref<2x1x32x1xf32> to memref<32xf32, strided<[1], offset: ?>>
// IR:             %[[SUBVIEW_1:.*]] = memref.subview %[[EXPAND_SHAPE_1]]{{\[}}%[[INDEX_CAST_0]], 0, 0, 0] [1, 1, 32, 1] [1, 1, 1, 1] : memref<2x1x32x1xf32> to memref<32xf32, strided<[1], offset: ?>>
// IR:             %[[SUBVIEW_2:.*]] = memref.subview %[[ARG4]]{{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// IR:             scf.for %[[VAL_1:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
// IR:               %[[LOAD_0:.*]] = memref.load %[[SUBVIEW_0]]{{\[}}%[[VAL_1]]] : memref<32xf32, strided<[1], offset: ?>>
// IR:               %[[BROADCAST_0:.*]] = vector.broadcast %[[LOAD_0]] : f32 to vector<16xf32>
// IR:               scf.for %[[VAL_2:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_3]] step %[[CONSTANT_2]] {
// IR:                 %[[TRANSFER_READ_0:.*]] = vector.transfer_read %[[ARG0]]{{\[}}%[[VAL_1]], %[[VAL_2]]], %[[CONSTANT_1]] : memref<32x32xi32>, vector<16xi32>
// IR:                 %[[TRANSFER_READ_1:.*]] = vector.transfer_read %[[SUBVIEW_1]]{{\[}}%[[VAL_2]]], %[[CONSTANT_0]] : memref<32xf32, strided<[1], offset: ?>>, vector<16xf32>
// IR:                 %[[MULF_0:.*]] = arith.mulf %[[BROADCAST_0]], %[[TRANSFER_READ_1]] : vector<16xf32>
// IR:                 %[[SITOFP_0:.*]] = arith.sitofp %[[TRANSFER_READ_0]] : vector<16xi32> to vector<16xf32>
// IR:                 %[[MULF_1:.*]] = arith.mulf %[[SITOFP_0]], %[[MULF_0]] : vector<16xf32>
// IR:                 vector.transfer_write %[[MULF_1]], %[[SUBVIEW_2]]{{\[}}%[[VAL_1]], %[[VAL_2]]] {in_bounds = [true]} : vector<16xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// IR:               }
// IR:             }
// IR:           }
// IR:           return
// IR:         }