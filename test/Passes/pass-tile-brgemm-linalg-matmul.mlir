// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=8,32,1" --split-input-file  | FileCheck -check-prefix=CONF1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=32,32,32" --canonicalize --split-input-file  | FileCheck -check-prefix=CONF2 %s

module {
  func.func @gemm_do_register_tiling(%arg0: memref<16x32x16x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<16x32x16x32xf32>) {
    scf.forall (%arg3, %arg4) in (16, 32) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_1 : memref<16x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}

// CONF1-LABEL:   func.func @gemm_do_register_tiling(
// CONF1-SAME:                     %[[VAL_0:.*]]: memref<16x32x16x32xf32>,
// CONF1-SAME:                     %[[VAL_1:.*]]: memref<32x32x32x32xf32>,
// CONF1-SAME:                     %[[VAL_2:.*]]: memref<16x32x16x32xf32>) {
// CONF1:           %[[VAL_3:.*]] = arith.constant 1 : index
// CONF1:           %[[VAL_4:.*]] = arith.constant 32 : index
// CONF1:           %[[VAL_5:.*]] = arith.constant 8 : index
// CONF1:           %[[VAL_6:.*]] = arith.constant 16 : index
// CONF1:           %[[VAL_7:.*]] = arith.constant 0 : index
// CONF1:           scf.forall (%[[VAL_8:.*]], %[[VAL_9:.*]]) in (16, 32) {
// CONF1:             %[[VAL_10:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_8]], 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CONF1:             %[[VAL_11:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_9]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:             %[[VAL_12:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_8]], %[[VAL_9]], 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
// CONF1:             scf.for %[[VAL_13:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CONF1:               scf.for %[[VAL_14:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_4]] {
// CONF1:                 scf.for %[[VAL_15:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] {
// CONF1:                   scf.for %[[VAL_16:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] {
// CONF1:                     %[[VAL_17:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_15]], %[[VAL_13]], %[[VAL_16]]] [1, 8, 1] [1, 1, 1] : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[512, 32, 1], offset: ?>>
// CONF1:                     %[[VAL_18:.*]] = memref.subview %[[VAL_11]]{{\[}}%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]] [1, 1, 32] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:                     %[[VAL_19:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_13]], %[[VAL_14]]] [8, 32] [1, 1] : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CONF1:                     linalg.batch_reduce_matmul ins(%[[VAL_17]], %[[VAL_18]] : memref<1x8x1xf32, strided<[512, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_19]] : memref<8x32xf32, strided<[32, 1], offset: ?>>)
// CONF1:                   }
// CONF1:                 }
// CONF1:               }
// CONF1:             }
// CONF1:           }
// CONF1:           return
// CONF1:         }

// -----

module {
  memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @chainned_gemm_do_register_tiling(%arg0: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
      %subview_1 = memref.subview %alloc_0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview_1, %0 : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<48x32x32xf32>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    }
    return %alloc : memref<8x48x32x32xf32>
  }
}

// CONF1-LABEL:   memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
// CONF1-LABEL:   func.func @chainned_gemm_do_register_tiling(
// CONF1-SAME:                     %[[VAL_0:.*]]: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
// CONF1:           %[[VAL_1:.*]] = arith.constant 1 : index
// CONF1:           %[[VAL_2:.*]] = arith.constant 48 : index
// CONF1:           %[[VAL_3:.*]] = arith.constant 8 : index
// CONF1:           %[[VAL_4:.*]] = arith.constant 32 : index
// CONF1:           %[[VAL_5:.*]] = arith.constant 0 : index
// CONF1:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CONF1:           %[[VAL_7:.*]] = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
// CONF1:           %[[VAL_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CONF1:           scf.forall (%[[VAL_9:.*]], %[[VAL_10:.*]]) in (8, 48) {
// CONF1:             %[[VAL_11:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_9]], %[[VAL_10]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CONF1:             linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_11]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CONF1:             %[[VAL_12:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_9]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:             scf.for %[[VAL_13:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CONF1:               scf.for %[[VAL_14:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_4]] {
// CONF1:                 scf.for %[[VAL_15:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_1]] {
// CONF1:                   scf.for %[[VAL_16:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_1]] {
// CONF1:                     %[[VAL_17:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_15]], %[[VAL_13]], %[[VAL_16]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:                     %[[VAL_18:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:                     %[[VAL_19:.*]] = memref.subview %[[VAL_11]]{{\[}}%[[VAL_13]], %[[VAL_14]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CONF1:                     linalg.batch_reduce_matmul ins(%[[VAL_17]], %[[VAL_18]] : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_19]] : memref<8x32xf32, strided<[32, 1], offset: ?>>)
// CONF1:                   }
// CONF1:                 }
// CONF1:               }
// CONF1:             }
// CONF1:           }
// CONF1:           %[[VAL_20:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CONF1:           scf.forall (%[[VAL_21:.*]], %[[VAL_22:.*]]) in (8, 48) {
// CONF1:             %[[VAL_23:.*]] = memref.subview %[[VAL_20]]{{\[}}%[[VAL_21]], %[[VAL_22]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CONF1:             linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_23]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CONF1:             %[[VAL_24:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_21]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:             scf.for %[[VAL_25:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CONF1:               scf.for %[[VAL_26:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_4]] {
// CONF1:                 scf.for %[[VAL_27:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_1]] {
// CONF1:                   scf.for %[[VAL_28:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_1]] {
// CONF1:                     %[[VAL_29:.*]] = memref.subview %[[VAL_24]]{{\[}}%[[VAL_27]], %[[VAL_25]], %[[VAL_28]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:                     %[[VAL_30:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_27]], %[[VAL_28]], %[[VAL_26]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:                     %[[VAL_31:.*]] = memref.subview %[[VAL_23]]{{\[}}%[[VAL_25]], %[[VAL_26]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CONF1:                     linalg.batch_reduce_matmul ins(%[[VAL_29]], %[[VAL_30]] : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_31]] : memref<8x32xf32, strided<[32, 1], offset: ?>>)
// CONF1:                   }
// CONF1:                 }
// CONF1:               }
// CONF1:             }
// CONF1:           }
// CONF1:           scf.forall (%[[VAL_32:.*]], %[[VAL_33:.*]]) in (8, 48) {
// CONF1:             %[[VAL_34:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_32]], %[[VAL_33]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CONF1:             linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_34]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CONF1:             %[[VAL_35:.*]] = memref.subview %[[VAL_20]]{{\[}}%[[VAL_32]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:             scf.for %[[VAL_36:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CONF1:               scf.for %[[VAL_37:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_4]] {
// CONF1:                 scf.for %[[VAL_38:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_1]] {
// CONF1:                   scf.for %[[VAL_39:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_1]] {
// CONF1:                     %[[VAL_40:.*]] = memref.subview %[[VAL_35]]{{\[}}%[[VAL_38]], %[[VAL_36]], %[[VAL_39]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:                     %[[VAL_41:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_38]], %[[VAL_39]], %[[VAL_37]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1:                     %[[VAL_42:.*]] = memref.subview %[[VAL_34]]{{\[}}%[[VAL_36]], %[[VAL_37]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CONF1:                     linalg.batch_reduce_matmul ins(%[[VAL_40]], %[[VAL_41]] : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_42]] : memref<8x32xf32, strided<[32, 1], offset: ?>>)
// CONF1:                   }
// CONF1:                 }
// CONF1:               }
// CONF1:             }
// CONF1:           }
// CONF1:           return %[[VAL_8]] : memref<8x48x32x32xf32>
// CONF1:         }

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @gemm_32tiles_do_tiling(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
    scf.forall (%arg1, %arg2) in (8, 32) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
      %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<32x16x32x2xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
      ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_1 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
      }
    }
    return %alloc : memref<8x32x32x32xbf16>
  }
}

// CONF2: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
// CONF2: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
// CONF2: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

// CONF2-LABEL:   memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
// CONF2-LABEL:   func.func @gemm_32tiles_do_tiling(
// CONF2-SAME:                     %[[VAL_0:.*]]: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
// CONF2:           %[[VAL_1:.*]] = arith.constant 1 : index
// CONF2:           %[[VAL_2:.*]] = arith.constant 32 : index
// CONF2:           %[[VAL_3:.*]] = arith.constant 0 : index
// CONF2:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : bf16
// CONF2:           %[[VAL_5:.*]] = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
// CONF2:           %[[VAL_6:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
// CONF2:           %[[VAL_7:.*]] = memref.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
// CONF2:           scf.forall (%[[VAL_8:.*]], %[[VAL_9:.*]]) in (8, 32) {
// CONF2:             %[[VAL_10:.*]] = memref.subview %[[VAL_6]]{{\[}}%[[VAL_8]], %[[VAL_9]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CONF2:             linalg.fill ins(%[[VAL_4]] : bf16) outs(%[[VAL_10]] : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
// CONF2:             %[[VAL_11:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_8]], 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF2:             scf.for %[[VAL_12:.*]] = %[[VAL_3]] to %[[VAL_2]] step %[[VAL_1]] {
// CONF2:               %[[VAL_13:.*]] = memref.subview %[[VAL_11]]{{\[}}%[[VAL_12]], 0, 0, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF2:               %[[VAL_14:.*]] = memref.subview %[[VAL_5]]{{\[}}%[[VAL_12]], 0, 0, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CONF2:               linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%[[VAL_13]], %[[VAL_14]] : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%[[VAL_10]] : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
// CONF2:               ^bb0(%[[VAL_15:.*]]: bf16, %[[VAL_16:.*]]: bf16, %[[VAL_17:.*]]: bf16):
// CONF2:                 %[[VAL_18:.*]] = arith.mulf %[[VAL_15]], %[[VAL_16]] : bf16
// CONF2:                 %[[VAL_19:.*]] = arith.addf %[[VAL_17]], %[[VAL_18]] : bf16
// CONF2:                 linalg.yield %[[VAL_19]] : bf16
// CONF2:               }
// CONF2:             }
// CONF2:           }
// CONF2:           return %[[VAL_6]] : memref<8x32x32x32xbf16>
// CONF2:         }

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @gemm_64tiles_do_tiling(%arg0: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x64x64xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
    scf.forall (%arg1, %arg2) in (4, 16) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      linalg.fill ins(%cst : bf16) outs(%subview : memref<64x64xbf16, strided<[64, 1], offset: ?>>)
      %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, memref<16x32x64x2xbf16>) outs(%subview : memref<64x64xbf16, strided<[64, 1], offset: ?>>) {
      ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_1 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
      }
    }
    return %alloc : memref<4x16x64x64xbf16>
  }
}

// CONF2: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
// CONF2: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
// CONF2: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
// CONF2-LABEL:   memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
// CONF2-LABEL:   func.func @gemm_64tiles_do_tiling(
// CONF2-SAME:                     %[[VAL_0:.*]]: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
// CONF2:           %[[VAL_1:.*]] = arith.constant 1 : index
// CONF2:           %[[VAL_2:.*]] = arith.constant 16 : index
// CONF2:           %[[VAL_3:.*]] = arith.constant 32 : index
// CONF2:           %[[VAL_4:.*]] = arith.constant 64 : index
// CONF2:           %[[VAL_5:.*]] = arith.constant 0 : index
// CONF2:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : bf16
// CONF2:           %[[VAL_7:.*]] = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
// CONF2:           %[[VAL_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x16x64x64xbf16>
// CONF2:           %[[VAL_9:.*]] = memref.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
// CONF2:           scf.forall (%[[VAL_10:.*]], %[[VAL_11:.*]]) in (4, 16) {
// CONF2:             %[[VAL_12:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
// CONF2:             linalg.fill ins(%[[VAL_6]] : bf16) outs(%[[VAL_12]] : memref<64x64xbf16, strided<[64, 1], offset: ?>>)
// CONF2:             %[[VAL_13:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_10]], 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CONF2:             scf.for %[[VAL_14:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CONF2:               scf.for %[[VAL_15:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CONF2:                 scf.for %[[VAL_16:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_1]] {
// CONF2:                   scf.for %[[VAL_17:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_2]] {
// CONF2:                     %[[VAL_18:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_16]], %[[VAL_14]], %[[VAL_17]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CONF2:                     %[[VAL_19:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_16]], %[[VAL_17]], %[[VAL_15]], 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x32x64x2xbf16> to memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
// CONF2:                     %[[VAL_20:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_15]]] [32, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
// CONF2:                     linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%[[VAL_18]], %[[VAL_19]] : memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>) outs(%[[VAL_20]] : memref<32x32xbf16, strided<[64, 1], offset: ?>>) {
// CONF2:                     ^bb0(%[[VAL_21:.*]]: bf16, %[[VAL_22:.*]]: bf16, %[[VAL_23:.*]]: bf16):
// CONF2:                       %[[VAL_24:.*]] = arith.mulf %[[VAL_21]], %[[VAL_22]] : bf16
// CONF2:                       %[[VAL_25:.*]] = arith.addf %[[VAL_23]], %[[VAL_24]] : bf16
// CONF2:                       linalg.yield %[[VAL_25]] : bf16
// CONF2:                     }
// CONF2:                   }
// CONF2:                 }
// CONF2:               }
// CONF2:             }
// CONF2:           }
// CONF2:           return %[[VAL_8]] : memref<4x16x64x64xbf16>
// CONF2:         }
