// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=8,32" --split-input-file  | FileCheck %s

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

// CHECK-LABEL:   func.func @gemm_do_register_tiling(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<16x32x16x32xf32>,
// CHECK-SAME:                     %[[VAL_1:.*]]: memref<32x32x32x32xf32>,
// CHECK-SAME:                     %[[VAL_2:.*]]: memref<16x32x16x32xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           scf.forall (%[[VAL_8:.*]], %[[VAL_9:.*]]) in (16, 32) {
// CHECK:             %[[VAL_10:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_8]], 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CHECK:             %[[VAL_11:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_9]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             %[[VAL_12:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_8]], %[[VAL_9]], 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:               scf.for %[[VAL_14:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK:                 scf.for %[[VAL_15:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:                   scf.for %[[VAL_16:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:                     %[[VAL_17:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_15]], %[[VAL_13]], %[[VAL_16]]] [1, 8, 1] [1, 1, 1] : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[512, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_18:.*]] = memref.subview %[[VAL_11]]{{\[}}%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]] [1, 1, 32] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_19:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_13]], %[[VAL_14]]] [8, 32] [1, 1] : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:                     linalg.batch_reduce_matmul ins(%[[VAL_17]], %[[VAL_18]] : memref<1x8x1xf32, strided<[512, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_19]] : memref<8x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

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

// CHECK-LABEL:   memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
// CHECK-LABEL:   func.func @chainned_gemm_do_register_tiling(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 48 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:           scf.forall (%[[VAL_9:.*]], %[[VAL_10:.*]]) in (8, 48) {
// CHECK:             %[[VAL_11:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_9]], %[[VAL_10]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_11]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:             %[[VAL_12:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_9]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:               scf.for %[[VAL_14:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK:                 scf.for %[[VAL_15:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_1]] {
// CHECK:                   scf.for %[[VAL_16:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_1]] {
// CHECK:                     %[[VAL_17:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_15]], %[[VAL_13]], %[[VAL_16]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_18:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_15]], %[[VAL_16]], %[[VAL_14]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_19:.*]] = memref.subview %[[VAL_11]]{{\[}}%[[VAL_13]], %[[VAL_14]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:                     linalg.batch_reduce_matmul ins(%[[VAL_17]], %[[VAL_18]] : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_19]] : memref<8x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_20:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:           scf.forall (%[[VAL_21:.*]], %[[VAL_22:.*]]) in (8, 48) {
// CHECK:             %[[VAL_23:.*]] = memref.subview %[[VAL_20]]{{\[}}%[[VAL_21]], %[[VAL_22]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_23]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:             %[[VAL_24:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_21]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_25:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:               scf.for %[[VAL_26:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK:                 scf.for %[[VAL_27:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_1]] {
// CHECK:                   scf.for %[[VAL_28:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_1]] {
// CHECK:                     %[[VAL_29:.*]] = memref.subview %[[VAL_24]]{{\[}}%[[VAL_27]], %[[VAL_25]], %[[VAL_28]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_30:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_27]], %[[VAL_28]], %[[VAL_26]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_31:.*]] = memref.subview %[[VAL_23]]{{\[}}%[[VAL_25]], %[[VAL_26]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:                     linalg.batch_reduce_matmul ins(%[[VAL_29]], %[[VAL_30]] : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_31]] : memref<8x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           scf.forall (%[[VAL_32:.*]], %[[VAL_33:.*]]) in (8, 48) {
// CHECK:             %[[VAL_34:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_32]], %[[VAL_33]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             linalg.fill ins(%[[VAL_6]] : f32) outs(%[[VAL_34]] : memref<32x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:             %[[VAL_35:.*]] = memref.subview %[[VAL_20]]{{\[}}%[[VAL_32]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_36:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_3]] {
// CHECK:               scf.for %[[VAL_37:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK:                 scf.for %[[VAL_38:.*]] = %[[VAL_5]] to %[[VAL_2]] step %[[VAL_1]] {
// CHECK:                   scf.for %[[VAL_39:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_1]] {
// CHECK:                     %[[VAL_40:.*]] = memref.subview %[[VAL_35]]{{\[}}%[[VAL_38]], %[[VAL_36]], %[[VAL_39]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_41:.*]] = memref.subview %[[VAL_7]]{{\[}}%[[VAL_38]], %[[VAL_39]], %[[VAL_37]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_42:.*]] = memref.subview %[[VAL_34]]{{\[}}%[[VAL_36]], %[[VAL_37]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:                     linalg.batch_reduce_matmul ins(%[[VAL_40]], %[[VAL_41]] : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%[[VAL_42]] : memref<8x32xf32, strided<[32, 1], offset: ?>>)
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_8]] : memref<8x48x32x32xf32>
// CHECK:         }
