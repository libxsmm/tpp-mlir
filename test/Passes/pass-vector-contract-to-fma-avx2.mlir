

// RUN: tpp-opt %s  --vector-contract-to-fma="target-feature=avx2" --split-input-file | FileCheck %s --check-prefix=AVX2


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
func.func @entry(%arg0: memref<4x32x24x4xf32>, %arg1: memref<32x32x4x64xf32>, %arg2: memref<4x32x24x64xf32>) -> memref<4x32x24x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c3 = arith.constant 3 : index
  %c24 = arith.constant 24 : index
  %c0 = arith.constant 0 : index
  scf.forall (%arg3, %arg4) in (4, 32) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 24, 4] [1, 1, 1, 1] : memref<4x32x24x4xf32> to memref<32x24x4xf32, strided<[96, 4, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 4, 64] [1, 1, 1, 1] : memref<32x32x4x64xf32> to memref<32x4x64xf32, strided<[256, 64, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 24, 64] [1, 1, 1, 1] : memref<4x32x24x64xf32> to memref<24x64xf32, strided<[64, 1], offset: ?>>
    scf.for %arg5 = %c0 to %c24 step %c3 {
      scf.for %arg6 = %c0 to %c64 step %c32 {
        %subview_2 = memref.subview %subview_1[%arg5, %arg6] [3, 32] [1, 1] : memref<24x64xf32, strided<[64, 1], offset: ?>> to memref<3x32xf32, strided<[64, 1], offset: ?>>
        %0 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<3x32xf32>
        %1 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %0) -> (vector<3x32xf32>) {
          %2 = scf.for %arg9 = %c0 to %c4 step %c1 iter_args(%arg10 = %arg8) -> (vector<3x32xf32>) {
            %subview_3 = memref.subview %subview[%arg7, %arg5, %arg9] [1, 3, 1] [1, 1, 1] : memref<32x24x4xf32, strided<[96, 4, 1], offset: ?>> to memref<1x3x1xf32, strided<[96, 4, 1], offset: ?>>
            %subview_4 = memref.subview %subview_0[%arg7, %arg9, %arg6] [1, 1, 32] [1, 1, 1] : memref<32x4x64xf32, strided<[256, 64, 1], offset: ?>> to memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>
            %3 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x3x1xf32, strided<[96, 4, 1], offset: ?>>, vector<1x3x1xf32>
            %4 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>, vector<1x1x32xf32>
            %5 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg10 : vector<1x3x1xf32>, vector<1x1x32xf32> into vector<3x32xf32>
            scf.yield %5 : vector<3x32xf32>
          }
          scf.yield %2 : vector<3x32xf32>
        }
        vector.transfer_write %1, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<3x32xf32>, memref<3x32xf32, strided<[64, 1], offset: ?>>
      }
    }
  }
  return %arg2 : memref<4x32x24x64xf32>
}


// AVX2-LABEL:   func.func @entry(
// AVX2-SAME:                     %[[ARG0:.*]]: memref<4x32x24x4xf32>,
// AVX2-SAME:                     %[[ARG1:.*]]: memref<32x32x4x64xf32>,
// AVX2-SAME:                     %[[ARG2:.*]]: memref<4x32x24x64xf32>) -> memref<4x32x24x64xf32> {
// AVX2:           %[[VAL_0:.*]] = arith.constant 2 : index
// AVX2:           %[[VAL_1:.*]] = arith.constant 16 : index
// AVX2:           %[[VAL_2:.*]] = arith.constant 8 : index
// AVX2:           %[[VAL_3:.*]] = arith.constant 4 : index
// AVX2:           %[[VAL_4:.*]] = arith.constant 1 : index
// AVX2:           %[[VAL_5:.*]] = arith.constant 32 : index
// AVX2:           %[[VAL_6:.*]] = arith.constant 64 : index
// AVX2:           %[[VAL_7:.*]] = arith.constant 3 : index
// AVX2:           %[[VAL_8:.*]] = arith.constant 24 : index
// AVX2:           %[[VAL_9:.*]] = arith.constant 0 : index
// AVX2:           scf.forall (%[[VAL_10:.*]], %[[VAL_11:.*]]) in (4, 32) {
// AVX2:             %[[VAL_12:.*]] = memref.subview %[[ARG0]]{{\[}}%[[VAL_10]], 0, 0, 0] [1, 32, 24, 4] [1, 1, 1, 1] : memref<4x32x24x4xf32> to memref<32x24x4xf32, strided<[96, 4, 1], offset: ?>>
// AVX2:             %[[VAL_13:.*]] = memref.subview %[[ARG1]]{{\[}}%[[VAL_11]], 0, 0, 0] [1, 32, 4, 64] [1, 1, 1, 1] : memref<32x32x4x64xf32> to memref<32x4x64xf32, strided<[256, 64, 1], offset: ?>>
// AVX2:             %[[VAL_14:.*]] = memref.subview %[[ARG2]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0, 0] [1, 1, 24, 64] [1, 1, 1, 1] : memref<4x32x24x64xf32> to memref<24x64xf32, strided<[64, 1], offset: ?>>
// AVX2:             scf.for %[[VAL_15:.*]] = %[[VAL_9]] to %[[VAL_8]] step %[[VAL_7]] {
// AVX2:               scf.for %[[VAL_16:.*]] = %[[VAL_9]] to %[[VAL_6]] step %[[VAL_5]] {
// AVX2:                 %[[VAL_17:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_15]], %[[VAL_16]]] [3, 32] [1, 1] : memref<24x64xf32, strided<[64, 1], offset: ?>> to memref<3x32xf32, strided<[64, 1], offset: ?>>
// AVX2:                 %[[VAL_18:.*]] = memref.subview %[[VAL_17]][0, 0] [1, 32] [1, 1] : memref<3x32xf32, strided<[64, 1], offset: ?>> to memref<1x32xf32, strided<[64, 1], offset: ?>>
// AVX2:                 %[[VAL_19:.*]] = memref.subview %[[VAL_17]][1, 0] [1, 32] [1, 1] : memref<3x32xf32, strided<[64, 1], offset: ?>> to memref<1x32xf32, strided<[64, 1], offset: ?>>
// AVX2:                 %[[VAL_20:.*]] = memref.subview %[[VAL_17]][2, 0] [1, 32] [1, 1] : memref<3x32xf32, strided<[64, 1], offset: ?>> to memref<1x32xf32, strided<[64, 1], offset: ?>>
// AVX2:                 %[[VAL_21:.*]] = vector.load %[[VAL_18]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_22:.*]] = vector.load %[[VAL_18]]{{\[}}%[[VAL_9]], %[[VAL_2]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_23:.*]] = vector.load %[[VAL_18]]{{\[}}%[[VAL_9]], %[[VAL_1]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_24:.*]] = vector.load %[[VAL_18]]{{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_25:.*]] = vector.load %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_26:.*]] = vector.load %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_2]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_27:.*]] = vector.load %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_1]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_28:.*]] = vector.load %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_29:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_30:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_2]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_31:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_1]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_32:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 %[[VAL_33:.*]]:12 = scf.for %[[VAL_34:.*]] = %[[VAL_9]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_35:.*]] = %[[VAL_21]], %[[VAL_36:.*]] = %[[VAL_22]], %[[VAL_37:.*]] = %[[VAL_23]], %[[VAL_38:.*]] = %[[VAL_24]], %[[VAL_39:.*]] = %[[VAL_25]], %[[VAL_40:.*]] = %[[VAL_26]], %[[VAL_41:.*]] = %[[VAL_27]], %[[VAL_42:.*]] = %[[VAL_28]], %[[VAL_43:.*]] = %[[VAL_29]], %[[VAL_44:.*]] = %[[VAL_30]], %[[VAL_45:.*]] = %[[VAL_31]], %[[VAL_46:.*]] = %[[VAL_32]]) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
// AVX2:                   %[[VAL_47:.*]]:12 = scf.for %[[VAL_48:.*]] = %[[VAL_9]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_49:.*]] = %[[VAL_35]], %[[VAL_50:.*]] = %[[VAL_36]], %[[VAL_51:.*]] = %[[VAL_37]], %[[VAL_52:.*]] = %[[VAL_38]], %[[VAL_53:.*]] = %[[VAL_39]], %[[VAL_54:.*]] = %[[VAL_40]], %[[VAL_55:.*]] = %[[VAL_41]], %[[VAL_56:.*]] = %[[VAL_42]], %[[VAL_57:.*]] = %[[VAL_43]], %[[VAL_58:.*]] = %[[VAL_44]], %[[VAL_59:.*]] = %[[VAL_45]], %[[VAL_60:.*]] = %[[VAL_46]]) -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>) {
// AVX2:                     %[[VAL_61:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_34]], %[[VAL_15]], %[[VAL_48]]] [1, 3, 1] [1, 1, 1] : memref<32x24x4xf32, strided<[96, 4, 1], offset: ?>> to memref<1x3x1xf32, strided<[96, 4, 1], offset: ?>>
// AVX2:                     %[[VAL_62:.*]] = memref.load %[[VAL_61]]{{\[}}%[[VAL_9]], %[[VAL_9]], %[[VAL_9]]] : memref<1x3x1xf32, strided<[96, 4, 1], offset: ?>>
// AVX2:                     %[[VAL_63:.*]] = vector.broadcast %[[VAL_62]] : f32 to vector<8xf32>
// AVX2:                     %[[VAL_64:.*]] = memref.load %[[VAL_61]]{{\[}}%[[VAL_9]], %[[VAL_4]], %[[VAL_9]]] : memref<1x3x1xf32, strided<[96, 4, 1], offset: ?>>
// AVX2:                     %[[VAL_65:.*]] = vector.broadcast %[[VAL_64]] : f32 to vector<8xf32>
// AVX2:                     %[[VAL_66:.*]] = memref.load %[[VAL_61]]{{\[}}%[[VAL_9]], %[[VAL_0]], %[[VAL_9]]] : memref<1x3x1xf32, strided<[96, 4, 1], offset: ?>>
// AVX2:                     %[[VAL_67:.*]] = vector.broadcast %[[VAL_66]] : f32 to vector<8xf32>
// AVX2:                     %[[VAL_68:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_34]], %[[VAL_48]], %[[VAL_16]]] [1, 1, 32] [1, 1, 1] : memref<32x4x64xf32, strided<[256, 64, 1], offset: ?>> to memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>
// AVX2:                     %[[VAL_69:.*]] = vector.load %[[VAL_68]]{{\[}}%[[VAL_9]], %[[VAL_9]], %[[VAL_9]]] : memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>, vector<8xf32>
// AVX2:                     %[[VAL_70:.*]] = vector.fma %[[VAL_63]], %[[VAL_69]], %[[VAL_49]] : vector<8xf32>
// AVX2:                     %[[VAL_71:.*]] = vector.fma %[[VAL_65]], %[[VAL_69]], %[[VAL_53]] : vector<8xf32>
// AVX2:                     %[[VAL_72:.*]] = vector.fma %[[VAL_67]], %[[VAL_69]], %[[VAL_57]] : vector<8xf32>
// AVX2:                     %[[VAL_73:.*]] = vector.load %[[VAL_68]]{{\[}}%[[VAL_9]], %[[VAL_9]], %[[VAL_2]]] : memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>, vector<8xf32>
// AVX2:                     %[[VAL_74:.*]] = vector.fma %[[VAL_63]], %[[VAL_73]], %[[VAL_50]] : vector<8xf32>
// AVX2:                     %[[VAL_75:.*]] = vector.fma %[[VAL_65]], %[[VAL_73]], %[[VAL_54]] : vector<8xf32>
// AVX2:                     %[[VAL_76:.*]] = vector.fma %[[VAL_67]], %[[VAL_73]], %[[VAL_58]] : vector<8xf32>
// AVX2:                     %[[VAL_77:.*]] = vector.load %[[VAL_68]]{{\[}}%[[VAL_9]], %[[VAL_9]], %[[VAL_1]]] : memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>, vector<8xf32>
// AVX2:                     %[[VAL_78:.*]] = vector.fma %[[VAL_63]], %[[VAL_77]], %[[VAL_51]] : vector<8xf32>
// AVX2:                     %[[VAL_79:.*]] = vector.fma %[[VAL_65]], %[[VAL_77]], %[[VAL_55]] : vector<8xf32>
// AVX2:                     %[[VAL_80:.*]] = vector.fma %[[VAL_67]], %[[VAL_77]], %[[VAL_59]] : vector<8xf32>
// AVX2:                     %[[VAL_81:.*]] = vector.load %[[VAL_68]]{{\[}}%[[VAL_9]], %[[VAL_9]], %[[VAL_8]]] : memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>, vector<8xf32>
// AVX2:                     %[[VAL_82:.*]] = vector.fma %[[VAL_63]], %[[VAL_81]], %[[VAL_52]] : vector<8xf32>
// AVX2:                     %[[VAL_83:.*]] = vector.fma %[[VAL_65]], %[[VAL_81]], %[[VAL_56]] : vector<8xf32>
// AVX2:                     %[[VAL_84:.*]] = vector.fma %[[VAL_67]], %[[VAL_81]], %[[VAL_60]] : vector<8xf32>
// AVX2:                     scf.yield %[[VAL_70]], %[[VAL_74]], %[[VAL_78]], %[[VAL_82]], %[[VAL_71]], %[[VAL_75]], %[[VAL_79]], %[[VAL_83]], %[[VAL_72]], %[[VAL_76]], %[[VAL_80]], %[[VAL_84]] : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
// AVX2:                   }
// AVX2:                   scf.yield %[[VAL_85:.*]]#0, %[[VAL_85]]#1, %[[VAL_85]]#2, %[[VAL_85]]#3, %[[VAL_85]]#4, %[[VAL_85]]#5, %[[VAL_85]]#6, %[[VAL_85]]#7, %[[VAL_85]]#8, %[[VAL_85]]#9, %[[VAL_85]]#10, %[[VAL_85]]#11 : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
// AVX2:                 }
// AVX2:                 vector.store %[[VAL_86:.*]]#0, %[[VAL_18]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#1, %[[VAL_18]]{{\[}}%[[VAL_9]], %[[VAL_2]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#2, %[[VAL_18]]{{\[}}%[[VAL_9]], %[[VAL_1]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#3, %[[VAL_18]]{{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#4, %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#5, %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_2]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#6, %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_1]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#7, %[[VAL_19]]{{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#8, %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_9]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#9, %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_2]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#10, %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_1]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:                 vector.store %[[VAL_86]]#11, %[[VAL_20]]{{\[}}%[[VAL_9]], %[[VAL_8]]] : memref<1x32xf32, strided<[64, 1], offset: ?>>, vector<8xf32>
// AVX2:               }
// AVX2:             }
// AVX2:           }
// AVX2:           return %[[ARG2]] : memref<4x32x24x64xf32>
// AVX2:         }

