// RUN: tpp-opt %s  --vector-contract-to-fma="target-feature=avx512" --split-input-file | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @entry(%arg0: memref<8x16x32x64xf32>, %arg1: memref<16x16x64x64xf32>, %arg2: memref<8x16x32x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg3, %arg4) in (8, 16) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 16, 32, 64] [1, 1, 1, 1] : memref<8x16x32x64xf32> to memref<16x32x64xf32, strided<[2048, 64, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<16x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x16x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      scf.for %arg5 = %c0 to %c32 step %c4 {
        
        scf.for %arg6 = %c0 to %c64 step %c64 {
          %subview_2 = memref.subview %subview_1[%arg5, %arg6] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
          %2 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>

          %con = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%argcon = %2) -> vector<4x64xf32> {

            %con1 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%argcon1 = %argcon) -> vector<4x64xf32> {
              %subview_3 = memref.subview %subview[%arg7, %arg5, %arg8] [1, 4, 1] [1, 1, 1] : memref<16x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
              %subview_4 = memref.subview %subview_0[%arg7, %arg8, %arg6] [1, 1, 64] [1, 1, 1] : memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
              %0 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x4x1xf32>
              %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<1x1x64xf32>       
              %3 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %argcon1 : vector<1x4x1xf32>, vector<1x1x64xf32> into vector<4x64xf32>
			  scf.yield %3 : vector<4x64xf32>
            }
			scf.yield %con1 : vector<4x64xf32>
          }
          vector.transfer_write %con, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
        }
      }
    }
    return
  }
}

// CHECK-LABEL:   module {
// CHECK:           func.func @entry(%[[VAL_0:.*]]: memref<8x16x32x64xf32>, %[[VAL_1:.*]]: memref<16x16x64x64xf32>, %[[VAL_2:.*]]: memref<8x16x32x64xf32>) {
// CHECK:             %[[VAL_3:.*]] = arith.constant 3 : index
// CHECK:             %[[VAL_4:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_5:.*]] = arith.constant 48 : index
// CHECK:             %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_7:.*]] = arith.constant 16 : index
// CHECK:             %[[VAL_8:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_9:.*]] = arith.constant 4 : index
// CHECK:             %[[VAL_10:.*]] = arith.constant 32 : index
// CHECK:             %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:             scf.forall (%[[VAL_12:.*]], %[[VAL_13:.*]]) in (8, 16) {
// CHECK:               %[[VAL_14:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_12]], 0, 0, 0] [1, 16, 32, 64] [1, 1, 1, 1] : memref<8x16x32x64xf32> to memref<16x32x64xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:               %[[VAL_15:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_13]], 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<16x16x64x64xf32> to memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>>
// CHECK:               %[[VAL_16:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_12]], %[[VAL_13]], 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x16x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:               scf.for %[[VAL_17:.*]] = %[[VAL_11]] to %[[VAL_10]] step %[[VAL_9]] {
// CHECK:                 scf.for %[[VAL_18:.*]] = %[[VAL_11]] to %[[VAL_8]] step %[[VAL_8]] {
// CHECK:                   %[[VAL_19:.*]] = memref.subview %[[VAL_16]]{{\[}}%[[VAL_17]], %[[VAL_18]]] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                   %[[VAL_20:.*]] = memref.subview %[[VAL_19]][0, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                   %[[VAL_21:.*]] = memref.subview %[[VAL_19]][1, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                   %[[VAL_22:.*]] = memref.subview %[[VAL_19]][2, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                   %[[VAL_23:.*]] = memref.subview %[[VAL_19]][3, 0] [1, 64] [1, 1] : memref<4x64xf32, strided<[64, 1], offset: ?>> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                   %[[VAL_24:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_11]], %[[VAL_11]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_25:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_11]], %[[VAL_7]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_26:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_27:.*]] = vector.load %[[VAL_20]]{{\[}}%[[VAL_11]], %[[VAL_5]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_28:.*]] = vector.load %[[VAL_21]]{{\[}}%[[VAL_11]], %[[VAL_11]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_29:.*]] = vector.load %[[VAL_21]]{{\[}}%[[VAL_11]], %[[VAL_7]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_30:.*]] = vector.load %[[VAL_21]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_31:.*]] = vector.load %[[VAL_21]]{{\[}}%[[VAL_11]], %[[VAL_5]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_32:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_11]], %[[VAL_11]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_33:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_11]], %[[VAL_7]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_34:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_35:.*]] = vector.load %[[VAL_22]]{{\[}}%[[VAL_11]], %[[VAL_5]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_36:.*]] = vector.load %[[VAL_23]]{{\[}}%[[VAL_11]], %[[VAL_11]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_37:.*]] = vector.load %[[VAL_23]]{{\[}}%[[VAL_11]], %[[VAL_7]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_38:.*]] = vector.load %[[VAL_23]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_39:.*]] = vector.load %[[VAL_23]]{{\[}}%[[VAL_11]], %[[VAL_5]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   %[[VAL_40:.*]]:16 = scf.for %[[VAL_41:.*]] = %[[VAL_11]] to %[[VAL_7]] step %[[VAL_6]] iter_args(%[[VAL_42:.*]] = %[[VAL_24]], %[[VAL_43:.*]] = %[[VAL_25]], %[[VAL_44:.*]] = %[[VAL_26]], %[[VAL_45:.*]] = %[[VAL_27]], %[[VAL_46:.*]] = %[[VAL_28]], %[[VAL_47:.*]] = %[[VAL_29]], %[[VAL_48:.*]] = %[[VAL_30]], %[[VAL_49:.*]] = %[[VAL_31]], %[[VAL_50:.*]] = %[[VAL_32]], %[[VAL_51:.*]] = %[[VAL_33]], %[[VAL_52:.*]] = %[[VAL_34]], %[[VAL_53:.*]] = %[[VAL_35]], %[[VAL_54:.*]] = %[[VAL_36]], %[[VAL_55:.*]] = %[[VAL_37]], %[[VAL_56:.*]] = %[[VAL_38]], %[[VAL_57:.*]] = %[[VAL_39]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CHECK:                     %[[VAL_58:.*]]:16 = scf.for %[[VAL_59:.*]] = %[[VAL_11]] to %[[VAL_8]] step %[[VAL_6]] iter_args(%[[VAL_60:.*]] = %[[VAL_42]], %[[VAL_61:.*]] = %[[VAL_43]], %[[VAL_62:.*]] = %[[VAL_44]], %[[VAL_63:.*]] = %[[VAL_45]], %[[VAL_64:.*]] = %[[VAL_46]], %[[VAL_65:.*]] = %[[VAL_47]], %[[VAL_66:.*]] = %[[VAL_48]], %[[VAL_67:.*]] = %[[VAL_49]], %[[VAL_68:.*]] = %[[VAL_50]], %[[VAL_69:.*]] = %[[VAL_51]], %[[VAL_70:.*]] = %[[VAL_52]], %[[VAL_71:.*]] = %[[VAL_53]], %[[VAL_72:.*]] = %[[VAL_54]], %[[VAL_73:.*]] = %[[VAL_55]], %[[VAL_74:.*]] = %[[VAL_56]], %[[VAL_75:.*]] = %[[VAL_57]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CHECK:                       %[[VAL_76:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_41]], %[[VAL_17]], %[[VAL_59]]] [1, 4, 1] [1, 1, 1] : memref<16x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                       %[[VAL_77:.*]] = memref.load %[[VAL_76]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_11]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                       %[[VAL_78:.*]] = vector.broadcast %[[VAL_77]] : f32 to vector<16xf32>
// CHECK:                       %[[VAL_79:.*]] = memref.load %[[VAL_76]]{{\[}}%[[VAL_11]], %[[VAL_6]], %[[VAL_11]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                       %[[VAL_80:.*]] = vector.broadcast %[[VAL_79]] : f32 to vector<16xf32>
// CHECK:                       %[[VAL_81:.*]] = memref.load %[[VAL_76]]{{\[}}%[[VAL_11]], %[[VAL_4]], %[[VAL_11]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                       %[[VAL_82:.*]] = vector.broadcast %[[VAL_81]] : f32 to vector<16xf32>
// CHECK:                       %[[VAL_83:.*]] = memref.load %[[VAL_76]]{{\[}}%[[VAL_11]], %[[VAL_3]], %[[VAL_11]]] : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                       %[[VAL_84:.*]] = vector.broadcast %[[VAL_83]] : f32 to vector<16xf32>
// CHECK:                       %[[VAL_85:.*]] = memref.subview %[[VAL_15]]{{\[}}%[[VAL_41]], %[[VAL_59]], %[[VAL_18]]] [1, 1, 64] [1, 1, 1] : memref<16x64x64xf32, strided<[4096, 64, 1], offset: ?>> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
// CHECK:                       %[[VAL_86:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_11]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_87:.*]] = vector.fma %[[VAL_78]], %[[VAL_86]], %[[VAL_60]] : vector<16xf32>
// CHECK:                       %[[VAL_88:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_7]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_89:.*]] = vector.fma %[[VAL_78]], %[[VAL_88]], %[[VAL_61]] : vector<16xf32>
// CHECK:                       %[[VAL_90:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_10]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_91:.*]] = vector.fma %[[VAL_78]], %[[VAL_90]], %[[VAL_62]] : vector<16xf32>
// CHECK:                       %[[VAL_92:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_5]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_93:.*]] = vector.fma %[[VAL_78]], %[[VAL_92]], %[[VAL_63]] : vector<16xf32>
// CHECK:                       %[[VAL_94:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_11]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_95:.*]] = vector.fma %[[VAL_80]], %[[VAL_94]], %[[VAL_64]] : vector<16xf32>
// CHECK:                       %[[VAL_96:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_7]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_97:.*]] = vector.fma %[[VAL_80]], %[[VAL_96]], %[[VAL_65]] : vector<16xf32>
// CHECK:                       %[[VAL_98:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_10]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_99:.*]] = vector.fma %[[VAL_80]], %[[VAL_98]], %[[VAL_66]] : vector<16xf32>
// CHECK:                       %[[VAL_100:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_5]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_101:.*]] = vector.fma %[[VAL_80]], %[[VAL_100]], %[[VAL_67]] : vector<16xf32>
// CHECK:                       %[[VAL_102:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_11]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_103:.*]] = vector.fma %[[VAL_82]], %[[VAL_102]], %[[VAL_68]] : vector<16xf32>
// CHECK:                       %[[VAL_104:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_7]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_105:.*]] = vector.fma %[[VAL_82]], %[[VAL_104]], %[[VAL_69]] : vector<16xf32>
// CHECK:                       %[[VAL_106:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_10]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_107:.*]] = vector.fma %[[VAL_82]], %[[VAL_106]], %[[VAL_70]] : vector<16xf32>
// CHECK:                       %[[VAL_108:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_5]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_109:.*]] = vector.fma %[[VAL_82]], %[[VAL_108]], %[[VAL_71]] : vector<16xf32>
// CHECK:                       %[[VAL_110:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_11]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_111:.*]] = vector.fma %[[VAL_84]], %[[VAL_110]], %[[VAL_72]] : vector<16xf32>
// CHECK:                       %[[VAL_112:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_7]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_113:.*]] = vector.fma %[[VAL_84]], %[[VAL_112]], %[[VAL_73]] : vector<16xf32>
// CHECK:                       %[[VAL_114:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_10]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_115:.*]] = vector.fma %[[VAL_84]], %[[VAL_114]], %[[VAL_74]] : vector<16xf32>
// CHECK:                       %[[VAL_116:.*]] = vector.load %[[VAL_85]]{{\[}}%[[VAL_11]], %[[VAL_11]], %[[VAL_5]]] : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<16xf32>
// CHECK:                       %[[VAL_117:.*]] = vector.fma %[[VAL_84]], %[[VAL_116]], %[[VAL_75]] : vector<16xf32>
// CHECK:                       scf.yield %[[VAL_87]], %[[VAL_89]], %[[VAL_91]], %[[VAL_93]], %[[VAL_95]], %[[VAL_97]], %[[VAL_99]], %[[VAL_101]], %[[VAL_103]], %[[VAL_105]], %[[VAL_107]], %[[VAL_109]], %[[VAL_111]], %[[VAL_113]], %[[VAL_115]], %[[VAL_117]] : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_118:.*]]#0, %[[VAL_118]]#1, %[[VAL_118]]#2, %[[VAL_118]]#3, %[[VAL_118]]#4, %[[VAL_118]]#5, %[[VAL_118]]#6, %[[VAL_118]]#7, %[[VAL_118]]#8, %[[VAL_118]]#9, %[[VAL_118]]#10, %[[VAL_118]]#11, %[[VAL_118]]#12, %[[VAL_118]]#13, %[[VAL_118]]#14, %[[VAL_118]]#15 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CHECK:                   }
// CHECK:                   vector.store %[[VAL_119:.*]]#0, %[[VAL_20]]{{\[}}%[[VAL_11]], %[[VAL_11]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#1, %[[VAL_20]]{{\[}}%[[VAL_11]], %[[VAL_7]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#2, %[[VAL_20]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#3, %[[VAL_20]]{{\[}}%[[VAL_11]], %[[VAL_5]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#4, %[[VAL_21]]{{\[}}%[[VAL_11]], %[[VAL_11]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#5, %[[VAL_21]]{{\[}}%[[VAL_11]], %[[VAL_7]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#6, %[[VAL_21]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#7, %[[VAL_21]]{{\[}}%[[VAL_11]], %[[VAL_5]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#8, %[[VAL_22]]{{\[}}%[[VAL_11]], %[[VAL_11]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#9, %[[VAL_22]]{{\[}}%[[VAL_11]], %[[VAL_7]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#10, %[[VAL_22]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#11, %[[VAL_22]]{{\[}}%[[VAL_11]], %[[VAL_5]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#12, %[[VAL_23]]{{\[}}%[[VAL_11]], %[[VAL_11]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#13, %[[VAL_23]]{{\[}}%[[VAL_11]], %[[VAL_7]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#14, %[[VAL_23]]{{\[}}%[[VAL_11]], %[[VAL_10]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                   vector.store %[[VAL_119]]#15, %[[VAL_23]]{{\[}}%[[VAL_11]], %[[VAL_5]]] : memref<1x64xf32, strided<[64, 1], offset: ?>>, vector<16xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:             return
// CHECK:           }
// CHECK:         }

//-----

#mapA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapB = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapC = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @matmul_without_iterarg_accumulator(%arg0: tensor<4x1xf32>, %arg1: tensor<1x64xf32>, %arg2: tensor<4x64xf32>) -> tensor<4x64xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x1xf32>, vector<4x1xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
    %3 = vector.contract {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x1xf32>, vector<1x64xf32> into vector<4x64xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
    return %4 : tensor<4x64xf32>
  }
}

// CHECK-NOT: vector.fma

//-----

#mapTransposeB = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>

module {
  func.func @entry(%arg0: memref<16x32x128xf32>, %arg1: memref<16x128x64xf32>, %arg2: memref<32x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index

    scf.for %arg5 = %c0 to %c32 step %c4 {
      scf.for %arg6 = %c0 to %c128 step %c64 {
        %subview_2 = memref.subview %arg2[%arg5, %arg6] [4, 64] [1, 1] : memref<32x64xf32> to memref<4x64xf32, strided<[64, 1], offset: ?>>
        %2 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>
        %con = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%argcon = %2) -> vector<4x64xf32> {
          %con1 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%argcon1 = %argcon) -> vector<4x64xf32> {
            %subview_3 = memref.subview %arg0[%arg7, %arg5, %arg8] [1, 4, 1] [1, 1, 1] : memref<16x32x128xf32> to memref<1x4x1xf32, strided<[4096, 128, 1], offset: ?>>
            %subview_4 = memref.subview %arg1[%arg7, %arg8, %arg6] [1, 1, 64] [1, 1, 1] : memref<16x128x64xf32> to memref<1x1x64xf32, strided<[8192, 64, 1], offset: ?>>
            %0 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[4096, 128, 1], offset: ?>>, vector<1x4x1xf32>
            %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {permutation_map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>, in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[8192, 64, 1], offset: ?>>, vector<1x64x1xf32>
            %3 = vector.contract {indexing_maps = [#map, #mapTransposeB, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %argcon1 : vector<1x4x1xf32>, vector<1x64x1xf32> into vector<4x64xf32>
            scf.yield %3 : vector<4x64xf32>
          }
          scf.yield %con1 : vector<4x64xf32>
        }
        vector.transfer_write %con, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
      }
    }
    return
  }
}

// CHECK-NOT: vector.fma

// -----
