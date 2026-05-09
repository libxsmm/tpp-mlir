// RUN: tpp-opt %s --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=4,32,2" --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF2 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=2,16,2" --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF3 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=2,32,4" --loop-invariant-code-motion --vectorization-pass --vector-contract-to-bf16dp --split-input-file  | FileCheck -check-prefix=CONF4 %s

module {
 func.func @gemm_lower_to_bf16dp(%arg0: memref<32x32x16x2xbf16>, %arg1: memref<32x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
  %cst = arith.constant 0.000000e+00 : bf16
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %arg3 = %c0 to %c32 step %c2 {
    scf.for %arg4 = %c0 to %c32 step %c32 {
      %subview = memref.subview %arg2[%arg3, %arg4] [2, 32] [1, 1] : memref<32x32xbf16> to memref<2x32xbf16, strided<[32, 1], offset: ?>>
      scf.for %arg5 = %c0 to %c32 step %c1 {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg6, 0] [1, 2, 1, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16> to memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
          %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4, 0] [1, 1, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
          %0 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<1x2x1x2xbf16>
          %1 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<1x1x32x2xbf16>
          %2 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x32xbf16, strided<[32, 1], offset: ?>>, vector<2x32xbf16>
          %3 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<1x2x1x2xbf16>, vector<1x1x32x2xbf16> into vector<2x32xbf16>
          vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x32xbf16>, memref<2x32xbf16, strided<[32, 1], offset: ?>>
        }
      }
    }
  }
  return
 }
}

// CONF1-LABEL:   func.func @gemm_lower_to_bf16dp(
// CONF1-SAME:                     %[[VAL_0:.*]]: memref<32x32x16x2xbf16>,
// CONF1-SAME:                     %[[VAL_1:.*]]: memref<32x16x32x2xbf16>,
// CONF1-SAME:                     %[[VAL_2:.*]]: memref<32x32xbf16>) {
// CONF1:           %[[VAL_3:.*]] = arith.constant 16 : i32
// CONF1:           %[[VAL_4:.*]] = arith.constant 0 : index
// CONF1:           %[[VAL_5:.*]] = arith.constant 32 : index
// CONF1:           %[[VAL_6:.*]] = arith.constant 2 : index
// CONF1:           %[[VAL_7:.*]] = arith.constant 1 : index
// CONF1:           %[[VAL_8:.*]] = arith.constant 16 : index
// CONF1:           scf.for %[[VAL_9:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_6]] {
// CONF1:             scf.for %[[VAL_10:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_5]] {
// CONF1:               %[[VAL_11:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_9]], %[[VAL_10]]] [2, 32] [1, 1] : memref<32x32xbf16> to memref<2x32xbf16, strided<[32, 1], offset: ?>>
// CONF1:               %[[VAL_12:.*]] = memref.subview %[[VAL_11]][0, 0] [1, 32] [1, 1] : memref<2x32xbf16, strided<[32, 1], offset: ?>> to memref<1x32xbf16, strided<[32, 1], offset: ?>>
// CONF1:               %[[VAL_13:.*]] = vector.load %[[VAL_12]]{{\[}}%[[VAL_4]], %[[VAL_4]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:               %[[VAL_14:.*]] = vector.bitcast %[[VAL_13]] : vector<16xbf16> to vector<16xi16>
// CONF1:               %[[VAL_15:.*]] = arith.extui %[[VAL_14]] : vector<16xi16> to vector<16xi32>
// CONF1:               %[[VAL_16:.*]] = vector.broadcast %[[VAL_3]] : i32 to vector<16xi32>
// CONF1:               %[[VAL_17:.*]] = arith.shli %[[VAL_15]], %[[VAL_16]] : vector<16xi32>
// CONF1:               %[[VAL_18:.*]] = vector.bitcast %[[VAL_17]] : vector<16xi32> to vector<16xf32>
// CONF1:               %[[VAL_19:.*]] = vector.load %[[VAL_12]]{{\[}}%[[VAL_4]], %[[VAL_8]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:               %[[VAL_20:.*]] = vector.bitcast %[[VAL_19]] : vector<16xbf16> to vector<16xi16>
// CONF1:               %[[VAL_21:.*]] = arith.extui %[[VAL_20]] : vector<16xi16> to vector<16xi32>
// CONF1:               %[[VAL_22:.*]] = vector.broadcast %[[VAL_3]] : i32 to vector<16xi32>
// CONF1:               %[[VAL_23:.*]] = arith.shli %[[VAL_21]], %[[VAL_22]] : vector<16xi32>
// CONF1:               %[[VAL_24:.*]] = vector.bitcast %[[VAL_23]] : vector<16xi32> to vector<16xf32>
// CONF1:               %[[VAL_25:.*]] = memref.subview %[[VAL_11]][1, 0] [1, 32] [1, 1] : memref<2x32xbf16, strided<[32, 1], offset: ?>> to memref<1x32xbf16, strided<[32, 1], offset: ?>>
// CONF1:               %[[VAL_26:.*]] = vector.load %[[VAL_25]]{{\[}}%[[VAL_4]], %[[VAL_4]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:               %[[VAL_27:.*]] = vector.bitcast %[[VAL_26]] : vector<16xbf16> to vector<16xi16>
// CONF1:               %[[VAL_28:.*]] = arith.extui %[[VAL_27]] : vector<16xi16> to vector<16xi32>
// CONF1:               %[[VAL_29:.*]] = vector.broadcast %[[VAL_3]] : i32 to vector<16xi32>
// CONF1:               %[[VAL_30:.*]] = arith.shli %[[VAL_28]], %[[VAL_29]] : vector<16xi32>
// CONF1:               %[[VAL_31:.*]] = vector.bitcast %[[VAL_30]] : vector<16xi32> to vector<16xf32>
// CONF1:               %[[VAL_32:.*]] = vector.load %[[VAL_25]]{{\[}}%[[VAL_4]], %[[VAL_8]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:               %[[VAL_33:.*]] = vector.bitcast %[[VAL_32]] : vector<16xbf16> to vector<16xi16>
// CONF1:               %[[VAL_34:.*]] = arith.extui %[[VAL_33]] : vector<16xi16> to vector<16xi32>
// CONF1:               %[[VAL_35:.*]] = vector.broadcast %[[VAL_3]] : i32 to vector<16xi32>
// CONF1:               %[[VAL_36:.*]] = arith.shli %[[VAL_34]], %[[VAL_35]] : vector<16xi32>
// CONF1:               %[[VAL_37:.*]] = vector.bitcast %[[VAL_36]] : vector<16xi32> to vector<16xf32>
// CONF1:               %[[VAL_38:.*]]:4 = scf.for %[[VAL_39:.*]] = %[[VAL_4]] to %[[VAL_5]] step %[[VAL_7]] iter_args(%[[VAL_40:.*]] = %[[VAL_18]], %[[VAL_41:.*]] = %[[VAL_24]], %[[VAL_42:.*]] = %[[VAL_31]], %[[VAL_43:.*]] = %[[VAL_37]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CONF1:                 %[[VAL_44:.*]]:4 = scf.for %[[VAL_45:.*]] = %[[VAL_4]] to %[[VAL_8]] step %[[VAL_7]] iter_args(%[[VAL_46:.*]] = %[[VAL_40]], %[[VAL_47:.*]] = %[[VAL_41]], %[[VAL_48:.*]] = %[[VAL_42]], %[[VAL_49:.*]] = %[[VAL_43]]) -> (vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>) {
// CONF1:                   %[[VAL_50:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_39]], %[[VAL_9]], %[[VAL_45]], 0] [1, 2, 1, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16> to memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF1:                   %[[VAL_51:.*]] = vector.load %[[VAL_50]]{{\[}}%[[VAL_4]], %[[VAL_4]], %[[VAL_4]], %[[VAL_4]]] : memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CONF1:                   %[[VAL_52:.*]] = vector.bitcast %[[VAL_51]] : vector<2xbf16> to vector<1xi32>
// CONF1:                   %[[VAL_53:.*]] = vector.broadcast %[[VAL_52]] : vector<1xi32> to vector<16xi32>
// CONF1:                   %[[VAL_54:.*]] = vector.bitcast %[[VAL_53]] : vector<16xi32> to vector<32xbf16>
// CONF1:                   %[[VAL_55:.*]] = vector.load %[[VAL_50]]{{\[}}%[[VAL_4]], %[[VAL_7]], %[[VAL_4]], %[[VAL_4]]] : memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<2xbf16>
// CONF1:                   %[[VAL_56:.*]] = vector.bitcast %[[VAL_55]] : vector<2xbf16> to vector<1xi32>
// CONF1:                   %[[VAL_57:.*]] = vector.broadcast %[[VAL_56]] : vector<1xi32> to vector<16xi32>
// CONF1:                   %[[VAL_58:.*]] = vector.bitcast %[[VAL_57]] : vector<16xi32> to vector<32xbf16>
// CONF1:                   %[[VAL_59:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_39]], %[[VAL_45]], %[[VAL_10]], 0] [1, 1, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CONF1:                   %[[VAL_60:.*]] = vector.load %[[VAL_59]]{{\[}}%[[VAL_4]], %[[VAL_4]], %[[VAL_4]], %[[VAL_4]]] : memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<32xbf16>
// CONF1:                   %[[VAL_61:.*]] = vector.load %[[VAL_59]]{{\[}}%[[VAL_4]], %[[VAL_4]], %[[VAL_8]], %[[VAL_4]]] : memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<32xbf16>
// CONF1:                   %[[VAL_62:.*]] = x86.avx512.dot %[[VAL_46]], %[[VAL_54]], %[[VAL_60]] : vector<32xbf16> -> vector<16xf32>
// CONF1:                   %[[VAL_63:.*]] = x86.avx512.dot %[[VAL_47]], %[[VAL_54]], %[[VAL_61]] : vector<32xbf16> -> vector<16xf32>
// CONF1:                   %[[VAL_64:.*]] = x86.avx512.dot %[[VAL_48]], %[[VAL_58]], %[[VAL_60]] : vector<32xbf16> -> vector<16xf32>
// CONF1:                   %[[VAL_65:.*]] = x86.avx512.dot %[[VAL_49]], %[[VAL_58]], %[[VAL_61]] : vector<32xbf16> -> vector<16xf32>
// CONF1:                   scf.yield %[[VAL_62]], %[[VAL_63]], %[[VAL_64]], %[[VAL_65]] : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CONF1:                 }
// CONF1:                 scf.yield %[[VAL_66:.*]]#0, %[[VAL_66]]#1, %[[VAL_66]]#2, %[[VAL_66]]#3 : vector<16xf32>, vector<16xf32>, vector<16xf32>, vector<16xf32>
// CONF1:               }
// CONF1:               %[[VAL_67:.*]] = arith.truncf %[[VAL_68:.*]]#0 : vector<16xf32> to vector<16xbf16>
// CONF1:               vector.store %[[VAL_67]], %[[VAL_12]]{{\[}}%[[VAL_4]], %[[VAL_4]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:               %[[VAL_69:.*]] = arith.truncf %[[VAL_68]]#1 : vector<16xf32> to vector<16xbf16>
// CONF1:               vector.store %[[VAL_69]], %[[VAL_12]]{{\[}}%[[VAL_4]], %[[VAL_8]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:               %[[VAL_70:.*]] = arith.truncf %[[VAL_68]]#2 : vector<16xf32> to vector<16xbf16>
// CONF1:               vector.store %[[VAL_70]], %[[VAL_25]]{{\[}}%[[VAL_4]], %[[VAL_4]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:               %[[VAL_71:.*]] = arith.truncf %[[VAL_68]]#3 : vector<16xf32> to vector<16xbf16>
// CONF1:               vector.store %[[VAL_71]], %[[VAL_25]]{{\[}}%[[VAL_4]], %[[VAL_8]]] : memref<1x32xbf16, strided<[32, 1], offset: ?>>, vector<16xbf16>
// CONF1:             }
// CONF1:           }
// CONF1:           return
// CONF1:         }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d1)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  func.func @TransposeB_no_lowering(%arg0: memref<32x32x16x2xbf16>, %arg1: memref<32x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %arg3 = %c0 to %c32 step %c2 {
      scf.for %arg4 = %c0 to %c32 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [2, 32] [1, 1] : memref<32x32xbf16> to memref<2x32xbf16, strided<[32, 1], offset: ?>>
        scf.for %arg5 = %c0 to %c32 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg6, 0] [1, 2, 1, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16> to memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg6, %arg4, 0] [1, 1, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
            %0 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<1x2x1x2xbf16>
            %1 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true], permutation_map = #map} : memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<1x32x1x2xbf16>
            %2 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x32xbf16, strided<[32, 1], offset: ?>>, vector<2x32xbf16>
            %3 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<1x2x1x2xbf16>, vector<1x32x1x2xbf16> into vector<2x32xbf16>
            vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x32xbf16>, memref<2x32xbf16, strided<[32, 1], offset: ?>>
          }
        }
      }
    }
    return
  }
}

// CONF1-LABEL: func.func @TransposeB_no_lowering
// CONF1-NOT: x86.avx512.dot

// -----

module {
 func.func @gemm_64_tiles_testing_different_cases(%arg0: memref<32x64x32x2xbf16>, %arg1: memref<32x32x64x2xbf16>, %arg2: memref<64x64xbf16>) {
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<32x64x32x2xbf16>, memref<32x32x64x2xbf16>) outs(%arg2 : memref<64x64xbf16>) {
    ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
      %0 = arith.mulf %in, %in_2 : bf16
      %1 = arith.addf %out, %0 : bf16
      linalg.yield %1 : bf16
    }
  return
 }
}

// CONF2-LABEL: func.func @gemm_64_tiles_testing_different_cases
// CONF2: x86.avx512.dot
// CONF2-NEXT: x86.avx512.dot
// CONF2-NEXT: x86.avx512.dot
// CONF2-NEXT: x86.avx512.dot
// CONF2-NEXT: x86.avx512.dot
// CONF2-NEXT: x86.avx512.dot
// CONF2-NEXT: x86.avx512.dot
// CONF2-NEXT: x86.avx512.dot
// CONF2-NEXT: scf.yield

// CONF3-LABEL: func.func @gemm_64_tiles_testing_different_cases
// CONF3-NOT: x86.avx512.dot

// CONF4-LABEL: func.func @gemm_64_tiles_testing_different_cases
// CONF4-NOT: x86.avx512.dot

// -----

module {
  func.func @gemm_no_bf16dp_lowering(%arg0: memref<32x16x32xf32>, %arg1: memref<32x32x32xf32>, %arg2: memref<16x32xf32>) {
      linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<32x16x32xf32>, memref<32x32x32xf32>) outs(%arg2 : memref<16x32xf32>)
    return
  }
}

// CONF2-LABEL: func.func @gemm_no_bf16dp_lowering
// CONF2-NOT: x86.avx512.dot

