// RUN: tpp-opt %s  --hoist-vector-transfer --split-input-file  | FileCheck %s


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @simple_gemm(%arg0: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x64xf32>
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
    scf.forall (%arg1, %arg2) in (8, 24) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c4 {
        scf.for %arg4 = %c0 to %c64 step %c64 {
          %subview_2 = memref.subview %subview[%arg3, %arg4] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c24 step %c1 {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg6] [1, 4, 1] [1, 1, 1] : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
              %subview_4 = memref.subview %0[%arg5, %arg6, %arg4] [1, 1, 64] [1, 1, 1] : memref<24x64x64xf32> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
              %1 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x4x1xf32>
              %2 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<1x1x64xf32>
              %3 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x4x1xf32>, vector<1x1x64xf32> into vector<4x64xf32>
              vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
            }
          }
        }
      }
    }
    return %alloc : memref<8x24x32x64xf32>
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-LABEL:   memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @simple_gemm(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
// CHECK:           scf.forall (%[[VAL_11:.*]], %[[VAL_12:.*]]) in (8, 24) {
// CHECK:             %[[VAL_13:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_11]], %[[VAL_12]], 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_13]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             %[[VAL_14:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_11]], 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_8]] to %[[VAL_7]] step %[[VAL_6]] {
// CHECK:               scf.for %[[VAL_16:.*]] = %[[VAL_8]] to %[[VAL_5]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_17:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_15]], %[[VAL_16]]] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_18:.*]] = vector.transfer_read %[[VAL_17]]{{\[}}%[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<4x64xf32, strided<[64, 1], offset: ?>>, vector<4x64xf32>
// CHECK:                 %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_8]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (vector<4x64xf32>) {
// CHECK:                   %[[VAL_22:.*]] = scf.for %[[VAL_23:.*]] = %[[VAL_8]] to %[[VAL_5]] step %[[VAL_3]] iter_args(%[[VAL_24:.*]] = %[[VAL_21]]) -> (vector<4x64xf32>) {
// CHECK:                     %[[VAL_25:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_20]], %[[VAL_15]], %[[VAL_23]]] [1, 4, 1] [1, 1, 1] : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_26:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_20]], %[[VAL_23]], %[[VAL_16]]] [1, 1, 64] [1, 1, 1] : memref<24x64x64xf32> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
// CHECK:                     %[[VAL_27:.*]] = vector.transfer_read %[[VAL_25]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x4x1xf32>
// CHECK:                     %[[VAL_28:.*]] = vector.transfer_read %[[VAL_26]]{{\[}}%[[VAL_8]], %[[VAL_8]], %[[VAL_8]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<1x1x64xf32>
// CHECK:                     %[[VAL_29:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x4x1xf32>, vector<1x1x64xf32> into vector<4x64xf32>
// CHECK:                     scf.yield %[[VAL_29]] : vector<4x64xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_22]] : vector<4x64xf32>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_19]], %[[VAL_17]]{{\[}}%[[VAL_8]], %[[VAL_8]]] {in_bounds = [true, true]} : vector<4x64xf32>, memref<4x64xf32, strided<[64, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_10]] : memref<8x24x32x64xf32>
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @chainned_gemm(%arg0: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
    %c1 = arith.constant 1 : index
    %c48 = arith.constant 48 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c8 {
        scf.for %arg4 = %c0 to %c32 step %c32 {
          %subview_3 = memref.subview %subview[%arg3, %arg4] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c48 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c1 {
              %subview_4 = memref.subview %subview_2[%arg5, %arg3, %arg6] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
              %subview_5 = memref.subview %0[%arg5, %arg6, %arg4] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
              %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x8x1xf32>
              %2 = vector.transfer_read %subview_5[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x1x32xf32>
              %3 = vector.transfer_read %subview_3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x32xf32, strided<[32, 1], offset: ?>>, vector<8x32xf32>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x8x1xf32>, vector<1x1x32xf32> into vector<8x32xf32>
              vector.transfer_write %4, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, memref<8x32xf32, strided<[32, 1], offset: ?>>
            }
          }
        }
      }
    }
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc_1[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c8 {
        scf.for %arg4 = %c0 to %c32 step %c32 {
          %subview_3 = memref.subview %subview[%arg3, %arg4] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c48 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c1 {
              %subview_4 = memref.subview %subview_2[%arg5, %arg3, %arg6] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
              %subview_5 = memref.subview %0[%arg5, %arg6, %arg4] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
              %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x8x1xf32>
              %2 = vector.transfer_read %subview_5[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x1x32xf32>
              %3 = vector.transfer_read %subview_3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x32xf32, strided<[32, 1], offset: ?>>, vector<8x32xf32>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x8x1xf32>, vector<1x1x32xf32> into vector<8x32xf32>
              vector.transfer_write %4, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, memref<8x32xf32, strided<[32, 1], offset: ?>>
            }
          }
        }
      }
    }
    scf.forall (%arg1, %arg2) in (8, 48) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %alloc_1[%arg1, 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c8 {
        scf.for %arg4 = %c0 to %c32 step %c32 {
          %subview_3 = memref.subview %subview[%arg3, %arg4] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c48 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c1 {
              %subview_4 = memref.subview %subview_2[%arg5, %arg3, %arg6] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
              %subview_5 = memref.subview %0[%arg5, %arg6, %arg4] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
              %1 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x8x1xf32>
              %2 = vector.transfer_read %subview_5[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x1x32xf32>
              %3 = vector.transfer_read %subview_3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<8x32xf32, strided<[32, 1], offset: ?>>, vector<8x32xf32>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x8x1xf32>, vector<1x1x32xf32> into vector<8x32xf32>
              vector.transfer_write %4, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, memref<8x32xf32, strided<[32, 1], offset: ?>>
            }
          }
        }
      }
    }
    return %alloc : memref<8x48x32x32xf32>
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK-LABEL:   memref.global "private" constant @__constant_48x32x32xf32 : memref<48x32x32xf32> = dense<1.000000e+00> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @chainned_gemm(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x48x32x32xf32>) -> memref<8x48x32x32xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<32x32xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 48 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = memref.get_global @__constant_48x32x32xf32 : memref<48x32x32xf32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:           scf.forall (%[[VAL_10:.*]], %[[VAL_11:.*]]) in (8, 48) {
// CHECK:             %[[VAL_12:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_12]]{{\[}}%[[VAL_7]], %[[VAL_7]]] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             %[[VAL_13:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_10]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_14:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:               scf.for %[[VAL_15:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_6]] {
// CHECK:                 %[[VAL_16:.*]] = memref.subview %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_15]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:                 %[[VAL_17:.*]] = vector.transfer_read %[[VAL_16]]{{\[}}%[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<8x32xf32, strided<[32, 1], offset: ?>>, vector<8x32xf32>
// CHECK:                 %[[VAL_18:.*]] = scf.for %[[VAL_19:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_20:.*]] = %[[VAL_17]]) -> (vector<8x32xf32>) {
// CHECK:                   %[[VAL_21:.*]] = scf.for %[[VAL_22:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_3]] iter_args(%[[VAL_23:.*]] = %[[VAL_20]]) -> (vector<8x32xf32>) {
// CHECK:                     %[[VAL_24:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_19]], %[[VAL_14]], %[[VAL_22]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_25:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_19]], %[[VAL_22]], %[[VAL_15]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_26:.*]] = vector.transfer_read %[[VAL_24]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x8x1xf32>
// CHECK:                     %[[VAL_27:.*]] = vector.transfer_read %[[VAL_25]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x1x32xf32>
// CHECK:                     %[[VAL_28:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x8x1xf32>, vector<1x1x32xf32> into vector<8x32xf32> 
// CHECK:                     scf.yield %[[VAL_28]] : vector<8x32xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_21]] : vector<8x32xf32>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_18]], %[[VAL_16]]{{\[}}%[[VAL_7]], %[[VAL_7]]] {in_bounds = [true, true]} : vector<8x32xf32>, memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_29:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x48x32x32xf32>
// CHECK:           scf.forall (%[[VAL_30:.*]], %[[VAL_31:.*]]) in (8, 48) {
// CHECK:             %[[VAL_32:.*]] = memref.subview %[[VAL_29]]{{\[}}%[[VAL_30]], %[[VAL_31]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_32]]{{\[}}%[[VAL_7]], %[[VAL_7]]] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             %[[VAL_33:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_30]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_34:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:               scf.for %[[VAL_35:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_6]] {
// CHECK:                 %[[VAL_36:.*]] = memref.subview %[[VAL_32]]{{\[}}%[[VAL_34]], %[[VAL_35]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:                 %[[VAL_37:.*]] = vector.transfer_read %[[VAL_36]]{{\[}}%[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<8x32xf32, strided<[32, 1], offset: ?>>, vector<8x32xf32>
// CHECK:                 %[[VAL_38:.*]] = scf.for %[[VAL_39:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_40:.*]] = %[[VAL_37]]) -> (vector<8x32xf32>) {
// CHECK:                   %[[VAL_41:.*]] = scf.for %[[VAL_42:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_3]] iter_args(%[[VAL_43:.*]] = %[[VAL_40]]) -> (vector<8x32xf32>) {
// CHECK:                     %[[VAL_44:.*]] = memref.subview %[[VAL_33]]{{\[}}%[[VAL_39]], %[[VAL_34]], %[[VAL_42]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_45:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_39]], %[[VAL_42]], %[[VAL_35]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_46:.*]] = vector.transfer_read %[[VAL_44]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x8x1xf32>
// CHECK:                     %[[VAL_47:.*]] = vector.transfer_read %[[VAL_45]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x1x32xf32>
// CHECK:                     %[[VAL_48:.*]] =  vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x8x1xf32>, vector<1x1x32xf32> into vector<8x32xf32>
// CHECK:                     scf.yield %[[VAL_48]] : vector<8x32xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_41]] : vector<8x32xf32>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_38]], %[[VAL_36]]{{\[}}%[[VAL_7]], %[[VAL_7]]] {in_bounds = [true, true]} : vector<8x32xf32>, memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           scf.forall (%[[VAL_49:.*]], %[[VAL_50:.*]]) in (8, 48) {
// CHECK:             %[[VAL_51:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_49]], %[[VAL_50]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_51]]{{\[}}%[[VAL_7]], %[[VAL_7]]] {in_bounds = [true, true]} : vector<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:             %[[VAL_52:.*]] = memref.subview %[[VAL_29]]{{\[}}%[[VAL_49]], 0, 0, 0] [1, 48, 32, 32] [1, 1, 1, 1] : memref<8x48x32x32xf32> to memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_53:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:               scf.for %[[VAL_54:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_6]] {
// CHECK:                 %[[VAL_55:.*]] = memref.subview %[[VAL_51]]{{\[}}%[[VAL_53]], %[[VAL_54]]] [8, 32] [1, 1] : memref<32x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:                 %[[VAL_56:.*]] = vector.transfer_read %[[VAL_55]]{{\[}}%[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<8x32xf32, strided<[32, 1], offset: ?>>, vector<8x32xf32>
// CHECK:                 %[[VAL_57:.*]] = scf.for %[[VAL_58:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_59:.*]] = %[[VAL_56]]) -> (vector<8x32xf32>) {
// CHECK:                   %[[VAL_60:.*]] = scf.for %[[VAL_61:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_3]] iter_args(%[[VAL_62:.*]] = %[[VAL_59]]) -> (vector<8x32xf32>) {
// CHECK:                     %[[VAL_63:.*]] = memref.subview %[[VAL_52]]{{\[}}%[[VAL_58]], %[[VAL_53]], %[[VAL_61]]] [1, 8, 1] [1, 1, 1] : memref<48x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_64:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_58]], %[[VAL_61]], %[[VAL_54]]] [1, 1, 32] [1, 1, 1] : memref<48x32x32xf32> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:                     %[[VAL_65:.*]] = vector.transfer_read %[[VAL_63]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x8x1xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x8x1xf32>
// CHECK:                     %[[VAL_66:.*]] = vector.transfer_read %[[VAL_64]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>, vector<1x1x32xf32>
// CHECK:                     %[[VAL_67:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x8x1xf32>, vector<1x1x32xf32> into vector<8x32xf32>
// CHECK:                     scf.yield %[[VAL_67]] : vector<8x32xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_60]] : vector<8x32xf32>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_57]], %[[VAL_55]]{{\[}}%[[VAL_7]], %[[VAL_7]]] {in_bounds = [true, true]} : vector<8x32xf32>, memref<8x32xf32, strided<[32, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_9]] : memref<8x48x32x32xf32>
// CHECK:         }



// -----


#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @hoist_gemm_bf16_vnni_layout(%arg0: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64x64xbf16>
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x16x64x64xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
    scf.forall (%arg1, %arg2) in (4, 16) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xbf16>, memref<64x64xbf16, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c64 step %c32 {
        scf.for %arg4 = %c0 to %c64 step %c32 {
          %subview_2 = memref.subview %subview[%arg3, %arg4] [32, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c16 step %c1 {
            scf.for %arg6 = %c0 to %c32 step %c16 {
              %subview_3 = memref.subview %subview_1[%arg5, %arg3, %arg6, 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
              %subview_4 = memref.subview %0[%arg5, %arg6, %arg4, 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x32x64x2xbf16> to memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
              %1 = vector.transfer_read %subview_3[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, vector<1x32x16x2xbf16>
              %2 = vector.transfer_read %subview_4[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>, vector<1x16x32x2xbf16>
              %3 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x32xbf16, strided<[64, 1], offset: ?>>, vector<32x32xbf16>
              %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<1x32x16x2xbf16>, vector<1x16x32x2xbf16> into vector<32x32xbf16>
              vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[64, 1], offset: ?>>
            }
          }
        }
      }
    }
    return %alloc : memref<4x16x64x64xbf16>
  }
}



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
// CHECK-LABEL:   memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
// CHECK-LABEL:   func.func @hoist_gemm_bf16_vnni_layout(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<64x64xbf16>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x16x64x64xbf16>
// CHECK:           %[[VAL_10:.*]] = memref.expand_shape %[[VAL_0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
// CHECK:           scf.forall (%[[VAL_11:.*]], %[[VAL_12:.*]]) in (4, 16) {
// CHECK:             %[[VAL_13:.*]] = memref.subview %[[VAL_9]]{{\[}}%[[VAL_11]], %[[VAL_12]], 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_13]]{{\[}}%[[VAL_7]], %[[VAL_7]]] {in_bounds = [true, true]} : vector<64x64xbf16>, memref<64x64xbf16, strided<[64, 1], offset: ?>>
// CHECK:             %[[VAL_14:.*]] = memref.subview %[[VAL_10]]{{\[}}%[[VAL_11]], 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CHECK:             scf.for %[[VAL_15:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:               scf.for %[[VAL_16:.*]] = %[[VAL_7]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:                 %[[VAL_17:.*]] = memref.subview %[[VAL_13]]{{\[}}%[[VAL_15]], %[[VAL_16]]] [32, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
// CHECK:                 %[[VAL_18:.*]] = vector.transfer_read %[[VAL_17]]{{\[}}%[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<32x32xbf16, strided<[64, 1], offset: ?>>, vector<32x32xbf16>
// CHECK:                 %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_3]] iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (vector<32x32xbf16>) {
// CHECK:                   %[[VAL_22:.*]] = scf.for %[[VAL_23:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_4]] iter_args(%[[VAL_24:.*]] = %[[VAL_21]]) -> (vector<32x32xbf16>) {
// CHECK:                     %[[VAL_25:.*]] = memref.subview %[[VAL_14]]{{\[}}%[[VAL_20]], %[[VAL_15]], %[[VAL_23]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CHECK:                     %[[VAL_26:.*]] = memref.subview %[[VAL_8]]{{\[}}%[[VAL_20]], %[[VAL_23]], %[[VAL_16]], 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x32x64x2xbf16> to memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
// CHECK:                     %[[VAL_27:.*]] = vector.transfer_read %[[VAL_25]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true, true, true]} : memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, vector<1x32x16x2xbf16>
// CHECK:                     %[[VAL_28:.*]] = vector.transfer_read %[[VAL_26]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]], %[[VAL_1]] {in_bounds = [true, true, true, true]} : memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>, vector<1x16x32x2xbf16>
// CHECK:                     %[[VAL_29:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x32x16x2xbf16>, vector<1x16x32x2xbf16> into vector<32x32xbf16>
// CHECK:                     scf.yield %[[VAL_29]] : vector<32x32xbf16>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_22]] : vector<32x32xbf16>
// CHECK:                 }
// CHECK:                 vector.transfer_write %[[VAL_19]], %[[VAL_17]]{{\[}}%[[VAL_7]], %[[VAL_7]]] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[64, 1], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[VAL_9]] : memref<4x16x64x64xbf16>
// CHECK:         }

// -----


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @gemm_without_tiling(%arg0: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x64xf32>
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
    scf.forall (%arg1, %arg2) in (8, 24) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      vector.transfer_write %cst_0, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
      %subview_1 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
      %1 = vector.transfer_read %subview_1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>, vector<24x32x64xf32>
      %2 = vector.transfer_read %0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<24x64x64xf32>, vector<24x64x64xf32>
      %3 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x64xf32, strided<[64, 1], offset: ?>>, vector<32x64xf32>
      %4 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<24x32x64xf32>, vector<24x64x64xf32> into vector<32x64xf32>
      vector.transfer_write %4, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
    }
    return %alloc : memref<8x24x32x64xf32>
  }
}



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-LABEL:   memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}

// CHECK-LABEL:   func.func @gemm_without_tiling(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
// CHECK:           scf.forall (%[[VAL_6:.*]], %[[VAL_7:.*]]) in (8, 24) {
// CHECK:             %[[VAL_8:.*]] = memref.subview %[[VAL_5]]{{\[}}%[[VAL_6]], %[[VAL_7]], 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             vector.transfer_write %[[VAL_2]], %[[VAL_8]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:             %[[VAL_9:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_6]], 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
// CHECK:             %[[VAL_10:.*]] = vector.transfer_read %[[VAL_9]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>, vector<24x32x64xf32>
// CHECK:             %[[VAL_11:.*]] = vector.transfer_read %[[VAL_4]]{{\[}}%[[VAL_3]], %[[VAL_3]], %[[VAL_3]]], %[[VAL_1]] {in_bounds = [true, true, true]} : memref<24x64x64xf32>, vector<24x64x64xf32>
// CHECK:             %[[VAL_12:.*]] = vector.transfer_read %[[VAL_8]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_1]] {in_bounds = [true, true]} : memref<32x64xf32, strided<[64, 1], offset: ?>>, vector<32x64xf32>
// CHECK:             %[[VAL_13:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %1, %2, %3 : vector<24x32x64xf32>, vector<24x64x64xf32> into vector<32x64xf32> 
// CHECK:             vector.transfer_write %[[VAL_13]], %[[VAL_8]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<32x64xf32>, memref<32x64xf32, strided<[64, 1], offset: ?>>
// CHECK:           }
// CHECK:           return %[[VAL_5]] : memref<8x24x32x64xf32>
// CHECK:         }

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @gemm_with_args(%arg0: tensor<4x1xf32>, %arg1: tensor<1x64xf32>, %arg2: tensor<4x64xf32>) -> tensor<4x64xf32> {
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


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>


// CHECK-LABEL:   func.func @gemm_with_args(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<4x1xf32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: tensor<1x64xf32>,
// CHECK-SAME:                            %[[VAL_2:.*]]: tensor<4x64xf32>) -> tensor<4x64xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_5:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<4x1xf32>, vector<4x1xf32>
// CHECK:           %[[VAL_6:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<4x64xf32>, vector<4x64xf32>
// CHECK:           %[[VAL_8:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x1xf32>, vector<1x64xf32> into vector<4x64xf32> 
// CHECK:           %[[VAL_9:.*]] = vector.transfer_write %[[VAL_8]], %[[VAL_2]]{{\[}}%[[VAL_3]], %[[VAL_3]]] {in_bounds = [true, true]} : vector<4x64xf32>, tensor<4x64xf32>
// CHECK:           return %[[VAL_9]] : tensor<4x64xf32>
// CHECK:         }
