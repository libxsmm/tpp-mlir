// RUN: tpp-opt %s --scf-forall-loop-flatten-sfc -split-input-file | FileCheck %s

func.func @flatten_2d_forall() {
  %work = memref.alloca() : memref<4x8xi32>
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  
  scf.forall (%i, %j) in (%c4, %c8) {
    %temp = arith.muli %i, %c8 : index
    %workid = arith.addi %temp, %j : index
    %workid_i32 = index.casts %workid : index to i32
    memref.store %workid_i32, %work[%i, %j] : memref<4x8xi32>
  }
  
  return
}

// CHECK:         memref.global "private" constant @[[IV0:.*]] : memref<32xi8> = dense<[0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 1, 0, 0, 0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 1, 0, 0]>
// CHECK:         memref.global "private" constant @[[IV1:.*]] : memref<32xi8> = dense<[0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 6, 6, 7]>
// CHECK:         func.func @flatten_2d_forall()
// CHECK:         %[[WORK:.*]] = memref.alloca() : memref<4x8xi32>
// CHECK:         %[[C8:.*]] = arith.constant 8 : index
// CHECK:         %[[TBL_I:.*]] = memref.get_global @[[IV0]] : memref<32xi8>
// CHECK:         %[[TBL_J:.*]] = memref.get_global @[[IV1]] : memref<32xi8>
// CHECK:         scf.forall (%[[IDX:.*]]) in (32) {
// CHECK:           %[[I_I8:.*]] = memref.load %[[TBL_I]][%[[IDX]]] : memref<32xi8>
// CHECK:           %[[J_I8:.*]] = memref.load %[[TBL_J]][%[[IDX]]] : memref<32xi8>
// CHECK:           %[[I:.*]] = arith.index_cast %[[I_I8]] : i8 to index
// CHECK:           %[[J:.*]] = arith.index_cast %[[J_I8]] : i8 to index
// CHECK:           %[[TEMP:.*]] = arith.muli %[[I]], %[[C8]] : index
// CHECK:           %[[WORKID:.*]] = arith.addi %[[TEMP]], %[[J]] : index
// CHECK:           %[[WORKID_I32:.*]] = index.casts %[[WORKID]] : index to i32
// CHECK:           memref.store %[[WORKID_I32]], %[[WORK]][%[[I]], %[[J]]] : memref<4x8xi32>
// CHECK:         }
// CHECK:         return

// -----

func.func @flatten_different_bounds() {
  %work = memref.alloca() : memref<2x3xi32>
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c100 = arith.constant 100 : i32
  
  scf.forall (%i, %j) in (%c2, %c3) {
    memref.store %c100, %work[%i, %j] : memref<2x3xi32>
  }
  
  return
}

// CHECK:         memref.global "private" constant @[[IV0:.*]] : memref<6xi8> = dense<[0, 1, 1, 1, 0, 0]>
// CHECK:         memref.global "private" constant @[[IV1:.*]] : memref<6xi8> = dense<[0, 0, 1, 2, 2, 1]>
// CHECK:         func.func @flatten_different_bounds()
// CHECK:         %[[WORK:.*]] = memref.alloca() : memref<2x3xi32>
// CHECK:         %[[C100:.*]] = arith.constant 100 : i32
// CHECK:         %[[TBL_I:.*]] = memref.get_global @[[IV0]] : memref<6xi8>
// CHECK:         %[[TBL_J:.*]] = memref.get_global @[[IV1]] : memref<6xi8>
// CHECK:         scf.forall (%[[IDX:.*]]) in (6) {
// CHECK:           %[[I_I8:.*]] = memref.load %[[TBL_I]][%[[IDX]]] : memref<6xi8>
// CHECK:           %[[J_I8:.*]] = memref.load %[[TBL_J]][%[[IDX]]] : memref<6xi8>
// CHECK:           %[[I:.*]] = arith.index_cast %[[I_I8]] : i8 to index
// CHECK:           %[[J:.*]] = arith.index_cast %[[J_I8]] : i8 to index
// CHECK:           memref.store %[[C100]], %[[WORK]][%[[I]], %[[J]]] : memref<2x3xi32>
// CHECK:         }
// CHECK:         return

// -----

// Test that the index element type widens to i16 once the linearized tile
// count exceeds the i8 range (16 * 16 = 256 tiles > 127).
func.func @flatten_widens_to_i16() {
  %work = memref.alloca() : memref<16x16xi32>
  %c16 = arith.constant 16 : index
  %c100 = arith.constant 100 : i32

  scf.forall (%i, %j) in (%c16, %c16) {
    memref.store %c100, %work[%i, %j] : memref<16x16xi32>
  }

  return
}

// CHECK:         memref.global "private" constant @[[IV0:.*]] : memref<256xi16> = dense<{{.*}}>
// CHECK:         memref.global "private" constant @[[IV1:.*]] : memref<256xi16> = dense<{{.*}}>
// CHECK:         func.func @flatten_widens_to_i16()
// CHECK:         %[[TBL_I:.*]] = memref.get_global @[[IV0]] : memref<256xi16>
// CHECK:         %[[TBL_J:.*]] = memref.get_global @[[IV1]] : memref<256xi16>
// CHECK:         scf.forall (%[[IDX:.*]]) in (256) {
// CHECK:           %[[I_I16:.*]] = memref.load %[[TBL_I]][%[[IDX]]] : memref<256xi16>
// CHECK:           %[[J_I16:.*]] = memref.load %[[TBL_J]][%[[IDX]]] : memref<256xi16>
// CHECK:           %[[I:.*]] = arith.index_cast %[[I_I16]] : i16 to index
// CHECK:           %[[J:.*]] = arith.index_cast %[[J_I16]] : i16 to index
// CHECK:         }
// CHECK:         return

// -----

// Test that 3D forall loops are not flattened
func.func @no_flatten_3d_forall() {
  %work = memref.alloca() : memref<2x3x4xi32>
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c100 = arith.constant 100 : i32
  
  scf.forall (%i, %j, %k) in (%c2, %c3, %c4) {
    memref.store %c100, %work[%i, %j, %k] : memref<2x3x4xi32>
  }
  
  return
}

// CHECK-LABEL: func.func @no_flatten_3d_forall()
// CHECK:         scf.forall (%{{.*}}, %{{.*}}, %{{.*}}) in
// CHECK-NOT:     vector.extract
