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

// CHECK-LABEL: func.func @flatten_2d_forall()
// CHECK:         %[[WORK:.*]] = memref.alloca() : memref<4x8xi32>
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK:         %[[C8:.*]] = arith.constant 8 : index
// CHECK:         %[[IV_I:.*]] = arith.constant dense<[0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 1, 0, 0, 0, 0, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 1, 0, 0]> : vector<32xi16>
// CHECK:         %[[IV_J:.*]] = arith.constant dense<[0, 1, 1, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 2, 2, 3, 4, 5, 5, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 6, 6, 7]> : vector<32xi16>
// CHECK:         scf.forall (%[[IDX:.*]]) in (32) {
// CHECK:           %[[I_I16:.*]] = vector.extract %[[IV_I]][%[[IDX]]] : i16 from vector<32xi16>
// CHECK:           %[[J_I16:.*]] = vector.extract %[[IV_J]][%[[IDX]]] : i16 from vector<32xi16>
// CHECK:           %[[I:.*]] = arith.index_cast %[[I_I16]] : i16 to index
// CHECK:           %[[J:.*]] = arith.index_cast %[[J_I16]] : i16 to index
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

// CHECK-LABEL: func.func @flatten_different_bounds()
// CHECK:         %[[WORK:.*]] = memref.alloca() : memref<2x3xi32>
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C100:.*]] = arith.constant 100 : i32
// CHECK:         %[[IV_I:.*]] = arith.constant dense<[0, 1, 1, 1, 0, 0]> : vector<6xi16>
// CHECK:         %[[IV_J:.*]] = arith.constant dense<[0, 0, 1, 2, 2, 1]> : vector<6xi16>
// CHECK:         scf.forall (%[[IDX:.*]]) in (6) {
// CHECK:           %[[I_I16:.*]] = vector.extract %[[IV_I]][%[[IDX]]] : i16 from vector<6xi16>
// CHECK:           %[[J_I16:.*]] = vector.extract %[[IV_J]][%[[IDX]]] : i16 from vector<6xi16>
// CHECK:           %[[I:.*]] = arith.index_cast %[[I_I16]] : i16 to index
// CHECK:           %[[J:.*]] = arith.index_cast %[[J_I16]] : i16 to index
// CHECK:           memref.store %[[C100]], %[[WORK]][%[[I]], %[[J]]] : memref<2x3xi32>
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
