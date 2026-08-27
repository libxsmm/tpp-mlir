//RUN: tpp-opt %s  --scf-parallel-loop-tiling-pass -split-input-file | FileCheck %s

  func.func @entry() {
    %work = memref.alloca():memref<8x32xi32>
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c32_i32 = arith.constant 32: i32
    scf.parallel (%i, %j) = (%c0, %c0) to (%c8, %c32)  step(%c1, %c1) { 
        %temp = arith.muli %i, %c32 :index
	%workid_ij = arith.addi %temp, %j: index
	%workid_ij_i32 = index.casts %workid_ij :index to i32
        memref.store %workid_ij_i32, %work[%i, %j]:memref<8x32xi32>	
   }
  %d1 = arith.constant 0 : i32
   
  %v1 = vector.transfer_read %work[%c0, %c0], %d1 : memref<8x32xi32>, vector<8x32xi32>
  vector.print %v1 : vector<8x32xi32>
 

  return
}


//CHECK:module {
//CHECK: func.func @entry() {
//CHECK:   %[[alloca:.*]] = memref.alloca() : memref<8x32xi32>
//CHECK:   %[[c0:.*]] = arith.constant 0 : index
//CHECK:   %[[c8:.*]] = arith.constant 8 : index
//CHECK:   %[[c32:.*]] = arith.constant 32 : index
//CHECK:   %[[c1:.*]] = arith.constant 1 : index
//CHECK:   %[[c32_i32:.*]] = arith.constant 32 : i32
//CHECK:   %[[c0_0:.*]] = arith.constant 0 : index
//CHECK:   %[[c1_1:.*]] = arith.constant 1 : index
//CHECK:   %[[c1_2:.*]] = arith.constant 1 : index
//CHECK:   %[[temp0:.*]] = arith.muli %[[c1]], %[[c1_1]] : index
//CHECK:   %[[temp1:.*]] = arith.muli %[[c1]], %[[c1_2]] : index
//CHECK:   scf.parallel (%[[ARG0:.*]], %[[ARG1:.*]]) = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[temp0]], %[[temp1]]) 
//CHECK:     scf.for %[[ARG2:.*]] = %[[c0_0]] to %[[temp0]] step %[[c1]] {
//CHECK:       scf.for %[[ARG3:.*]] = %[[c0_0]] to %[[temp1]] step %[[c1]] {
//CHECK:         %[[temp3:.*]] = arith.addi %[[ARG2]], %[[ARG0]] : index
//CHECK:         %[[temp4:.*]] = arith.addi %[[ARG3]], %[[ARG1]] : index
//CHECK:         %[[temp5:.*]] = arith.muli %[[temp3]], %[[c32]] : index
//CHECK:         %[[temp6:.*]] = arith.addi %[[temp5]], %[[temp4]] : index
//CHECK:         %[[temp7:.*]] = index.casts %[[temp6]] : index to i32
//CHECK:         memref.store %[[temp7]], %alloca[%[[temp3]], %[[temp4]]] : memref<8x32xi32>
