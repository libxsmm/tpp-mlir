// RUN: tpp-opt %s -convert-tpp-to-loops -split-input-file | FileCheck %s

// CHECK: func.func @identity_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>) {
func.func @identity_to_loops(%arg0: memref<3x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[fill:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     memref.store %[[fill]], %[[ARG0]][%[[i]], %[[j]]] : memref<3x3xf32>
  %cst = arith.constant 0.000000e+00 : f32
  tpp.identity ins(%cst: f32) outs(%arg0: memref<3x3xf32>)
  return
}

// -----

// CHECK: func.func @relu_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>) {
func.func @relu_to_loops(%arg0: memref<3x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[relu:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[load:.*]] = memref.load %[[ARG0]][%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[max:.*]] = arith.maxf %[[load]], %[[relu]] : f32
  // CHECK:     memref.store %[[max]], %[[ARG0]][%[[i]], %[[j]]] : memref<3x3xf32>
  tpp.relu ins(%arg0: memref<3x3xf32>) outs(%arg0: memref<3x3xf32>)
  return
}

// -----

// CHECK: func.func @relu_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>)
func.func @relu_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[relu:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[load:.*]] = memref.load %[[ARG0]][%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[max:.*]] = arith.maxf %[[load]], %[[relu]] : f32
  // CHECK:     memref.store %[[max]], %[[ARG1]][%[[i]], %[[j]]] : memref<3x3xf32>
  tpp.relu ins(%arg0: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>)
  return 
}

// -----

// CHECK: func.func @add_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<3x3xf32>) {
func.func @add_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) {
  // CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:     %[[load1:.*]] = memref.load %[[ARG0]][%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[load2:.*]] = memref.load %[[ARG1]][%[[i]], %[[j]]] : memref<3x3xf32>
  // CHECK:     %[[add:.*]] = arith.addf %[[load1]], %[[load2]] : f32
  // CHECK:     memref.store %[[add]], %[[ARG1]][%[[i]], %[[j]]] : memref<3x3xf32>
  tpp.add ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) outs(%arg1: memref<3x3xf32>)
  return
}

// -----

func.func @identity_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<3xf32>) {
  tpp.identity ins(%arg1: memref<3xf32>) outs(%arg0: memref<3x3xf32>)
  return
}

// CHECK: func.func @identity_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<3xf32>) {
// CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:     %[[tostore:.*]] = memref.load %[[ARG1]][%[[j]]] : memref<3xf32>
// CHECK:     memref.store %[[tostore]], %[[ARG0]][%[[i]], %[[j]]] : memref<3x3xf32>


// -----

func.func @identity_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<1x3xf32>) { 
  tpp.identity ins(%arg1: memref<1x3xf32>) outs(%arg0: memref<3x3xf32>)
  return
}

// CHECK: func.func @identity_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<1x3xf32>) {
// CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:     %[[tostore:.*]] = memref.load %arg1[%[[lb]], %[[j]]] : memref<1x3xf32>
// CHECK:     memref.store %[[tostore]], %arg0[%[[i]], %[[j]]] : memref<3x3xf32>

// -----

func.func @identity_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<1x1xf32>) { 
  tpp.identity ins(%arg1: memref<1x1xf32>) outs(%arg0: memref<3x3xf32>)
  return
}

// CHECK: func.func @identity_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<1x1xf32>) {
// CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:     %[[tostore:.*]] = memref.load %[[ARG1]][%[[lb]], %[[lb]]] : memref<1x1xf32>
// CHECK:     memref.store %[[tostore]], %[[ARG0]][%[[i]], %[[j]]] : memref<3x3xf32>

// -----

func.func @identity_to_loops(%arg0: memref<5x1xf32>, %arg1: memref<5x6xf32>) { 
  tpp.identity ins(%arg0: memref<5x1xf32>) outs(%arg1: memref<5x6xf32>)
  return
}

// CHECK: func.func @identity_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<5x1xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<5x6xf32>) {
// CHECK-DAG: %[[five:.*]] = arith.constant 5 : index
// CHECK-DAG: %[[six:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[i:.*]] = %[[zero]] to %[[five]] step %[[one]] {
// CHECK:   scf.for %[[j:.*]] = %[[zero]] to %[[six]] step %[[one]] {
// CHECK:     %[[tostore:.*]] = memref.load %[[ARG0]][%[[i]], %[[zero]]] : memref<5x1xf32>
// CHECK:     memref.store %[[tostore]], %[[ARG1]][%[[i]], %[[j]]] : memref<5x6xf32>

// -----

func.func @brgemm_to_loops(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) { 
  tpp.brgemm ins(%arg0: memref<2x3x4xf32>, %arg1: memref<2x4x3xf32>, %arg2: memref<3x3xf32>) 
             outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: func.func @brgemm_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<2x3x4xf32>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<2x4x3xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<3x3xf32>) {
// CHECK-DAG: %[[three:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[four:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[two:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[one:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[b:.*]] = %[[zero]] to %[[two]] step %[[one]] {
// CHECK: scf.for %[[i:.*]] = %[[zero]] to %[[three]] step %[[one]] {
// CHECK: scf.for %[[j:.*]] = %[[zero]] to %[[three]] step %[[one]] {
// CHECK: scf.for %[[k:.*]] = %[[zero]] to %[[four]] step %[[one]] {
// CHECK: %[[ma:.*]] = memref.load %[[ARG0]][%[[b]], %[[i]], %[[k]]] : memref<2x3x4xf32>
// CHECK: %[[mb:.*]] = memref.load %[[ARG1]][%[[b]], %[[k]], %[[j]]] : memref<2x4x3xf32>
// CHECK: %[[mc:.*]] = memref.load %[[ARG2]][%[[i]], %[[j]]] : memref<3x3xf32>
// CHECK: %[[mul:.*]] = arith.mulf %[[ma]], %[[mb]] : f32
// CHECK: %[[add:.*]] = arith.addf %[[mc]], %[[mul]] : f32
// CHECK: memref.store %[[add]], %[[ARG2]][%[[i]], %[[j]]] : memref<3x3xf32>

// -----

func.func @add_to_loops(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  tpp.add ins(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>) outs(%arg2: memref<3x3xf32>)
  return
}

// CHECK: func.func @add_to_loops(
// CHECK-SAME:  %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>, %[[ARG2:.+]]: memref<3x3xf32>) {
// CHECK-DAG: %[[ub:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[i:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:   scf.for %[[j:.*]] = %[[lb]] to %[[ub]] step %[[step]] {
// CHECK:     %[[load1:.*]] = memref.load %[[ARG0]][%[[i]], %[[j]]] : memref<3x3xf32>
// CHECK:     %[[load2:.*]] = memref.load %[[ARG1]][%[[i]], %[[j]]] : memref<3x3xf32>
// CHECK:     %[[add:.*]] = arith.addf %[[load1]], %[[load2]] : f32
// CHECK:     memref.store %[[add]], %[[ARG2]][%[[i]], %[[j]]] : memref<3x3xf32>
