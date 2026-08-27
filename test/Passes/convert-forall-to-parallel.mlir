// RUN: tpp-opt %s -convert-forall-to-parallel -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @tensor_forall
func.func @tensor_forall(%arg0: tensor<32x32xbf16>) -> tensor<8x112x32x32xbf16> {
  %c8 = arith.constant 8 : index
  %c112 = arith.constant 112 : index
  %0 = tensor.empty() : tensor<8x112x32x32xbf16>
  // CHECK-NOT: scf.parallel
  %1 = scf.forall (%i, %j) in (%c8, %c112) shared_outs(%k = %0) -> (tensor<8x112x32x32xbf16>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %arg0 into %k[%i, %j, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xbf16> into tensor<8x112x32x32xbf16>
    }
  }
  return %1 : tensor<8x112x32x32xbf16>
}

// -----

// CHECK-LABEL: func.func @memref_forall
func.func @memref_forall(%arg0: memref<32x32xbf16>) -> memref<8x112x32x32xbf16> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK-DAG: %[[C112:.+]] = arith.constant 112 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  %alloc = memref.alloc() alignment = 64 : memref<8x112x32x32xbf16>
  // CHECK-NOT: scf.forall
  // CHECK: scf.parallel (%[[ARG1:.+]], %[[ARG2:.+]]) =
  // CHECK-SAME:  (%[[C0]], %[[C0]]) to (%[[C8]], %[[C112]]) step (%[[C1]], %[[C1]])
  scf.forall (%arg1, %arg2) in (8, 112) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x112x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    memref.copy %arg0, %subview : memref<32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
  }
  return %alloc : memref<8x112x32x32xbf16>
}

// -----

// CHECK-LABEL: func.func @thread_forall
// CHECK-SAME:  %{{.+}}: memref<1x5xf32>,
// CHECK-SAME:  %{{.+}}: memref<?x?xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: index
func.func @thread_forall(%arg0: memref<1x5xf32>, %arg1: memref<?x?xf32>, %arg2: index) -> index {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.parallel (%{{.+}}) = (%[[C0]]) to (%[[ARG2]]) step (%[[C1]])
  scf.forall (%arg3) in (%arg2) {
    %subview = memref.subview %arg1[%arg3, 0] [1, 5] [1, 1] : memref<?x?xf32> to memref<1x5xf32, strided<[?, 1], offset: ?>>
    memref.copy %arg0, %subview : memref<1x5xf32> to memref<1x5xf32, strided<[?, 1], offset: ?>>
  }
  %dim = memref.dim %arg1, %c1 : memref<?x?xf32>
  return %dim : index
}
