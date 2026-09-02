// RUN: tpp-opt -rewrite-batch-matmul-to-matmul -split-input-file %s | FileCheck %s

func.func @batch_matmul_rewrite(%arg0: tensor<512x32x64xf32>, %arg1: tensor<512x64x32xf32>) -> tensor<512x32x32xf32> {
  %0 = tensor.empty() : tensor<512x32x32xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<512x32x64xf32>, tensor<512x64x32xf32>)
                           outs(%0 : tensor<512x32x32xf32>) -> tensor<512x32x32xf32>
  return %1 : tensor<512x32x32xf32>
}

// CHECK-LABEL: batch_matmul_rewrite
// CHECK-SAME: %[[ARG0:.+]]: tensor<512x32x64xf32>, %[[ARG1:.+]]: tensor<512x64x32xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<512x32x32xf32>
// CHECK: %{{.+}} = scf.forall (%[[ARG2:.+]]) in (512)
// CHECK-SAME:  shared_outs(%[[ARG3:.+]] = %[[EMPTY]]) -> (tensor<512x32x32xf32>) {
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG2]], 0, 0] [1, 32, 64] [1, 1, 1]
// CHECK-SAME:  : tensor<512x32x64xf32> to tensor<32x64xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG2]], 0, 0] [1, 64, 32] [1, 1, 1]
// CHECK-SAME:  : tensor<512x64x32xf32> to tensor<64x32xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG3]][%[[ARG2]], 0, 0] [1, 32, 32] [1, 1, 1]
// CHECK-SAME:  : tensor<512x32x32xf32> to tensor<32x32xf32>
// CHECK: %{{.+}} = linalg.matmul ins(%[[SLICE]], %[[SLICE1]] : tensor<32x64xf32>, tensor<64x32xf32>)
// CHECK-SAME:  outs(%[[SLICE2]] : tensor<32x32xf32>) -> tensor<32x32xf32>

// -----

func.func @batch_matmul_rewrite(%arg0: tensor<512x?x?xf32>,
  %arg1: tensor<512x?x?xf32>, %dim0: index, %dim1: index) -> tensor<512x?x?xf32> {
  %0 = tensor.empty(%dim0, %dim1) : tensor<512x?x?xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<512x?x?xf32>, tensor<512x?x?xf32>)
                           outs(%0 : tensor<512x?x?xf32>) -> tensor<512x?x?xf32>
  return %1 : tensor<512x?x?xf32>
}

// CHECK-LABEL: batch_matmul_rewrite
// CHECK-SAME:  %[[ARG0:.+]]: tensor<512x?x?xf32>, %[[ARG1:.+]]: tensor<512x?x?xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: index, %[[ARG3:.+]]: index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[EMPTY:.+]] = tensor.empty(%[[ARG2]], %[[ARG3]]) : tensor<512x?x?xf32>
// CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<512x?x?xf32>
// CHECK: %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<512x?x?xf32>
// CHECK: %[[DIM1:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<512x?x?xf32>
// CHECK: %{{.+}} = scf.forall (%[[ARG4:.+]]) in (512)
// CHECK-SAME:  shared_outs(%[[ARG5:.+]] = %[[EMPTY]]) -> (tensor<512x?x?xf32>) {
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG4]], 0, 0] [1, %[[DIM]], %[[DIM0]]] [1, 1, 1]
// CHECK-SAME:  : tensor<512x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG4]], 0, 0] [1, %[[DIM0]], %[[DIM1]]] [1, 1, 1]
// CHECK-SAME:  : tensor<512x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG5]][%[[ARG4]], 0, 0] [1, %[[DIM]], %[[DIM1]]] [1, 1, 1]
// CHECK-SAME:  : tensor<512x?x?xf32> to tensor<?x?xf32>
// CHECK: %{{.+}} = linalg.matmul ins(%[[SLICE]], %[[SLICE1]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:  outs(%[[SLICE2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

func.func @batch_matmul_rewrite(%arg0: tensor<?x?x?xf32>,
  %arg1: tensor<?x?x?xf32>, %dim0: index, %dim1: index, %batch: index) -> tensor<?x?x?xf32> {
  %0 = tensor.empty(%batch, %dim0, %dim1) : tensor<?x?x?xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
                           outs(%0 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>
}

// CHECK-LABEL: batch_matmul_rewrite
// CHECK-SAME:  %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: index, %[[ARG3:.+]]: index, %[[ARG4:.+]]: index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[EMPTY:.+]] = tensor.empty(%[[ARG4]], %[[ARG2]], %[[ARG3]]) : tensor<?x?x?xf32>
// CHECK: %[[DIM:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?xf32>
// CHECK: %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?xf32>
// CHECK: %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?xf32>
// CHECK: %[[DIM2:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<?x?x?xf32>
// CHECK: %{{.+}} = scf.forall (%[[ARG5:.+]]) in (%[[DIM]])
// CHECK-SAME:  shared_outs(%[[ARG6:.+]] = %[[EMPTY]]) -> (tensor<?x?x?xf32>) {
// CHECK: %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG5]], 0, 0] [1, %[[DIM0]], %[[DIM1]]] [1, 1, 1]
// CHECK-SAME:  : tensor<?x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[SLICE1:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG5]], 0, 0] [1, %[[DIM1]], %[[DIM2]]] [1, 1, 1]
// CHECK-SAME:  : tensor<?x?x?xf32> to tensor<?x?xf32>
// CHECK: %[[SLICE2:.+]] = tensor.extract_slice %[[ARG6]][%[[ARG5]], 0, 0] [1, %[[DIM0]], %[[DIM2]]] [1, 1, 1]
// CHECK-SAME:  : tensor<?x?x?xf32> to tensor<?x?xf32>
// CHECK: %{{.+}} = linalg.matmul ins(%[[SLICE]], %[[SLICE1]] : tensor<?x?xf32>, tensor<?x?xf32>)
// CHECK-SAME:  outs(%[[SLICE2]] : tensor<?x?xf32>) -> tensor<?x?xf32>
