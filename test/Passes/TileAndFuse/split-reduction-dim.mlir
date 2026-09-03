// RUN: tpp-opt %s -split-reduction-dim=tile=8 -canonicalize -split-input-file | FileCheck %s

func.func @tile_matmul_memref(%A: memref<32x64xf32>, %B: memref<64x16xf32>,
    %C: memref<32x16xf32>) {
  linalg.matmul ins(%A, %B: memref<32x64xf32>, memref<64x16xf32>) outs(%C: memref<32x16xf32>)
  return
}

// CHECK-LABEL: @tile_matmul_memref(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: memref<32x64xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: memref<64x16xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: memref<32x16xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]] {
// CHECK:   %[[SUBVIEW_A:.+]] = memref.subview %[[A]][0, %[[IV]]] [32, 8] [1, 1]
// CHECK:   %[[SUBVIEW_B:.+]] = memref.subview %[[B]][%[[IV]], 0] [8, 16] [1, 1]
// CHECK:   linalg.matmul
// CHECK-SAME: ins(%[[SUBVIEW_A]], %[[SUBVIEW_B]]
// CHECK-SAME: outs(%[[C]]

// -----

func.func @tile_matmul_tensor(%A: tensor<32x64xf32>, %B: tensor<64x16xf32>,
    %C: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<32x64xf32>, tensor<64x16xf32>)
    outs(%C: tensor<32x16xf32>) -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// CHECK-LABEL: @tile_matmul_tensor(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: tensor<32x64xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: tensor<64x16xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: tensor<32x16xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]]
// CHECK-SAME: iter_args(%[[ACC:.+]] = %[[C]])
// CHECK:   %[[SLICE_A:.+]] = tensor.extract_slice %[[A]][0, %[[IV]]] [32, 8] [1, 1]
// CHECK:   %[[SLICE_B:.+]] = tensor.extract_slice %[[B]][%[[IV]], 0] [8, 16] [1, 1]
// CHECK:   %[[RES:.+]] = linalg.matmul
// CHECK-SAME: ins(%[[SLICE_A]], %[[SLICE_B]]
// CHECK-SAME: outs(%[[ACC]]
// CHECK:    scf.yield %[[RES]]

// -----

func.func @tile_reduction_not_divisible(%A: memref<32x60xf32>, %B: memref<60x16xf32>,
    %C: memref<32x16xf32>) {
  linalg.matmul ins(%A, %B: memref<32x60xf32>, memref<60x16xf32>) outs(%C: memref<32x16xf32>)
  return
}

// CHECK-DAG: #[[TILE_MAP:.+]] = affine_map<(d0) -> (-d0 + 60, 8)>
// CHECK-LABEL: @tile_reduction_not_divisible(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: memref<32x60xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: memref<60x16xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: memref<32x16xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.+]] = arith.constant 60 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]] {
// CHECK:   %[[TILE_SIZE:.+]] = affine.min #[[TILE_MAP]](%[[IV]])
// CHECK:   %[[SUBVIEW_A:.+]] = memref.subview %[[A]][0, %[[IV]]] [32, %[[TILE_SIZE]]] [1, 1]
// CHECK:   %[[SUBVIEW_B:.+]] = memref.subview %[[B]][%[[IV]], 0] [%[[TILE_SIZE]], 16] [1, 1]
// CHECK:   linalg.matmul
// CHECK-SAME: ins(%[[SUBVIEW_A]], %[[SUBVIEW_B]]
// CHECK-SAME: outs(%[[C]]

// -----

func.func @tile_dynamic_memref(%A: memref<?x?xf32>, %B: memref<?x?xf32>,
    %C: memref<?x?xf32>) {
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>) outs(%C: memref<?x?xf32>)
  return
}

// CHECK-DAG: #[[TILE_MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
// CHECK-LABEL: @tile_dynamic_memref(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: memref<?x?xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: memref<?x?xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: memref<?x?xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[UB:.+]] = memref.dim %[[A]], %[[C1]]
// CHECK-DAG: %[[DIM_M:.+]] = memref.dim %[[A]], %[[C0]]
// CHECK-DAG: %[[DIM_N:.+]] = memref.dim %[[B]], %[[C1]]
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]] {
// CHECK:   %[[TILE_SIZE:.+]] = affine.min #[[TILE_MAP]](%[[IV]])[%[[UB]]]
// CHECK:   %[[SUBVIEW_A:.+]] = memref.subview %[[A]][0, %[[IV]]] [%[[DIM_M]], %[[TILE_SIZE]]] [1, 1]
// CHECK:   %[[SUBVIEW_B:.+]] = memref.subview %[[B]][%[[IV]], 0] [%[[TILE_SIZE]], %[[DIM_N]]] [1, 1]
// CHECK:   %[[SUBVIEW_C:.+]] = memref.subview %[[C]][0, 0] [%[[DIM_M]], %[[DIM_N]]] [1, 1]
// CHECK:   linalg.matmul
// CHECK-SAME: ins(%[[SUBVIEW_A]], %[[SUBVIEW_B]]
// CHECK-SAME: outs(%[[SUBVIEW_C]]

// -----

func.func @tile_dynamic_tensor(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-DAG: #[[TILE_MAP:.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
// CHECK-LABEL: @tile_dynamic_tensor(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: tensor<?x?xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[UB:.+]] = tensor.dim %[[A]], %[[C1]]
// CHECK-DAG: %[[DIM_M:.+]] = tensor.dim %[[A]], %[[C0]]
// CHECK-DAG: %[[DIM_N:.+]] = tensor.dim %[[B]], %[[C1]]
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]]
// CHECK-SAME: iter_args(%[[ACC:.+]] = %[[C]])
// CHECK:   %[[TILE_SIZE:.+]] = affine.min #[[TILE_MAP]](%[[IV]])[%[[UB]]]
// CHECK:   %[[SLICE_A:.+]] = tensor.extract_slice %[[A]][0, %[[IV]]] [%[[DIM_M]], %[[TILE_SIZE]]] [1, 1]
// CHECK:   %[[SLICE_B:.+]] = tensor.extract_slice %[[B]][%[[IV]], 0] [%[[TILE_SIZE]], %[[DIM_N]]] [1, 1]
// CHECK:   %[[SLICE_C:.+]] = tensor.extract_slice %[[ACC]][0, 0] [%[[DIM_M]], %[[DIM_N]]] [1, 1]
// CHECK:   %[[RES:.+]] = linalg.matmul
// CHECK-SAME: ins(%[[SLICE_A]], %[[SLICE_B]]
// CHECK-SAME: outs(%[[SLICE_C]]
// CHECK:   %[[UPDATE_C:.+]] = tensor.insert_slice %[[RES]]
// CHECK-SAME: into %[[ACC]][0, 0] [%[[DIM_M]], %[[DIM_N]]] [1, 1]
// CHECK:   scf.yield %[[UPDATE_C]]

// -----

func.func @tile_batch_matmul(%A: memref<2x32x64xf32>, %B: memref<2x64x16xf32>,
    %C: memref<2x32x16xf32>) {
  linalg.batch_matmul ins(%A, %B: memref<2x32x64xf32>, memref<2x64x16xf32>)
    outs(%C: memref<2x32x16xf32>)
  return
}

// CHECK-LABEL: @tile_batch_matmul(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: memref<2x32x64xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: memref<2x64x16xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: memref<2x32x16xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]] {
// CHECK:   %[[SUBVIEW_A:.+]] = memref.subview %[[A]][0, 0, %[[IV]]] [2, 32, 8] [1, 1, 1]
// CHECK:   %[[SUBVIEW_B:.+]] = memref.subview %[[B]][0, %[[IV]], 0] [2, 8, 16] [1, 1, 1]
// CHECK:   linalg.batch_matmul
// CHECK-SAME: ins(%[[SUBVIEW_A]], %[[SUBVIEW_B]]
// CHECK-SAME: outs(%[[C]]

// -----

func.func @tile_batch_reduce_matmul(%A: memref<2x32x64xf32>, %B: memref<2x64x16xf32>,
    %C: memref<32x16xf32>) {
  linalg.batch_reduce_matmul ins(%A, %B: memref<2x32x64xf32>, memref<2x64x16xf32>)
    outs(%C: memref<32x16xf32>)
  return
}

// CHECK-LABEL: @tile_batch_reduce_matmul(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: memref<2x32x64xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: memref<2x64x16xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: memref<32x16xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]] {
// CHECK:   %[[SUBVIEW_A:.+]] = memref.subview %[[A]][0, 0, %[[IV]]] [2, 32, 8] [1, 1, 1]
// CHECK:   %[[SUBVIEW_B:.+]] = memref.subview %[[B]][0, %[[IV]], 0] [2, 8, 16] [1, 1, 1]
// CHECK:   linalg.batch_reduce_matmul
// CHECK-SAME: ins(%[[SUBVIEW_A]], %[[SUBVIEW_B]]
// CHECK-SAME: outs(%[[C]]

// -----

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @tile_generic_1D(%A: memref<32xf32>, %B: memref<32xf32>, %C: memref<f32>) {
  linalg.generic {indexing_maps = [#map, #map, #map1],
  iterator_types = ["reduction"]}
  ins(%A, %B : memref<32xf32>, memref<32xf32>) outs(%C : memref<f32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %0 = arith.subf %in, %in_1 : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> ()>
// CHECK-LABEL: @tile_generic_1D(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: memref<32xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: memref<32xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: memref<f32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]] {
// CHECK:   %[[SUBVIEW_A:.+]] = memref.subview %[[A]][%[[IV]]] [8] [1]
// CHECK:   %[[SUBVIEW_B:.+]] = memref.subview %[[B]][%[[IV]]] [8] [1]
// CHECK:   linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["reduction"]
// CHECK-SAME: ins(%[[SUBVIEW_A]], %[[SUBVIEW_B]]
// CHECK-SAME: outs(%[[C]]
// CHECK:      arith.subf
// CHECK:      arith.addf
// CHECK:      linalg.yield

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> ()>
func.func @tile_generic_multireduction(%A: memref<32x16xf32>, %B: memref<16x32xf32>,
    %C: memref<f32>) {
  linalg.generic {indexing_maps = [#map, #map1, #map2],
  iterator_types = ["reduction", "reduction"]}
  ins(%A, %B : memref<32x16xf32>, memref<16x32xf32>) outs(%C : memref<f32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %0 = arith.subf %in, %in_1 : f32
    %1 = arith.addf %out, %0 : f32
    linalg.yield %1 : f32
  }
  return
}

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: @tile_generic_multireduction(
// CHECK-SAME:  %[[A:[0-9a-z]+]]: memref<32x16xf32>
// CHECK-SAME:  %[[B:[0-9a-z]+]]: memref<16x32xf32>
// CHECK-SAME:  %[[C:[0-9a-z]+]]: memref<f32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[UB:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[K_TILE:.+]] = arith.constant 8 : index
// CHECK: scf.for %[[IV:.+]] = %[[C0]] to %[[UB]] step %[[K_TILE]] {
// CHECK:   %[[SUBVIEW_A:.+]] = memref.subview %[[A]][0, %[[IV]]] [32, 8] [1, 1]
// CHECK:   %[[SUBVIEW_B:.+]] = memref.subview %[[B]][%[[IV]], 0] [8, 32] [1, 1]
// CHECK:   linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "reduction"]
// CHECK-SAME: ins(%[[SUBVIEW_A]], %[[SUBVIEW_B]]
// CHECK-SAME: outs(%[[C]]
// CHECK:      arith.subf
// CHECK:      arith.addf
// CHECK:      linalg.yield
