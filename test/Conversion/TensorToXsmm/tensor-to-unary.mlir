// RUN: tpp-opt --tpp-mapping -cleanup -tpp-conversion --bufferize --convert-memref-to-tpp --convert-tpp-to-xsmm %s --split-input-file | FileCheck %s

func.func @pack1(%in: tensor<4x4xf32>, %out: tensor<2x2x2x2xf32>) ->  tensor<2x2x2x2xf32> {
  %1 = tensor.pack %in inner_dims_pos = [0, 1] inner_tiles = [2,2] into %out : tensor<4x4xf32> -> tensor<2x2x2x2xf32>
  return %1 : tensor<2x2x2x2xf32>
}

// CHECK: func.func @pack1(%[[ARG1:.+]]: memref<4x4xf32>, %[[ARG1:.+]]: memref<2x2x2x2xf32>)  ->  memref<2x2x2x2xf32> {
// CHECK: scf.for
// CHECK:  scf.for
// CHECK:    %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [2, 2, 4, 2] flags = (none) data_type = f32 
// CHECK:    xsmm.unary identity(data_type = f32, %[[DISPATCH]], %{{[^:]+}}, {{[^:]+}}) : (i64, memref<2x2xf32, strided<[4, 1], offset: ?>>, memref<2x2xf32, strided<[2, 1], offset: ?>>) -> ()

// -----

func.func @pack2(%0: tensor<1x2x2x4xf32>, %1:  tensor<1x2x2x2x2xf32>)-> tensor<1x2x2x2x2xf32>{
 %2 = tensor.pack %0  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %1 : tensor<1x2x2x4xf32> -> tensor<1x2x2x2x2xf32>
  return %2: tensor<1x2x2x2x2xf32>
}

// CHECK: func.func @pack2(%[[ARG0:.+]]: memref<1x2x2x4xf32>, %[[ARG1:.+]]: memref<1x2x2x2x2xf32>) -> memref<1x2x2x2x2xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:      memref.subview
// CHECK:      memref.subview
// CHECK:      memref.copy 
// -----

func.func @pack3(%in: tensor<8x2x2x2xf32>, %out: tensor<2x2x1x4x2x2xf32>)-> tensor<2x2x1x4x2x2xf32>{
  %2 = tensor.pack %in outer_dims_perm = [3, 2, 1, 0] inner_dims_pos=[1, 0] inner_tiles = [2, 2] into %out:   tensor<8x2x2x2xf32>->tensor<2x2x1x4x2x2xf32>
  return %2: tensor<2x2x1x4x2x2xf32>
}

// CHECK: func.func @pack3(%[[ARG0:.+]] memref<8x2x2x2xf32>, %[[ARG1:.+]] memref<2x2x1x4x2x2xf32>) -> memref<2x2x1x4x2x2xf32> {
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       memref.subview
// CHECK:       linalg.transpose
// CHECK:       memref.subview
// CHECK:       %[[DISPATCH:.+]] = xsmm.unary.dispatch identity [2, 2, 2, 2] flags = (none) data_type = f32 
// CHECK:       xsmm.unary identity(data_type = f32, %{{[^:]+}}, {{[^:]+}}, {{[^:]+}}) : (i64, memref<2x2xf32>, memref<2x2xf32, strided<[2, 1], offset: ?>>) -> ()
