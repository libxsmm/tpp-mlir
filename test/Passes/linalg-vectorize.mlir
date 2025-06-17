// RUN: tpp-opt %s -linalg-vectorize -split-input-file | FileCheck %s

// Check a few relevant patterns to validate vectorization driver.
// Core logic is driven by upstream utilities.

func.func @vectorize_matmul(%arg0: tensor<256x256xf32>,
    %arg1: tensor<256x256xf32>, %arg2: tensor<256x256xf32>
    ) -> tensor<256x256xf32> {
  %0 = linalg.matmul
    ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>)
    outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
}

// CHECK-LABEL: @vectorize_matmul
// CHECK: vector.contract

// -----

func.func @vectorize_eltwise(
    %arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>
    ) -> tensor<256x256xf32> {
  %e = tensor.empty() : tensor<256x256xf32>
  %0 = linalg.add
    ins(%arg0, %arg1 : tensor<256x256xf32>, tensor<256x256xf32>)
    outs(%e : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
}

// CHECK-LABEL: @vectorize_eltwise
// CHECK: arith.addf

// -----

func.func @negative_vectorize_pack(
    %arg0: tensor<512x1024xf32>, %arg1: tensor<16x32x32x32xf32>)
    -> tensor<16x32x32x32xf32> {
  %pack = linalg.pack %arg0
    inner_dims_pos = [0, 1]
    inner_tiles = [32, 32]
    into %arg1 : tensor<512x1024xf32> -> tensor<16x32x32x32xf32>
  return %pack : tensor<16x32x32x32xf32>
}

// CHECK-LABEL: @negative_vectorize_pack
// CHECK: linalg.pack

// -----

func.func @negative_vectorize_unpack(
    %arg0: tensor<16x16x32x32xf32>, %arg1: tensor<512x512xf32>
    ) -> tensor<512x512xf32> {
  %unpack = linalg.unpack %arg0
    inner_dims_pos = [0, 1]
    inner_tiles = [32, 32]
    into %arg1 : tensor<16x16x32x32xf32> -> tensor<512x512xf32>
  return %unpack : tensor<512x512xf32>
}

// CHECK-LABEL: @negative_vectorize_unpack
// CHECK: linalg.unpack

// -----

func.func @negative_vectorize_insert_slice(
    %arg0: tensor<16xf32>, %arg1: tensor<8x16xf32>
    ) -> tensor<8x16xf32> {
  %0 = tensor.insert_slice %arg0 into %arg1[0, 0][1, 16][1, 1] :
      tensor<16xf32> into tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @negative_vectorize_insert_slice
// CHECK: tensor.insert_slice
