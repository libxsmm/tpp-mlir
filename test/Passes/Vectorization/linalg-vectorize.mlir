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

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @vectorize_contract_mixed_precision_float(
    %arg0: tensor<256x128x2xbf16>, %arg1: tensor<128x256x2xbf16>,
    %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
  %0 = linalg.contract
    indexing_maps = [#map, #map1, #map2]
    ins(%arg0, %arg1 : tensor<256x128x2xbf16>, tensor<128x256x2xbf16>)
    outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
  return %0 : tensor<256x256xf32>
}

// Ensure that mixed precision contraction vectorizes cleanly
// without extra operations and/or dimensions.

// CHECK-LABEL: @vectorize_contract_mixed_precision_float
// CHECK: vector.transfer_read{{.*}}: tensor<256x128x2xbf16>, vector<256x128x2xbf16>
// CHECK-NOT: vector.broadcast
// CHECK-NOT: vector.transpose
// CHECK: vector.transfer_read{{.*}}: tensor<128x256x2xbf16>, vector<128x256x2xbf16>
// CHECK: vector.transfer_read{{.*}}: tensor<256x256xf32>, vector<256x256xf32>
// CHECK-NOT: arith.extf
// CHECK: vector.contract
// CHECK: vector.transfer_write

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
module {
  func.func @vectorize_contract_mixed_precision_int(
      %arg0: tensor<1x2x32x8x4xi8>, %arg1: tensor<2x2x8x32x4xi8>,
      %arg2: tensor<1x2x32x32xi32>) -> tensor<1x2x32x32xi32> {
    %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<1x2x32x8x4xi8>, tensor<2x2x8x32x4xi8>)
      outs(%arg2 : tensor<1x2x32x32xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %0 = arith.extsi %in : i8 to i32
      %1 = arith.extsi %in_0 : i8 to i32
      %2 = arith.muli %0, %1 : i32
      %3 = arith.addi %out, %2 : i32
      linalg.yield %3 : i32
    } -> tensor<1x2x32x32xi32>
    return %0 : tensor<1x2x32x32xi32>
  }
}

// Validate mixed precision case with more complex affine maps
// and integer types.

// CHECK-LABEL: @vectorize_contract_mixed_precision_int
// CHECK: vector.transfer_read{{.*}}: tensor<1x2x32x8x4xi8>, vector<1x2x32x8x4xi8>
// CHECK-NOT: vector.broadcast
// CHECK-NOT: vector.transpose
// CHECK: vector.transfer_read{{.*}}: tensor<2x2x8x32x4xi8>, vector<2x2x8x32x4xi8>
// CHECK: vector.transfer_read{{.*}}: tensor<1x2x32x32xi32>, vector<1x2x32x32xi32>
// CHECK-NOT: arith.extsi
// CHECK: vector.contract
// CHECK: vector.transfer_write

// -----

func.func @vectorize_memref(%arg0: memref<256x256xf32>,
    %arg1: memref<256x256xf32>, %arg2: memref<256x256xf32>,
    %arg3: memref<256x256xf32>) {
  linalg.matmul
    ins(%arg0, %arg1 : memref<256x256xf32>, memref<256x256xf32>)
    outs(%arg2 : memref<256x256xf32>)
  linalg.add
    ins(%arg2, %arg3 : memref<256x256xf32>, memref<256x256xf32>)
    outs(%arg2 : memref<256x256xf32>)
  return
}

// CHECK-LABEL: @vectorize_memref
// CHECK: vector.contract
// CHECK: arith.addf

// -----

func.func @negative_vectorize_dynamic_shapes(
    %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>
    ) -> tensor<?x?xf32> {
  %0 = linalg.add
    ins(%arg0, %arg0 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @negative_vectorize_dynamic_shapes
// CHECK: linalg.add

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
