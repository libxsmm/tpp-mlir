// RUN: tpp-opt %s  --vectorization-pass --split-input-file | FileCheck %s

module {
  func.func @entry(%arg0: memref<2x4x8x2xf32>, %arg1: memref<2x4x2x4xf32>, %arg2: memref<2x2x8x4xf32>) {
    scf.forall (%arg3, %arg4) in (2, 2) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 4, 8, 2] [1, 1, 1, 1] : memref<2x4x8x2xf32> to memref<4x8x2xf32, strided<[16, 2, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 4, 2, 4] [1, 1, 1, 1] : memref<2x4x2x4xf32> to memref<4x2x4xf32, strided<[8, 4, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<2x2x8x4xf32> to memref<8x4xf32, strided<[4, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<4x8x2xf32, strided<[16, 2, 1], offset: ?>>, memref<4x2x4xf32, strided<[8, 4, 1], offset: ?>>) outs(%subview_1 : memref<8x4xf32, strided<[4, 1], offset: ?>>)
    }
    return
  }
}

// CHECK: func.func @entry(%[[ARG0:.*]]: memref<2x4x8x2xf32>, %[[ARG1:.*]]: memref<2x4x2x4xf32>, %[[ARG2:.*]]: memref<2x2x8x4xf32>) {
// CHECK: %[[ub0:.*]] = ub.poison : f32
// CHECK: arith.constant 0 : index
// CHECK: scf.forall (%[[ARG3:.*]], %[[ARG4:.*]]) in (2, 2) {
// CHECK:       %[[subview:.*]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 4, 8, 2] [1, 1, 1, 1] : memref<2x4x8x2xf32> to memref<4x8x2xf32, strided<[16, 2, 1], offset: ?>>
// CHECK:       %[[subview_0:.*]] = memref.subview %[[ARG1]][%[[ARG4]], 0, 0, 0] [1, 4, 2, 4] [1, 1, 1, 1] : memref<2x4x2x4xf32> to memref<4x2x4xf32, strided<[8, 4, 1], offset: ?>>
// CHECK:       %[[subview_1:.*]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<2x2x8x4xf32> to memref<8x4xf32, strided<[4, 1], offset: ?>>
// CHECK:       %[[vec0:.*]] = vector.transfer_read %[[subview]][%c0, %c0, %c0], %[[ub0]] {in_bounds = [true, true, true]} : memref<4x8x2xf32, strided<[16, 2, 1], offset: ?>>, vector<4x8x2xf32>
// CHECK:       %[[vec1:.*]] = vector.transfer_read %[[subview_0]][%c0, %c0, %c0], %[[ub0]] {in_bounds = [true, true, true]} : memref<4x2x4xf32, strided<[8, 4, 1], offset: ?>>, vector<4x2x4xf32>
// CHECK:       %[[vec2:.*]] = vector.transfer_read %[[subview_1]][%c0, %c0], %[[ub0]] {in_bounds = [true, true]} : memref<8x4xf32, strided<[4, 1], offset: ?>>, vector<8x4xf32>
// CHECK:       %[[vec3:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[vec0]], %[[vec1]], %[[vec2]] : vector<4x8x2xf32>, vector<4x2x4xf32> into vector<8x4xf32>
// CHECK:       vector.transfer_write %[[vec3]], %[[subview_1]][%c0, %c0] {in_bounds = [true, true]} : vector<8x4xf32>, memref<8x4xf32, strided<[4, 1], offset: ?>>
// CHECK:     }

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_4x1x4x2xbf16 : memref<4x1x4x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<2x4x8x2xbf16>) -> memref<2x2x8x4xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_4x1x4x2xbf16 : memref<4x1x4x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x2x8x4xbf16>
    scf.forall (%arg1, %arg2) in (2, 2) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<2x2x8x4xbf16> to memref<8x4xbf16, strided<[4, 1], offset: ?>>
      linalg.fill ins(%cst : bf16) outs(%subview : memref<8x4xbf16, strided<[4, 1], offset: ?>>)
      %subview_0 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 4, 8, 2] [1, 1, 1, 1] : memref<2x4x8x2xbf16> to memref<4x8x2xbf16, strided<[16, 2, 1], offset: ?>>
      %expanded = memref.expand_shape %subview_0 [[0], [1], [2, 3]] output_shape [4, 8, 1, 2]
        : memref<4x8x2xbf16, strided<[16, 2, 1], offset: ?>> into memref<4x8x1x2xbf16, strided<[16, 2, 2, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2],
        iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]}
        ins(%expanded, %0 : memref<4x8x1x2xbf16, strided<[16, 2, 2, 1], offset: ?>>, memref<4x1x4x2xbf16>)
        outs(%subview : memref<8x4xbf16, strided<[4, 1], offset: ?>>) {
      ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_1 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
      }
    }
    return %alloc : memref<2x2x8x4xbf16>
  }
}

// CHECK:   func.func @entry(%[[ARG0:.*]]: memref<2x4x8x2xbf16>) -> memref<2x2x8x4xbf16> {
// CHECK:     %[[ub0:.*]] = ub.poison : bf16
// CHECK:     scf.forall (%[[ARG1:.*]], %[[ARG2:.*]]) in (2, 2) {
// CHECK:       %[[subview:.*]] = memref.subview %alloc[%[[ARG1]], %[[ARG2]], 0, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<2x2x8x4xbf16> to memref<8x4xbf16, strided<[4, 1], offset: ?>>
// CHECK:       vector.transfer_write %cst, %[[subview]][%c0, %c0] {in_bounds = [true, true]} : vector<8x4xbf16>, memref<8x4xbf16, strided<[4, 1], offset: ?>>
// CHECK:       %[[subview_1:.*]] = memref.subview %[[ARG0]][%[[ARG1]], 0, 0, 0] [1, 4, 8, 2] [1, 1, 1, 1] : memref<2x4x8x2xbf16> to memref<4x8x2xbf16, strided<[16, 2, 1], offset: ?>>
// CHECK:       %[[expand_shape:.*]] = memref.expand_shape %[[subview_1]] {{\[}}[0], [1], [2, 3]] output_shape [4, 8, 1, 2] : memref<4x8x2xbf16, strided<[16, 2, 1], offset: ?>> into memref<4x8x1x2xbf16, strided<[16, 2, 2, 1], offset: ?>>
// CHECK:       %[[vec1:.*]] = vector.transfer_read %[[expand_shape]][%c0, %c0, %c0, %c0], %[[ub0]] {in_bounds = [true, true, true, true]} : memref<4x8x1x2xbf16, strided<[16, 2, 2, 1], offset: ?>>, vector<4x8x1x2xbf16>
// CHECK:       %[[vec2:.*]] = vector.transfer_read %1[%c0, %c0, %c0, %c0], %[[ub0]] {in_bounds = [true, true, true, true]} : memref<4x1x4x2xbf16>, vector<4x1x4x2xbf16>
// CHECK:       %[[vec3:.*]] = vector.transfer_read %[[subview]][%c0, %c0], %[[ub0]] {in_bounds = [true, true]} : memref<8x4xbf16, strided<[4, 1], offset: ?>>, vector<8x4xbf16>
// CHECK:       %[[vec4:.*]] = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[vec1]], %[[vec2]], %[[vec3]] : vector<4x8x1x2xbf16>, vector<4x1x4x2xbf16> into vector<8x4xbf16>
// CHECK:       vector.transfer_write %[[vec4]], %[[subview]][%c0, %c0] {in_bounds = [true, true]} : vector<8x4xbf16>, memref<8x4xbf16, strided<[4, 1], offset: ?>>

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module {
  func.func @entry(%arg0: memref<4x8x16x32x64xbf16>) {
    linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} outs(%arg0 : memref<4x8x16x32x64xbf16>) {
    ^bb0(%out: bf16):
      %0 = math.absf %out : bf16
      linalg.yield %0 : bf16
    }
    return
  }
}

// CHECK: func.func @entry(%[[ARG0:.*]]: memref<4x8x16x32x64xbf16>) {
// CHECK:     %[[vec0:.*]] = vector.transfer_read %[[ARG0]][%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true, true, true, true, true]} : memref<4x8x16x32x64xbf16>, vector<4x8x16x32x64xbf16>
// CHECK:     %[[vec1:.*]] = math.absf %[[vec0]] : vector<4x8x16x32x64xbf16>
// CHECK:     vector.transfer_write %[[vec1]], %[[ARG0]][%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true]} : vector<4x8x16x32x64xbf16>, memref<4x8x16x32x64xbf16>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
module {
  func.func @entry(%arg0: tensor<2x4x8x1x2xbf16>) -> tensor<2x2x8x4xbf16> {
    %cst = arith.constant dense<1.000000e+00> : tensor<2x4x1x4x2xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<2x2x8x4xbf16>
    %1 = linalg.fill ins(%cst_0 : bf16) outs(%0 : tensor<2x2x8x4xbf16>) -> tensor<2x2x8x4xbf16>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]}
      ins(%arg0, %cst : tensor<2x4x8x1x2xbf16>, tensor<2x4x1x4x2xbf16>)
      outs(%1 : tensor<2x2x8x4xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %3 = arith.mulf %in, %in_1 : bf16
      %4 = arith.addf %out, %3 : bf16
      linalg.yield %4 : bf16
    } -> tensor<2x2x8x4xbf16>
    return %2 : tensor<2x2x8x4xbf16>
  }
}


// CHECK:   func.func @entry(%[[ARG0:.*]]: tensor<2x4x8x1x2xbf16>) -> tensor<2x2x8x4xbf16> {
// CHECK:       vector.transfer_write
// CHECK:       vector.transfer_read
// CHECK:       vector.transfer_read
// CHECK:       vector.contract
// CHECK:       vector.transfer_write

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
