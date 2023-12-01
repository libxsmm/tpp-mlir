// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

func.func @simple_gemm(%arg0: memref<32x64xf32, strided<[64, 1], offset: ?>>,
                       %arg1: memref<64x32xf32, strided<[32, 1], offset: ?>>,
                       %arg2: memref<32x32xf32, strided<[32, 1], offset: ?>>) {
  linalg.matmul ins(%arg0, %arg1 : memref<32x64xf32, strided<[64, 1], offset: ?>>,
                                   memref<64x32xf32, strided<[32, 1], offset: ?>>)
                outs(%arg2 : memref<32x32xf32, strided<[32, 1], offset: ?>>)
  return
}

// CHECK-LABEL: simple_gemm
// CHECK-SAME: %[[ARG0:.+]]: memref<32x64xf32, strided<[64, 1], offset: ?>>,
// CHECK-SAME: %[[ARG1:.+]]: memref<64x32xf32, strided<[32, 1], offset: ?>>,
// CHECK-SAME: %[[ARG2:.+]]: memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [32, 32, 64, 64, 32, 32] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d0)>

func.func @mha_query_times_key(%arg0: memref<64x32x8x64xf32>, %arg1: memref<64x32x8x64xf32>,
                               %arg2: memref<64x8x32x32xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  scf.forall (%arg3, %arg4) in (64, 8) {
    %subview = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<64x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    linalg.fill ins(%cst : f32) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>)
    %subview_0 = memref.subview %arg0[%arg3, 0, %arg4, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
    %subview_1 = memref.subview %arg1[%arg3, 0, %arg4, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%subview_0, %subview_1 : memref<32x64xf32, strided<[512, 1], offset: ?>>, memref<32x64xf32, strided<[512, 1], offset: ?>>) outs(%subview : memref<32x32xf32, strided<[32, 1], offset: ?>>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %0 = arith.mulf %in, %in_2 : f32
        %1 = arith.addf %out, %0 : f32
        linalg.yield %1 : f32
    }
  }
  return
}

// CHECK-LABEL: mha_query_times_key
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32x8x64xf32>, %[[ARG1:.+]]: memref<64x32x8x64xf32>, %[[ARG2:.+]]: memref<64x8x32x32xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (64, 8)
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG2]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : memref<64x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: %[[FILL:.+]] = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[FILL]], %[[CST]], %[[SUB]])
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1]
// CHECK-SAME:  : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG1]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1]
// CHECK-SAME:  : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<64x32xf32>
// CHECK: %[[TRAN:.+]] = xsmm.unary.dispatch transpose [32, 64, 512, 32] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[TRAN]], %[[SUB_0]], %[[ALLOC]])
// CHECK: %[[GEMM:.+]] = xsmm.gemm.dispatch [32, 32, 64, 512, 32, 32] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[GEMM]], %[[SUB_1]], %[[ALLOC]], %[[SUB]])

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

func.func @mha_out_softmax_times_value(%arg0: memref<64x8x32x32xf32>, %arg1: memref<64x32x8x64xf32>,
                                   %arg2: memref<64x32x8x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  scf.forall (%arg3, %arg4) in (64, 8) {
    %subview = memref.subview %arg2[%arg3, 0, %arg4, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x64xf32, strided<[512, 1], offset: ?>>)
      %subview_0 = memref.subview %arg0[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<64x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      %subview_1 = memref.subview %arg1[%arg3, 0, %arg4, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
      linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%subview_0, %subview_1 : memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x64xf32, strided<[512, 1], offset: ?>>) outs(%subview : memref<32x64xf32, strided<[512, 1], offset: ?>>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %0 = arith.mulf %in, %in_2 : f32
        %1 = arith.addf %out, %0 : f32
        linalg.yield %1 : f32
      }
    }
  return
}

// CHECK-LABEL: mha_out_softmax_times_value
// CHECK-SAME: %[[ARG0:.+]]: memref<64x8x32x32xf32>, %[[ARG1:.+]]: memref<64x32x8x64xf32>, %[[ARG2:.+]]: memref<64x32x8x64xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (64, 8)
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1]
// CHECK-SAME:  : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[FILL:.+]] = xsmm.unary.dispatch zero [32, 64, 1, 512] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[FILL]], %[[CST]], %[[SUB]])
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG0]][%[[ARG3]], %[[ARG4]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK-SAME:  : memref<64x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG1]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1]
// CHECK-SAME:  : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[GEMM:.+]] = xsmm.gemm.dispatch [32, 64, 32, 32, 512, 512] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[GEMM]], %[[SUB_0]], %[[SUB_1]], %[[SUB]])

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

func.func @mha_projection(%arg0: memref<512x8x64xf32>, %arg1: memref<64x32x512xf32>, %arg2: memref<64x32x8x64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  scf.forall (%arg3, %arg4) in (64, 8) {
    %subview = memref.subview %arg2[%arg3, 0, %arg4, 0] [1, 32, 1, 64] [1, 1, 1, 1] : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
    linalg.fill ins(%cst : f32) outs(%subview : memref<32x64xf32, strided<[512, 1], offset: ?>>)
    %subview_0 = memref.subview %arg1[%arg3, 0, 0] [1, 32, 512] [1, 1, 1] : memref<64x32x512xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
    %subview_1 = memref.subview %arg0[0, %arg4, 0] [512, 1, 64] [1, 1, 1] : memref<512x8x64xf32> to memref<512x64xf32, strided<[512, 1], offset: ?>>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "reduction", "parallel"]}
      ins(%subview_0, %subview_1 : memref<32x512xf32, strided<[512, 1], offset: ?>>, memref<512x64xf32, strided<[512, 1], offset: ?>>) outs(%subview : memref<32x64xf32, strided<[512, 1], offset: ?>>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %0 = arith.mulf %in, %in_2 : f32
        %1 = arith.addf %out, %0 : f32
        linalg.yield %1 : f32
      }
    }
    return
}

// CHECK-LABEL: mha_projection
// CHECK-SAME: %[[ARG0:.+]]: memref<512x8x64xf32>, %[[ARG1:.+]]: memref<64x32x512xf32>, %[[ARG2:.+]]: memref<64x32x8x64xf32>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (64, 8)
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 32, 1, 64] [1, 1, 1, 1]
// CHECK-SAME:  : memref<64x32x8x64xf32> to memref<32x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[FILL:.+]] = xsmm.unary.dispatch zero [32, 64, 1, 512] flags = (bcast_scalar) data_type = f32
// CHECK: xsmm.unary zero(data_type = f32, %[[FILL]], %[[CST]], %[[SUB]])
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][%[[ARG3]], 0, 0] [1, 32, 512] [1, 1, 1]
// CHECK-SAME:  : memref<64x32x512xf32> to memref<32x512xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG0]][0, %[[ARG4]], 0] [512, 1, 64] [1, 1, 1]
// CHECK-SAME:  : memref<512x8x64xf32> to memref<512x64xf32, strided<[512, 1], offset: ?>>
// CHECK: %[[GEMM:.+]] = xsmm.gemm.dispatch [32, 64, 512, 512, 512, 512] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[GEMM]], %[[SUB_0]], %[[SUB_1]], %[[SUB]])
