// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

#map = affine_map<(i, k, kk, j) -> (i, k, kk)>
#map1 = affine_map<(i, k, kk, j) -> (k, kk, j)>
#map2 = affine_map<(i, k, kk, j) -> (i, j)>

func.func @brgemm(%arg0: memref<2x2x2x4xf32>, %arg1: memref<2x4x8x2xf32>,
                  %arg2: memref<2x2x8x2xf32>) {
  scf.forall (%arg3, %arg4) in (2, 8) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
      : memref<2x2x2x4xf32> to memref<2x2x4xf32, strided<[8, 4, 1], offset: ?>>
    %subview_2 = memref.subview %arg1[0, 0, %arg4, 0] [2, 4, 1, 2] [1, 1, 1, 1]
      : memref<2x4x8x2xf32> to memref<2x4x2xf32, strided<[64, 16, 1], offset: ?>>
    %subview_3 = memref.subview %arg2[%arg3, 0, %arg4, 0] [1, 2, 1, 2] [1, 1, 1, 1]
      : memref<2x2x8x2xf32> to memref<2x2xf32, strided<[16, 1], offset: ?>>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "reduction", "reduction", "parallel"]}
      ins(%subview, %subview_2 : memref<2x2x4xf32, strided<[8, 4, 1], offset: ?>>, memref<2x4x2xf32, strided<[64, 16, 1], offset: ?>>)
      outs(%subview_3 : memref<2x2xf32, strided<[16, 1], offset: ?>>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
  }
  return
}

// CHECK-LABEL: brgemm
// CHECK-SAME: %[[ARG0:.+]]: memref<2x2x2x4xf32>, %[[ARG1:.+]]: memref<2x4x8x2xf32>, %[[ARG2:.+]]: memref<2x2x8x2xf32>
// CHECK: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: scf.forall (%[[ARG3:.+]], %[[ARG4:.+]]) in (2, 8) {
// CHECK: %[[SUB:.+]] = memref.subview %[[ARG0]][%[[ARG3]], 0, 0, 0] [1, 2, 2, 4] [1, 1, 1, 1]
// CHECK-SAME:  : memref<2x2x2x4xf32> to memref<2x2x4xf32, strided<[8, 4, 1], offset: ?>>
// CHECK: %[[SUB_0:.+]] = memref.subview %[[ARG1]][0, 0, %[[ARG4]], 0] [2, 4, 1, 2] [1, 1, 1, 1]
// CHECK-SAME:  : memref<2x4x8x2xf32> to memref<2x4x2xf32, strided<[64, 16, 1], offset: ?>>
// CHECK: %[[SUB_1:.+]] = memref.subview %[[ARG2]][%[[ARG3]], 0, %[[ARG4]], 0] [1, 2, 1, 2] [1, 1, 1, 1]
// CHECK-SAME:  : memref<2x2x8x2xf32> to memref<2x2xf32, strided<[16, 1], offset: ?>>
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [2, 2, 4, 8, 16, 16, 4, 64] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[SUB]], %[[SUB_0]], %[[SUB_1]], %[[C2]])

// m = 2
// n = 2
// k = 4
// lda = 8
// ldb = 16
// ldc = 16
// stride_a = 4
// stride_b = 64

// -----

#map = affine_map<(i, j, kk, k) -> (kk, i, k)>
#map1 = affine_map<(i, j, kk, k) -> (kk, k, j)>
#map2 = affine_map<(i, j, kk, k) -> (i, j)>

func.func @brgemm_1(%arg0: memref<9x4x5xf32>, %arg1: memref<9x5x8xf32>, %arg2: memref<4x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : memref<9x4x5xf32>, memref<9x5x8xf32>)
    outs(%arg2: memref<4x8xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: brgemm_1
// CHECK-SAME: %[[ARG0:.+]]: memref<9x4x5xf32>, %[[ARG1:.+]]: memref<9x5x8xf32>, %[[ARG2:.+]]: memref<4x8xf32>
// CHECK: %[[C9:.+]] = arith.constant 9 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [4, 8, 5, 5, 8, 8, 20, 40] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %0, %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C9]])

// -----

#map = affine_map<(kk, k, i, j) -> (kk, i, k)>
#map1 = affine_map<(kk, k, i, j) -> (kk, k, j)>
#map2 = affine_map<(kk, k, i, j) -> (i, j)>

func.func @brgemm_2(%arg0: memref<9x4x5xf32>, %arg1: memref<9x5x8xf32>, %arg2: memref<4x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<9x4x5xf32>, memref<9x5x8xf32>)
    outs(%arg2: memref<4x8xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: brgemm_2
// CHECK-SAME: %[[ARG0:.+]]: memref<9x4x5xf32>, %[[ARG1:.+]]: memref<9x5x8xf32>, %[[ARG2:.+]]: memref<4x8xf32>
// CHECK: %[[C9:.+]] = arith.constant 9 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [4, 8, 5, 5, 8, 8, 20, 40] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %0, %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C9]])

// -----

#map = affine_map<(kk, k, i, j) -> (kk, i, k)>
#map1 = affine_map<(kk, k, i, j) -> (kk, k, j)>
#map2 = affine_map<(kk, k, i, j) -> (i, j)>

// non unit stride.
func.func @brgemm_3(%arg0: memref<9x4x5xf32>, %arg1: memref<9x5x8xf32, strided<[40, 8, 2], offset: ?>>, %arg2: memref<4x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel"]}
    ins(%arg0, %arg1 : memref<9x4x5xf32>, memref<9x5x8xf32, strided<[40, 8, 2], offset: ?>>)
    outs(%arg2: memref<4x8xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: brgemm_3
// CHECK-NOT: xsmm.brgemm
// CHECK: linalg.generic

// -----

#map = affine_map<(i, j, kk, k) -> (kk, i, k)>
#map1 = affine_map<(i, j, kk, k) -> (kk, j, k)>
#map2 = affine_map<(i, j, kk, k) -> (i, j)>

func.func @brgemm_5(%arg0: memref<9x4x5xf32>, %arg1: memref<9x8x5xf32>, %arg2: memref<4x8xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%arg0, %arg1 : memref<9x4x5xf32>, memref<9x8x5xf32>)
    outs(%arg2: memref<4x8xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %5 = arith.mulf %in, %in_8 : f32
        %6 = arith.addf %out, %5 : f32
        linalg.yield %6 : f32
  }
  return
}

// CHECK-LABEL: brgemm_5
// CHECK-SAME: %[[ARG0:.+]]: memref<9x4x5xf32>, %[[ARG1:.+]]: memref<9x8x5xf32>, %[[ARG2:.+]]: memref<4x8xf32>
// CHECK: %[[C9:.+]] = arith.constant 9 : i64
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<9x5x8xf32>
// CHECK: linalg.transpose ins(%[[ARG1]] : memref<9x8x5xf32>)
// CHECK-SAME:  outs(%[[ALLOC]] : memref<9x5x8xf32>) permutation = [0, 2, 1]
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [4, 8, 5, 5, 8, 8, 20, 40] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ALLOC]], %[[ARG2]], %[[C9]])


// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (k, j)>
#map2 = affine_map<(i, j, k) -> (i, j)>

func.func @gemm_1(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: memref<64x32xf32>, memref<32x64xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_1
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<32x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [64, 64, 32, 32, 64, 64] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(k, i, j) -> (i, k)>
#map1 = affine_map<(k, i, j) -> (k, j)>
#map2 = affine_map<(k, i, j) -> (i, j)>

// permutation on outerloop is not relevant.
func.func @gemm_2(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel"]}
    ins(%arg0, %arg1: memref<64x32xf32>, memref<32x64xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_2
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<32x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [64, 64, 32, 32, 64, 64] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]])

// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (k, j)>
#map2 = affine_map<(i, j, k) -> (j, i)>

func.func @gemm_3(%arg0: memref<64x32xf32>, %arg1: memref<32x64xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: memref<64x32xf32>, memref<32x64xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_3
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<32x64xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32>
// CHECK: %[[DIS_TRANS:.+]] = xsmm.unary.dispatch transpose [64, 32, 32, 64] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[DIS_TRANS]], %[[ARG0]], %[[ALLOC]])
// CHECK: %[[ALLOC_0:.+]] = memref.alloc() : memref<64x32xf32>
// CHECK: %[[DIS_TRANS_1:.+]] = xsmm.unary.dispatch transpose [32, 64, 64, 32] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[DIS_TRANS_1]], %[[ARG1]], %[[ALLOC_0]])
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [64, 64, 32, 32, 64, 64] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[DIS]], %[[ALLOC_0]], %[[ALLOC]], %[[ARG2]])

// -----

#map = affine_map<(i, j, k) -> (i, k)>
#map1 = affine_map<(i, j, k) -> (j, k)>
#map2 = affine_map<(i, j, k) -> (j, i)>

func.func @gemm_4(%arg0: memref<64x32xf32>, %arg1: memref<64x32xf32>, %arg2: memref<64x64xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1: memref<64x32xf32>, memref<64x32xf32>)
    outs(%arg2: memref<64x64xf32>) {
  ^bb0(%in: f32, %in_4: f32, %out: f32):
      %1 = arith.mulf %in, %in_4 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  }
  return
}

// CHECK-LABEL: gemm_4
// CHECK-SAME: %[[ARG0:.+]]: memref<64x32xf32>, %[[ARG1:.+]]: memref<64x32xf32>, %[[ARG2:.+]]: memref<64x64xf32>
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32>
// CHECK: %[[DIS_TRAN:.+]] = xsmm.unary.dispatch transpose [64, 32, 32, 64] flags = (none) data_type = f32
// CHECK: xsmm.unary transpose(data_type = f32, %[[DIS_TRAN]], %[[ARG0]], %[[ALLOC]])
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [64, 64, 32, 32, 64, 64] flags = (none) data_type = f32
// CHECK: xsmm.gemm(data_type = f32, %[[DIS]], %[[ARG1]], %[[ALLOC]], %[[ARG2]])

// -----

func.func @simple_brgemm(%arg0: memref<2x32x32xf32>, %arg1: memref<2x32x32xf32>, %arg2: memref<32x32xf32>) {
  linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<2x32x32xf32>, memref<2x32x32xf32>)
                                  outs(%arg2: memref<32x32xf32>)
  return
}

// CHECK-LABEL: simple_brgemm
// CHECK-SAME: %[[ARG0:.+]]: memref<2x32x32xf32>, %[[ARG1:.+]]: memref<2x32x32xf32>, %[[ARG2:.+]]: memref<32x32xf32>
// CHECK: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (none) data_type = f32
// CHECK: xsmm.brgemm(data_type = f32, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C2]])

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @vnni_brgemm_interchanged(%arg0: memref<16x32x32xbf16>, %arg1: memref<16x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
    %expanded = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [16, 32, 16, 2]
      : memref<16x32x32xbf16> into memref<16x32x16x2xbf16>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]}
      ins(%expanded, %arg1 : memref<16x32x16x2xbf16>, memref<16x16x32x2xbf16>)
      outs(%arg2 : memref<32x32xbf16>) {
        ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
          %5 = arith.mulf %in, %in_5 : bf16
          %6 = arith.addf %out, %5 : bf16
          linalg.yield %6 : bf16
    }
    return
  }
}

// CHECK-LABEL: vnni_brgemm_interchanged
// CHECK-SAME:  %[[ARG0:.+]]: memref<16x32x32xbf16>, %[[ARG1:.+]]: memref<16x16x32x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<32x32xbf16>
// CHECK: %[[C16:.+]] = arith.constant 16 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024]
// CHECK-SAME:  flags = (vnni_b) data_type = bf16
// CHECK: xsmm.brgemm(data_type = bf16, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C16]])

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @vnni_brgemm(%arg0: memref<16x32x32xbf16>, %arg1: memref<16x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
    %expanded = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [16, 32, 16, 2]
      : memref<16x32x32xbf16> into memref<16x32x16x2xbf16>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
      ins(%expanded, %arg1 : memref<16x32x16x2xbf16>, memref<16x16x32x2xbf16>)
      outs(%arg2 : memref<32x32xbf16>) {
        ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
          %5 = arith.mulf %in, %in_5 : bf16
          %6 = arith.addf %out, %5 : bf16
          linalg.yield %6 : bf16
    }
    return
  }
}

// CHECK-LABEL: vnni_brgemm
// CHECK-SAME:  %[[ARG0:.+]]: memref<16x32x32xbf16>, %[[ARG1:.+]]: memref<16x16x32x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<32x32xbf16>
// CHECK: %[[C16:.+]] = arith.constant 16 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024]
// CHECK-SAME:  flags = (vnni_b) data_type = bf16
// CHECK: xsmm.brgemm(data_type = bf16, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C16]])

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
// Mixed-precision: bf16 inputs are extended to f32 and accumulated into an f32
// output. The kernel input `data_type` stays bf16 while the wider output type
// is carried via `out_type` (f32 = 1).
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @vnni_brgemm_f32_acc(%arg0: memref<16x32x16x2xbf16>, %arg1: memref<16x16x32x2xbf16>, %arg2: memref<32x32xf32>) {
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1 : memref<16x32x16x2xbf16>, memref<16x16x32x2xbf16>)
      outs(%arg2 : memref<32x32xf32>) {
        ^bb0(%in: bf16, %in_5: bf16, %out: f32):
          %e0 = arith.extf %in : bf16 to f32
          %e1 = arith.extf %in_5 : bf16 to f32
          %5 = arith.mulf %e0, %e1 : f32
          %6 = arith.addf %out, %5 : f32
          linalg.yield %6 : f32
    }
    return
  }
}

// CHECK-LABEL: vnni_brgemm_f32_acc
// CHECK-SAME:  %[[ARG0:.+]]: memref<16x32x16x2xbf16>, %[[ARG1:.+]]: memref<16x16x32x2xbf16>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<32x32xf32>
// CHECK: %[[C16:.+]] = arith.constant 16 : i64
// CHECK: %[[COLLAPSE:.+]] = memref.collapse_shape %[[ARG0]]
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024]
// CHECK-SAME:  flags = (vnni_b) data_type = bf16 {out_type = 1 : i64}
// CHECK: xsmm.brgemm(data_type = bf16, %[[DIS]], %[[COLLAPSE]], %[[ARG1]], %[[ARG2]], %[[C16]]) {out_type = 1 : i64}

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @vnni_brgemm_strided(%arg0: memref<8x8x8xbf16, strided<[64, 8, 1], offset: ?>>,
                                %arg1: memref<8x4x8x2xbf16, strided<[64, 16, 2, 1], offset: ?>>,
                                %arg2: memref<8x8xbf16>) {
    %expanded = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [8, 8, 4, 2]
      : memref<8x8x8xbf16, strided<[64, 8, 1], offset: ?>> into memref<8x8x4x2xbf16, strided<[64, 8, 2, 1], offset: ?>>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
      ins(%expanded, %arg1 : memref<8x8x4x2xbf16, strided<[64, 8, 2, 1], offset: ?>>, memref<8x4x8x2xbf16, strided<[64, 16, 2, 1], offset: ?>>)
      outs(%arg2 : memref<8x8xbf16>) {
        ^bb0(%in: bf16, %in_9: bf16, %out: bf16):
          %11 = arith.mulf %in, %in_9 : bf16
          %12 = arith.addf %out, %11 : bf16
          linalg.yield %12 : bf16
    }
    return
  }
}

// CHECK-LABEL: vnni_brgemm_strided
// CHECK-SAME:  %[[ARG0:.+]]: memref<8x8x8xbf16, strided<[64, 8, 1], offset: ?>>,
// CHECK-SAME:  %[[ARG1:.+]]: memref<8x4x8x2xbf16, strided<[64, 16, 2, 1], offset: ?>>,
// CHECK-SAME:  %[[ARG2:.+]]: memref<8x8xbf16>
// CHECK: %[[C8:.+]] = arith.constant 8 : i64
// CHECK: %[[DIS:.+]] = xsmm.brgemm.dispatch [8, 8, 8, 8, 8, 8, 64, 64]
// CHECK-SAME:  flags = (vnni_b) data_type = bf16
// CHECK: xsmm.brgemm(data_type = bf16, %[[DIS]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[C8]])

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d2)>

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @vnni_brgemm_require_transpose_on_C(%arg0: memref<16x32x32xbf16>, %arg1: memref<16x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
    %expanded = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [16, 32, 16, 2]
      : memref<16x32x32xbf16> into memref<16x32x16x2xbf16>
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]}
      ins(%expanded, %arg1 : memref<16x32x16x2xbf16>, memref<16x16x32x2xbf16>)
      outs(%arg2 : memref<32x32xbf16>) {
        ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
          %5 = arith.mulf %in, %in_5 : bf16
          %6 = arith.addf %out, %5 : bf16
          linalg.yield %6 : bf16
    }
    return
  }
}

// CHECK-LABEL: vnni_brgemm_require_transpose_on_C
// CHECK-NOT: xsmm.brgemm
// CHECK: linalg.generic

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

func.func @brgemm_not_vnni(%arg0: memref<16x32x160xbf16>, %arg1: memref<16x32x32x5xbf16>, %arg2: memref<32x32xbf16>) {
  %expanded = memref.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [16, 32, 32, 5]
    : memref<16x32x160xbf16> into memref<16x32x32x5xbf16>
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]}
    ins(%expanded, %arg1 : memref<16x32x32x5xbf16>, memref<16x32x32x5xbf16>)
    outs(%arg2 : memref<32x32xbf16>) {
      ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
        %5 = arith.mulf %in, %in_5 : bf16
        %6 = arith.addf %out, %5 : bf16
        linalg.yield %6 : bf16
  }
  return
}

// CHECK-LABEL: brgemm_not_vnni
// CHECK-NOT: xsmm.brgemm
// CHECK: linalg.generic
