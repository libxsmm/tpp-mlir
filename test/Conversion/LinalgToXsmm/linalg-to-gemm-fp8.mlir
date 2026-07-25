// RUN: tpp-opt %s -convert-linalg-to-xsmm -split-input-file | FileCheck %s

// FP8 uses a VNNI blocking factor of 4. libxsmm names E5M2 as BF8 and E4M3 as
// HF8.

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// Fix VNNI blocking factor for lit testing.
// Prevent mismatches due to target specific VNNI factors.
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 4 : i32>>
} {
  func.func @square_vnni_gemm_e5m2(%arg0: memref<64x16x4xf8E5M2, strided<[64, 4, 1], offset: ?>>,
    %arg1: memref<16x64x4xf8E5M2>, %arg2: memref<64x64xf8E5M2, strided<[64, 1], offset: ?>>) {
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : memref<64x16x4xf8E5M2, strided<[64, 4, 1], offset: ?>>, memref<16x64x4xf8E5M2>)
      outs(%arg2 : memref<64x64xf8E5M2, strided<[64, 1], offset: ?>>) {
        ^bb0(%in: f8E5M2, %in_2: f8E5M2, %out: f8E5M2):
          %1 = arith.mulf %in, %in_2 : f8E5M2
          %2 = arith.addf %out, %1 : f8E5M2
          linalg.yield %2 : f8E5M2
      }
    return
  }
}

// CHECK-LABEL: square_vnni_gemm_e5m2
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [64, 64, 64, 64, 64, 64] flags = (vnni_b) data_type = bf8
// CHECK: xsmm.gemm(data_type = bf8, %[[DIS]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 4 : i32>>
} {
  func.func @square_vnni_gemm_e4m3(%arg0: memref<64x16x4xf8E4M3FN, strided<[64, 4, 1], offset: ?>>,
    %arg1: memref<16x64x4xf8E4M3FN>, %arg2: memref<64x64xf8E4M3FN, strided<[64, 1], offset: ?>>) {
    linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : memref<64x16x4xf8E4M3FN, strided<[64, 4, 1], offset: ?>>, memref<16x64x4xf8E4M3FN>)
      outs(%arg2 : memref<64x64xf8E4M3FN, strided<[64, 1], offset: ?>>) {
        ^bb0(%in: f8E4M3FN, %in_2: f8E4M3FN, %out: f8E4M3FN):
          %1 = arith.mulf %in, %in_2 : f8E4M3FN
          %2 = arith.addf %out, %1 : f8E4M3FN
          linalg.yield %2 : f8E4M3FN
      }
    return
  }
}

// CHECK-LABEL: square_vnni_gemm_e4m3
// CHECK: %[[DIS:.+]] = xsmm.gemm.dispatch [64, 64, 64, 64, 64, 64] flags = (vnni_b) data_type = hf8
// CHECK: xsmm.gemm(data_type = hf8, %[[DIS]]
