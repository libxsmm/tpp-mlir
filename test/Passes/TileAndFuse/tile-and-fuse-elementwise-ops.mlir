// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=0,32,32" -split-input-file | FileCheck %s
// RUN: tpp-opt %s -tile-elementwise-ops -split-input-file | FileCheck %s --check-prefix=FOLD

// Flat bf16 batch-reduce GEMM with a bias-add + truncf epilogue: the
// contraction is tiled 32x32 on M and N, and the zero fill and the elementwise
// epilogue are fused into the same tile.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @brgemm_bf16_bias_add_truncf(%arg0: tensor<48x128x128xbf16>, %arg1: tensor<48x128x128xbf16>, %arg2: tensor<128x128xbf16>) -> tensor<128x128xbf16> attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"reg_gemm_unroll" = [16, 16, 32]>>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<48x128x128xbf16>, tensor<48x128x128xbf16>) outs(%1 : tensor<128x128xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %4 = arith.extf %in : bf16 to f32
      %5 = arith.extf %in_0 : bf16 to f32
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<128x128xf32>, tensor<128x128xbf16>) outs(%arg2 : tensor<128x128xbf16>) {
    ^bb0(%in: f32, %in_0: bf16, %out: bf16):
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.addf %in, %4 : f32
      %6 = arith.truncf %5 : f32 to bf16
      linalg.yield %6 : bf16
    } -> tensor<128x128xbf16>
    return %3 : tensor<128x128xbf16>
  }
}

// CHECK-LABEL: func.func @brgemm_bf16_bias_add_truncf(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<48x128x128xbf16>, %[[ARG1:.+]]: tensor<48x128x128xbf16>, %[[ARG2:.+]]: tensor<128x128xbf16>
// CHECK:       %[[EMPTY:.+]] = tensor.empty() : tensor<128x128xf32>
// CHECK:       %[[RES:.+]] = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (128, 128) step (32, 32)
// CHECK-SAME:    shared_outs(%[[OUT:.+]] = %[[ARG2]]) -> (tensor<128x128xbf16>)
// CHECK:         %[[A:.+]] = tensor.extract_slice %[[ARG0]][0, %[[I]], 0] [48, 32, 128] [1, 1, 1]
// CHECK:         %[[B:.+]] = tensor.extract_slice %[[ARG1]][0, 0, %[[J]]] [48, 128, 32] [1, 1, 1]
// CHECK:         %[[ACC:.+]] = tensor.extract_slice %[[EMPTY]][%[[I]], %[[J]]] [32, 32] [1, 1]
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[ACC]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:         %[[GEMM:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<48x32x128xbf16>, tensor<48x128x32xbf16>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<32x32xf32>)
// CHECK:         %[[BIAS:.+]] = tensor.extract_slice %[[OUT]][%[[I]], %[[J]]] [32, 32] [1, 1]
// CHECK:         %[[DEST:.+]] = tensor.extract_slice %[[OUT]][%[[I]], %[[J]]] [32, 32] [1, 1]
// CHECK:         %[[EPI:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      ins(%[[GEMM]], %[[BIAS]] : tensor<32x32xf32>, tensor<32x32xbf16>)
// CHECK-SAME:      outs(%[[DEST]] : tensor<32x32xbf16>)
// CHECK:           arith.truncf %{{.+}} : f32 to bf16
// CHECK:         scf.forall.in_parallel
// CHECK-NEXT:      tensor.parallel_insert_slice %[[EPI]] into %[[OUT]][%[[I]], %[[J]]] [32, 32] [1, 1]
// CHECK-NOT:   linalg.generic
// CHECK:       return %[[RES]] : tensor<128x128xbf16>

// -----

// Flat i8 batch-reduce GEMM with a dequantization epilogue (row and column
// scales broadcast along N and M): the contraction is tiled 32x32 on M and N,
// and the zero fill and the elementwise epilogue are fused into the same tile.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @brgemm_i8_dequantize(%arg0: tensor<8x64x64xi8>, %arg1: tensor<64xf32>, %arg2: tensor<8x64x64xi8>, %arg3: tensor<64xf32>, %arg4: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {dlti.target_system_spec = #dlti.target_system_spec<"CPU" = #dlti.target_device_spec<"reg_gemm_unroll" = [16, 16, 32]>>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<64x64xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg2 : tensor<8x64x64xi8>, tensor<8x64x64xi8>) outs(%1 : tensor<64x64xi32>) {
    ^bb0(%in: i8, %in_0: i8, %out: i32):
      %4 = arith.extsi %in : i8 to i32
      %5 = arith.extsi %in_0 : i8 to i32
      %6 = arith.muli %4, %5 : i32
      %7 = arith.addi %out, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<64x64xi32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%2, %arg1, %arg3 : tensor<64x64xi32>, tensor<64xf32>, tensor<64xf32>) outs(%arg4 : tensor<64x64xf32>) {
    ^bb0(%in: i32, %in_0: f32, %in_1: f32, %out: f32):
      %4 = arith.sitofp %in : i32 to f32
      %5 = arith.mulf %in_0, %in_1 : f32
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %6, %out : f32
      linalg.yield %7 : f32
    } -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
}

// CHECK-LABEL: func.func @brgemm_i8_dequantize(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<8x64x64xi8>, %[[ARG1:.+]]: tensor<64xf32>, %[[ARG2:.+]]: tensor<8x64x64xi8>, %[[ARG3:.+]]: tensor<64xf32>, %[[ARG4:.+]]: tensor<64x64xf32>
// CHECK:       %[[EMPTY:.+]] = tensor.empty() : tensor<64x64xi32>
// CHECK:       %[[RES:.+]] = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (64, 64) step (32, 32)
// CHECK-SAME:    shared_outs(%[[OUT:.+]] = %[[ARG4]]) -> (tensor<64x64xf32>)
// CHECK:         %[[A:.+]] = tensor.extract_slice %[[ARG0]][0, %[[I]], 0] [8, 32, 64] [1, 1, 1]
// CHECK:         %[[B:.+]] = tensor.extract_slice %[[ARG2]][0, 0, %[[J]]] [8, 64, 32] [1, 1, 1]
// CHECK:         %[[ACC:.+]] = tensor.extract_slice %[[EMPTY]][%[[I]], %[[J]]] [32, 32] [1, 1]
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%{{.+}} : i32) outs(%[[ACC]] : tensor<32x32xi32>) -> tensor<32x32xi32>
// CHECK:         %[[GEMM:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["reduction", "parallel", "parallel", "reduction"]
// CHECK-SAME:      ins(%[[A]], %[[B]] : tensor<8x32x64xi8>, tensor<8x64x32xi8>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<32x32xi32>)
// CHECK:         %[[SCALE_M:.+]] = tensor.extract_slice %[[ARG1]][%[[I]]] [32] [1]
// CHECK:         %[[SCALE_N:.+]] = tensor.extract_slice %[[ARG3]][%[[J]]] [32] [1]
// CHECK:         %[[DEST:.+]] = tensor.extract_slice %[[OUT]][%[[I]], %[[J]]] [32, 32] [1, 1]
// CHECK:         %[[EPI:.+]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      ins(%[[GEMM]], %[[SCALE_M]], %[[SCALE_N]] : tensor<32x32xi32>, tensor<32xf32>, tensor<32xf32>)
// CHECK-SAME:      outs(%[[DEST]] : tensor<32x32xf32>)
// CHECK:           arith.sitofp %{{.+}} : i32 to f32
// CHECK:         scf.forall.in_parallel
// CHECK-NEXT:      tensor.parallel_insert_slice %[[EPI]] into %[[OUT]][%[[I]], %[[J]]] [32, 32] [1, 1]
// CHECK-NOT:   linalg.generic
// CHECK:       return %[[RES]] : tensor<64x64xf32>

// -----

// Negative: on memrefs the contraction is not tiled and the epilogue is not
// fused.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @brgemm_memref_no_fusion(%arg0: memref<48x128x128xbf16>, %arg1: memref<48x128x128xbf16>, %arg2: memref<128x128xbf16>, %arg3: memref<128x128xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst : f32) outs(%arg3 : memref<128x128xf32>)
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<48x128x128xbf16>, memref<48x128x128xbf16>) outs(%arg3 : memref<128x128xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %0 = arith.extf %in : bf16 to f32
      %1 = arith.extf %in_0 : bf16 to f32
      %2 = arith.mulf %0, %1 : f32
      %3 = arith.addf %out, %2 : f32
      linalg.yield %3 : f32
    }
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg3, %arg2 : memref<128x128xf32>, memref<128x128xbf16>) outs(%arg2 : memref<128x128xbf16>) {
    ^bb0(%in: f32, %in_0: bf16, %out: bf16):
      %0 = arith.extf %in_0 : bf16 to f32
      %1 = arith.addf %in, %0 : f32
      %2 = arith.truncf %1 : f32 to bf16
      linalg.yield %2 : bf16
    }
    return
  }
}

// CHECK-LABEL: func.func @brgemm_memref_no_fusion(
// CHECK-SAME:  %[[ARG0:.+]]: memref<48x128x128xbf16>, %[[ARG1:.+]]: memref<48x128x128xbf16>, %[[ARG2:.+]]: memref<128x128xbf16>, %[[ARG3:.+]]: memref<128x128xf32>
// CHECK-NOT:   scf.forall
// CHECK:       linalg.fill ins(%{{.+}} : f32) outs(%[[ARG3]] : memref<128x128xf32>)
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[ARG0]], %[[ARG1]] : memref<48x128x128xbf16>, memref<48x128x128xbf16>)
// CHECK-SAME:    outs(%[[ARG3]] : memref<128x128xf32>)
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[ARG3]], %[[ARG2]] : memref<128x128xf32>, memref<128x128xbf16>)
// CHECK-SAME:    outs(%[[ARG2]] : memref<128x128xbf16>)
// CHECK-NOT:   scf.forall
// CHECK:       return

// -----

// Negative (-tile-elementwise-ops): the epilogue reads the output tensor
// through a transposed map, so that input is not folded into the init; it
// would read a different element than the one being updated.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @epilogue_transposed_output_read(%arg0: tensor<48x128x128xbf16>, %arg1: tensor<48x128x128xbf16>, %arg2: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<48x128x128xbf16>, tensor<48x128x128xbf16>) outs(%1 : tensor<128x128xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %4 = arith.extf %in : bf16 to f32
      %5 = arith.extf %in_0 : bf16 to f32
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<128x128xf32>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<128x128xf32>, tensor<128x128xbf16>) outs(%arg2 : tensor<128x128xbf16>) {
    ^bb0(%in: f32, %in_0: bf16, %out: bf16):
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.addf %in, %4 : f32
      %6 = arith.truncf %5 : f32 to bf16
      linalg.yield %6 : bf16
    } -> tensor<128x128xbf16>
    return %3 : tensor<128x128xbf16>
  }
}

// CHECK-LABEL: func.func @epilogue_transposed_output_read(

// FOLD:        #[[TMAP:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// FOLD-LABEL:  func.func @epilogue_transposed_output_read(
// FOLD-SAME:   %{{.+}}: tensor<48x128x128xbf16>, %{{.+}}: tensor<48x128x128xbf16>, %[[ARG2:.+]]: tensor<128x128xbf16>
// FOLD:        scf.for
// FOLD:          %[[BIAS:.+]] = tensor.extract_slice %[[ARG2]][%{{.+}}, %{{.+}}] [128, 1] [1, 1]
// FOLD:          linalg.generic {indexing_maps = [#{{.+}}, #[[TMAP]], #{{.+}}]
// FOLD-SAME:       ins(%{{.+}}, %[[BIAS]] : tensor<1x128xf32>, tensor<128x1xbf16>)
// FOLD-NEXT:     ^bb0(%{{.+}}: f32, %{{.+}}: bf16, %{{.+}}: bf16):
