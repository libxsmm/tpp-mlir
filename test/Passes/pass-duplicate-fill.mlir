// RUN: tpp-opt %s -duplicate-fill -split-input-file | FileCheck %s

// Check we do not introduce additional allocations or copies.
// RUN: tpp-opt %s -bufferize -split-input-file | FileCheck %s -check-prefix=BUFF
// RUN: tpp-opt %s -bufferize="duplicate-fill=false" -split-input-file | FileCheck %s -check-prefix=BUFFNOTDUP

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: duplicate_zero_fill_on_contractions
// BUFF-LABEL: duplicate_zero_fill_on_contractions
// BUFFNOTDUP-LABEL: duplicate_zero_fill_on_contractions
func.func @duplicate_zero_fill_on_contractions(%arg0: tensor<32x512xf32>,
      %arg1: tensor<512x64xf32>) -> tensor<32x64xf32> {
  // BUFF-COUNT-2: memref.alloc
  // BUFF-COUNT-1: memref.dealloc
  // BUFF-NOT: linalg.copy
  //
  // BUFFNOTDUP-COUNT-2: memref.alloc
  // BUFFNOTDUP-COUNT-1: memref.dealloc
  // BUFFNOTDUP-NOT: linalg.copy
  %cst_2 = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<32x64xf32>
  %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
  // CHECK: linalg.fill
  // CHECK-NEXT: linalg.generic
  %3 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<32x512xf32>, tensor<512x64xf32>) outs(%1 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %9 = arith.mulf %in, %in_5 : f32
        %10 = arith.addf %out, %9 : f32
        linalg.yield %10 : f32
  } -> tensor<32x64xf32>
  // CHECK: linalg.fill
  // CHECK-NEXT: linalg.generic
  %4 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<32x512xf32>, tensor<512x64xf32>) outs(%1 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %9 = arith.mulf %in, %in_5 : f32
        %10 = arith.addf %out, %9 : f32
        linalg.yield %10 : f32
  } -> tensor<32x64xf32>
  // CHECK-NOT: linalg.fill
  %5 = linalg.add ins(%3, %4 : tensor<32x64xf32>, tensor<32x64xf32>)
                  outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %5 : tensor<32x64xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d2, d0)>

func.func @mha_contractions(%arg0: tensor<64x32x512xf32>, %arg1: tensor<64x32x512xf32>,
                            %arg2: tensor<64x32x512xf32>) -> tensor<64x8x32x32xf32> {
  %cst = arith.constant dense<2.000000e-01> : tensor<512x64xf32>
  %cst_0 = arith.constant dense<1.000000e-01> : tensor<512x64xf32>
  %cst_1 = arith.constant dense<1.250000e-01> : tensor<32x64xf32>
  %cst_2 = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<64x8x32x32xf32>
  %1 = scf.forall (%arg3, %arg4) in (64, 8) shared_outs(%arg5 = %0) -> (tensor<64x8x32x32xf32>) {
    // BUFF-COUNT-2: memref.alloc
    // BUFF-COUNT-2: memref.dealloc
    //
    // BUFFNOTDUP-COUNT-2: memref.alloc
    // BUFFNOTDUP-COUNT-2: memref.dealloc
    %2 = tensor.empty() : tensor<32x64xf32>
    // CHECK: linalg.fill
    // CHECK-NEXT: tensor.extract_slice
    // CHECK-NEXT: linalg.generic
    %3 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<32x64xf32>) -> tensor<32x64xf32>
    %extracted_slice = tensor.extract_slice %arg1[%arg3, 0, 0] [1, 32, 512] [1, 1, 1] : tensor<64x32x512xf32> to tensor<32x512xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%extracted_slice, %cst : tensor<32x512xf32>, tensor<512x64xf32>) outs(%3 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %9 = arith.mulf %in, %in_5 : f32
        %10 = arith.addf %out, %9 : f32
        linalg.yield %10 : f32
    } -> tensor<32x64xf32>
    %extracted_slice_3 = tensor.extract_slice %arg0[%arg3, 0, 0] [1, 32, 512] [1, 1, 1] : tensor<64x32x512xf32> to tensor<32x512xf32>
    // CHECK: linalg.fill
    // CHECK-NEXT: linalg.generic
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%extracted_slice_3, %cst_0 : tensor<32x512xf32>, tensor<512x64xf32>) outs(%3 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %9 = arith.mulf %in, %in_5 : f32
        %10 = arith.addf %out, %9 : f32
        linalg.yield %10 : f32
    } -> tensor<32x64xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%5, %cst_1 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%2 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %9 = arith.mulf %in, %in_5 : f32
        linalg.yield %9 : f32
    } -> tensor<32x64xf32>
    %extracted_slice_4 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<64x8x32x32xf32> to tensor<32x32xf32>
    %7 = linalg.fill ins(%cst_2 : f32) outs(%extracted_slice_4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map4, #map5], iterator_types = ["parallel", "reduction", "parallel"]} ins(%4, %6 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%7 : tensor<32x32xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %9 = arith.mulf %in, %in_5 : f32
        %10 = arith.addf %out, %9 : f32
        linalg.yield %10 : f32
    } -> tensor<32x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %8 into %arg5[%arg3, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<64x8x32x32xf32>
    }
  }
  return %1 : tensor<64x8x32x32xf32>
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK-LABEL: duplicate_non_zero_fill_on_contractions
// BUFF-LABEL: duplicate_non_zero_fill_on_contractions
// BUFFNOTDUP-LABEL: duplicate_non_zero_fill_on_contractions
func.func @duplicate_non_zero_fill_on_contractions(%arg0: tensor<32x512xf32>,
      %arg1: tensor<512x64xf32>) -> tensor<32x64xf32> {
  %cst_2 = arith.constant 1.0 : f32
  %0 = tensor.empty() : tensor<32x64xf32>
  %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32x64xf32>) -> tensor<32x64xf32>
  // CHECK: linalg.fill
  // CHECK-NEXT: linalg.generic
  %3 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<32x512xf32>, tensor<512x64xf32>) outs(%1 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %9 = arith.mulf %in, %in_5 : f32
        %10 = arith.addf %out, %9 : f32
        linalg.yield %10 : f32
  } -> tensor<32x64xf32>
  // CHECK-NOT: linalg.fill
  // CHECK: linalg.generic
  %4 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "reduction", "parallel"]}
    ins(%arg0, %arg1 : tensor<32x512xf32>, tensor<512x64xf32>) outs(%1 : tensor<32x64xf32>) {
      ^bb0(%in: f32, %in_5: f32, %out: f32):
        %9 = arith.mulf %in, %in_5 : f32
        %10 = arith.addf %out, %9 : f32
        linalg.yield %10 : f32
  } -> tensor<32x64xf32>
  // CHECK-NOT: linalg.fill
  %5 = linalg.add ins(%3, %4 : tensor<32x64xf32>, tensor<32x64xf32>)
                  outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %5 : tensor<32x64xf32>
}

// -----

func.func @matmuls(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // BUFF-COUNT-2: memref.alloc
  // BUFFNOTDUP-COUNT-2: memref.alloc
  %0 = tensor.empty() : tensor<32x32xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %3 = linalg.matmul ins(%arg0, %2 : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %3 : tensor<32x32xf32>
}

// CHECK-LABEL: matmuls
// CHECK-SAME: %[[ARG0:.+]]: tensor<32x32xf32>, %[[ARG1:.+]]: tensor<32x32xf32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[MUL:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK-SAME:  outs(%[[FILL]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %[[FILL_1:.+]] = linalg.fill ins(%{{.+}} : f32) outs(%[[EMPTY]] : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK: %{{.+}} = linalg.matmul ins(%[[ARG0]], %[[MUL]] : tensor<32x32xf32>, tensor<32x32xf32>)
// CHECK-SAME:  outs(%[[FILL_1]] : tensor<32x32xf32>) -> tensor<32x32xf32>

// -----

// CHECK-LABEL: matmuls_1
func.func @matmuls_1(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // BUFF-COUNT-3: memref.alloc
  // BUFF-COUNT-1: memref.copy
  // BUFFNOTDUP-COUNT-3: memref.alloc
  // BUFFNOTDUP-COUNT-1: memref.copy
  // CHECK-COUNT-2: linalg.fill
  %0 = tensor.empty() : tensor<32x32xf32>
  %cst = arith.constant 0.0 : f32
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %3 = linalg.add ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
                  outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %4 = linalg.matmul ins(%3, %2 : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%1 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %4 : tensor<32x32xf32>
}
