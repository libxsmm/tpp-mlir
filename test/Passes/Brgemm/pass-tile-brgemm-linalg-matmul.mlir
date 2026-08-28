// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=8,32,1" --split-input-file  | FileCheck -check-prefix=CONF1 %s
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=32,32,32" --split-input-file  | FileCheck -check-prefix=CONF2 %s

module {
  func.func @gemm_do_register_tiling(%arg0: memref<16x32x16x32xf32>, %arg1: memref<32x32x32x32xf32>, %arg2: memref<16x32x16x32xf32>) {
    scf.forall (%arg3, %arg4) in (16, 32) {
      %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
      %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
      %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
      linalg.batch_reduce_matmul ins(%subview, %subview_0 : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_1 : memref<16x32xf32, strided<[32, 1], offset: ?>>)
    }
    return
  }
}


// CONF1-LABEL: func.func @gemm_do_register_tiling
// CONF1-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF1-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF1-DAG: %[[C8:.+]] = arith.constant 8 : index
// CONF1-DAG: %[[C16:.+]] = arith.constant 16 : index
// CONF1-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF1: scf.forall (%arg3, %arg4) in (16, 32) {
// CONF1-NEXT: %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>>
// CONF1-NEXT: %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<32x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1-NEXT: %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<16x32x16x32xf32> to memref<16x32xf32, strided<[32, 1], offset: ?>>
// CONF1-NEXT: scf.for %[[I:.+]] = %[[C0]] to %[[C16]] step %[[C8]] {
// CONF1-NEXT:  scf.for %[[J:.+]] = %[[C0]] to %[[C32]] step %[[C32]] {
// CONF1-NEXT:   scf.for %[[K:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CONF1-NEXT:    scf.for %[[L:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CONF1-NEXT:     %subview_2 = memref.subview %subview[%[[K]], %[[I]], %[[L]]] [1, 8, 1] [1, 1, 1] : memref<32x16x32xf32, strided<[512, 32, 1], offset: ?>> to memref<1x8x1xf32, strided<[512, 32, 1], offset: ?>>
// CONF1-NEXT:     %subview_3 = memref.subview %subview_0[%[[K]], %[[L]], %[[J]]] [1, 1, 32] [1, 1, 1] : memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>> to memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>
// CONF1-NEXT:     %subview_4 = memref.subview %subview_1[%[[I]], %[[J]]] [8, 32] [1, 1] : memref<16x32xf32, strided<[32, 1], offset: ?>> to memref<8x32xf32, strided<[32, 1], offset: ?>>
// CONF1-NEXT:     linalg.batch_reduce_matmul ins(%subview_2, %subview_3 : memref<1x8x1xf32, strided<[512, 32, 1], offset: ?>>, memref<1x1x32xf32, strided<[1024, 32, 1], offset: ?>>) outs(%subview_4 : memref<8x32xf32, strided<[32, 1], offset: ?>>)

// -----

module {
  func.func @brgemm_tensor_type_tiling(%arg0: tensor<128x256x512xf32>, %arg1: tensor<128x512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<128x256x512xf32>, tensor<128x512x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}


// CONF1-LABEL: func.func @brgemm_tensor_type_tiling
// CONF1-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF1-DAG: %[[C256:.+]] = arith.constant 256 : index
// CONF1-DAG: %[[C8:.+]] = arith.constant 8 : index
// CONF1-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF1-DAG: %[[C128:.+]] = arith.constant 128 : index
// CONF1-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF1-DAG: %[[C512:.+]] = arith.constant 512 : index
// CONF1: %0 = scf.for %[[I:.+]] = %[[C0]] to %[[C256]] step %[[C8]] iter_args(%arg4 = %arg2) -> (tensor<256x256xf32>) {
// CONF1-NEXT:  %1 = scf.for %[[J:.+]] = %[[C0]] to %[[C256]] step %[[C32]] iter_args(%arg6 = %arg4) -> (tensor<256x256xf32>) {
// CONF1-NEXT:   %2 = scf.for %[[K:.+]] = %[[C0]] to %[[C128]] step %[[C1]] iter_args(%arg8 = %arg6) -> (tensor<256x256xf32>) {
// CONF1-NEXT:    %3 = scf.for %[[L:.+]] = %[[C0]] to %[[C512]] step %[[C1]] iter_args(%arg10 = %arg8) -> (tensor<256x256xf32>) {
// CONF1-NEXT:     %extracted_slice = tensor.extract_slice %arg0[%[[K]], %[[I]], %[[L]]] [1, 8, 1] [1, 1, 1] : tensor<128x256x512xf32> to tensor<1x8x1xf32>
// CONF1-NEXT:     %extracted_slice_0 = tensor.extract_slice %arg1[%[[K]], %[[L]], %[[J]]] [1, 1, 32] [1, 1, 1] : tensor<128x512x256xf32> to tensor<1x1x32xf32>
// CONF1-NEXT:     %extracted_slice_1 = tensor.extract_slice %arg10[%[[I]], %[[J]]] [8, 32] [1, 1] : tensor<256x256xf32> to tensor<8x32xf32>
// CONF1-NEXT:     %4 = linalg.batch_reduce_matmul ins(%extracted_slice, %extracted_slice_0 : tensor<1x8x1xf32>, tensor<1x1x32xf32>) outs(%extracted_slice_1 : tensor<8x32xf32>) -> tensor<8x32xf32>
// CONF1-NEXT:     %inserted_slice = tensor.insert_slice %4 into %arg10[%[[I]], %[[J]]] [8, 32] [1, 1] : tensor<8x32xf32> into tensor<256x256xf32>

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> alignment = 64
  func.func @gemm_32tiles_do_tiling_bf16(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
    %alloc = memref.alloc() alignment = 64 : memref<8x32x32x32xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
    scf.forall (%arg1, %arg2) in (8, 32) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
      %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<32x16x32x2xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
      ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_1 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
      }
    }
    return %alloc : memref<8x32x32x32xbf16>
  }
}

// CONF2-LABEL: func.func @gemm_32tiles_do_tiling_bf16
// CONF2-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF2-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF2-DAG: %[[C16:.+]] = arith.constant 16 : index
// CONF2-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF2: %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CONF2-NEXT: linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
// CONF2-NEXT: %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF2-NEXT:  scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C32]] {
// CONF2-NEXT:   scf.for %[[J:.+]] = %[[C0]] to %[[C32]] step %[[C32]] {
// CONF2-NEXT:    scf.for %[[K:.+]] = %[[C0]] to %[[C32]] step %[[C1]] {
// CONF2-NEXT:     scf.for %[[L:.+]] = %[[C0]] to %[[C16]] step %[[C16]] {
// CONF2-NEXT:      %subview_1 = memref.subview %subview_0[%[[K]], %[[I]], %[[L]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
// CONF2-NEXT:      %subview_2 = memref.subview %0[%[[K]], %[[L]], %[[J]], 0]  [1, 16, 32, 2] [1, 1, 1, 1] : memref<32x16x32x2xbf16> to memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
// CONF2-NEXT:      %subview_3 = memref.subview %subview[%[[I]], %[[J]]]  [32, 32] [1, 1] : memref<32x32xbf16, strided<[32, 1], offset: ?>> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
// CONF2-NEXT:      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_1, %subview_2 : memref<1x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<1x16x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>) outs(%subview_3 : memref<32x32xbf16, strided<[32, 1], offset: ?>>)

// -----

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d2)>
module {
  memref.global "private" constant @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16> = dense<1.000000e+00> alignment = 64
  func.func @gemm_64tiles_do_tiling_bf16(%arg0: memref<4x16x64x64xbf16>) -> memref<4x16x64x64xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_16x32x64x2xbf16 : memref<16x32x64x2xbf16>
    %alloc = memref.alloc() alignment = 64 : memref<4x16x64x64xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : memref<4x16x64x64xbf16> into memref<4x16x64x32x2xbf16>
    scf.forall (%arg1, %arg2) in (4, 16) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
      linalg.fill ins(%cst : bf16) outs(%subview : memref<64x64xbf16, strided<[64, 1], offset: ?>>)
      %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, memref<16x32x64x2xbf16>) outs(%subview : memref<64x64xbf16, strided<[64, 1], offset: ?>>) {
      ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_1 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
      }
    }
    return %alloc : memref<4x16x64x64xbf16>
  }
}

// CONF2-LABEL: func.func @gemm_64tiles_do_tiling_bf16
// CONF2-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF2-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF2-DAG: %[[C64:.+]] = arith.constant 64 : index
// CONF2-DAG: %[[C16:.+]] = arith.constant 16 : index
// CONF2-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF2: %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
// CONF2-NEXT: linalg.fill ins(%cst : bf16) outs(%subview : memref<64x64xbf16, strided<[64, 1], offset: ?>>)
// CONF2-NEXT: %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x16x64x32x2xbf16> to memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CONF2-NEXT: scf.for %[[I:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CONF2-NEXT:  scf.for %[[J:.+]] = %[[C0]] to %[[C64]] step %[[C32]] {
// CONF2-NEXT:   scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CONF2-NEXT:    scf.for %[[L:.+]] = %[[C0]] to %[[C32]] step %[[C16]] {
// CONF2-NEXT:     %subview_1 = memref.subview %subview_0[%[[K]], %[[I]], %[[L]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : memref<16x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>> to memref<1x32x16x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
// CONF2-NEXT:     %subview_2 = memref.subview %0[%[[K]], %[[L]], %[[J]], 0] [1, 16, 32, 2] [1, 1, 1, 1] : memref<16x32x64x2xbf16> to memref<1x16x32x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
// CONF2-NEXT:     %subview_3 = memref.subview %subview[%[[I]], %[[J]]] [32, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
// CONF2-NEXT:     linalg.generic

// -----


#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  func.func @gemm_64tiles_do_tiling_bf16_tensor(%arg0: tensor<4x16x64x64xbf16>) -> tensor<4x16x64x64xbf16> {
    %cst = arith.constant dense<1.000000e+00> : tensor<16x32x64x2xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %0 = bufferization.alloc_tensor() : tensor<4x16x64x64xbf16>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 16, 64, 32, 2] : tensor<4x16x64x64xbf16> into tensor<4x16x64x32x2xbf16>
    %1 = scf.forall (%arg1, %arg2) in (4, 16) shared_outs(%arg3 = %0) -> (tensor<4x16x64x64xbf16>) {
      %extracted_slice = tensor.extract_slice %arg3[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<4x16x64x64xbf16> to tensor<64x64xbf16>
      %2 = linalg.fill ins(%cst_0 : bf16) outs(%extracted_slice : tensor<64x64xbf16>) -> tensor<64x64xbf16>
      %extracted_slice_1 = tensor.extract_slice %expanded[%arg1, 0, 0, 0, 0] [1, 16, 64, 32, 2] [1, 1, 1, 1, 1] : tensor<4x16x64x32x2xbf16> to tensor<16x64x32x2xbf16>
      %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_1, %cst : tensor<16x64x32x2xbf16>, tensor<16x32x64x2xbf16>) outs(%2 : tensor<64x64xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %4 = arith.mulf %in, %in_2 : bf16
        %5 = arith.addf %out, %4 : bf16
        linalg.yield %5 : bf16
      } -> tensor<64x64xbf16>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %3 into %arg3[%arg1, %arg2, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : tensor<64x64xbf16> into tensor<4x16x64x64xbf16>
      }
    }
    return %1 : tensor<4x16x64x64xbf16>
  }
}

// CONF2-LABEL: func.func @gemm_64tiles_do_tiling_bf16_tensor
// CONF2-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF2-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF2-DAG: %[[C64:.+]] = arith.constant 64 : index
// CONF2-DAG: %[[C16:.+]] = arith.constant 16 : index
// CONF2-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF2:      %3 = scf.for %[[I:.+]] = %[[C0]] to %[[C64]] step %[[C32]] iter_args(%arg5 = %2) -> (tensor<64x64xbf16>) 
// CONF2-NEXT:        %4 = scf.for %[[J:.+]] = %[[C0]] to %[[C64]] step %[[C32]] iter_args(%arg7 = %arg5) -> (tensor<64x64xbf16>) 
// CONF2-NEXT:          %5 = scf.for %[[K:.+]] = %[[C0]] to %[[C16]] step %[[C1]] iter_args(%arg9 = %arg7) -> (tensor<64x64xbf16>) 
// CONF2-NEXT:            %6 = scf.for %[[L:.+]] = %[[C0]] to %[[C32]] step %[[C16]] iter_args(%arg11 = %arg9) -> (tensor<64x64xbf16>) 
// CONF2-NEXT:              %extracted_slice_2 = tensor.extract_slice %extracted_slice_1[%[[K]], %[[I]], %[[L]], 0] [1, 32, 16, 2] [1, 1, 1, 1] : tensor<16x64x32x2xbf16> to tensor<1x32x16x2xbf16>
// CONF2-NEXT:              %extracted_slice_3 = tensor.extract_slice %cst[%[[K]], %[[L]], %[[J]], 0] [1, 16, 32, 2] [1, 1, 1, 1] : tensor<16x32x64x2xbf16> to tensor<1x16x32x2xbf16>
// CONF2-NEXT:              %extracted_slice_4 = tensor.extract_slice %arg11[%[[I]], %[[J]]] [32, 32] [1, 1] : tensor<64x64xbf16> to tensor<32x32xbf16>
// CONF2-NEXT:              %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x32x16x2xbf16>, tensor<1x16x32x2xbf16>) outs(%extracted_slice_4 : tensor<32x32xbf16>)

// -----

module {
  func.func @matmul_no_tiling(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) {
     linalg.matmul ins(%arg0, %arg1 : memref<64x64xf32>, memref<64x64xf32>)
                outs(%arg2 : memref<64x64xf32>)
     return
  }
}

// CONF1-LABEL: func.func @matmul_no_tiling
// CONF1-NOT: scf.for
// CONF2-LABEL: func.func @matmul_no_tiling
// CONF2-NOT: scf.for

// -----

func.func @batch_matmul_no_tiling(%arg0: tensor<512x32x64xf32>, %arg1: tensor<512x64x32xf32>) -> tensor<512x32x32xf32> {
  %0 = tensor.empty() : tensor<512x32x32xf32>
  %1 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<512x32x64xf32>, tensor<512x64x32xf32>)
                           outs(%0 : tensor<512x32x32xf32>) -> tensor<512x32x32xf32>
  return %1 : tensor<512x32x32xf32>
}

// CONF1-LABEL: func.func @batch_matmul_no_tiling
// CONF1-NOT: scf.for
// CONF2-LABEL: func.func @batch_matmul_no_tiling
// CONF2-NOT: scf.for

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @generic_matmul_no_tiling(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  %c0 = arith.constant 0.0 : f32
  %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%0: tensor<128x128xf32>) {
    ^bb0(%out: f32):
      %2 = arith.maximumf %out, %c0 : f32
      linalg.yield %2 : f32
    } -> tensor<128x128xf32>
  return %1 : tensor<128x128xf32>
}

// CONF1-LABEL: func.func @generic_matmul_no_tiling
// CONF1-NOT: scf.for
// CONF2-LABEL: func.func @generic_matmul_no_tiling
// CONF2-NOT: scf.for

// -----
