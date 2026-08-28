// RUN: tpp-opt %s -bufferize | FileCheck %s

#map = affine_map<(d0) -> (d0 * 32)>

func.func @pack_fusion(%arg0: tensor<1024x512xbf16>, %arg1: tensor<16x32x16x32x2xbf16>) -> tensor<16x32x16x32x2xbf16> {
  %0 = scf.forall (%arg2, %arg3) in (16, 32) shared_outs(%arg4 = %arg1) -> (tensor<16x32x16x32x2xbf16>) {
    // CHECK-NOT: memref.alloc
    // CHECK-NOT: memref.copy
    %1 = affine.apply #map(%arg3)
    %2 = affine.apply #map(%arg2)
    %extracted_slice = tensor.extract_slice %arg0[%1, %2] [32, 32] [1, 1] : tensor<1024x512xbf16> to tensor<32x32xbf16>
    %3 = tensor.empty() : tensor<16x32x2xbf16>
    %expanded = tensor.expand_shape %extracted_slice [[0, 1], [2]] output_shape [16, 2, 32] : tensor<32x32xbf16> into tensor<16x2x32xbf16>
    %transposed = linalg.transpose ins(%expanded : tensor<16x2x32xbf16>) outs(%3 : tensor<16x32x2xbf16>) permutation = [0, 2, 1]
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %transposed into %arg4[%arg2, %arg3, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] : tensor<16x32x2xbf16> into tensor<16x32x16x32x2xbf16>
    }
  }
  return %0 : tensor<16x32x16x32x2xbf16>
}
