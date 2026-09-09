// End-to-end correctness of quantized i8 nano (AMX) K-cache-blocking. The
// reduction (K = 4*64 = 256) has 4 batch-reduce blocks; blocking by 2 threads
// the wide i32 partial through a stationary i32 carrier (beta=1) and lets the
// last K-block requantize the full reduction back to i8.

// RUN: tpp-run %s -e entry --entry-point-result=void -print --splat-to-random --init-type quant -seed 123 > %t.1
// RUN: tpp-run %s -e entry --entry-point-result=void --nano-kernels --registerBlocking=32,32,64 --gemm-unroll=16,16,16 --k-cache-blocking=2 -print --splat-to-random --init-type quant -seed 123 > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3, 0)>
module {
  func.func @entry(%arg0: tensor<2x4x32x64xi8>, %arg1: tensor<64xf8E8M0FNU>, %arg2: tensor<2x4x16x32x4xi8>, %arg3: tensor<64xf8E8M0FNU>, %arg4: tensor<64xf8E8M0FNU>, %arg5: tensor<2x2x32x32xi8>) -> tensor<2x2x32x32xi8> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<2x2x32x32xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<2x2x32x32xi32>) -> tensor<2x2x32x32xi32>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [2, 4, 32, 16, 4] : tensor<2x4x32x64xi8> into tensor<2x4x32x16x4xi8>
    %2 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%expanded, %arg2 : tensor<2x4x32x16x4xi8>, tensor<2x4x16x32x4xi8>) outs(%1 : tensor<2x2x32x32xi32>) -> tensor<2x2x32x32xi32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 1, 32, 1] : tensor<64xf8E8M0FNU> into tensor<2x1x32x1xf8E8M0FNU>
    %expanded_1 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [2, 1, 32, 1] : tensor<64xf8E8M0FNU> into tensor<2x1x32x1xf8E8M0FNU>
    %expanded_2 = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [2, 1, 32, 1] : tensor<64xf8E8M0FNU> into tensor<2x1x32x1xf8E8M0FNU>
    %cst = arith.constant -1.280000e+02 : f32
    %cst_3 = arith.constant 1.270000e+02 : f32
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %expanded_0, %expanded_1, %expanded_2 : tensor<2x2x32x32xi32>, tensor<2x1x32x1xf8E8M0FNU>, tensor<2x1x32x1xf8E8M0FNU>, tensor<2x1x32x1xf8E8M0FNU>) outs(%arg5 : tensor<2x2x32x32xi8>) {
    ^bb0(%in: i32, %in_4: f8E8M0FNU, %in_5: f8E8M0FNU, %in_6: f8E8M0FNU, %out: i8):
      %4 = arith.sitofp %in : i32 to f32
      %5 = arith.extf %in_4 : f8E8M0FNU to f32
      %6 = arith.extf %in_5 : f8E8M0FNU to f32
      %7 = arith.extf %in_6 : f8E8M0FNU to f32
      %8 = arith.mulf %5, %6 : f32
      %cst_7 = arith.constant 1.000000e+00 : f32
      %9 = arith.divf %cst_7, %7 : f32
      %10 = arith.mulf %8, %9 : f32
      %11 = arith.mulf %4, %10 : f32
      %12 = arith.maximumf %11, %cst : f32
      %13 = arith.minimumf %12, %cst_3 : f32
      %14 = arith.fptosi %13 : f32 to i8
      linalg.yield %14 : i8
    } -> tensor<2x2x32x32xi8>
    return %3 : tensor<2x2x32x32xi8>
  }
}
