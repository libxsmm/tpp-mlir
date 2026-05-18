// RUN: tpp-opt --default-tpp-passes="vector-to-kernel registerBlocking=32,32,64" %s | FileCheck %s -check-prefix=IR

// RUN: tpp-run -e entry --entry-point-result=void --linalg-to-loops -print --splat-to-random --init-type quant -seed 123  %s > %t.1
// RUN: tpp-run -e entry --entry-point-result=void -print --vector-to-kernels --registerBlocking=32,32,64 --splat-to-random -seed 123 --init-type quant %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

// IR-COUNT-4: amx.tile_muli

// IR: scf.for
// IR-COUNT-2:     vector.transfer_read
// IR: vector.broadcast
// IR-COUNT-1:     vector.transfer_read
// IR: vector.shape_cast
// IR:     arith.extf
// IR:     arith.mulf
// IR:     arith.sitofp
// IR:     arith.mulf
// IR:     vector.transfer_write
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3, 0)>
module {
  func.func @entry(%arg0: tensor<4x36x32x64xi8>, %arg1: tensor<128xf8E8M0FNU>, %arg2: tensor<24x36x16x32x4xi8>, %arg3: tensor<768xf8E8M0FNU>, %arg4: tensor<4x24x32x32xf32>) -> tensor<4x24x32x32xf32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<4x24x32x32xi32>
    %1 = linalg.fill ins(%c0_i32 : i32) outs(%0 : tensor<4x24x32x32xi32>) -> tensor<4x24x32x32xi32>
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 36, 32, 16, 4] : tensor<4x36x32x64xi8> into tensor<4x36x32x16x4xi8>
    %2 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%expanded, %arg2 : tensor<4x36x32x16x4xi8>, tensor<24x36x16x32x4xi8>) outs(%1 : tensor<4x24x32x32xi32>) -> tensor<4x24x32x32xi32>
    %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [4, 1, 32, 1] : tensor<128xf8E8M0FNU> into tensor<4x1x32x1xf8E8M0FNU>
    %expanded_1 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [24, 1, 32, 1] : tensor<768xf8E8M0FNU> into tensor<24x1x32x1xf8E8M0FNU>
    %3 = linalg.generic {indexing_maps = [#map3, #map4, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %expanded_0, %expanded_1 : tensor<4x24x32x32xi32>, tensor<4x1x32x1xf8E8M0FNU>, tensor<24x1x32x1xf8E8M0FNU>) outs(%arg4 : tensor<4x24x32x32xf32>) {
    ^bb0(%in: i32, %in_2: f8E8M0FNU, %in_3: f8E8M0FNU, %out: f32):
      %4 = arith.extf %in_2 fastmath<nnan> : f8E8M0FNU to f32
      %5 = arith.extf %in_3 fastmath<nnan> : f8E8M0FNU to f32
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.sitofp %in : i32 to f32
      %8 = arith.mulf %7, %6 : f32
      linalg.yield %8 : f32
    } -> tensor<4x24x32x32xf32>
    return %3 : tensor<4x24x32x32xf32>
  }
}
