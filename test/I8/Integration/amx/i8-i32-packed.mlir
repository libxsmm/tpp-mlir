// RUN: tpp-opt --default-tpp-passes="vector-to-kernel registerBlocking=32,32,64" %s | FileCheck %s -check-prefix=IR

// RUN: tpp-run -e entry --entry-point-result=void -print --splat-to-random --init-type mixed -seed 123  %s > %t.1
// RUN: tpp-opt --default-tpp-passes="vector-to-kernel registerBlocking=32,32,64" %s | tpp-run -e entry --target-feature="-avx, -avx2, +amx_tile, +amx_int8" --entry-point-result=void --print --splat-to-random -seed 123 --init-type mixed %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

// IR-COUNT-4: amx.tile_muli
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
module {
  func.func @entry(%arg0: tensor<2x2x32x64xi8>, %arg1: tensor<2x2x16x32x4xi8>, %arg2: tensor<2x2x32x32xi32>) -> tensor<2x2x32x32xi32> {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [2, 2, 32, 16, 4] : tensor<2x2x32x64xi8> into tensor<2x2x32x16x4xi8>
    %2 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%expanded, %arg1 : tensor<2x2x32x16x4xi8>, tensor<2x2x16x32x4xi8>) outs(%arg2 : tensor<2x2x32x32xi32>) -> tensor<2x2x32x32xi32>
    return %2 : tensor<2x2x32x32xi32>
  }
}