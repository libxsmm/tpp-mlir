// RUN: tpp-opt --default-tpp-passes="nano-kernel registerBlocking=32,32,64 gemm-unroll=16,16,16" %s | FileCheck %s -check-prefix=IR

// RUN: tpp-run -e entry --entry-point-result=void -print --splat-to-random --init-type quant -seed 123  %s > %t.1
// RUN: tpp-run -e entry --entry-point-result=void -print --nano-kernels --gemm-unroll=16,16,16 --registerBlocking=32,32,64 --splat-to-random -seed 123 --init-type quant %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

// IR-COUNT-4: amx.tile_zero
// IR-COUNT-4: amx.tile_muli
// IR-COUNT-4: amx.tile_store

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3, 0)>

// Requantize an i8xi8->i32 GEMM back to i8. The i32 accumulator is dequantized
// with the per-row input scale and per-output-channel weight scale, rescaled
// with the per-output-channel output scale, saturated to [-128, 127] and
// truncated to i8.
func.func @entry(
    %arg0: tensor<2x2x64x64xi8>,
    %arg1: tensor<128xf32>,
    %arg2: tensor<2x2x16x64x4xi8>,
    %arg3: tensor<128xf32>,
    %arg4: tensor<128xf32>,
    %arg5: tensor<2x2x64x64xi8>) -> tensor<2x2x64x64xi8> {

  %c0_i32 = arith.constant 0 : i32
  %lo = arith.constant -1.280000e+02 : f32
  %hi = arith.constant 1.270000e+02 : f32

  %0 = tensor.empty() : tensor<2x2x64x64xi32>

  %1 = linalg.fill
      ins(%c0_i32 : i32)
      outs(%0 : tensor<2x2x64x64xi32>)
      -> tensor<2x2x64x64xi32>

  %expanded = tensor.expand_shape %arg0
      [[0], [1], [2], [3, 4]]
      output_shape [2, 2, 64, 16, 4]
      : tensor<2x2x64x64xi8>
     into tensor<2x2x64x16x4xi8>

  %2 = linalg.contract
      indexing_maps = [#map, #map1, #map2]
      ins(%expanded, %arg2
          : tensor<2x2x64x16x4xi8>,
            tensor<2x2x16x64x4xi8>)
      outs(%1 : tensor<2x2x64x64xi32>)
      -> tensor<2x2x64x64xi32>

  %expanded_0 = tensor.expand_shape %arg1
      [[0, 1, 2, 3]]
      output_shape [2, 1, 64, 1]
      : tensor<128xf32>
     into tensor<2x1x64x1xf32>

  %expanded_1 = tensor.expand_shape %arg3
      [[0, 1, 2, 3]]
      output_shape [2, 1, 64, 1]
      : tensor<128xf32>
     into tensor<2x1x64x1xf32>

  %expanded_2 = tensor.expand_shape %arg4
      [[0, 1, 2, 3]]
      output_shape [2, 1, 64, 1]
      : tensor<128xf32>
     into tensor<2x1x64x1xf32>

  %3 = linalg.generic {
      indexing_maps = [#map3, #map4, #map5, #map5, #map3],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
    ins(%2, %expanded_0, %expanded_1, %expanded_2
        : tensor<2x2x64x64xi32>,
          tensor<2x1x64x1xf32>,
          tensor<2x1x64x1xf32>,
          tensor<2x1x64x1xf32>)
    outs(%arg5 : tensor<2x2x64x64xi8>) {
  ^bb0(%in: i32, %inS: f32, %wS: f32, %oS: f32, %out: i8):
    %4 = arith.sitofp %in : i32 to f32
    %5 = arith.mulf %inS, %wS : f32
    %6 = arith.mulf %5, %oS : f32
    %7 = arith.mulf %4, %6 : f32
    %8 = arith.maximumf %7, %lo : f32
    %9 = arith.minimumf %8, %hi : f32
    %10 = arith.fptosi %9 : f32 to i8
    linalg.yield %10 : i8
  } -> tensor<2x2x64x64xi8>

  return %3 : tensor<2x2x64x64xi8>
}
