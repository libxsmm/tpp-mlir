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

// Requantize an i8xi8->i32 GEMM back to i8 using narrow (f8E8M0FNU) scales.
// The i32 accumulator is dequantized with the per-row input scale and
// per-output-channel weight scale, rescaled with the per-output-channel output
// scale, saturated to [-128, 127] and truncated to i8. Each narrow scale is
// extended to f32 before being combined and applied.
func.func @entry(
    %arg0: tensor<2x2x64x64xi8>,
    %arg1: tensor<128xf8E8M0FNU> {tpp.quant_scale},
    %arg2: tensor<2x2x16x64x4xi8>,
    %arg3: tensor<128xf8E8M0FNU> {tpp.quant_scale},
    %arg4: tensor<128xf8E8M0FNU> {tpp.quant_scale},
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
      : tensor<128xf8E8M0FNU>
     into tensor<2x1x64x1xf8E8M0FNU>

  %expanded_1 = tensor.expand_shape %arg3
      [[0, 1, 2, 3]]
      output_shape [2, 1, 64, 1]
      : tensor<128xf8E8M0FNU>
     into tensor<2x1x64x1xf8E8M0FNU>

  %expanded_2 = tensor.expand_shape %arg4
      [[0, 1, 2, 3]]
      output_shape [2, 1, 64, 1]
      : tensor<128xf8E8M0FNU>
     into tensor<2x1x64x1xf8E8M0FNU>

  %3 = linalg.generic {
      indexing_maps = [#map3, #map4, #map5, #map5, #map3],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
    ins(%2, %expanded_0, %expanded_1, %expanded_2
        : tensor<2x2x64x64xi32>,
          tensor<2x1x64x1xf8E8M0FNU>,
          tensor<2x1x64x1xf8E8M0FNU>,
          tensor<2x1x64x1xf8E8M0FNU>)
    outs(%arg5 : tensor<2x2x64x64xi8>) {
  ^bb0(%in: i32, %inS: f8E8M0FNU, %wS: f8E8M0FNU, %oS: f8E8M0FNU, %out: i8):
    %4 = arith.sitofp %in : i32 to f32
    %5 = arith.extf %inS : f8E8M0FNU to f32
    %6 = arith.extf %wS : f8E8M0FNU to f32
    %7 = arith.extf %oS : f8E8M0FNU to f32
    %8 = arith.mulf %5, %6 : f32
    %9 = arith.mulf %8, %7 : f32
    %10 = arith.mulf %4, %9 : f32
    %11 = arith.maximumf %10, %lo : f32
    %12 = arith.minimumf %11, %hi : f32
    %13 = arith.fptosi %12 : f32 to i8
    linalg.yield %13 : i8
  } -> tensor<2x2x64x64xi8>

  return %3 : tensor<2x2x64x64xi8>
}
