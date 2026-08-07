// RUN: tpp-run -e entry --entry-point-result=void -print --splat-to-random --init-type quant -seed 123  %s > %t.1
// RUN: tpp-run -e entry --entry-point-result=void -print --nano-kernels --gemm-unroll=16,16,16 --registerBlocking=32,32,64 --splat-to-random -seed 123 --init-type quant %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

// IR-COUNT-4: amx.tile_zero
// IR-COUNT-4: amx.tile_muli
// IR-COUNT-4: amx.tile_store

// IR: scf.for
// IR-COUNT-2:     vector.transfer_read
// IR: vector.broadcast
// IR-COUNT-1:     vector.transfer_read
// IR: vector.shape_cast
// IR:     arith.mulf
// IR:     arith.sitofp
// IR:     arith.mulf
// IR:     vector.transfer_write
#afmap = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
#afmap1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
#afmap2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
#afmap3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#afmap4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
#afmap5 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3, 0)>

func.func @entry(
    %arg0: tensor<2x2x64x64xi8>,
    %arg1: tensor<128xf32> {tpp.quant_scale},
    %arg2: tensor<4x2x16x64x4xi8>,
    %arg3: tensor<256xf32> {tpp.quant_scale},
    %arg4: tensor<2x4x64x64xf32>) -> tensor<2x4x64x64xf32> {

  %c0_i32 = arith.constant 0 : i32

  %0 = tensor.empty() : tensor<2x4x64x64xi32>

  %1 = linalg.fill
      ins(%c0_i32 : i32)
      outs(%0 : tensor<2x4x64x64xi32>)
      -> tensor<2x4x64x64xi32>

  %expanded = tensor.expand_shape %arg0
      [[0], [1], [2], [3, 4]]
      output_shape [2, 2, 64, 16, 4]
      : tensor<2x2x64x64xi8>
     into tensor<2x2x64x16x4xi8>

  %2 = linalg.contract
      indexing_maps = [#afmap, #afmap1, #afmap2]
      ins(%expanded, %arg2
          : tensor<2x2x64x16x4xi8>,
            tensor<4x2x16x64x4xi8>)
      outs(%1 : tensor<2x4x64x64xi32>)
      -> tensor<2x4x64x64xi32>

  %expanded_0 = tensor.expand_shape %arg1
      [[0, 1, 2, 3]]
      output_shape [2, 1, 64, 1]
      : tensor<128xf32>
     into tensor<2x1x64x1xf32>

  %expanded_1 = tensor.expand_shape %arg3
      [[0, 1, 2, 3]]
      output_shape [4, 1, 64, 1]
      : tensor<256xf32>
     into tensor<4x1x64x1xf32>

  %3 = linalg.generic {
      indexing_maps = [#afmap3, #afmap4, #afmap5, #afmap3],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    }
    ins(%2, %expanded_0, %expanded_1
        : tensor<2x4x64x64xi32>,
          tensor<2x1x64x1xf32>,
          tensor<4x1x64x1xf32>)
    outs(%arg4 : tensor<2x4x64x64xf32>) {
  ^bb0(%in: i32, %in_2: f32, %in_3: f32, %out: f32):
    %4 = arith.mulf %in_2, %in_3 : f32
    %5 = arith.sitofp %in : i32 to f32
    %6 = arith.mulf %5, %4 : f32
    linalg.yield %6 : f32
  } -> tensor<2x4x64x64xf32>

  return %3 : tensor<2x4x64x64xf32>
}
