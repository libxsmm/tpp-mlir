// RUN: tpp-run -e gemm32 --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e gemm32 --entry-point-result=void --disable-vnni-packing --vector-to-kernels --registerBlocking=32,32,32 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

func.func @gemm32(%arg0: memref<4x32x32xbf16>, %arg1: memref<4x32x32xbf16>, %arg2: memref<32x32xbf16>) -> memref<32x32xbf16> {
    linalg.generic {indexing_maps = [affine_map<(d0, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d2, d3, d4) -> (d0, d4, d3)>, affine_map<(d0, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<4x32x32xbf16>, memref<4x32x32xbf16>) outs(%arg2 : memref<32x32xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_1 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    }
  return %arg2 : memref<32x32xbf16>
}

// RUN: tpp-run -e gemm96 --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e gemm96 --entry-point-result=void --disable-vnni-packing --vector-to-kernels --registerBlocking=32,32,32 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

func.func @gemm96(%arg0: memref<4x96x96xbf16>, %arg1: memref<4x96x96xbf16>, %arg2: memref<96x96xbf16>) -> memref<96x96xbf16> {
    linalg.generic {indexing_maps = [affine_map<(d0, d2, d3, d4) -> (d0, d2, d4)>, affine_map<(d0, d2, d3, d4) -> (d0, d4, d3)>, affine_map<(d0, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<4x96x96xbf16>, memref<4x96x96xbf16>) outs(%arg2 : memref<96x96xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_1 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    }
  return %arg2 : memref<96x96xbf16>
}

// RUN: tpp-run -e mlp64 --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e mlp64 --entry-point-result=void --disable-vnni-packing --vector-to-kernels --registerBlocking=32,32,32 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.2
// RUN: tpp-run -e mlp64 --entry-point-result=void --disable-vnni-packing --vector-to-kernels --registerBlocking=32,64,32 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.3
// RUN: fpcmp -r 0.001 %t.1 %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.3

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @mlp64(%arg0: tensor<8x2x64x64xbf16>, %arg1: tensor<2x2x64x64xbf16>, %arg2: tensor<2x64xbf16>, %arg3: tensor<8x2x64x64xbf16>) -> tensor<8x2x64x64xbf16> {
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x2x64x64xbf16>, tensor<2x2x64x64xbf16>) outs(%arg3 : tensor<8x2x64x64xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %3 = arith.mulf %in, %in_0 : bf16
    %4 = arith.addf %out, %3 : bf16
    linalg.yield %4 : bf16
  } -> tensor<8x2x64x64xbf16>
  %1 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<2x64xbf16>) outs(%0 : tensor<8x2x64x64xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %3 = arith.addf %in, %out : bf16
    linalg.yield %3 : bf16
  } -> tensor<8x2x64x64xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %2 = linalg.generic {indexing_maps = [#map4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<8x2x64x64xbf16>) {
  ^bb0(%out: bf16):
    %3 = arith.maximumf %out, %cst : bf16
    linalg.yield %3 : bf16
  } -> tensor<8x2x64x64xbf16>
  return %2 : tensor<8x2x64x64xbf16>
}
