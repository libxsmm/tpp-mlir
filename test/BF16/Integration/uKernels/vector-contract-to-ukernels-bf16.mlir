// RUN: tpp-run  -e gemm_bf16_dp_random_AB --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run  -e gemm_bf16_dp_random_AB --entry-point-result=void --vector-to-kernels --registerBlocking=8,32,2 -print  --splat-to-random --init-type normal  -seed 123 %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

memref.global "private" constant @__constant_2x16x32x2xbf16 : memref<2x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
func.func @gemm_bf16_dp_random_AB(%arg0: memref<8x2x32x32xbf16>) -> memref<8x2x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = memref.get_global @__constant_2x16x32x2xbf16 : memref<2x16x32x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x2x32x32xbf16>
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 2, 32, 16, 2] : memref<8x2x32x32xbf16> into memref<8x2x32x16x2xbf16>
  scf.forall (%arg1, %arg2) in (8, 2) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
    %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 2, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x2x32x16x2xbf16> to memref<2x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<2x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<2x16x32x2xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_1 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    }
  }
  return %alloc : memref<8x2x32x32xbf16>
}


// RUN: tpp-run -e gemm_bf16_args --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e gemm_bf16_args --entry-point-result=void  --vector-to-kernels --registerBlocking=8,32,2 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2

func.func @gemm_bf16_args(%arg0: memref<4x2x64x64xbf16>, %arg1: memref<2x2x32x64x2xbf16>, %arg2: memref<4x2x64x64xbf16>) -> memref<4x2x64x64xbf16> {
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [4, 2, 64, 32, 2] : memref<4x2x64x64xbf16> into memref<4x2x64x32x2xbf16>
  scf.forall (%arg3, %arg4) in (4, 2) {
    %subview = memref.subview %expand_shape[%arg3, 0, 0, 0, 0] [1, 2, 64, 32, 2] [1, 1, 1, 1, 1] : memref<4x2x64x32x2xbf16> to memref<2x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0, 0] [1, 2, 32, 64, 2] [1, 1, 1, 1, 1] : memref<2x2x32x64x2xbf16> to memref<2x32x64x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<4x2x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview, %subview_0 : memref<2x64x32x2xbf16, strided<[4096, 64, 2, 1], offset: ?>>, memref<2x32x64x2xbf16, strided<[4096, 128, 2, 1], offset: ?>>) outs(%subview_1 : memref<64x64xbf16, strided<[64, 1], offset: ?>>) {
    ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
      %0 = arith.mulf %in, %in_2 : bf16
      %1 = arith.addf %out, %0 : bf16
      linalg.yield %1 : bf16
    }
  }
  %alloc = memref.alloc() : memref<4x2x64x64xbf16>
  memref.copy %arg2, %alloc : memref<4x2x64x64xbf16> to memref<4x2x64x64xbf16>
  return %alloc : memref<4x2x64x64xbf16>
}


// RUN: tpp-run -e mlp_bf16 --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e  mlp_bf16 --entry-point-result=void --vector-to-kernels --registerBlocking=4,32,2 -print  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2

memref.global "private" constant @__constant_32xbf16 : memref<32xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
func.func @mlp_bf16(%arg0: memref<8x2x32x32xbf16>) -> memref<8x2x32x32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = memref.get_global @__constant_32xbf16 : memref<32xbf16>
  %1 = memref.get_global @__constant_2x16x32x2xbf16 : memref<2x16x32x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x2x32x32xbf16>
  %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 2, 32, 16, 2] : memref<8x2x32x32xbf16> into memref<8x2x32x16x2xbf16>
  scf.forall (%arg1, %arg2) in (8, 2) {
    %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x2x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
    linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
    %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 2, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x2x32x16x2xbf16> to memref<2x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %1 : memref<2x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<2x16x32x2xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %2 = arith.mulf %in, %in_1 : bf16
      %3 = arith.addf %out, %2 : bf16
      linalg.yield %3 : bf16
    }
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<32xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%in: bf16, %out: bf16):
      %2 = arith.addf %in, %out : bf16
      linalg.yield %2 : bf16
    }
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
    ^bb0(%out: bf16):
      %2 = arith.maximumf %out, %cst : bf16
      linalg.yield %2 : bf16
    }
  }
  return %alloc : memref<8x2x32x32xbf16>
}

// RUN: tpp-run -e optimal_register_blocking --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e optimal_register_blocking --entry-point-result=void --vector-to-kernels --registerBlocking=6,64,2 -print  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: tpp-run -e optimal_register_blocking --entry-point-result=void --vector-to-kernels --registerBlocking=3,32,2 -print  --splat-to-random --init-type normal  -seed 123 %s  > %t.3
// RUN: fpcmp -r 0.01 %t.1 %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.3

memref.global "private" constant @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
func.func @optimal_register_blocking(%arg0: memref<2x24x16x2xbf16>) -> memref<24x128xbf16> {
  %0 = memref.get_global @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<24x128xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  linalg.fill ins(%cst : bf16) outs(%alloc : memref<24x128xbf16>)

    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %0 : memref<2x24x16x2xbf16>, memref<2x16x128x2xbf16>) outs(%alloc : memref<24x128xbf16>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
      %1 = arith.mulf %in, %in_1 : bf16
      %2 = arith.addf %out, %1 : bf16
      linalg.yield %2 : bf16
    }
  return %alloc : memref<24x128xbf16>
}

// RUN: tpp-run -e gemm_splat --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-opt --tile-brgemm-linalg="registerBlocking=9,48,2"  --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer -vector-contract-to-micro-kernels | tpp-run -e gemm_splat --entry-point-result=void -print  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2
func.func @gemm_splat(%arg0: memref<1x9x32xbf16>, %arg1: memref<1x32x48xbf16>, %arg2: memref<9x48xbf16>) -> memref<9x48xbf16> {
      linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<1x9x32xbf16>, memref<1x32x48xbf16>) outs(%arg2 : memref<9x48xbf16>)
      return %arg2 : memref<9x48xbf16>
}

// RUN: tpp-run -e mlp_splat --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e mlp_splat --entry-point-result=void -print --disable-vnni-packing --vector-to-kernels --registerBlocking=2,32,2  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2
#map_splat = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map_splat1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map_splat2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map_splat3 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map_splat4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @mlp_splat(%arg0: tensor<8x32x32x32xbf16>, %arg1: tensor<32x32x32x32xbf16>, %arg2: tensor<32x32xbf16>, %arg3: tensor<8x32x32x32xbf16>) -> tensor<8x32x32x32xbf16> {
  %0 = linalg.generic {indexing_maps = [#map_splat, #map_splat1, #map_splat2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%arg3 : tensor<8x32x32x32xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %3 = arith.mulf %in, %in_0 : bf16
    %4 = arith.addf %out, %3 : bf16
    linalg.yield %4 : bf16
  } -> tensor<8x32x32x32xbf16>
  %1 = linalg.generic {indexing_maps = [#map_splat3, #map_splat4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<32x32xbf16>) outs(%0 : tensor<8x32x32x32xbf16>) {
  ^bb0(%in: bf16, %out: bf16):
    %3 = arith.addf %in, %out : bf16
    linalg.yield %3 : bf16
  } -> tensor<8x32x32x32xbf16>
  %cst = arith.constant 0.000000e+00 : bf16
  %2 = linalg.generic {indexing_maps = [#map_splat4], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%1 : tensor<8x32x32x32xbf16>) {
  ^bb0(%out: bf16):
    %3 = arith.maximumf %out, %cst : bf16
    linalg.yield %3 : bf16
  } -> tensor<8x32x32x32xbf16>
  return %2 : tensor<8x32x32x32xbf16>
}
