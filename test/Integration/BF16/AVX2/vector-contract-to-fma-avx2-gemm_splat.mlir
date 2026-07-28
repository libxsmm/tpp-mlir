// RUN: tpp-run -e gemm_splat --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e gemm_splat --entry-point-result=void -print --disable-vnni-packing --nano-kernels --gemm-unroll=1,8,1 --registerBlocking=4,16,1  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.2

func.func @gemm_splat(%arg0: tensor<8x32x32x32xbf16>, %arg1: tensor<32x32x32x32xbf16>, %arg2: tensor<8x32x32x32xbf16>) -> tensor<8x32x32x32xbf16> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x32x32x32xbf16>, tensor<32x32x32x32xbf16>) outs(%arg2 : tensor<8x32x32x32xbf16>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
    %1 = arith.mulf %in, %in_0 : bf16
    %2 = arith.addf %out, %1 : bf16
    linalg.yield %2 : bf16
  } -> tensor<8x32x32x32xbf16>
  return %0 : tensor<8x32x32x32xbf16>
}
