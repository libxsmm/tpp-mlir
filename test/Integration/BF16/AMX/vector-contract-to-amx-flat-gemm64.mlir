// RUN: tpp-run -e gemm64 --entry-point-result=void --disable-vnni-packing -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e gemm64 --entry-point-result=void --disable-vnni-packing --nano-kernels --gemm-unroll=16,16,32 --registerBlocking=32,32,32 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

func.func @gemm64(
    %arg0: tensor<4x64x64xbf16>,
    %arg1: tensor<4x64x64xbf16>,
    %arg2: tensor<64x64xbf16>
) -> tensor<64x64xbf16> {
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d2, d3, d4) -> (d0, d2, d4)>,
        affine_map<(d0, d2, d3, d4) -> (d0, d4, d3)>,
        affine_map<(d0, d2, d3, d4) -> (d2, d3)>
      ],
      iterator_types = ["reduction", "parallel", "parallel", "reduction"]
    }
    ins(%arg0, %arg1 : tensor<4x64x64xbf16>, tensor<4x64x64xbf16>)
    outs(%arg2 : tensor<64x64xbf16>) {
  ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
    %mul = arith.mulf %in, %in_1 : bf16
    %add = arith.addf %out, %mul : bf16
    linalg.yield %add : bf16
  } -> tensor<64x64xbf16>

  return %result : tensor<64x64xbf16>
}
