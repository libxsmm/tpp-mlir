// RUN: tpp-run -e optimal_register_blocking --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e optimal_register_blocking --entry-point-result=void --nano-kernels --gemm-unroll=1,16,1 --registerBlocking=6,64,2 -print  --splat-to-random --init-type normal  -seed 123 %s  > %t.2
// RUN: tpp-run -e optimal_register_blocking --entry-point-result=void --nano-kernels --gemm-unroll=1,16,1 --registerBlocking=3,32,2 -print  --splat-to-random --init-type normal  -seed 123 %s  > %t.3
// RUN: fpcmp -r 0.01 %t.1 %t.2
// RUN: fpcmp -r 0.01 %t.1 %t.3
func.func @optimal_register_blocking(
    %arg0: tensor<2x24x16x2xbf16>) -> tensor<24x128xbf16> {
  %weights = arith.constant
      dense<1.000000e+00> : tensor<2x16x128x2xbf16>
  %empty = tensor.empty() : tensor<24x128xbf16>
  %zero = arith.constant 0.000000e+00 : bf16

  %init = linalg.fill
      ins(%zero : bf16)
      outs(%empty : tensor<24x128xbf16>)
      -> tensor<24x128xbf16>

  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>,
        affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
      ],
      iterator_types = [
        "reduction",
        "reduction",
        "parallel",
        "parallel",
        "reduction"
      ]
    }
    ins(%arg0, %weights
        : tensor<2x24x16x2xbf16>,
          tensor<2x16x128x2xbf16>)
    outs(%init : tensor<24x128xbf16>) {
  ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
    %mul = arith.mulf %in, %in_1 : bf16
    %sum = arith.addf %out, %mul : bf16
    linalg.yield %sum : bf16
  } -> tensor<24x128xbf16>

  return %result : tensor<24x128xbf16>
}
