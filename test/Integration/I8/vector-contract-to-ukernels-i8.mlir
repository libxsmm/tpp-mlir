// RUN: tpp-run  -e gemm_i8 --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run  -e gemm_i8 --entry-point-result=void --nano-kernels --registerBlocking=3,32,4 --gemm-unroll=1,8,1 -print  --splat-to-random --init-type normal  -seed 123 %s > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

func.func @gemm_i8(
    %arg0: tensor<2x24x8x4xi8>,
    %arg1: tensor<2x8x128x4xi8>,
    %arg2: tensor<24x128xi32>
) -> tensor<24x128xi32> {
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
    ins(%arg0, %arg1
      : tensor<2x24x8x4xi8>,
        tensor<2x8x128x4xi8>)
    outs(%arg2 : tensor<24x128xi32>) {
  ^bb0(%in: i8, %in_1: i8, %out: i32):
    %0 = arith.extsi %in : i8 to i32
    %1 = arith.extsi %in_1 : i8 to i32
    %2 = arith.muli %0, %1 : i32
    %3 = arith.addi %out, %2 : i32
    linalg.yield %3 : i32
  } -> tensor<24x128xi32>

  return %result : tensor<24x128xi32>
}
