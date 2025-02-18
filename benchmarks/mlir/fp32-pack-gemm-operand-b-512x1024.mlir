// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 2097152

func.func @entry(%arg0: tensor<1024x512xf32>, %arg1: tensor<16x32x32x32xf32>) -> tensor<16x32x32x32xf32> {
  %0 = linalg.pack %arg0
    outer_dims_perm = [1, 0]
    inner_dims_pos = [0, 1]
    inner_tiles = [32, 32]
    into %arg1 : tensor<1024x512xf32> -> tensor<16x32x32x32xf32>
  return %0 : tensor<16x32x32x32xf32>
}
