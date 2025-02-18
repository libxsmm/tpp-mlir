// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 2097152

func.func @entry(%arg0: tensor<512x1024xf32>, %arg1: tensor<16x32x32x32xf32>) -> tensor<16x32x32x32xf32> {
  %pack = linalg.pack %arg0
    inner_dims_pos = [0, 1]
    inner_tiles = [32, 32]
    into %arg1 : tensor<512x1024xf32> -> tensor<16x32x32x32xf32>
  return %pack : tensor<16x32x32x32xf32>
}
