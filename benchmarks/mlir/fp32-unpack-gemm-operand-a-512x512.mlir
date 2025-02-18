// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void

// BENCH_TOTAL_FLOPS: 1048576

func.func @entry(%arg0: tensor<16x16x32x32xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %unpack = linalg.unpack %arg0
    inner_dims_pos = [0, 1]
    inner_tiles = [32, 32]
    into %arg1 : tensor<16x16x32x32xf32> -> tensor<512x512xf32>
  return %unpack : tensor<512x512xf32>
}
