// RUN: tpp-run %s -n 10 \
// RUN:  -e entry -entry-point-result=void -print \
//
// Total flops = O(2*n*k*m) = 2*64x96x48 = 589824
// BENCH_TOTAL_FLOPS: 589824

func.func @entry(%A: tensor<64x96xf32>, %B: tensor<96x48xf32>,
                  %C: tensor<64x48xf32>) -> tensor<64x48xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<64x96xf32>, tensor<96x48xf32>) outs(%C: tensor<64x48xf32>) -> tensor<64x48xf32>
  return %D : tensor<64x48xf32>
}
