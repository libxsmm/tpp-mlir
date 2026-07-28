// RUN: tpp-run %s -e big_matmul --entry-point-result=void -print -n 1 --seed=123 2>&1 > %S/lower-pack-unpack-without-transpose-big_matmul-only-default-passes.out
// RUN: tpp-run --lower-pack-unpack-without-transpose %s -e big_matmul  --entry-point-result=void -print -n 1 --seed=123 2>&1 > %S/lower-pack-unpack-without-transpose-big_matmul-default-passes-lower-pack-unpack-without-transpose.out
// RUN: fpcmp -r 0.001 %S/lower-pack-unpack-without-transpose-big_matmul-only-default-passes.out %S/lower-pack-unpack-without-transpose-big_matmul-default-passes-lower-pack-unpack-without-transpose.out
// RUN: rm %S/lower-pack-unpack-without-transpose-big_matmul-only-default-passes.out %S/lower-pack-unpack-without-transpose-big_matmul-default-passes-lower-pack-unpack-without-transpose.out

func.func @big_matmul(%A: tensor<256x128xf32>, %B: tensor<128x64xf32>, %C: tensor<256x64xf32>) -> tensor<256x64xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<256x128xf32>, tensor<128x64xf32>) outs(%C: tensor<256x64xf32>) -> tensor<256x64xf32>
  return %D : tensor<256x64xf32>
}
