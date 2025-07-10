// RUN: tpp-run %s -e entry --entry-point-result=void -seed 123 -print > %t.1
// RUN: tpp-opt %s -tpp-mapping -x86-vectorizer | \
// RUN:  tpp-run -e entry --entry-point-result=void -seed 123 -print > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

func.func @entry(
  %arg0: tensor<256x512xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<256x128xf32>)
    -> tensor<256x128xf32>
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [4, 16, 1]>,
      #dlti.dl_entry<"reg_gemm_unroll", [1, 16, 1]>
  >>}
{
  %0 = linalg.matmul
    ins(%arg0, %arg1: tensor<256x512xf32>, tensor<512x128xf32>)
    outs(%arg2: tensor<256x128xf32>)
    -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}
