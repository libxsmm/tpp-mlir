// RUN: tpp-run %s -e entry --entry-point-result=void -seed 123 -print > %t.1
// RUN: tpp-opt %s -tpp-mapping -x86-vectorizer | \
// RUN:  tpp-run -e entry --entry-point-result=void -seed 123 -print > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @entry(
  %arg0: tensor<256x512xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<256x128xf32>,
  %arg3: tensor<128xf32>) -> tensor<256x128xf32>
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [4, 16, 1]>,
      #dlti.dl_entry<"reg_gemm_unroll", [1, 16, 1]>
  >>}
{
  %0 = linalg.matmul
    ins(%arg0, %arg1: tensor<256x512xf32>, tensor<512x128xf32>)
    outs(%arg2: tensor<256x128xf32>) -> tensor<256x128xf32>
  %1 = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg3 : tensor<128xf32>)
    outs(%0 : tensor<256x128xf32>) {
  ^bb0(%in: f32, %out: f32):
      %3 = arith.addf %in, %out : f32
      linalg.yield %3 : f32
  } -> tensor<256x128xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel"]}
    outs(%1 : tensor<256x128xf32>) {
  ^bb0(%out: f32):
    %3 = arith.maximumf %out, %cst : f32
    linalg.yield %3 : f32
  } -> tensor<256x128xf32>
  return %2 : tensor<256x128xf32>
}
