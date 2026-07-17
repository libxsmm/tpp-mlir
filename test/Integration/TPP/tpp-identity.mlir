// RUN: tpp-opt %s -default-tpp-passes | FileCheck -check-prefix=IR %s

// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

// RUN: tpp-run %s -linalg-to-loops \
// RUN:  -e entry -entry-point-result=void

#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// IR-LABEL: entry
func.func @entry(){
  %init_source = tensor.empty() : tensor<64xf32>
  %cst = arith.constant 23.0 : f32
  %input_tensor = linalg.fill ins(%cst : f32) outs(%init_source : tensor<64xf32>) -> tensor<64xf32>
  %1 = tensor.empty() : tensor<56x64xf32>
  // IR: xsmm_unary_invoke
  %2 = linalg.generic {indexing_maps=[#map, #map1],
                       iterator_types = ["parallel", "parallel"]}
    ins(%input_tensor : tensor<64xf32> ) outs(%1 : tensor<56x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
  } -> tensor<56x64xf32>
  %3 = tensor.empty() : tensor<56x64xf32>
  %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<56x64xf32>) -> tensor<56x64xf32>

  %threshold = arith.constant 0.0: f32
  check.expect_almost_eq(%2, %4, %threshold) : tensor<56x64xf32>, tensor<56x64xf32>, f32
  return
}
