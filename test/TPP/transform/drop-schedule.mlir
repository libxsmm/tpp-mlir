// RUN: tpp-opt -transform-dialect-interpreter -transform-drop-schedule %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @relu(%arg0: tensor<128x128xf32>) -> tensor<128x128xf32> {
  // CHECK: tpp.relu
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%arg0: tensor<128x128xf32>) {
    ^bb0(%out: f32):
      %1 = arith.maxf %out, %c0 : f32
      linalg.yield %1 : f32
  } -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-NOT: transform.sequence
transform.sequence failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["func.func"]} in %arg1
    transform.structured.map_linalg_to_tpp %0
}
