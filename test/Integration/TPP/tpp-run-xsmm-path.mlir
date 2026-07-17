// RUN: tpp-run %s -e entry -entry-point-result=void | FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes | \
// RUN:   FileCheck %s -check-prefix=IR

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @add(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // IR: xsmm_binary_dispatch
  // IR: xsmm_binary_invoke
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1: tensor<2x2xf32>, tensor<2x2xf32>)
    outs(%arg1: tensor<2x2xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %0 = arith.addf %in, %in_1 : f32
        linalg.yield %0 : f32
  } -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

func.func @entry() {
  %cst_one = arith.constant 1.0 : f32
  %arg0 = tensor.empty() : tensor<2x2xf32>
  %fill = linalg.fill ins(%cst_one : f32) outs(%arg0 : tensor<2x2xf32>) -> tensor<2x2xf32>

  %cst_two = arith.constant 2.0 : f32
  %arg1 = bufferization.alloc_tensor() : tensor<2x2xf32>
  %fill_1 = linalg.fill ins(%cst_two : f32) outs(%arg1 : tensor<2x2xf32>) -> tensor<2x2xf32>

  %added = call @add(%fill, %fill_1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  // CHECK: ( ( 3, 3 ), ( 3, 3 ) )
  %v1 = vector.transfer_read %added[%c0, %c0], %d1 : tensor<2x2xf32>, vector<2x2xf32>
  vector.print %v1 : vector<2x2xf32>

  return
}
