// This should really be in the passes directory, not here
// RUN: tpp-opt %s -default-tpp-passes | FileCheck -check-prefix=IR %s

// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// IR-LABEL: gemm_tpp
func.func @gemm_tpp(%A: tensor<4x8xf32>,
           %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // IR: xsmm_gemm_invoke
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2],
                       iterator_types = ["parallel", "parallel", "reduction"]}
  ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %0 = arith.mulf %a, %b : f32
      %1 = arith.addf %c, %0 : f32
      linalg.yield %1 : f32
  } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

func.func @entry() {

  // Initialize various matrices.
  %cst_one = arith.constant 1.0 : f32
  %da = tensor.empty() : tensor<4x8xf32>
  %0 = linalg.fill ins(%cst_one : f32) outs(%da : tensor<4x8xf32>) -> tensor<4x8xf32>

  %cst_two = arith.constant 2.0 : f32
  %db = tensor.empty() : tensor<8x4xf32>
  %1 = linalg.fill ins(%cst_two : f32) outs(%db : tensor<8x4xf32>) -> tensor<8x4xf32>

  %cst_zero = arith.constant 0.0 : f32
  %C = tensor.empty() : tensor<4x4xf32>
  %2 = linalg.fill ins(%cst_zero : f32) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>

  // Call kernel.
  %res = call @gemm_tpp(%0, %1, %2)
       : (tensor<4x8xf32>, tensor<8x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

  // Print result.
  %zeroCst = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %res[%zeroCst, %zeroCst], %d1 : tensor<4x4xf32>, vector<4x4xf32>
  vector.print %v0 : vector<4x4xf32>
  //
  // CHECK:       ( 16,   16,   16,   16 ),
  // CHECK-SAME:  ( 16,   16,   16,   16 ),
  // CHECK-SAME:  ( 16,   16,   16,   16 ),
  // CHECK-SAME:  ( 16,   16,   16,   16 )
  //

  return
}
