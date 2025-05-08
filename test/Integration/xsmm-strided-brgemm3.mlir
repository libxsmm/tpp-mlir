// RUN: tpp-run %s -e entry -entry-point-result=void | FileCheck %s
// RUN: tpp-run %s -linalg-to-loops -e entry -entry-point-result=void | FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes | FileCheck %s -check-prefix=IR
// RUN: tpp-sched --bundle=default-tpp-passes %s | FileCheck %s -check-prefix=IR

!A_tensor_t = tensor<4x8xf32>
!B_tensor_t = tensor<16x8xf32>
!C_tensor_t = tensor<4x16xf32>

#map = affine_map<(b, i, h, k, j) -> (b, i, h, k)>
#map1 = affine_map<(b, i, h, k, j) -> (b, j, h, k)>
#map2 = affine_map<(b, i, h, k, j) -> (b, h, j, i)>

// IR-LABEL: matmul_static
func.func @matmul_static(%A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t) {
  %A_exp = tensor.expand_shape %A [[0, 1], [2, 3]] output_shape[2, 2, 2, 4] :
    !A_tensor_t into tensor<2x2x2x4xf32>
  %B_exp = tensor.expand_shape %B [[0, 1], [2, 3]] output_shape[2, 8, 2, 4] :
    !B_tensor_t into tensor<2x8x2x4xf32>
  %C_exp = tensor.expand_shape %C [[0, 1], [2, 3]] output_shape[2, 2, 8, 2] :
    !C_tensor_t into tensor<2x2x8x2xf32>

  %cst_fill = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x8x2xf32>
  %fill = linalg.fill ins(%cst_fill : f32) outs(%empty: tensor<2x2x8x2xf32>) -> tensor<2x2x8x2xf32>

  // IR-DAG: %[[C2:.+]] = arith.constant 2 : i64
  // IR-DAG: %[[C1:.+]] = arith.constant 1 : i64
  // IR-DAG: %[[C8:.+]] = arith.constant 8 : i64
  // IR-DAG: %[[C4:.+]] = arith.constant 4 : i64
  // IR-DAG: %[[C0:.+]] = arith.constant 0 : i64
  // The unary dispatch in this case is the transpose.
  // IR-COUNT-1: xsmm_unary_dispatch
  // IR: xsmm_gemm_dispatch(%[[C1]], %[[C8]], %[[C2]], %[[C4]], %[[C8]], %[[C2]], %[[C2]], %[[C4]])
  %gemm = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%A_exp, %B_exp : tensor<2x2x2x4xf32>, tensor<2x8x2x4xf32>)
    outs(%fill : tensor<2x2x8x2xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.mulf %in, %in_2 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
  } -> tensor<2x2x8x2xf32>

  %gemm_clps = tensor.collapse_shape %gemm [[0, 1], [2, 3]] :
    tensor<2x2x8x2xf32> into !C_tensor_t
  %cst = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %gemm_clps[%cst, %cst], %d1 : tensor<4x16xf32>, vector<4x16xf32>

  //
  // CHECK:     ( ( 32.04, 33.08, 33.08, 34.16, 34.12, 35.24, 35.16, 36.32, 36.2, 37.4, 37.24, 38.48, 38.28, 39.56, 39.32, 40.64 ),
  // CHECK-SAME:  ( 179.24, 181.88, 181.88, 184.56, 184.52, 187.24, 187.16, 189.92, 189.8, 192.6, 192.44, 195.28, 195.08, 197.96, 197.72, 200.64 ),
  // CHECK-SAME:  ( 43.08, 44.44, 34.12, 35.16, 34.232, 35.276, 34.344, 35.392, 34.456, 35.508, 34.568, 35.624, 34.68, 35.74, 34.792, 35.856 ),
  // CHECK-SAME:  ( 206.28, 209.24, 184.52, 187.16, 184.792, 187.436, 185.064, 187.712, 185.336, 187.988, 185.608, 188.264, 185.88, 188.54, 186.152, 188.816 ) )
  //
  vector.print %v0 : vector<4x16xf32>

  return
}

func.func @entry() {
  %C = arith.constant dense<0.0> : !C_tensor_t
  %A = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ]
  ]> : !A_tensor_t
  %B = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ],
        [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 ],
        [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7 ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8 ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9 ],
        [ 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10 ],
        [ 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11 ],
        [ 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12 ],
        [ 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13 ],
        [ 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14 ],
        [ 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15 ],
        [ 1.16, 2.16, 3.16, 4.16, 5.16, 6.16, 7.16, 8.16 ]
  ]> : !B_tensor_t
  call @matmul_static(%A, %B, %C) : (!A_tensor_t, !B_tensor_t, !C_tensor_t) -> ()
  return
}
