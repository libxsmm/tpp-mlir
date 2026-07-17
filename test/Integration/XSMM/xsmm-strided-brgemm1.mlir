// RUN: tpp-run %s -e entry -entry-point-result=void | FileCheck %s
// RUN: tpp-run %s -linalg-to-loops -e entry -entry-point-result=void | FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes | FileCheck %s -check-prefix=IR

!A_tensor_t = tensor<16x8xf32>
!B_tensor_t = tensor<8x16xf32>
!C_tensor_t = tensor<4x16xf32>

#map = affine_map<(b, i, h, k, j) -> (b, h, i, k)>
#map1 = affine_map<(b, i, h, k, j) -> (b, k, h, j)>
#map2 = affine_map<(b, i, h, k, j) -> (b, i, h, j)>

// IR-LABEL: matmul_static
func.func @matmul_static(%A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t) {
  %A_exp = tensor.expand_shape %A [[0, 1], [2, 3]] output_shape[2, 8, 2, 4] :
    !A_tensor_t into tensor<2x8x2x4xf32>
  %B_exp = tensor.expand_shape %B [[0, 1], [2, 3]] output_shape[2, 4, 8, 2] :
    !B_tensor_t into tensor<2x4x8x2xf32>
  %C_exp = tensor.expand_shape %C [[0, 1], [2, 3]] output_shape[2, 2, 8, 2] :
    !C_tensor_t into tensor<2x2x8x2xf32>

  %cst_fill = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x2x8x2xf32>
  %fill = linalg.fill ins(%cst_fill : f32) outs(%empty: tensor<2x2x8x2xf32>) -> tensor<2x2x8x2xf32>

  // IR-DAG: %[[C2:.+]] = arith.constant 2 : i64
  // IR-DAG: %[[C1:.+]] = arith.constant 1 : i64
  // IR-DAG: %[[C16:.+]] = arith.constant 16 : i64
  // IR-DAG: %[[C4:.+]] = arith.constant 4 : i64
  // IR-NOT: xsmm_unary_dispatch
  // IR: xsmm_gemm_dispatch(%[[C1]], %[[C2]], %[[C2]], %[[C4]], %[[C4]], %[[C16]], %[[C16]], %[[C4]])
  %gemm = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel"]}
    ins(%A_exp, %B_exp : tensor<2x8x2x4xf32>, tensor<2x4x8x2xf32>)
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
  // CHECK:     ( ( 13.5, 23.9, 35.6, 46.4, 59.3, 70.5, 84.6, 96.2, 111.5, 123.5, 140, 152.4, 170.1, 182.9, 201.8, 215 ),
  // CHECK-SAME:  ( 33.5, 59.9, 87.6, 114.4, 143.3, 170.5, 200.6, 228.2, 259.5, 287.5, 320, 348.4, 382.1, 410.9, 445.8, 475 ),
  // CHECK-SAME:  ( 22.94, 36.54, 38.46, 48.86, 59.486, 69.926, 80.672, 91.152, 102.018, 112.538, 123.524, 134.084, 145.19, 155.79, 167.016, 177.656 ),
  // CHECK-SAME:  ( 49.34, 78.94, 96.86, 123.26, 149.886, 176.326, 203.072, 229.552, 256.418, 282.938, 309.924, 336.484, 363.59, 390.19, 417.416, 444.056 ) )
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
  ]> : !A_tensor_t
  %B = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 12.3, 13.3, 14.3, 15.3, 16.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4, 16.4 ],
        [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5 ],
        [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6, 16.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 9.7, 10.7, 11.7, 12.7, 13.7, 14.7, 15.7, 16.7 ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8, 16.8 ]
  ]> : !B_tensor_t
  call @matmul_static(%A, %B, %C) : (!A_tensor_t, !B_tensor_t, !C_tensor_t) -> ()
  return
}
