// RUN: tpp-run %s -e entry -entry-point-result=void | FileCheck %s
// RUN: tpp-run %s -linalg-to-loops -e entry -entry-point-result=void | FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes | FileCheck %s -check-prefix=IR
// RUN: tpp-sched --bundle=default-tpp-passes %s | FileCheck %s -check-prefix=IR

!A_tensor_t = tensor<4x8xf32>
!B_tensor_t = tensor<8x16xf32>
!C_tensor_t = tensor<4x16xf32>
!D_tensor_t = tensor<4x16xf32>

#map = affine_map<(i, ii, k, kk, j, jj) -> (i, ii, k, kk)>
#map1 = affine_map<(i, ii, k, kk, j, jj) -> (k, kk, j, jj)>
#map2 = affine_map<(i, ii, k, kk, j, jj) -> (i, ii, j, jj)>
#map3 = affine_map<(i, ii, j, jj) -> (i, ii, j, jj)>

func.func @matmul_static(%A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t, %D : !D_tensor_t) {
  %A_exp = tensor.expand_shape %A [[0, 1], [2, 3]] output_shape[2, 2, 2, 4] :
    !A_tensor_t into tensor<2x2x2x4xf32>
  %B_exp = tensor.expand_shape %B [[0, 1], [2, 3]] output_shape[2, 4, 8, 2] :
    !B_tensor_t into tensor<2x4x8x2xf32>
  %C_exp = tensor.expand_shape %C [[0, 1], [2, 3]] output_shape[2, 2, 8, 2] :
    !C_tensor_t into tensor<2x2x8x2xf32>
  %D_exp = tensor.expand_shape %D [[0, 1], [2, 3]] output_shape[2, 2, 8, 2] :
    !D_tensor_t into tensor<2x2x8x2xf32>

  // IR-DAG: %[[C1:.+]] = arith.constant 1 : i64
  // IR-DAG: %[[C2:.+]] = arith.constant 2 : i64
  // IR-DAG: %[[C4:.+]] = arith.constant 4 : i64
  // IR-DAG: %[[C8:.+]] = arith.constant 8 : i64
  // IR-DAG: %[[C16:.+]] = arith.constant 16 : i64
  // IR-DAG: %[[C64:.+]] = arith.constant 64 : i64
  // IR-DAG: %[[C0:.+]] = arith.constant 0 : i64
  // IR: xsmm_brgemm_dispatch(%[[C1]], %[[C2]], %[[C2]], %[[C4]], %[[C8]], %[[C16]], %[[C2]], %[[C4]], %[[C64]], %[[C0]])
  // Parameters:
  // 1) kind
  // 2) m = 2
  // 3) n = 2
  // 4) k = 4
  // 5) lda = 8
  // 6) ldb = 16
  // 7) ldc = 2
  // 8) stride on A = 4
  // 9) stride on B = 64
  // 10) data type
  %gemm = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel"]}
    ins(%A_exp, %B_exp : tensor<2x2x2x4xf32>, tensor<2x4x8x2xf32>)
    outs(%C_exp : tensor<2x2x8x2xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %4 = arith.mulf %in, %in_2 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
  } -> tensor<2x2x8x2xf32>

  %bias = linalg.generic {
    indexing_maps = [#map3, #map3],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%gemm : tensor<2x2x8x2xf32>)
    outs(%D_exp : tensor<2x2x8x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %4 = arith.addf %in, %out : f32
      linalg.yield %4 : f32
    } -> tensor<2x2x8x2xf32>

  %bias_clps = tensor.collapse_shape %bias [[0, 1], [2, 3]] :
    tensor<2x2x8x2xf32> into !D_tensor_t
  %cst = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %bias_clps[%cst, %cst], %d1 : tensor<4x16xf32>, vector<4x16xf32>

  //
  // CHECK:     ( ( 59.46, 97.26, 135.06, 172.86, 210.66, 248.46, 286.26, 324.06, 361.86, 399.66, 437.46, 475.26, 513.06, 550.86, 588.66, 626.46 ),
  // CHECK-SAME:  ( 60.62, 99.22, 137.82, 176.42, 215.02, 253.62, 292.22, 330.82, 369.42, 408.02, 446.62, 485.22, 523.82, 562.42, 601.02, 639.62 ),
  // CHECK-SAME:  ( 61.78, 101.18, 140.58, 179.98, 219.38, 258.78, 298.18, 337.58, 376.98, 416.38, 455.78, 495.18, 534.58, 573.98, 613.38, 652.78 ),
  // CHECK-SAME:  ( 62.94, 103.14, 143.34, 183.54, 223.74, 263.94, 304.14, 344.34, 384.54, 424.74, 464.94, 505.14, 545.34, 585.54, 625.74, 665.94 ) )
  //
  vector.print %v0 : vector<4x16xf32>

  return
}

func.func @entry() {
  %C = arith.constant dense<0.0> : !C_tensor_t
  %D = arith.constant dense<[
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9, 11.9, 12.9, 13.9, 14.9, 15.9, 16.9 ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9, 11.9, 12.9, 13.9, 14.9, 15.9, 16.9 ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9, 11.9, 12.9, 13.9, 14.9, 15.9, 16.9 ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9, 11.9, 12.9, 13.9, 14.9, 15.9, 16.9 ]
  ]> : !D_tensor_t
  %A = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ]
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
  call @matmul_static(%A, %B, %C, %D) : (!A_tensor_t, !B_tensor_t, !C_tensor_t, !D_tensor_t) -> ()
  return
}
