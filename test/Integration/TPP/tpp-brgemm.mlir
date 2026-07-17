// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes | FileCheck %s -check-prefix=IR

// IR-LABEL: brgemm_tpp
func.func @brgemm_tpp(%A: tensor<1x4x8xf32>,
                     %B: tensor<1x8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32>  {
  // IR: xsmm_gemm_invoke
  %D = linalg.batch_reduce_matmul ins(%A, %B: tensor<1x4x8xf32>, tensor<1x8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D: tensor<4x4xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  // Initialize various tensors.
  %da = arith.constant dense<[[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ]
  ]]> : tensor<1x4x8xf32>

  %db = arith.constant dense<[[
        [ 10.1, 11.1, 12.1, 13.1 ],
        [ 10.2, 11.2, 12.2, 13.2 ],
        [ 10.3, 11.3, 12.3, 13.3 ],
        [ 10.4, 11.4, 12.4, 13.4 ],
        [ 10.5, 11.5, 12.5, 13.5 ],
        [ 10.6, 11.6, 12.6, 13.6 ],
        [ 10.7, 11.7, 12.7, 13.7 ],
        [ 10.8, 11.8, 12.8, 13.8 ]
  ]]> : tensor<1x8x4xf32>

  // Call kernel.
  %C = arith.constant dense<0.0> : tensor<4x4xf32>
  %0 = call @brgemm_tpp(%da, %db, %C)
      : (tensor<1x4x8xf32>, tensor<1x8x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>

  //
  // CHECK:    ( ( 388.76, 425.56, 462.36, 499.16 ),
  // CHECK-SAME: ( 397.12, 434.72, 472.32, 509.92 ),
  // CHECK-SAME: ( 405.48, 443.88, 482.28, 520.68 ),
  // CHECK-SAME: ( 413.84, 453.04, 492.24, 531.44 ) )
  //
  %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<4x4xf32>, vector<4x4xf32>
  vector.print %v0 : vector<4x4xf32>

  return
}
