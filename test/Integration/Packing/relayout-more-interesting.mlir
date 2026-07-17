// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#accessesToBlock = [
  affine_map<(n1, c1, n2, c2) -> (n1 * 2 + n2, c1 * 2 + c2)>,
  affine_map<(n1, c1, n2, c2) -> (n1, c1, n2, c2)>
]

#traitToBlock = {
  iterator_types = ["parallel", "parallel", "parallel", "parallel"],
  indexing_maps = #accessesToBlock,
  library_call = "to-block-layout"
}

// To show that what we want to do is not a simple reshape but it requires copying.

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  %d = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 12.3, 13.3, 14.3, 15.3, 16.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4, 16.4 ],
        [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5 ],
        [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6, 16.6 ]
  ]> : tensor<6x16xf32>

  %0 = bufferization.alloc_tensor() : tensor<3x8x2x2xf32>
  %1 = linalg.generic #traitToBlock
      ins(%d: tensor<6x16xf32>)
      outs(%0: tensor<3x8x2x2xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):
          linalg.yield %arg2 : f32
  } -> tensor<3x8x2x2xf32>

  %v0 = vector.transfer_read %1[%c0, %c0, %c0, %c0], %d1 : tensor<3x8x2x2xf32>, vector<3x8x2x2xf32>
  //
  // CHECK:       ( ( ( ( 1.1, 2.1 ), ( 1.2, 2.2 ) ),
  // CHECK-SAME:      ( ( 3.1, 4.1 ), ( 3.2, 4.2 ) ),
  // CHECK-SAME:      ( ( 5.1, 6.1 ), ( 5.2, 6.2 ) ),
  // CHECK-SAME:      ( ( 7.1, 8.1 ), ( 7.2, 8.2 ) ),
  // CHECK-SAME:      ( ( 9.1, 10.1 ), ( 9.2, 10.2 ) ),
  // CHECK-SAME:      ( ( 11.1, 12.1 ), ( 11.2, 12.2 ) ),
  // CHECK-SAME:      ( ( 13.1, 14.1 ), ( 13.2, 14.2 ) ),
  // CHECK-SAME:      ( ( 15.1, 16.1 ), ( 15.2, 16.2 ) ) ),
  // CHECK-SAME:    ( ( ( 1.3, 2.3 ), ( 1.4, 2.4 ) ),
  // CHECK-SAME:      ( ( 3.3, 4.3 ), ( 3.4, 4.4 ) ),
  // CHECK-SAME:      ( ( 5.3, 6.3 ), ( 5.4, 6.4 ) ),
  // CHECK-SAME:      ( ( 7.3, 8.3 ), ( 7.4, 8.4 ) ),
  // CHECK-SAME:      ( ( 9.3, 10.3 ), ( 9.4, 10.4 ) ),
  // CHECK-SAME:      ( ( 11.3, 12.3 ), ( 11.4, 12.4 ) ),
  // CHECK-SAME:      ( ( 13.3, 14.3 ), ( 13.4, 14.4 ) ),
  // CHECK-SAME:      ( ( 15.3, 16.3 ), ( 15.4, 16.4 ) ) ),
  // CHECK-SAME:    ( ( ( 1.5, 2.5 ), ( 1.6, 2.6 ) ),
  // CHECK-SAME:      ( ( 3.5, 4.5 ), ( 3.6, 4.6 ) ),
  // CHECK-SAME:      ( ( 5.5, 6.5 ), ( 5.6, 6.6 ) ),
  // CHECK-SAME:      ( ( 7.5, 8.5 ), ( 7.6, 8.6 ) ),
  // CHECK-SAME:      ( ( 9.5, 10.5 ), ( 9.6, 10.6 ) ),
  // CHECK-SAME:      ( ( 11.5, 12.5 ), ( 11.6, 12.6 ) ),
  // CHECK-SAME:      ( ( 13.5, 14.5 ), ( 13.6, 14.6 ) ),
  // CHECK-SAME:      ( ( 15.5, 16.5 ), ( 15.6, 16.6 ) ) ) )
  //
  vector.print %v0 : vector<3x8x2x2xf32>

  %2 = bufferization.alloc_tensor() : tensor<6x16xf32>
  %3 = linalg.copy ins(%d: tensor<6x16xf32>) outs(%2: tensor<6x16xf32>) -> tensor<6x16xf32>
  %4 = tensor.collapse_shape %3 [[0, 1]] : tensor<6x16xf32> into tensor<96xf32>
  %5 = tensor.expand_shape %4 [[0, 1, 2, 3]] output_shape [3, 8, 2, 2] : tensor<96xf32> into tensor<3x8x2x2xf32>
  %v1 = vector.transfer_read %5[%c0, %c0, %c0, %c0], %d1 : tensor<3x8x2x2xf32>, vector<3x8x2x2xf32>
  //
  // CHECK:     ( ( ( ( 1.1, 2.1 ), ( 3.1, 4.1 ) ),
  // CHECK-SAME:    ( ( 5.1, 6.1 ), ( 7.1, 8.1 ) ),
  // CHECK-SAME:    ( ( 9.1, 10.1 ), ( 11.1, 12.1 ) ),
  // CHECK-SAME:    ( ( 13.1, 14.1 ), ( 15.1, 16.1 ) ),
  // CHECK-SAME:    ( ( 1.2, 2.2 ), ( 3.2, 4.2 ) ),
  // CHECK-SAME:    ( ( 5.2, 6.2 ), ( 7.2, 8.2 ) ),
  // CHECK-SAME:    ( ( 9.2, 10.2 ), ( 11.2, 12.2 ) ),
  // CHECK-SAME:    ( ( 13.2, 14.2 ), ( 15.2, 16.2 ) ) ),
  // CHECK-SAME:  ( ( ( 1.3, 2.3 ), ( 3.3, 4.3 ) ),
  // CHECK-SAME:    ( ( 5.3, 6.3 ), ( 7.3, 8.3 ) ),
  // CHECK-SAME:    ( ( 9.3, 10.3 ), ( 11.3, 12.3 ) ),
  // CHECK-SAME:    ( ( 13.3, 14.3 ), ( 15.3, 16.3 ) ),
  // CHECK-SAME:    ( ( 1.4, 2.4 ), ( 3.4, 4.4 ) ),
  // CHECK-SAME:    ( ( 5.4, 6.4 ), ( 7.4, 8.4 ) ),
  // CHECK-SAME:    ( ( 9.4, 10.4 ), ( 11.4, 12.4 ) ),
  // CHECK-SAME:    ( ( 13.4, 14.4 ), ( 15.4, 16.4 ) ) ),
  // CHECK-SAME:  ( ( ( 1.5, 2.5 ), ( 3.5, 4.5 ) ),
  // CHECK-SAME:    ( ( 5.5, 6.5 ), ( 7.5, 8.5 ) ),
  // CHECK-SAME:    ( ( 9.5, 10.5 ), ( 11.5, 12.5 ) ),
  // CHECK-SAME:    ( ( 13.5, 14.5 ), ( 15.5, 16.5 ) ),
  // CHECK-SAME:    ( ( 1.6, 2.6 ), ( 3.6, 4.6 ) ),
  // CHECK-SAME:    ( ( 5.6, 6.6 ), ( 7.6, 8.6 ) ),
  // CHECK-SAME:    ( ( 9.6, 10.6 ), ( 11.6, 12.6 ) ),
  // CHECK-SAME:    ( ( 13.6, 14.6 ), ( 15.6, 16.6 ) ) ) )
  //
  vector.print %v1 : vector<3x8x2x2xf32>

  return
}
