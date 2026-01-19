// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes | FileCheck %s -check-prefix=IR

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0, d1) -> (0, 0)>

// IR-LABEL: copytppbrcast
func.func @copytppbrcast(%A: tensor<1x6xf32>) -> tensor<9x6xf32>  {
  %B = tensor.empty() : tensor<9x6xf32>
  // IR: xsmm_unary_invoke
  %O = linalg.generic { indexing_maps = [#map1, #map0],
                        iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<1x6xf32>) outs(%B: tensor<9x6xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
  } -> tensor<9x6xf32>
  return %O: tensor<9x6xf32>
}

// IR-LABEL: copytppbrcastother
func.func @copytppbrcastother(%A: tensor<6x1xf32>) -> tensor<6x9xf32>  {
  %B = tensor.empty() : tensor<6x9xf32>
  // IR: xsmm_unary_invoke
  %O = linalg.generic { indexing_maps = [#map2, #map0],
                        iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<6x1xf32>) outs(%B: tensor<6x9xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
  } -> tensor<6x9xf32>
  return %O: tensor<6x9xf32>
}

// IR-LABEL: copyscalar
func.func @copyscalar(%A: f32) -> tensor<6x9xf32>  {
  %B = tensor.empty() : tensor<6x9xf32>
  // IR: linalg.fill
  %O = linalg.generic { indexing_maps = [#map3, #map0],
                        iterator_types = ["parallel", "parallel"] }
      ins(%A: f32) outs(%B: tensor<6x9xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
  } -> tensor<6x9xf32>
  return %O: tensor<6x9xf32>
}

// IR-LABEL: copyscalarother
func.func @copyscalarother(%A: tensor<1x1xf32>) -> tensor<6x9xf32>  {
  %B = tensor.empty() : tensor<6x9xf32>
  // Rank-0 is on input is not matched to xsmm.
  // IR: linalg.generic
  %O = linalg.generic { indexing_maps = [#map4, #map0],
                        iterator_types = ["parallel", "parallel"] }
      ins(%A: tensor<1x1xf32>) outs(%B: tensor<6x9xf32>) {
        ^bb0(%a: f32, %b: f32):
          linalg.yield %a: f32
  } -> tensor<6x9xf32>
  return %O: tensor<6x9xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  // Initialize various matrices.
  %bcastrow = arith.constant dense<[
      [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ]
  ]> : tensor<1x6xf32>

  %1 = call @copytppbrcast(%bcastrow) : (tensor<1x6xf32>) -> tensor<9x6xf32>

  //
  // CHECK:     ( ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:  ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ) )
  //

  %v1 = vector.transfer_read %1[%c0, %c0], %d1 : tensor<9x6xf32>, vector<9x6xf32>
  vector.print %v1 : vector<9x6xf32>

  %bcastcol = arith.constant dense<[
      [ 1.1 ],
      [ 2.1 ],
      [ 3.1 ],
      [ 4.1 ],
      [ 5.1 ],
      [ 6.1 ]
  ]> : tensor<6x1xf32>

  %2 = call @copytppbrcastother(%bcastcol) : (tensor<6x1xf32>) -> tensor<6x9xf32>

  //
  // CHECK:     ( ( 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1 ),
  // CHECK-SAME:  ( 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1 ),
  // CHECK-SAME:  ( 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1 ),
  // CHECK-SAME:  ( 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1 ),
  // CHECK-SAME:  ( 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1, 5.1 ),
  // CHECK-SAME:  ( 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1, 6.1 ) )
  //

  %v2 = vector.transfer_read %2[%c0, %c0], %d1 : tensor<6x9xf32>, vector<6x9xf32>
  vector.print %v2 : vector<6x9xf32>

  %s = arith.constant 23.1 : f32
  %3 = call @copyscalar(%s) : (f32) -> tensor<6x9xf32>

  //
  // CHECK:     ( ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ),
  // CHECK-SAME:  ( 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1, 23.1 ) )
  //

  %v3 = vector.transfer_read %3[%c0, %c0], %d1 : tensor<6x9xf32>, vector<6x9xf32>
  vector.print %v3 : vector<6x9xf32>

  %ss = arith.constant dense<[
      [43.1]
  ]> : tensor<1x1xf32>

  %4 = call @copyscalarother(%ss) : (tensor<1x1xf32>) -> tensor<6x9xf32>

  //
  // CHECK:     ( ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ),
  // CHECK-SAME:  ( 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1, 43.1 ) )
  //

  %v4 = vector.transfer_read %4[%c0, %c0], %d1 : tensor<6x9xf32>, vector<6x9xf32>
  vector.print %v4 : vector<6x9xf32>

  return
}
