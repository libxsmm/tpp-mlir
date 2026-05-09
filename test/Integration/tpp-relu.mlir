// RUN: tpp-opt %s -default-tpp-passes | FileCheck -check-prefix=IR %s

// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>

// IR-LABEL: relutpp
func.func @relutpp(%A: tensor<9x6xf32>) -> tensor<9x6xf32>  {
  %c0 = arith.constant 0.0 : f32
  // IR: xsmm_unary_invoke
  %O = linalg.generic { indexing_maps = [#map0], iterator_types = ["parallel", "parallel"] }
    outs(%A: tensor<9x6xf32>) {
      ^bb0(%a: f32):
        %0 = arith.maximumf %a, %c0 : f32
        linalg.yield %0: f32
  } -> tensor<9x6xf32>
  return %O: tensor<9x6xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  %da = arith.constant dense<[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1    ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2    ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3    ],
        [ 1.4, -2.4, -3.4, -4.4, 5.4, 6.6 ],
        [ 1.5, -2.5, -3.5, -4.5, 5.5, 6.5 ],
        [ 1.6, -2.6, -3.6, -4.6, 5.6, 6.6 ],
        [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7    ],
        [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8    ],
        [ 1.9, 2.9, 3.9, 4.9, 5.9, 6.9    ]
  ]> : tensor<9x6xf32>

  %B = arith.constant dense<0.0> : tensor<9x6xf32>
  %0 = call @relutpp(%da) : (tensor<9x6xf32>) -> tensor<9x6xf32>

  //
  // CHECK:       ( ( 1.1, 2.1, 3.1, 4.1, 5.1, 6.1 ),
  // CHECK-SAME:    ( 1.2, 2.2, 3.2, 4.2, 5.2, 6.2 ),
  // CHECK-SAME:    ( 1.3, 2.3, 3.3, 4.3, 5.3, 6.3 ),
  // CHECK-SAME:    ( 1.4, 0, 0, 0, 5.4, 6.6 ),
  // CHECK-SAME:    ( 1.5, 0, 0, 0, 5.5, 6.5 ),
  // CHECK-SAME:    ( 1.6, 0, 0, 0, 5.6, 6.6 ),
  // CHECK-SAME:    ( 1.7, 2.7, 3.7, 4.7, 5.7, 6.7 ),
  // CHECK-SAME:    ( 1.8, 2.8, 3.8, 4.8, 5.8, 6.8 ),
  // CHECK-SAME:    ( 1.9, 2.9, 3.9, 4.9, 5.9, 6.9 ) )
  //

  %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<9x6xf32>, vector<9x6xf32>
  vector.print %v0 : vector<9x6xf32>

  return
}
