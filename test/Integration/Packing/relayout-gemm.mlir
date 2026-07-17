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

#accessToOrig = [
  affine_map<(n1, c1, n2, c2) -> (n1, c1, n2, c2)>,
  affine_map<(n1, c1, n2, c2) -> (n1 * 2 + n2, c1 * 2 + c2)>
]

#traitToOrig = {
  iterator_types = ["parallel", "parallel", "parallel", "parallel"],
  indexing_maps = #accessToOrig,
  library_call = "from-block-layout"
}


func.func @matmulrelayouts(%A: tensor<6x8xf32>,
  %B: tensor<8x16xf32>, %C: tensor<6x16xf32>) -> tensor<6x16xf32>  {
  // MNmn += MKmk * NKkn
  // N = 16
  // M = 6
  // K = 8
  // The block is hardcoded into the map.
  // m = 2
  // n = 2
  // k = 2
  %0 = tensor.empty() : tensor<3x8x2x2xf32>
  %1 = linalg.generic #traitToBlock
      ins(%C: tensor<6x16xf32>)
      outs(%0: tensor<3x8x2x2xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):
          linalg.yield %arg2: f32
  } -> tensor<3x8x2x2xf32>
  %2 = tensor.empty() : tensor<3x4x2x2xf32>
  %3 = linalg.generic #traitToBlock
      ins(%A: tensor<6x8xf32>)
      outs(%2: tensor<3x4x2x2xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):
          linalg.yield %arg2: f32
  } -> tensor<3x4x2x2xf32>
  %4 = tensor.empty() : tensor<4x8x2x2xf32>
  %5 = linalg.generic #traitToBlock
      ins(%B: tensor<8x16xf32>)
      outs(%4: tensor<4x8x2x2xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):
          linalg.yield %arg2: f32
  } -> tensor<4x8x2x2xf32>

  %c0 = arith.constant 0 : index
  %step = arith.constant 1 : index
  %cbm = arith.constant 3 : index
  %cbn = arith.constant 8 : index
  %cbk = arith.constant 4 : index
  %6 = scf.for %p1 = %c0 to %cbm step %step iter_args(%init = %1) -> tensor<3x8x2x2xf32> {
    %7 = scf.for %p2 = %c0 to %cbn step %step iter_args(%init1 = %init) -> tensor<3x8x2x2xf32> {
      %8 = scf.for %r1 = %c0 to %cbk step %step iter_args(%init2 = %init1) -> tensor<3x8x2x2xf32> {
        %tsc = tensor.extract_slice %init2[%p1, %p2, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
            : tensor<3x8x2x2xf32> to tensor<2x2xf32>
        %tsa = tensor.extract_slice %3[%p1, %r1, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
            : tensor<3x4x2x2xf32> to tensor<2x2xf32>
        %tsb = tensor.extract_slice %5[%r1, %p2, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
            : tensor<4x8x2x2xf32> to tensor<2x2xf32>
        %mul = linalg.matmul ins(%tsa, %tsb: tensor<2x2xf32>, tensor<2x2xf32>)
                               outs(%tsc: tensor<2x2xf32>) -> tensor<2x2xf32>
        %yielded = tensor.insert_slice %mul into %init2[%p1, %p2, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1]
            : tensor<2x2xf32> into tensor<3x8x2x2xf32>
        scf.yield %yielded : tensor<3x8x2x2xf32>
      }
      scf.yield %8 : tensor<3x8x2x2xf32>
    }
    scf.yield %7 : tensor<3x8x2x2xf32>
  }
  %9 = linalg.generic #traitToOrig
      ins(%6: tensor<3x8x2x2xf32>)
      outs(%C: tensor<6x16xf32>) {
        ^bb0(%arg2: f32, %arg3: f32):
          linalg.yield %arg2: f32
  } -> tensor<6x16xf32>
  return %9: tensor<6x16xf32>
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  %da = arith.constant dense<[
      [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
      [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
      [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
      [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ],
      [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5 ],
      [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6 ]
  ]> : tensor<6x8xf32>

  %db = arith.constant dense<[
      [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1, 16.1 ],
      [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 16.2 ],
      [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 9.3, 10.3, 11.3, 12.3, 13.3, 14.3, 15.3, 16.3 ],
      [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4, 16.4 ],
      [ 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5 ],
      [ 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6, 16.6 ],
      [ 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 9.7, 10.7, 11.7, 12.7, 13.7, 14.7, 15.7, 16.7 ],
      [ 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8, 10.8, 11.8, 12.8, 13.8, 14.8, 15.8, 16.8 ]
  ]> : tensor<8x16xf32>

  %C = arith.constant dense<0.0> : tensor<6x16xf32>
  %0 = call @matmulrelayouts(%da, %db, %C)
      : (tensor<6x8xf32>, tensor<8x16xf32>, tensor<6x16xf32>) -> tensor<6x16xf32>
  %v0 = vector.transfer_read %0[%c0, %c0], %d1 : tensor<6x16xf32>, vector<6x16xf32>
  //
  // CHECK:     ( ( 57.56, 94.36, 131.16, 167.96, 204.76, 241.56, 278.36, 315.16, 351.96, 388.76, 425.56, 462.36, 499.16, 535.96, 572.76, 609.56 ),
  // CHECK-SAME:  ( 58.72, 96.32, 133.92, 171.52, 209.12, 246.72, 284.32, 321.92, 359.52, 397.12, 434.72, 472.32, 509.92, 547.52, 585.12, 622.72 ),
  // CHECK-SAME:  ( 59.88, 98.28, 136.68, 175.08, 213.48, 251.88, 290.28, 328.68, 367.08, 405.48, 443.88, 482.28, 520.68, 559.08, 597.48, 635.88 ),
  // CHECK-SAME:  ( 61.04, 100.24, 139.44, 178.64, 217.84, 257.04, 296.24, 335.44, 374.64, 413.84, 453.04, 492.24, 531.44, 570.64, 609.84, 649.04 ),
  // CHECK-SAME:  ( 62.2, 102.2, 142.2, 182.2, 222.2, 262.2, 302.2, 342.2, 382.2, 422.2, 462.2, 502.2, 542.2, 582.2, 622.2, 662.2 ),
  // CHECK-SAME:  ( 63.36, 104.16, 144.96, 185.76, 226.56, 267.36, 308.16, 348.96, 389.76, 430.56, 471.36, 512.16, 552.96, 593.76, 634.56, 675.36 ) )
  //
  vector.print %v0 : vector<6x16xf32>
  return
}
