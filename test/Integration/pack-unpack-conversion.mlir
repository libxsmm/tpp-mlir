// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-loops -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

// FIXME: This test fails to bufferize when lowering linalg to loops

func.func private @generate_1D_source(%width : index) -> tensor<?xf32> {
  %init_source = tensor.empty(%width) : tensor<?xf32>
  %source = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      outs(%init_source : tensor<?xf32>) {
    ^bb0(%b0 : f32):
      %inner = linalg.index 0 : index
      %inner_val_i32 = arith.index_cast %inner : index to i32
      %inner_val = arith.sitofp %inner_val_i32 : i32 to f32
      linalg.yield %inner_val :  f32
  } -> tensor<?xf32>
  return %source : tensor<?xf32>
}

func.func @entry() {
  %c2 = arith.constant 2 : index
  %input_tensor = call @generate_1D_source(%c2) : (index) -> (tensor<?xf32>)
  %input_tensor_cast = tensor.cast %input_tensor : tensor<?xf32> to tensor<2xf32>
  %bcast = tensor.empty() : tensor<2x8x8x2xf32>
  %input_tensor_bcast = linalg.broadcast ins(%input_tensor_cast: tensor<2xf32>)
                                             outs(%bcast: tensor<2x8x8x2xf32>)
                                             dimensions = [0, 1, 2]
  %c0 = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v1 = vector.transfer_read %input_tensor_bcast[%c0, %c0, %c0, %c0], %d1 : tensor<2x8x8x2xf32>, vector<2x8x8x2xf32>
  vector.print %v1 : vector<2x8x8x2xf32>

  //
  // CHECK: ( ( ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ) ),
  // CHECK-SAME:  ( ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ) ) )
  //

  %unpacked_tensor = bufferization.alloc_tensor() : tensor<13x15xf32>
  %unpack = linalg.unpack %input_tensor_bcast inner_dims_pos = [0, 1] inner_tiles = [8, 2]
    into %unpacked_tensor : tensor<2x8x8x2xf32> -> tensor<13x15xf32>

  %v0 = vector.transfer_read %unpack[%c0, %c0], %d1 : tensor<13x15xf32>, vector<13x15xf32>
  vector.print %v0 : vector<13x15xf32>

  //
  // CHECK: ( ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 ) )
  //


  %c32 = arith.constant 32 : index
  %input_tensor1 = call @generate_1D_source(%c32) : (index) -> (tensor<?xf32>)
  %input_tensor1_cast = tensor.cast %input_tensor1 : tensor<?xf32> to tensor<32xf32>
  %bcast1 = tensor.empty() : tensor<1x1x1x1x8x32xf32>
  %input_tensor_bcast1 = linalg.broadcast ins(%input_tensor1_cast: tensor<32xf32>)
                                          outs(%bcast1: tensor<1x1x1x1x8x32xf32>)
                                          dimensions = [0, 1, 2, 3, 4]
  %v2 = vector.transfer_read %input_tensor_bcast1[%c0, %c0, %c0, %c0, %c0, %c0], %d1
    : tensor<1x1x1x1x8x32xf32>, vector<1x1x1x1x8x32xf32>
  vector.print %v2 : vector<1x1x1x1x8x32xf32>

  //
  // CHECK: ( ( ( ( ( ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ),
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ),
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ),
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ),
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ),
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ),
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ),
  // CHECK-SAME:  ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 ) ) ) ) ) )
  //

  %unpacked_tensor1 = bufferization.alloc_tensor() : tensor<1x1x32x8xf32>
  %unpack1 = linalg.unpack %input_tensor_bcast1 inner_dims_pos = [3, 2] inner_tiles = [8, 32]
    into %unpacked_tensor1 : tensor<1x1x1x1x8x32xf32> -> tensor<1x1x32x8xf32>

  %v3 = vector.transfer_read %unpack1[%c0, %c0, %c0, %c0], %d1 : tensor<1x1x32x8xf32>, vector<1x1x32x8xf32>
  vector.print %v3 : vector<1x1x32x8xf32>

  //
  // CHECK: ( ( ( ( 0, 0, 0, 0, 0, 0, 0, 0 ),
  // CHECK-SAME:  ( 1, 1, 1, 1, 1, 1, 1, 1 ),
  // CHECK-SAME:  ( 2, 2, 2, 2, 2, 2, 2, 2 ),
  // CHECK-SAME:  ( 3, 3, 3, 3, 3, 3, 3, 3 ),
  // CHECK-SAME:  ( 4, 4, 4, 4, 4, 4, 4, 4 ),
  // CHECK-SAME:  ( 5, 5, 5, 5, 5, 5, 5, 5 ),
  // CHECK-SAME:  ( 6, 6, 6, 6, 6, 6, 6, 6 ),
  // CHECK-SAME:  ( 7, 7, 7, 7, 7, 7, 7, 7 ),
  // CHECK-SAME:  ( 8, 8, 8, 8, 8, 8, 8, 8 ),
  // CHECK-SAME:  ( 9, 9, 9, 9, 9, 9, 9, 9 ),
  // CHECK-SAME:  ( 10, 10, 10, 10, 10, 10, 10, 10 ),
  // CHECK-SAME:  ( 11, 11, 11, 11, 11, 11, 11, 11 ),
  // CHECK-SAME:  ( 12, 12, 12, 12, 12, 12, 12, 12 ),
  // CHECK-SAME:  ( 13, 13, 13, 13, 13, 13, 13, 13 ),
  // CHECK-SAME:  ( 14, 14, 14, 14, 14, 14, 14, 14 ),
  // CHECK-SAME:  ( 15, 15, 15, 15, 15, 15, 15, 15 ),
  // CHECK-SAME:  ( 16, 16, 16, 16, 16, 16, 16, 16 ),
  // CHECK-SAME:  ( 17, 17, 17, 17, 17, 17, 17, 17 ),
  // CHECK-SAME:  ( 18, 18, 18, 18, 18, 18, 18, 18 ),
  // CHECK-SAME:  ( 19, 19, 19, 19, 19, 19, 19, 19 ),
  // CHECK-SAME:  ( 20, 20, 20, 20, 20, 20, 20, 20 ),
  // CHECK-SAME:  ( 21, 21, 21, 21, 21, 21, 21, 21 ),
  // CHECK-SAME:  ( 22, 22, 22, 22, 22, 22, 22, 22 ),
  // CHECK-SAME:  ( 23, 23, 23, 23, 23, 23, 23, 23 ),
  // CHECK-SAME:  ( 24, 24, 24, 24, 24, 24, 24, 24 ),
  // CHECK-SAME:  ( 25, 25, 25, 25, 25, 25, 25, 25 ),
  // CHECK-SAME:  ( 26, 26, 26, 26, 26, 26, 26, 26 ),
  // CHECK-SAME:  ( 27, 27, 27, 27, 27, 27, 27, 27 ),
  // CHECK-SAME:  ( 28, 28, 28, 28, 28, 28, 28, 28 ),
  // CHECK-SAME:  ( 29, 29, 29, 29, 29, 29, 29, 29 ),
  // CHECK-SAME:  ( 30, 30, 30, 30, 30, 30, 30, 30 ),
  // CHECK-SAME:  ( 31, 31, 31, 31, 31, 31, 31, 31 ) ) ) )
  //


  %input_tensor2 = call @generate_1D_source(%c2) : (index) -> (tensor<?xf32>)
  %input_tensor2_cast = tensor.cast %input_tensor2 : tensor<?xf32> to tensor<2xf32>
  %bcast2 = tensor.empty() : tensor<1x4x6x6x2xf32>
  %input_tensor_bcast2 = linalg.broadcast ins(%input_tensor2_cast: tensor<2xf32>)
                                          outs(%bcast2: tensor<1x4x6x6x2xf32>)
                                          dimensions = [0, 1, 2, 3]
  %unpacked_tensor2 = bufferization.alloc_tensor() : tensor<1x6x6x8xf32>
  %unpack2 = linalg.unpack %input_tensor_bcast2 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2]
    into %unpacked_tensor2 : tensor<1x4x6x6x2xf32> -> tensor<1x6x6x8xf32>
  %v4 = vector.transfer_read %unpack2[%c0, %c0, %c0, %c0], %d1 : tensor<1x6x6x8xf32>, vector<1x6x6x8xf32>
  vector.print %v4 : vector<1x6x6x8xf32>

  //
  // CHECK: ( ( ( ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:  ( 0, 1, 0, 1, 0, 1, 0, 1 ),
  // CHECK-SAME:( 0, 1, 0, 1, 0, 1, 0, 1 ) ) ) )
  //

  %c8 = arith.constant 8 : index
  %input_tensor3 = call @generate_1D_source(%c8) : (index) -> (tensor<?xf32>)
  %input_tensor3_cast = tensor.cast %input_tensor3 : tensor<?xf32> to tensor<8xf32>
  %bcast3 = tensor.empty() : tensor<1x6x6x8xf32>
  %input_tensor_bcast3 = linalg.broadcast ins(%input_tensor3_cast: tensor<8xf32>)
                                          outs(%bcast3: tensor<1x6x6x8xf32>)
                                          dimensions = [0, 1, 2]
  %packed_tensor = bufferization.alloc_tensor() : tensor<1x4x6x6x2xf32>
  %pack = linalg.pack %input_tensor_bcast3 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2]
    into %packed_tensor : tensor<1x6x6x8xf32> -> tensor<1x4x6x6x2xf32>
  %v5 = vector.transfer_read %pack[%c0, %c0, %c0, %c0, %c0], %d1 : tensor<1x4x6x6x2xf32>, vector<1x4x6x6x2xf32>

  //
  // CHECK: ( ( ( ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ),
  // CHECK-SAME:  ( 0, 1 ) ), ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ),
  // CHECK-SAME:  ( ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ), ( 0, 1 ) ) ),
  // CHECK-SAME:( ( ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ) ),
  // CHECK-SAME:  ( ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ) ),
  // CHECK-SAME:  ( ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ) ),
  // CHECK-SAME:  ( ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ) ),
  // CHECK-SAME:  ( ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ) ),
  // CHECK-SAME:  ( ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ), ( 2, 3 ) ) ),
  // CHECK-SAME:( ( ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ) ),
  // CHECK-SAME:  ( ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ) ),
  // CHECK-SAME:  ( ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ) ),
  // CHECK-SAME:  ( ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ) ),
  // CHECK-SAME:  ( ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ) ),
  // CHECK-SAME:  ( ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ), ( 4, 5 ) ) ),
  // CHECK-SAME:( ( ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ) ),
  // CHECK-SAME:  ( ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ) ),
  // CHECK-SAME:  ( ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ) ),
  // CHECK-SAME:  ( ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ) ),
  // CHECK-SAME:  ( ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ) ),
  // CHECK-SAME:  ( ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ), ( 6, 7 ) ) ) ) )
  //

  vector.print %v5 : vector<1x4x6x6x2xf32>

  %bcast4 = tensor.empty() : tensor<1x1x32x8xf32>
  %input_tensor_bcast4 = linalg.broadcast ins(%input_tensor3_cast: tensor<8xf32>)
                                          outs(%bcast4: tensor<1x1x32x8xf32>)
                                          dimensions = [0, 1, 2]
  %packed_tensor1 = bufferization.alloc_tensor() : tensor<1x1x1x1x8x32xf32>
  %pack1 = linalg.pack %input_tensor_bcast4 inner_dims_pos = [3, 2] inner_tiles = [8, 32]
    into %packed_tensor1 : tensor<1x1x32x8xf32> -> tensor<1x1x1x1x8x32xf32>
  %v6 = vector.transfer_read %pack1[%c0, %c0, %c0, %c0, %c0, %c0], %d1 : tensor<1x1x1x1x8x32xf32>, vector<1x1x1x1x8x32xf32>

  //
  // CHECK: ( ( ( ( ( ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
  // CHECK-SAME:  ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ),
  // CHECK-SAME:  ( 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ),
  // CHECK-SAME:  ( 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 ),
  // CHECK-SAME:  ( 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4 ),
  // CHECK-SAME:  ( 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 ),
  // CHECK-SAME:  ( 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 ),
  // CHECK-SAME:  ( 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 ) ) ) ) ) )
  //

  vector.print %v6 : vector<1x1x1x1x8x32xf32>
  return
}
