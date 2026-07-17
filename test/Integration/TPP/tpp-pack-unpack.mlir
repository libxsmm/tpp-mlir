// RUN: tpp-run %s -e entry  -entry-point-result=void | FileCheck %s

func.func private @pack1(%in: tensor<4x4xf32>, %out: tensor<2x2x2x2xf32>) ->  tensor<2x2x2x2xf32> {
  %1 = linalg.pack %in inner_dims_pos = [0, 1] inner_tiles = [2,2] into %out : tensor<4x4xf32> -> tensor<2x2x2x2xf32>
  return %1 : tensor<2x2x2x2xf32>
}

func.func private @pack2(%0: tensor<1x2x2x4xf32>, %1:  tensor<1x2x2x2x2xf32>)-> tensor<1x2x2x2x2xf32>{
 %2 = linalg.pack %0  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %1 : tensor<1x2x2x4xf32> -> tensor<1x2x2x2x2xf32>
  return %2: tensor<1x2x2x2x2xf32>
}

func.func private @pack3(%in: tensor<8x2x2x2xf32>, %out: tensor<2x2x1x4x2x2xf32>)-> tensor<2x2x1x4x2x2xf32>{
  %2 = linalg.pack %in outer_dims_perm = [3, 2, 1, 0] inner_dims_pos=[1, 0] inner_tiles = [2, 2] into %out:   tensor<8x2x2x2xf32>->tensor<2x2x1x4x2x2xf32>
  return %2: tensor<2x2x1x4x2x2xf32>
}

func.func private @unpack1(%in:tensor<2x2x2x2xf32>, %out:  tensor<4x4xf32>) ->   tensor<4x4xf32> {
  %1 = linalg.unpack %in inner_dims_pos = [0, 1] inner_tiles = [2,2] into %out : tensor<2x2x2x2xf32> -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

func.func private @unpack2(%0:  tensor<1x2x2x2x2xf32>, %1: tensor<1x2x2x4xf32>)-> tensor<1x2x2x4xf32>{
 %2 = linalg.unpack %0  outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [2] into %1 : tensor<1x2x2x2x2xf32>->tensor<1x2x2x4xf32>
  return %2: tensor<1x2x2x4xf32>
}


func.func @entry(){
  %0 = arith.constant dense<[[0.0, 1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0, 7.0],
                               [8.0, 9.0, 10.0, 11.0],
                               [12.0, 13.0, 14.0, 15.0]]> : tensor<4x4xf32>
  %1 = tensor.empty():tensor<2x2x2x2xf32>
  %c0 = arith.constant 0 : index
  %2 = call @pack1(%0,%1) : (tensor<4x4xf32>, tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %1[%c0, %c0, %c0, %c0], %d1 : tensor<2x2x2x2xf32>, vector<2x2x2x2xf32>
  vector.print %v0 : vector<2x2x2x2xf32>
  // CHECK: ( ( ( ( 0, 1 ), ( 4, 5 ) ), ( ( 2, 3 ), ( 6, 7 ) ) ), ( ( ( 8, 9 ), ( 12, 13 ) ), ( ( 10, 11 ), ( 14, 15 ) ) ) )

  %3 = arith.constant dense <[[[[0.0, 1.0, 2.0, 3.0],  [4.0, 5.0, 6.0, 7.0]],
                                [[8.0, 9.0, 10.0, 11.0],[12.0, 13.0, 14.0, 15.0]]]]>: tensor<1x2x2x4xf32>
  %4 = tensor.empty():tensor<1x2x2x2x2xf32>
  %5 = call @pack2(%3, %4):(tensor<1x2x2x4xf32>,  tensor<1x2x2x2x2xf32>)->(tensor<1x2x2x2x2xf32>)
  %v1 = vector.transfer_read %5[%c0, %c0, %c0, %c0, %c0], %d1 : tensor<1x2x2x2x2xf32>, vector<1x2x2x2x2xf32>
  vector.print %v1 : vector<1x2x2x2x2xf32>
  // CHECK: ( ( ( ( ( 0, 1 ), ( 4, 5 ) ), ( ( 8, 9 ), ( 12, 13 ) ) ), ( ( ( 2, 3 ), ( 6, 7 ) ), ( ( 10, 11 ), ( 14, 15 ) ) ) ) )

  %6 = arith.constant dense<[[[[0.0, 1.0], [2.0, 3.0]],
                            [[3.0, 4.0], [5.0, 6.0]]],
                            [[[7.0, 8.0], [9.0,10.0]],
                            [[11.0,12.0],[13.0,14.0]]],
                            [[[14.0,15.0],[16.0, 17.0]],
                            [[18.0, 19.0],[20.0, 21.0]]],
                            [[[22.0, 23.0],[24.0, 25.0]],
                            [[26.0, 27.0],[28.0, 29.0]]],
                            [[[30.0, 31.0],[32.0, 33.0]],
                            [[34.0, 35.0],[36.0, 37.0]]],
                            [[[38.0, 39.0],[40.0, 41.0]],
                            [[42.0, 43.0],[44.0, 45.0]]],
                            [[[46.0, 47.0],[48.0, 49.0]],
                            [[50.0, 51.0],[52.0, 53.0]]],
                            [[[53.0, 54.0],[55.0, 56.0]],
                            [[57.0, 58.0],[59.0, 60.0]]]]> : tensor<8x2x2x2xf32>

  %7 = tensor.empty():tensor<2x2x1x4x2x2xf32>
  %8 = call @pack3(%6, %7):(tensor<8x2x2x2xf32>,  tensor<2x2x1x4x2x2xf32>)->(tensor<2x2x1x4x2x2xf32>)
  %v2 = vector.transfer_read %8[%c0, %c0, %c0, %c0, %c0, %c0], %d1 : tensor<2x2x1x4x2x2xf32>, vector<2x2x1x4x2x2xf32>
  vector.print %v2 : vector<2x2x1x4x2x2xf32>
  // CHECK-NOT: ( ( ( ( ( ( 0, 3 ), ( 7, 11 ) ), ( ( 14, 18 ), ( 22, 26 ) ), ( ( 30, 34 ), ( 38, 42 ) ), ( ( 46, 50 ), ( 53, 57 ) ) ) ), ( ( ( ( 2, 5 ), ( 9, 13 ) ), ( ( 16, 20 ), ( 24, 28 ) ), ( ( 32, 36 ), ( 40, 44 ) ), ( ( 48, 52 ), ( 55, 59 ) ) ) ) ), ( ( ( ( ( 1, 4 ), ( 8, 12 ) ), ( ( 15, 19 ), ( 23, 27 ) ), ( ( 31, 35 ), ( 39, 43 ) ), ( ( 47, 51 ), ( 54, 58 ) ) ) ), ( ( ( ( 3, 6 ), ( 10, 14 ) ), ( ( 17, 21 ), ( 25, 29 ) ), ( ( 33, 37 ), ( 41, 45 ) ), ( ( 49, 53 ), ( 56, 60 ) ) ) ) ) )

  %9 = arith.constant dense<((((0.0, 1.0), (4.0, 5.0)),
                               ((2.0, 3.0), (6.0, 7.0))),
                              (((8.0, 9.0), (12.0, 13.0)),
                               ((10.0, 11.0), (14.0, 15.0))))>: tensor<2x2x2x2xf32>
  %10 = tensor.empty():tensor<4x4xf32>
  %11 = call @unpack1(%9,%10) : (tensor<2x2x2x2xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %v3 = vector.transfer_read %11[%c0, %c0], %d1 : tensor<4x4xf32>, vector<4x4xf32>
  vector.print %v3 : vector<4x4xf32>
  // CHECK: ( ( 0, 1, 2, 3 ), ( 4, 5, 6, 7 ), ( 8, 9, 10, 11 ), ( 12, 13, 14, 15 ) )

  %12 = arith.constant dense < ( ( ( ( 0.0, 1.0 ), ( 4.0, 5.0 ) ), ( ( 8.0, 9.0 ), ( 12.0, 13.0 ) ) ), ( ( ( 2.0, 3.0 ), ( 6.0, 7.0 ) ), ( ( 10.0, 11.0 ), ( 14.0, 15.0 ) ) ) ) >:tensor<1x2x2x2x2xf32>
  %13 = tensor.empty():tensor<1x2x2x4xf32>
  %14 = call @unpack2(%12, %13):(tensor<1x2x2x2x2xf32>, tensor<1x2x2x4xf32>)->(tensor<1x2x2x4xf32>)
  %v4 = vector.transfer_read %14[%c0, %c0, %c0, %c0], %d1 : tensor<1x2x2x4xf32>, vector<1x2x2x4xf32>
  vector.print %v4 : vector<1x2x2x4xf32>
  // CHECK: ( ( ( ( 0, 1, 2, 3 ), ( 4, 5, 6, 7 ) ), ( ( 8, 9, 10, 11 ), ( 12, 13, 14, 15 ) ) ) )

  return
}
