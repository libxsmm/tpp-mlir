// RUN: tpp-run %s -e entry --entry-point-result=void --splat-to-random --init-type quant --print-input --seed 123 | FileCheck %s
// RUN: tpp-run %s -e unpacked --entry-point-result=void --splat-to-random --init-type quant --print-input --seed 123 | FileCheck %s --check-prefix=UNPACKED
// RUN: tpp-run %s -e packed --entry-point-result=void --splat-to-random --init-type quant --print-input --seed 123 | FileCheck %s --check-prefix=PACKED

// CHECK: ( -29, -94, 97 )
// CHECK: ( 69, 9, -45 )
// CHECK: ( 7, -5, 77 )
// CHECK: ( 0.000976562, 0.00390625, 0.0078125 )
// CHECK: ( 18, 83, 72, 6 )
// CHECK: ( 78, -30, 63, -52 )
// CHECK: ( 4, -27, -17, 73 )
// CHECK: ( 0.00390625, 0.00390625, 0.00390625, 0.00390625 )

!twoDimInputf32 = tensor<3x3xf32>
!twoDimWeightf32 = tensor<3x4xf32>
!twoDimInputi8 = tensor<3x3xi8>
!twoDimWeighti8 = tensor<3x4xi8>
!oneDimScaleInputf32 = tensor<3xf32>
!oneDimScaleWeightf32 = tensor<4xf32>


func.func @entry(%input : !twoDimInputi8, %iScale : !oneDimScaleInputf32, %weight : !twoDimWeighti8, %wScale : !oneDimScaleWeightf32) {
  return
}


// UNPACKED: ( -7, -24, 24, 69, 9, -45, 15, -11 )
// UNPACKED: ( 77, 9, 42, 36, 3, 39, -15, 31 )
// UNPACKED: ( -52, 4, -27, -17, 73, -23, -33, -2 )
// UNPACKED: ( 14, 108, -28, 82, 71, 7, -63, 8 )
// UNPACKED: ( 0.00390625, 0.0078125, 0.00390625, 0.00390625 )
// UNPACKED: ( 55, 9, -75, -35, 47, -61, 57, -45, 1, 4, -17, 39, -25, -10, 81, -18 )
// UNPACKED: ( -15, 7, -9, 19, -82, -65, -64, 46, 57, 21, 73, 1, 5, 12, 3, -21 )
// UNPACKED: ( -9, -11, 29, 44, -19, 28, -94, -3, -44, -2, 35, -22, -18, -20, 65, 9 )
// UNPACKED: ( 67, 21, 5, -3, 2, 10, -17, 8, 0, -7, 2, 5, 1, -16, -4, 22 )
// UNPACKED: ( 86, 17, 39, -31, 7, 9, -74, -6, -87, -100, -13, -15, 50, -40, -72, -4 )
// UNPACKED: ( 43, 9, -11, 102, 8, 6, -96, -2, 28, -41, -45, 38, -83, -80, 93, -44 )
// UNPACKED: ( 17, -22, 59, 64, 0, 63, 62, 90, 17, -25, -3, 2, 10, -16, 94, 55 )
// UNPACKED: ( 17, 70, -60, 28, 20, 49, -43, -21, 2, 121, -49, -89, 23, 53, -37, -69 )
// UNPACKED: ( 0.0078125, 0.0078125, 0.00390625, 0.00195312, 0.0078125, 0.00390625, 0.00195312, 0.00390625, 0.0078125, 0.00390625, 0.00390625, 0.0078125, 0.00390625, 0.0078125, 0.00195312, 0.00390625 )

func.func @unpacked(%arg0: tensor<4x8xi8>, %arg1: tensor<4xf32>, %arg2: tensor<8x16xi8>, %arg3: tensor<16xf32>) {
  return
}


// PACKED: ( -7, -24, 24, 69 )
// PACKED: ( 9, -45, 15, -11 )
// PACKED: ( 77, 9, 42, 36 )
// PACKED: ( 3, 39, -15, 31 )
// PACKED: ( -52, 4, -27, -17 )
// PACKED: ( 73, -23, -33, -2 )
// PACKED: ( 14, 108, -28, 82 )
// PACKED: ( 71, 7, -63, 8 )
// PACKED: ( 0.00390625, 0.0078125, 0.00390625, 0.00390625 )
// PACKED: ( 55, 9, -75, -35 )
// PACKED: ( 47, -61, 57, -45 )
// PACKED: ( 1, 4, -17, 39 )
// PACKED: ( -25, -10, 81, -18 )
// PACKED: ( -15, 7, -9, 19 )
// PACKED: ( -82, -65, -64, 46 )
// PACKED: ( 57, 21, 73, 1 )
// PACKED: ( 5, 12, 3, -21 )
// PACKED: ( -9, -11, 29, 44 )
// PACKED: ( -19, 28, -94, -3 )
// PACKED: ( -44, -2, 35, -22 )
// PACKED: ( -18, -20, 65, 9 )
// PACKED: ( 67, 21, 5, -3 )
// PACKED: ( 2, 10, -17, 8 )
// PACKED: ( 0, -7, 2, 5 )
// PACKED: ( 1, -16, -4, 22 )
// PACKED: ( 86, 17, 39, -31 )
// PACKED: ( 7, 9, -74, -6 )
// PACKED: ( -87, -100, -13, -15 )
// PACKED: ( 50, -40, -72, -4 )
// PACKED: ( 43, 9, -11, 102 )
// PACKED: ( 8, 6, -96, -2 )
// PACKED: ( 28, -41, -45, 38 )
// PACKED: ( -83, -80, 93, -44 )
// PACKED: ( 17, -22, 59, 64 )
// PACKED: ( 0, 63, 62, 90 )
// PACKED: ( 17, -25, -3, 2 )
// PACKED: ( 10, -16, 94, 55 )
// PACKED: ( 17, 70, -60, 28 )
// PACKED: ( 20, 49, -43, -21 )
// PACKED: ( 2, 121, -49, -89 )
// PACKED: ( 23, 53, -37, -69 )
// PACKED: ( 0.0078125, 0.0078125, 0.00390625, 0.00195312, 0.0078125, 0.00390625, 0.00195312, 0.00390625, 0.0078125, 0.00390625, 0.00390625, 0.0078125, 0.00390625, 0.0078125, 0.00195312, 0.00390625 )

func.func @packed(%arg0: tensor<2x2x2x4xi8>, %arg1: tensor<4xf32>, %arg2: tensor<8x2x1x2x4xi8>, %arg3: tensor<16xf32>) {
  return
}
