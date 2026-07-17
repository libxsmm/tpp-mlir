// RUN: tpp-run %s -e entry --entry-point-result=void --splat-to-random --init-type quant --print-input --seed 123 | FileCheck %s
// RUN: tpp-run %s -e unpacked --entry-point-result=void --splat-to-random --init-type quant --print-input --seed 123 | FileCheck %s --check-prefix=UNPACKED
// RUN: tpp-run %s -e packed --entry-point-result=void --splat-to-random --init-type quant --print-input --seed 123 | FileCheck %s --check-prefix=PACKED

// CHECK: ( -13, 67, 77 )
// CHECK: ( 3, 0, 76 )
// CHECK: ( 50, 6, -69 )
// CHECK: ( 0.00195312, 0.00390625, 0.00195312 )
// CHECK: ( -40, -59, 13, 66 )
// CHECK: ( 4, -23, -28, -40 )
// CHECK: ( -110, -99, -98, -67 )
// CHECK: ( 0.00390625, 0.00390625, 0.00390625, 0.00195312 )

!twoDimInputf32 = tensor<3x3xf32>
!twoDimWeightf32 = tensor<3x4xf32>
!twoDimInputi8 = tensor<3x3xi8>
!twoDimWeighti8 = tensor<3x4xi8>
!oneDimScaleInputf32 = tensor<3xf32>
!oneDimScaleWeightf32 = tensor<4xf32>


func.func @entry(%input : !twoDimInputi8, %iScale : !oneDimScaleInputf32, %weight : !twoDimWeighti8, %wScale : !oneDimScaleWeightf32) {
  return
}


// UNPACKED: ( -7, 33, 39, 3, 0, 76, 25, 3 )
// UNPACKED: ( -69, -81, -117, 26, 66, 8, -47, -57 )
// UNPACKED: ( -20, -110, -99, -98, -33, -92, -26, -34 )
// UNPACKED: ( 40, -26, -113, 34, 53, -17, 17, 39 )
// UNPACKED: ( 0.00390625, 0.00195312, 0.00390625, 0.00390625 )
// UNPACKED: ( 48, 29, 50, 48, 48, -20, 37, -119, -4, -70, -15, 29, -5, 127, -5, -2 )
// UNPACKED: ( -83, 1, 32, -72, -37, 13, -38, 26, -77, 27, -9, -19, -18, 51, 62, 44 )
// UNPACKED: ( -16, 64, 3, 32, -16, 36, -76, 62, -9, 19, 59, 30, -5, -19, -13, 6 )
// UNPACKED: ( 3, 102, 3, -88, 25, -5, -14, 123, -20, 9, -45, 29, -15, 1, 16, -73 )
// UNPACKED: ( 11, 57, -26, 48, 30, -17, 22, -25, -56, -57, 8, -58, -2, -10, 72, -2 )
// UNPACKED: ( 97, -23, -30, -11, 25, 67, -71, -96, -2, 5, -44, -72, 72, 7, -33, -78 )
// UNPACKED: ( -46, 10, 39, -22, 127, -10, 47, -5, -94, -122, -94, 61, 10, -14, 5, 3 )
// UNPACKED: ( 15, -48, 66, 1, -4, -31, 68, 82, 22, 15, 10, 30, -2, -53, -37, 70 )
// UNPACKED: ( 0.00390625, 0.00390625, 0.0078125, 0.00195312, 0.00390625, 0.0078125, 0.00390625, 0.000976562, 0.00390625, 0.00390625, 0.00390625, 0.0078125, 0.0078125, 0.00195312, 0.00390625, 0.00390625 )

func.func @unpacked(%arg0: tensor<4x8xi8>, %arg1: tensor<4xf32>, %arg2: tensor<8x16xi8>, %arg3: tensor<16xf32>) {
  return
}


// PACKED: ( -7, 33, 39, 3 )
// PACKED: ( 0, 76, 25, 3 )
// PACKED: ( -69, -81, -117, 26 )
// PACKED: ( 66, 8, -47, -57 )
// PACKED: ( -20, -110, -99, -98 )
// PACKED: ( -33, -92, -26, -34 )
// PACKED: ( 40, -26, -113, 34 )
// PACKED: ( 53, -17, 17, 39 )
// PACKED: ( 0.00390625, 0.00195312, 0.00390625, 0.00390625 )
// PACKED: ( 48, 29, 50, 48 )
// PACKED: ( 48, -20, 37, -119 )
// PACKED: ( -4, -70, -15, 29 )
// PACKED: ( -5, 127, -5, -2 )
// PACKED: ( -83, 1, 32, -72 )
// PACKED: ( -37, 13, -38, 26 )
// PACKED: ( -77, 27, -9, -19 )
// PACKED: ( -18, 51, 62, 44 )
// PACKED: ( -16, 64, 3, 32 )
// PACKED: ( -16, 36, -76, 62 )
// PACKED: ( -9, 19, 59, 30 )
// PACKED: ( -5, -19, -13, 6 )
// PACKED: ( 3, 102, 3, -88 )
// PACKED: ( 25, -5, -14, 123 )
// PACKED: ( -20, 9, -45, 29 )
// PACKED: ( -15, 1, 16, -73 )
// PACKED: ( 11, 57, -26, 48 )
// PACKED: ( 30, -17, 22, -25 )
// PACKED: ( -56, -57, 8, -58 )
// PACKED: ( -2, -10, 72, -2 )
// PACKED: ( 97, -23, -30, -11 )
// PACKED: ( 25, 67, -71, -96 )
// PACKED: ( -2, 5, -44, -72 )
// PACKED: ( 72, 7, -33, -78 )
// PACKED: ( -46, 10, 39, -22 )
// PACKED: ( 127, -10, 47, -5 )
// PACKED: ( -94, -122, -94, 61 )
// PACKED: ( 10, -14, 5, 3 )
// PACKED: ( 22, 15, 10, 30 )
// PACKED: ( -2, -53, -37, 70 )
// PACKED: ( 0.00390625, 0.00390625, 0.0078125, 0.00195312, 0.00390625, 0.0078125, 0.00390625, 0.000976562, 0.00390625, 0.00390625, 0.00390625, 0.0078125, 0.0078125, 0.00195312, 0.00390625, 0.00390625 )

func.func @packed(%arg0: tensor<2x2x2x4xi8>, %arg1: tensor<4xf32>, %arg2: tensor<8x2x1x2x4xi8>, %arg3: tensor<16xf32>) {
  return
}