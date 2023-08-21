// RUN: tpp-opt %s -tpp-mapping -split-input-file | FileCheck %s

// We don't expect to block as the blocking factor do not create full tiles.
func.func @conv_to_matmul(%img: tensor<1x5x5x3xf32>, %filter: tensor<3x3x3x8xf32>, %out: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf ins(%img, %filter: tensor<1x5x5x3xf32>, tensor<3x3x3x8xf32>) outs(%out: tensor<1x3x3x8xf32>) -> tensor<1x3x3x8xf32>
  return %0: tensor<1x3x3x8xf32>
}

// CHECK-LABEL: func.func @conv_to_matmul(
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:         tensor.extract_slice{{[^:]+}}: tensor<1x5x5x3xf32> to tensor<3x3xf32>
// CHECK:         tensor.extract_slice{{[^:]+}}: tensor<3x3x3x8xf32> to tensor<3x8xf32>
// CHECK:         tensor.extract_slice{{[^:]+}}: tensor<1x3x3x8xf32> to tensor<3x8xf32>
// CHECK:         linalg.matmul{{.*}} -> tensor<3x8xf32>
// CHECK:         tensor.insert_slice{{[^:]+}}: tensor<3x8xf32> into tensor<1x3x3x8xf32>
// CHECK:       }

// -----

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<1x113x113x64xf32>, %arg1: tensor<3x3x64x256xf32>, %arg2: tensor<1x111x111x256xf32>) -> tensor<1x111x111x256xf32> {
  %1 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>,
                                 strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1 : tensor<1x113x113x64xf32>, tensor<3x3x64x256xf32>)
    outs(%arg2: tensor<1x111x111x256xf32>) -> tensor<1x111x111x256xf32>
  return %1 : tensor<1x111x111x256xf32>
}

// CHECK: func.func @conv_2d_nhwc_hwcf(%[[ARG0:.+]]: tensor<1x113x113x64xf32>, %[[ARG1:.+]]: tensor<3x3x64x256xf32>, %[[ARG2:.+]]: tensor<1x111x111x256xf32>) -> tensor<1x111x111x256xf32>
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// Generalized pack of the first input
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[c3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[c8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[c32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[c111:.+]] = arith.constant 111 : index
// CHECK-DAG: %[[c113:.+]] = arith.constant 113 : index
// CHECK-DAG:  %[[ZERO:.+]] = tensor.empty() : tensor<1x2x113x113x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c113]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[ZERO]]) -> (tensor<1x2x113x113x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c113]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x2x113x113x32xf32>) { 
// CHECK:    scf.for %[[ARG7:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x2x113x113x32xf32>) {
// CHECK:       %[[MUL:.+]] =  arith.muli %[[ARG7]], %[[c32]] : index 
// CHECK:       %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][0, %[[ARG3]], %[[ARG5]], %[[MUL]]] [1, 1, 1, 32] [1, 1, 1, 1] : tensor<1x113x113x64xf32> to tensor<32xf32>
// CHECK:       tensor.insert_slice %[[EXTRACT]] into %[[ARG8]][0, %[[ARG7]], %[[ARG3]], %[[ARG5]], 0] [1, 1, 1, 1, 32] [1, 1, 1, 1, 1] : tensor<32xf32> into tensor<1x2x113x113x32xf32>
// Generalized pack of the second input
// CHECK: %[[TWO:.+]] = tensor.empty() : tensor<8x2x3x3x32x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c3]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[TWO]]) -> (tensor<8x2x3x3x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c3]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<8x2x3x3x32x32xf32>) {
// CHECK:     scf.for %[[ARG7:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<8x2x3x3x32x32xf32>) {
// CHECK:       scf.for %[[ARG9:.+]] = %[[c0]] to %[[c8]] step %[[c1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<8x2x3x3x32x32xf32>) { 
// CHECK:	  %[[CONST_ONE:.+]] =  arith.muli %[[ARG7]], %[[c32]] : index
// CHECK:	  %[[CONST_TWO:.+]] =  arith.muli %[[ARG9]], %[[c32]] : index
// CHECK:    	  %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], %[[ARG5]], %[[CONST_ONE]], %[[CONST_TWO]]] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<3x3x64x256xf32> to tensor<32x32xf32>
// CHECK:    	  tensor.insert_slice %[[EXTRACT]] into %[[ARG10]][%[[ARG9]], %[[ARG7]], %[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : tensor<32x32xf32> into tensor<8x2x3x3x32x32xf32>
// Generalized pack of the output
// CHECK: %[[FOUR:.+]] = tensor.empty() : tensor<1x8x111x111x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c111]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[FOUR]]) -> (tensor<1x8x111x111x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c111]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<1x8x111x111x32xf32>) {
// CHECK:     scf.for %[[ARG7:.+]] = %[[c0]] to %[[c8]] step %[[c1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<1x8x111x111x32xf32>) {
// CHECK:       %[[CONST_THREE:.+]] = arith.muli %[[ARG7]], %[[c32]] : index
// CHECK:     	%[[EXTRACT]] = tensor.extract_slice %[[ARG2]][0, %[[ARG3]], %[[ARG5]], %[[CONST_THREE]]] [1, 1, 1, 32] [1, 1, 1, 1] : tensor<1x111x111x256xf32> to tensor<32xf32>
// CHECK:       tensor.insert_slice %[[EXTRACT]] into %[[ARG8]][0, %[[ARG7]], %[[ARG3]], %[[ARG5]], 0] [1, 1, 1, 1, 32] [1, 1, 1, 1, 1] : tensor<32xf32> into tensor<1x8x111x111x32xf32>
// Conv as matmul
// CHECK: scf.for
// CHECK:   linalg.matmul

// -----

func.func @conv_2d_nchw_fchw(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

// CHECK-LABEL: func.func @conv_2d_nchw_fchw(
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// Generalized pack of the first input
// CHECK-DAG: %[[c1024:.+]] = arith.constant 1024 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c14:.+]] = arith.constant 14 : index
// CHECK-DAG: %[[c32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[c28:.+]] = arith.constant 28 : index
// CHECK-DAG: %[[c16:.+]] = arith.constant 16 : index
// CHECK: %[[ZERO:.+]] = tensor.empty() : tensor<14x16x28x28x32xf32>
// CHECK:  scf.for %[[ARG3:.+]] = %[[c0]] to %[[c14]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[ZERO]]) -> (tensor<14x16x28x28x32xf32>) { 
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c16]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<14x16x28x28x32xf32>) {
// CHECK:     scf.for %[[ARG7:.+]] = %[[c0]] to %[[c28]] step %[[c1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<14x16x28x28x32xf32>) {
// CHECK:      scf.for %[[ARG9:.+]] = %[[c0]] to %[[c28]] step %[[c1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<14x16x28x28x32xf32>) {
// CHECK:        %[[CONST_MAP:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:        %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], %[[CONST_MAP]], %[[ARG7]], %[[ARG9]]] [1, 32, 1, 1] [1, 1, 1, 1] : tensor<14x512x28x28xf32> to tensor<32xf32>
// CHECK:        %[[TRANSPOSE:.+]] = tensor.empty() : tensor<32xf32>
// CHECK:        %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32xf32>) outs(%[[TRANSPOSE]] : tensor<32xf32>) permutation = [0]
// CHECK:        tensor.insert_slice %[[TRANSPOSED]] into %[[ARG10]][%[[ARG3]], %[[ARG5]], %[[ARG7]], %[[ARG9]], 0] [1, 1, 1, 1, 32] [1, 1, 1, 1, 1] : tensor<32xf32> into tensor<14x16x28x28x32xf32>
// Generalized pack of the second input
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<32x16x1x1x32x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c32]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<32x16x1x1x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c16]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<32x16x1x1x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG1]][%[[MAP0]], %[[MAP1]], 0, 0] [32, 32, 1, 1] [1, 1, 1, 1] : tensor<1024x512x1x1xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [1, 0]
// CHECK:     tensor.insert_slice %[[TRANSPOSED]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0, 0, 0] [1, 1, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1] : tensor<32x32xf32> into tensor<32x16x1x1x32x32xf32>
// Generalized pack of the output
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<14x32x28x28x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c14]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<14x32x28x28x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c32]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<14x32x28x28x32xf32>) {
// CHECK:     scf.for %[[ARG7:.+]] = %[[c0]] to %[[c28]] step %[[c1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<14x32x28x28x32xf32>) {
// CHECK:       scf.for %[[ARG9:.+]] = %[[c0]] to %[[c28]] step %[[c1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<14x32x28x28x32xf32>) {
// CHECK: 	  %[[MAP:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     	  %[[EXTRACTED:.+]] = tensor.extract_slice %[[ARG2]][%[[ARG3]], %[[MAP]], %[[ARG7]], %[[ARG9]]] [1, 32, 1, 1] [1, 1, 1, 1] : tensor<14x1024x28x28xf32> to tensor<32xf32>
// CHECK:	  %[[TEMP:.+]] = tensor.empty() : tensor<32xf32>
// CHECK:	  %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACTED]] : tensor<32xf32>) outs(%[[TEMP]] : tensor<32xf32>) permutation = [0]
// CHECK:         tensor.insert_slice %[[TRANSPOSED]] into %[[ARG10]][%[[ARG3]], %[[ARG5]], %[[ARG7]], %[[ARG9]], 0] [1, 1, 1, 1, 32] [1, 1, 1, 1, 1] : tensor<32xf32> into tensor<14x32x28x28x32xf32>
// Conv as matmul
// CHECK: scf.for
// CHECK:   linalg.matmul

// -----

func.func @generalize_pack_unpack(%arg0: tensor<12x2x56x56x32xf32>, %arg1: tensor<512x1024xbf16>, %arg2: tensor<256x1024xbf16>)
                          -> (tensor<256x1024x2xbf16>, tensor<12x56x56x64xf32>) {
  %packOut = tensor.empty() : tensor<256x1024x2xbf16>
  %0 = tensor.pack %arg1 inner_dims_pos = [0] inner_tiles = [2] into %packOut : tensor<512x1024xbf16> -> tensor<256x1024x2xbf16>
  %unpackOut = tensor.empty() : tensor<12x56x56x64xf32>
  %1 = tensor.unpack %arg0 outer_dims_perm = [0, 3, 1, 2] inner_dims_pos = [3] inner_tiles = [32] into %unpackOut : tensor<12x2x56x56x32xf32> -> tensor<12x56x56x64xf32>
  return %0, %1 : tensor<256x1024x2xbf16>, tensor<12x56x56x64xf32>
}

// CHECK-LABEL: func.func @generalize_pack_unpack(
// CHECK-NOT: tensor.pack
// CHECK-DAG:  %[[c64:.+]] = arith.constant 64 : index
// CHECK-DAG:  %[[c56:.+]] = arith.constant 56 : index
// CHECK-DAG:  %[[c12:.+]] = arith.constant 12 : index
// CHECK-DAG:  %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:  %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:  %[[c256:.+]] = arith.constant 256 : index
// CHECK-DAG:  %[[c1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:  %[[c2:.+]] = arith.constant 2 : index
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<256x1024x2xbf16>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c256]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<256x1024x2xbf16>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c1024]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<256x1024x2xbf16>) {
// CHECK:     %[[MUL:.+]] = arith.muli %[[ARG3]], %[[c2]] : index
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG1]][%[[MUL]], %[[ARG5]]] [2, 1] [1, 1] : tensor<512x1024xbf16> to tensor<2xbf16>
// CHECK:     tensor.insert_slice %[[EXTRACT]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0] [1, 1, 2] [1, 1, 1] : tensor<2xbf16> into tensor<256x1024x2xbf16>
// CHECK-NOT: tensor.unpack
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<12x56x56x64xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c12]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<12x56x56x64xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c56]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<12x56x56x64xf32>) {
// CHECK:     scf.for %[[ARG7:.+]] = %[[c0]] to %[[c56]] step %[[c1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<12x56x56x64xf32>) {
// CHECK:       scf.for %[[ARG9:.+]] = %[[c0]] to %[[c64]] step %[[c1]] iter_args(%[[ARG10:.+]] = %[[ARG8]]) -> (tensor<12x56x56x64xf32>) {
// CHECK:         %[[ARGMAP_0:.+]] = affine.apply #map(%[[ARG9]])
// CHECK:         %[[TEMP:.+]] = tensor.empty() : tensor<1x1x1x32xf32>
// CHECK:          %[[ARGMAP_1:.+]] = affine.apply #map1(%[[ARG9]])
// CHECK:         %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[ARG3]], %[[ARGMAP_1]], %[[ARG5]], %[[ARG7]], 0] [1, 1, 1, 1, 32] [1, 1, 1, 1, 1] : tensor<12x2x56x56x32xf32> to tensor<32xf32>
// CHECK:	  %[[TRANSPOSE_BUF:.+]] = tensor.empty() : tensor<32xf32>
// CHECK:         %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32xf32>) outs(%[[TRANSPOSE_BUF]] : tensor<32xf32>) permutation = [0]
// CHECK:	  %[[INSERTED:.+]] = tensor.insert_slice %[[TRANSPOSED]] into %[[TEMP]][0, 0, 0, 0] [1, 1, 1, 32] [1, 1, 1, 1] : tensor<32xf32> into tensor<1x1x1x32xf32>
// CHECK:	  %[[EXTRACT_0:.+]] = tensor.extract_slice %[[INSERTED]][0, 0, 0, %[[ARGMAP_0]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x32xf32> to tensor<1x1x1x1xf32>
// CHECK:         tensor.insert_slice %[[EXTRACT_0]] into %[[ARG10]][%[[ARG3]], %[[ARG5]], %[[ARG7]], %[[ARG9]]] [1, 1, 1, 1] [1, 1, 1, 1] : tensor<1x1x1x1xf32> into tensor<12x56x56x64xf32>

// -----

func.func @pack_vnni(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>, %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1:tensor<32x4x4xbf16>, tensor<32x4x4xbf16>) outs(%arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0: tensor<4x4xbf16>
}

// CHECK-LABEL: func.func @pack_vnni(
// CHECK-NOT: linalg.batch_reduce_matmul
// CHECK-NOT: tensor.pack
// CHECK-DAG: %[[c4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[c32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<32x2x4x2xbf16>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c32]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<32x2x4x2xbf16>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<32x2x4x2xbf16>) {
// CHECK:     scf.for %[[ARG7:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG8:.+]] = %[[ARG6]]) -> (tensor<32x2x4x2xbf16>) {
// CHECK:       %[[AFFINEMAP:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:       %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG1]][%[[ARG3]], %[[AFFINEMAP]], %[[ARG7]]] [1, 2, 1] [1, 1, 1] : tensor<32x4x4xbf16> to tensor<2xbf16>
// CHECK:	%[[TEMP:.+]] = tensor.empty() : tensor<2xbf16>
// CHECK:	%[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<2xbf16>) outs(%[[TEMP]] : tensor<2xbf16>) permutation = [0]
// CHECK:       tensor.insert_slice %[[TRANSPOSED]] into %[[ARG8]][%[[ARG3]], %[[ARG5]], %[[ARG7]], 0] [1, 1, 1, 2] [1, 1, 1, 1] : tensor<2xbf16> into tensor<32x2x4x2xbf16>
// CHECK: tpp.brgemm

// -----

func.func @pack_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @pack_matmul(
// CHECK-NOT: linalg.matmul
// Generalized pack of the first input
// CHECK-DAG: %[[c128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[c4:.+]] = arith.constant 4 : index
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<4x4x32x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<4x4x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<4x4x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[MAP0]], %[[MAP1]]] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK:     tensor.insert_slice %[[TRANSPOSED]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x4x32x32xf32>
// Generalized pack of the second input
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<4x4x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<4x4x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG1]][%[[MAP0]], %[[MAP1]]] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK:     tensor.insert_slice %[[TRANSPOSED]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x4x32x32xf32>
// Generalized pack of the output
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<4x4x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<4x4x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG2]][%[[MAP0]], %[[MAP1]]] [32, 32] [1, 1] : tensor<128x128xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK:     tensor.insert_slice %[[TRANSPOSED]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x4x32x32xf32>
// Packed matmul
// CHECK: scf.forall
// CHECK: linalg.batch_reduce_matmul 

// -----

func.func @fold_const_pack() ->  tensor<8x2x1x1x32x32xi64> {
  %cst = arith.constant dense<1> : tensor<1x1x64x256xi64>
  %0 = tensor.empty() : tensor<8x2x1x1x32x32xi64>
  %pack = tensor.pack %cst outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %0 : tensor<1x1x64x256xi64> -> tensor<8x2x1x1x32x32xi64>
  return  %pack : tensor<8x2x1x1x32x32xi64>
}

// CHECK-LABEL: func.func @fold_const_pack(
// CHECK-NOT: tensor.pack
// CHECK: %[[CST:.+]] = arith.constant dense<1> : tensor<8x2x1x1x32x32xi64>
// CHECK-NEXT: return %[[CST]] : tensor<8x2x1x1x32x32xi64>

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>

func.func @propagate_pack_unpack(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<4x16x32x32xf32>
  %pack = tensor.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack_0 = tensor.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_1 = tensor.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_1 : tensor<4x8x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<4x8x32x32xf32>
  %unpack = tensor.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%unpack : tensor<128x256xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maxf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<128x256xf32>
  return %4 : tensor<128x256xf32>
}

// CHECK: func.func @propagate_pack_unpack(%[[ARG0:.+]]: tensor<128x512xf32>, %[[ARG1:.+]]: tensor<512x256xf32>, %[[ARG2:.+]]: tensor<128x256xf32>) ->  tensor<128x256xf32> {
// CHECK-DAG:   %[[c256:.+]] = arith.constant 256 : index
// CHECK-DAG:   %[[c128:.+]] = arith.constant 128 : index
// CHECK-DAG:   %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[c4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[c16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[c8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[c32:.+]] = arith.constant 32 : index
// CHECK-DAG:   %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<4x16x32x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<4x16x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c16]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<4x16x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = arith.muli %[[ARG3]], %[[c32]] : index
// CHECK:     %[[MAP1:.+]] = arith.muli %[[ARG5]], %[[c32]] : index
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[MAP0]], %[[MAP1]]] [32, 32] [1, 1] : tensor<128x512xf32> to tensor<32x32xf32>
// CHECK:     tensor.insert_slice %[[EXTRACT]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x16x32x32xf32>
// Generalized pack of the second input
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<8x16x32x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c8]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<8x16x32x32xf32>) {
// CHECK:   scf.for %[[ARG5]] = %[[c0]] to %[[c16]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<8x16x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG1]][%[[MAP0]], %[[MAP1]]] [32, 32] [1, 1] : tensor<512x256xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK:     tensor.insert_slice %[[TRANSPOSED]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<8x16x32x32xf32>
// Generalized pack of the output
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<4x8x32x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c4]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<4x8x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c8]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<4x8x32x32xf32>) {
// CHECK:     %[[MUL0:.+]] = arith.muli %[[ARG3]], %[[c32]] : index
// CHECK:     %[[MUL1:.+]] = arith.muli %[[ARG5]], %[[c32]] : index
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG2]][%[[MUL0]], %[[MUL1]]] [32, 32] [1, 1] : tensor<128x256xf32> to tensor<32x32xf32>
// CHECK:     tensor.insert_slice %[[EXTRACT]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<4x8x32x32xf32>
// Generic before unpack
// CHECK: linalg.generic
// Generalized unpack
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c128]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[ARG2]]) -> (tensor<128x256xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c256]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<128x256xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map2(%[[ARG3]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map2(%[[ARG5]])
// CHECK:     %[[MAP2:.+]] = affine.apply #map3(%[[ARG3]])
// CHECK:     %[[MAP3:.+]] = affine.apply #map3(%[[ARG5]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %{{[^:]+}}[%[[MAP2]], %[[MAP3]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<4x8x32x32xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK:     %[[EXTRACT1:.+]] = tensor.extract_slice %[[TRANSPOSED]][%[[MAP0]], %[[MAP1]]] [1, 1] [1, 1] : tensor<32x32xf32> to tensor<1x1xf32>
// CHECK:     tensor.insert_slice %[[EXTRACT1]] into %[[ARG6]][%[[ARG3]], %[[ARG5]]] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<128x256xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @conv_init_simplify(%arg0: tensor<1x56x56x64xf32>, %arg2: tensor<1x1x64x64xf32>, %arg3: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1x56x56x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg2 : tensor<1x56x56x64xf32>, tensor<1x1x64x64xf32>) outs(%1 : tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg3 : tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<1x56x56x64xf32>
  %4 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %3 : tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) outs(%0 : tensor<1x56x56x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
  } -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// CHECK-LABEL: func.func @conv_init_simplify(
// CHECK-NOT: linalg.fill
// CHECK-NOT: linalg.conv_2d_nhwc_hwcf
// Conv as matmul
// CHECK: scf.for
// CHECK:   linalg.matmul
// CHECK-NOT: linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @tile_and_fuse(%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>,
    %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %1 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%0: tensor<64x64xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maxf %out, %c0 : f32
        linalg.yield %2 : f32
    } -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// CHECK: func.func @tile_and_fuse(%[[ARG0:.+]]: tensor<64x64xf32>, %[[ARG1:.+]]: tensor<64x64xf32>, %[[ARG2:.+]]:  tensor<64x64xf32>) -> tensor<64x64xf32> {
// CHECK-DAG: %[[c64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[BUF:.+]] = tensor.empty() : tensor<2x2x32x32xf32>
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<2x2x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<2x2x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG0]][%[[MAP0]], %[[MAP1]]] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK:     tensor.insert_slice %[[TRANSPOSED]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<2x2x32x32xf32>
// Generalized pack of the second input
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<2x2x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<2x2x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG1]][%[[MAP0]], %[[MAP1]]] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK:     tensor.insert_slice %[[TRANSPOSED]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<2x2x32x32xf32>
// Generalized pack of the output
// CHECK: scf.for %[[ARG3:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG4:.+]] = %[[BUF]]) -> (tensor<2x2x32x32xf32>) {
// CHECK:   scf.for %[[ARG5:.+]] = %[[c0]] to %[[c2]] step %[[c1]] iter_args(%[[ARG6:.+]] = %[[ARG4]]) -> (tensor<2x2x32x32xf32>) {
// CHECK:     %[[MAP0:.+]] = affine.apply #map(%[[ARG3]])
// CHECK:     %[[MAP1:.+]] = affine.apply #map(%[[ARG5]])
// CHECK:     %[[EXTRACT:.+]] = tensor.extract_slice %[[ARG2]][%[[MAP0]], %[[MAP1]]] [32, 32] [1, 1] : tensor<64x64xf32> to tensor<32x32xf32>
// CHECK:     %[[TEMP:.+]] = tensor.empty() : tensor<32x32xf32>
// CHECK:     %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXTRACT]] : tensor<32x32xf32>) outs(%[[TEMP]] : tensor<32x32xf32>) permutation = [0, 1]
// CHECK:     tensor.insert_slice %[[TRANSPOSED]] into %[[ARG6]][%[[ARG3]], %[[ARG5]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<2x2x32x32xf32>
// Fused matmul and relu
// CHECK: scf.forall
// CHECK: linalg.batch_reduce_matmul{{.*}}ins(%{{.+}}, %{{.+}} : tensor<2x32x32xf32>, tensor<2x32x32xf32>)
// CHECK-SAME:{{.*}}outs(%{{.+}} : tensor<32x32xf32>)
// CHECK: linalg.generic{{.*}}outs(%{{.+}} : tensor<32x32xf32>)
// CHECK:   arith.maxf
