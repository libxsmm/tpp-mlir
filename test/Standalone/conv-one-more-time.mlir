// RUN: standalone-opt %s -split-input-file -decompose-conv-to-matmul | FileCheck %s

func.func @conv(%arg0: memref<1x4x4x3xf32>, %arg1: memref<1x1x3x8xf32>, %arg2: memref<1x4x4x8xf32>) {
  // CHECK: linalg.matmul
  linalg.conv_2d_nhwc_hwcf { dilations = dense<[1,1]> : tensor<2xi64>,
                                    strides = dense<[1,1]> : tensor<2xi64> }
    ins(%arg0, %arg1: memref<1x4x4x3xf32>, memref<1x1x3x8xf32>) outs(%arg2: memref<1x4x4x8xf32>)
  return
}

// -----

func.func @conv(%arg0: memref<1x4x4x3xf32>, %arg1: memref<2x2x3x8xf32>, %arg2: memref<1x2x2x8xf32>) {
  // CHECK: linalg.matmul
  linalg.conv_2d_nhwc_hwcf { dilations = dense<[1,1]> : tensor<2xi64>,
                                    strides = dense<[1,1]> : tensor<2xi64> }
    ins(%arg0, %arg1: memref<1x4x4x3xf32>, memref<2x2x3x8xf32>) outs(%arg2: memref<1x2x2x8xf32>)
  return 
}

// -----

// CHECK: lore
func.func @conv(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) 
                                outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}
