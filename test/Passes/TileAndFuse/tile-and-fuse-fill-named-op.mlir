// RUN: tpp-opt %s -tile-consumer-and-fuse-producers | FileCheck %s


// This test checks for the fusion of bias and relu with matmul.
func.func @fuse_fill(%arg0: tensor<8x32x32x32xf32>, %arg1: tensor<32x32x32x32xf32>, %arg4: tensor<32x32x32x32xf32>) -> tensor<8x32x32x32xf32> {
    %cst_d = arith.constant dense<1.000000e+00> : tensor<32x32x32x32xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<8x32x32x32xf32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<8x32x32x32xf32>
    %0 = tensor.empty() : tensor<32x32x32x32xf32>
    %emt = tensor.empty() : tensor<8x32x32x32xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%emt : tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32>
    %transposed = linalg.transpose ins(%arg1 : tensor<32x32x32x32xf32>) outs(%0 : tensor<32x32x32x32xf32>) permutation = [0, 1, 3, 2] 
    %1 = linalg.mmt4d ins(%arg0, %transposed : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%fill : tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32>
    %2 = tensor.empty() : tensor<8x32x32x32xf32>
    %3 = linalg.add ins(%cst_1, %1 : tensor<8x32x32x32xf32>, tensor<8x32x32x32xf32>) outs(%1 : tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32>
    %6 = linalg.max ins(%3, %cst_0 : tensor<8x32x32x32xf32>, tensor<8x32x32x32xf32>) outs(%3 : tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32>

    %7 = tensor.empty() : tensor<32x32x32x32xf32>
    %transposed_0 = linalg.transpose ins(%arg4 : tensor<32x32x32x32xf32>) outs(%7 : tensor<32x32x32x32xf32>) permutation = [0, 1, 3, 2] 
    %8 = linalg.mmt4d ins(%6, %transposed_0 : tensor<8x32x32x32xf32>, tensor<32x32x32x32xf32>) outs(%fill : tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32>
    %9 = tensor.empty() : tensor<8x32x32x32xf32>
    %10 = linalg.add ins(%cst_1, %8 : tensor<8x32x32x32xf32>, tensor<8x32x32x32xf32>) outs(%8 : tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %13 = linalg.max ins(%10, %cst_0 : tensor<8x32x32x32xf32>, tensor<8x32x32x32xf32>) outs(%10 : tensor<8x32x32x32xf32>) -> tensor<8x32x32x32xf32>
    
    return %13 : tensor<8x32x32x32xf32>
  }

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @fuse_fill

// CHECK:           %{{.+}} = linalg.transpose ins(%{{.+}} : tensor<32x32x32x32xf32>) outs(%{{.+}} : tensor<32x32x32x32xf32>) permutation = [0, 1, 3, 2]
// CHECK-NEXT:        %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) in (8, 32) shared_outs(%{{.+}} = %{{.+}}) -> (tensor<8x32x32x32xf32>) {
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.mulf
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.mulf
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.mulf
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.mulf
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.maximumf

// CHECK:           %{{.+}} = linalg.transpose ins(%{{.+}} : tensor<32x32x32x32xf32>) outs(%{{.+}} : tensor<32x32x32x32xf32>) permutation = [0, 1, 3, 2]
// CHECK-NEXT:        %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) in (8, 32) shared_outs(%{{.+}} = %{{.+}}) -> (tensor<8x32x32x32xf32>) {
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.mulf
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.mulf
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.mulf
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.fill ins(%{{.+}} : f32) outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.mulf
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.addf
// CHECK:               %{{.+}} = linalg.generic
// CHECK:                 %{{.+}} = arith.maximumf

// -----
