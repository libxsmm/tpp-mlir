// RUN: tpp-opt %s -element-wise-fusion -tile-consumer-and-fuse-producers="use-for-all=false" | FileCheck %s

func.func @mlp(%arg0: tensor<32x64x4x4xbf16>, %arg1: tensor<128x64x4x4xbf16>, %arg2: tensor<128x4xbf16>, %arg3: tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16> {
    %0 = tensor.empty() : tensor<128x64x4x4xbf16>
    %transposed = linalg.transpose ins(%arg1 : tensor<128x64x4x4xbf16>) outs(%0 : tensor<128x64x4x4xbf16>) permutation = [0, 1, 3, 2]
    %1 = linalg.mmt4d ins(%arg0, %transposed : tensor<32x64x4x4xbf16>, tensor<128x64x4x4xbf16>) outs(%arg3 : tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16>
    %2 = tensor.empty() : tensor<32x128x4x4xbf16>
    %broadcasted = linalg.broadcast ins(%arg2 : tensor<128x4xbf16>) outs(%2 : tensor<32x128x4x4xbf16>) dimensions = [0, 2]
    %3 = linalg.add ins(%broadcasted, %1 : tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) outs(%arg3 : tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %4 = tensor.empty() : tensor<32x128x4x4xbf16>
    %5 = linalg.fill ins(%cst : bf16) outs(%4 : tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16>
    %6 = linalg.max ins(%3, %5 : tensor<32x128x4x4xbf16>, tensor<32x128x4x4xbf16>) outs(%arg3 : tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16>
    return %6 : tensor<32x128x4x4xbf16>
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

// CHECK: func.func @mlp(
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %{{.+}} = scf.for %[[I:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK: %{{.+}} = scf.for %[[J:.+]] = %[[C0]] to %[[C128]] step %[[C1]]
// CHECK-COUNT-2: linalg.generic
// CHECK: ^bb0(
// CHECK-NEXT: %{{.+}} = arith.mulf
// CHECK-NEXT: %{{.+}} = arith.addf
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(
// CHECK: %{{.+}} = arith.addf
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]]
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-NEXT: ^bb0(
// CHECK-NEXT:  %{{.+}} = arith.maximumf

