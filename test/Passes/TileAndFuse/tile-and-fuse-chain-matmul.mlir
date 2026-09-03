// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=2,2 use-for-all=false" -canonicalize | FileCheck -check-prefix=CONF1 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=2,0 use-for-all=false" -canonicalize | FileCheck -check-prefix=CONF2 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=0,2 use-for-all=false" -canonicalize | FileCheck -check-prefix=CONF3 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=0,0 use-for-all=false" -canonicalize | FileCheck -check-prefix=CONF4 %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_sequence_fusion(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  %3 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%2: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %4 = arith.maximumf %out, %c0 : f32
        linalg.yield %4 : f32
  } -> tensor<32x32xf32>
  return %3 : tensor<32x32xf32>
}

// CONF1: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF1: func.func @matmul_sequence_fusion(
// CONF1-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF1-DAG: %[[C64:.+]] = arith.constant 64 : index
// CONF1-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF1-DAG: %[[C2:.+]] = arith.constant 2 : index
// CONF1: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// CONF1-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// CONF1: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x64xf32>, tensor<64x2xf32>)
// CONF1-SAME:  outs(%{{.+}} : tensor<2x2xf32>)
// CONF1: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// CONF1-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C64]] step %[[C2]]
// CONF1: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x32xf32>, tensor<32x2xf32>)
// CONF1-SAME:  outs({{.+}} : tensor<2x2xf32>)
// CONF1: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// CONF1-NEXT: %{{.+}} = scf.for %[[ARG8:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// CONF1: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [2, 64] [1, 1]
// CONF1-SAME       : tensor<32x64xf32> to tensor<2x64xf32>
// CONF1: %{{.+}} = tensor.extract_slice %{{.+}}[0, %[[ARG8]]] [64, 2] [1, 1]
// CONF1-SAME:      : tensor<64x32xf32> to tensor<64x2xf32>
// CONF1: %[[MUL:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x64xf32>, tensor<64x2xf32>)
// CONF1-SAME:  outs(%{{.+}} : tensor<2x2xf32>)
// CONF1: %{{.+}} = linalg.generic
// CONF1-SAME:  indexing_maps = [#[[MAP]]]
// CONF1-SAME:  outs(%[[MUL]]

// CONF2: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF2: func.func @matmul_sequence_fusion(
// CONF2-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF2-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF2-DAG: %[[C2:.+]] = arith.constant 2 : index
// CONF2: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// CONF2: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [2, 64] [1, 1]
// CONF2-SAME       : tensor<32x64xf32> to tensor<2x64xf32>
// CONF2: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [2, 32] [1, 1]
// CONF2-SAME:      : tensor<32x32xf32> to tensor<2x32xf32>
// CONF2: %[[MUL:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x64xf32>, tensor<64x32xf32>)
// CONF2-SAME:          outs(%{{.+}} : tensor<2x32xf32>)
// CONF2: %{{.+}} = tensor.extract_slice %{{.+}}[%[[ARG7]], 0] [2, 64] [1, 1]
// CONF2-SAME       : tensor<32x64xf32> to tensor<2x64xf32>
// CONF2: %[[MUL2:.+]] = linalg.matmul ins(%[[MUL]], %{{.+}} : tensor<2x32xf32>, tensor<32x64xf32>)
// CONF2-SAME:           outs(%{{.+}} : tensor<2x64xf32>)
// CONF2: %[[MUL3:.+]] = linalg.matmul ins(%[[MUL2]], %{{.+}} : tensor<2x64xf32>, tensor<64x32xf32>)
// CONF2-SAME:           outs(%{{.+}} : tensor<2x32xf32>)
// CONF2: %{{.+}} = linalg.generic
// CONF2-SAME:  indexing_maps = [#[[MAP]]]
// CONF2-SAME:  outs(%[[MUL3]]

// CONF3: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF3: func.func @matmul_sequence_fusion(
// CONF3-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF3-DAG: %[[C64:.+]] = arith.constant 64 : index
// CONF3-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF3-DAG: %[[C2:.+]] = arith.constant 2 : index
// CONF3: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// CONF3: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x64xf32>, tensor<64x2xf32>)
// CONF3-SAME:  outs(%{{.+}} : tensor<32x2xf32>)
// CONF3: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C64]] step %[[C2]]
// CONF3: %{{.+}} = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x32xf32>, tensor<32x2xf32>)
// CONF3-SAME:  outs({{.+}} : tensor<32x2xf32>)
// CONF3: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// CONF3: %{{.+}} = tensor.extract_slice %{{.+}}[0, %[[ARG7]]] [64, 2] [1, 1]
// CONF3-SAME:      : tensor<64x32xf32> to tensor<64x2xf32>
// CONF3: %[[MUL:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x64xf32>, tensor<64x2xf32>)
// CONF3-SAME:          outs(%{{.+}} : tensor<32x2xf32>)
// CONF3: %{{.+}} = linalg.generic
// CONF3-SAME:  indexing_maps = [#[[MAP]]]
// CONF3-SAME:  outs(%[[MUL]]

// CONF4: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF4: func.func @matmul_sequence_fusion(
// CONF4: %[[MUL:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<32x64xf32>, tensor<64x32xf32>)
// CONF4-SAME:                        outs(%{{.+}} : tensor<32x32xf32>)
// CONF4: %[[MUL1:.+]] = linalg.matmul ins(%[[MUL]], %{{.+}} : tensor<32x32xf32>, tensor<32x64xf32>)
// CONF4-SAME:                         outs(%{{.+}} : tensor<32x64xf32>)
// CONF4: %[[MUL2:.+]] = linalg.matmul ins(%[[MUL1]], %{{.+}} : tensor<32x64xf32>, tensor<64x32xf32>)
// CONF4-SAME:                         outs(%{{.+}} : tensor<32x32xf32>)
// CONF4: %{{.+}} = linalg.generic
// CONF4-SAME:  indexing_maps = [#[[MAP]]]
// CONF4-SAME:  outs(%[[MUL2]]
