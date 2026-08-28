// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=2,0 max-depth=1 use-for-all=false" | FileCheck -check-prefix=DEPTH1 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=2,0 max-depth=2 use-for-all=false" | FileCheck -check-prefix=DEPTH2 %s
// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="tile-sizes=2,0 max-depth=3 use-for-all=false" | FileCheck -check-prefix=DEPTH3 %s

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

// DEPTH1: func.func @matmul_sequence_fusion(
// DEPTH1-DAG: %[[C0:.+]] = arith.constant 0 : index
// DEPTH1-DAG: %[[C32:.+]] = arith.constant 32 : index
// DEPTH1-DAG: %[[C2:.+]] = arith.constant 2 : index
// DEPTH1: %{{.+}} = scf.for %[[ARG6:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// DEPTH1-COUNT-2: linalg.matmul
// DEPTH1: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// DEPTH1-COUNT-1: linalg.matmul
// DEPTH1-COUNT-1: linalg.generic

// DEPTH2: func.func @matmul_sequence_fusion(
// DEPTH2-DAG: %[[C0:.+]] = arith.constant 0 : index
// DEPTH2-DAG: %[[C32:.+]] = arith.constant 32 : index
// DEPTH2-DAG: %[[C2:.+]] = arith.constant 2 : index
// DEPTH2-COUNT-1: linalg.matmul
// DEPTH2: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// DEPTH2-COUNT-2: linalg.matmul
// DEPTH2-COUNT-1: linalg.generic

// DEPTH3: func.func @matmul_sequence_fusion(
// DEPTH3-DAG: %[[C0:.+]] = arith.constant 0 : index
// DEPTH3-DAG: %[[C32:.+]] = arith.constant 32 : index
// DEPTH3-DAG: %[[C2:.+]] = arith.constant 2 : index
// DEPTH3: %{{.+}} = scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C2]]
// DEPTH3-COUNT-3: linalg.matmul
// DEPTH3-COUNT-1: linalg.generic
