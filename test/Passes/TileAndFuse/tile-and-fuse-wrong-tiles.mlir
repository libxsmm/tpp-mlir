// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=5,5 use-for-all=false" -cse | FileCheck %s
// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=0,0 use-for-all=false" -cse | FileCheck %s
// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=5,5,5 use-for-all=false" -cse | FileCheck %s
// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=2,0 use-for-all=false" -cse | FileCheck -check-prefix=TILE %s

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_eletwise(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  %1 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%0: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %2 = arith.maximumf %out, %c0 : f32
        linalg.yield %2 : f32
    } -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK: func.func @matmul_eletwise
// CHECK-NOT: scf.for

// TILE: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILE: func.func @matmul_eletwise(
// TILE-DAG:  %[[C32:.+]] = arith.constant 32 : index
// TILE-DAG:  %[[C2:.+]] = arith.constant 2 : index
// TILE-DAG:  %[[C0:.+]] = arith.constant 0 : index
// TILE: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// TILE: linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<2x64xf32>, tensor<64x32xf32>)
// TILE-SAME:          outs(%{{.+}} : tensor<2x32xf32>)
// TILE: %{{.+}} = linalg.generic
// TILE-SAME: indexing_maps = [#[[MAP]]],
// TILE-SAME: iterator_types = ["parallel", "parallel"]
// TILE-SAME: outs(%{{.+}} : tensor<2x32xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Note that %2 has two users. If we decide to fuse %2 and %3 we will end doing
// some recomputation (%2). For now we bail out and do not fuse. Since we anchor
// the pattern to matmul ops no fusion will happen for the element-wise.
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
  %5 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%3: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %6 = arith.maximumf %out, %c0 : f32
        linalg.yield %6 : f32
  } -> tensor<32x32xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
    ins(%2: tensor<32x32xf32>) outs(%5: tensor<32x32xf32>) {
      ^bb0(%in_0: f32, %out_0: f32):
        %8 = arith.addf %in_0, %out_0 : f32
        linalg.yield %8: f32
  } -> tensor<32x32xf32>
  return %7 : tensor<32x32xf32>
}

// CHECK: func.func @matmul_sequence_fusion
// CHECK-NOT: scf.for

// TILE: func.func @matmul_sequence_fusion
// TILE-DAG:  %[[C32:.+]] = arith.constant 32 : index
// TILE-DAG:  %[[C2:.+]] = arith.constant 2 : index
// TILE-DAG:  %[[C0:.+]] = arith.constant 0 : index
// TILE-COUNT-1: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// TILE-COUNT-3: linalg.matmul
// TILE: scf.yield %{{.+}} : tensor<32x32xf32>
// TILE-COUNT-3: linalg.generic

// -----

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
  %5 = linalg.generic {indexing_maps = [#map],
                       iterator_types = ["parallel", "parallel"]}
    outs(%3: tensor<32x32xf32>) {
      ^bb0(%out: f32):
        %6 = arith.maximumf %out, %c0 : f32
        linalg.yield %6 : f32
  } -> tensor<32x32xf32>
  %7 = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
    ins(%arg6: tensor<32x32xf32>) outs(%5: tensor<32x32xf32>) {
      ^bb0(%in_0: f32, %out_0: f32):
        %8 = arith.addf %in_0, %out_0 : f32
        linalg.yield %8: f32
  } -> tensor<32x32xf32>
  return %7 : tensor<32x32xf32>
}

// CHECK: func.func @matmul_sequence_fusion
// CHECK-NOT: scf.for

// TILE: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// TILE: func.func @matmul_sequence_fusion
// TILE-DAG:  %[[C32:.+]] = arith.constant 32 : index
// TILE-DAG:  %[[C2:.+]] = arith.constant 2 : index
// TILE-DAG:  %[[C0:.+]] = arith.constant 0 : index
// TILE-COUNT-1: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C2]]
// TILE-COUNT-3: linalg.matmul
// TILE-COUNT-3: linalg.generic
// TILE: scf.yield %{{.+}} : tensor<32x32xf32>
