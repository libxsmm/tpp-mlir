// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=4,4 use-for-all=false" -cse | FileCheck -check-prefix=CONF1 %s

// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=1,1,1 use-for-all=false" -cse -canonicalize | FileCheck -check-prefix=CONF2 %s

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

// CONF1-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CONF1: func.func @matmul_eletwise
// CONF1-DAG: %[[C32:.+]] = arith.constant 32 : index
// CONF1-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF1-DAG: %[[C4:.+]] = arith.constant 4 : index
// CONF1: %[[LOOP:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C4]]
// CONF1-NEXT: %[[LOOP1:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C4]]
// CONF1-COUNT-1: linalg.matmul
// CONF1-COUNT-1: linalg.generic
// CONF1: scf.yield %{{.+}} : tensor<32x32xf32>
// CONF1-NEXT: }
// CONF1: scf.yield %{{.+}} : tensor<32x32xf32>
// CONF1-NEXT: }

// -----

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @blck_conv(%arg0: tensor<1x2x56x56x32xi64>, %arg1: tensor<2x2x1x1x32x32xi64>,
                     %arg2: tensor<1x2x56x56x32xi64>, %cst: tensor<2x32xi64>) -> tensor<1x2x56x56x32xi64> {
  %0 = linalg.generic {
    indexing_maps = [#map2, #map3, #map4],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "reduction"]}
    ins(%arg0, %arg1 : tensor<1x2x56x56x32xi64>, tensor<2x2x1x1x32x32xi64>) outs(%arg2 : tensor<1x2x56x56x32xi64>) {
    ^bb0(%in: i64, %in_51: i64, %out: i64):
      %151 = arith.muli %in, %in_51 : i64
      %152 = arith.addi %out, %151 : i64
      linalg.yield %152 : i64
  } -> tensor<1x2x56x56x32xi64>
  %2 = tensor.empty() : tensor<1x2x56x56x32xi64>
  %3 = linalg.generic {
    indexing_maps = [#map5, #map6],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%cst : tensor<2x32xi64>) outs(%2 : tensor<1x2x56x56x32xi64>) {
    ^bb0(%in: i64, %out: i64):
      linalg.yield %in : i64
  } -> tensor<1x2x56x56x32xi64>
  %4 = tensor.empty() : tensor<1x2x56x56x32xi64>
  %5 = linalg.generic {
    indexing_maps = [#map6, #map6, #map6],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]}
    ins(%0, %3 : tensor<1x2x56x56x32xi64>, tensor<1x2x56x56x32xi64>) outs(%4 : tensor<1x2x56x56x32xi64>) {
    ^bb0(%in: i64, %in_51: i64, %out: i64):
      %151 = arith.addi %in, %in_51 : i64
      linalg.yield %151 : i64
  } -> tensor<1x2x56x56x32xi64>
  return %5 : tensor<1x2x56x56x32xi64>
}

// CONF2: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d3)>
// CONF2-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CONF2-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CONF2-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d1)>
// CONF2-DAG: #[[MAP4:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CONF2: func.func @blck_conv(
// CONF2-SAME:  %[[ARG0:.+]]: tensor<1x2x56x56x32xi64>, %[[ARG1:.+]]: tensor<2x2x1x1x32x32xi64>,
// CONF2-SAME:  %[[ARG2:.+]]: tensor<1x2x56x56x32xi64>, %[[ARG3:.+]]: tensor<2x32xi64>)
// CONF2-DAG: %[[C56:.+]] = arith.constant 56 : index
// CONF2-DAG: %[[C2:.+]] = arith.constant 2 : index
// CONF2-DAG: %[[C0:.+]] = arith.constant 0 : index
// CONF2-DAG: %[[C1:.+]] = arith.constant 1 : index
// CONF2: %[[EMPTY_OUT_ADD:.+]] = tensor.empty() : tensor<1x2x56x56x32xi64>
// CONF2: %[[LOOP:.+]] = scf.for %[[ARG4:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CONF2-NEXT: %[[LOOP1:.+]] = scf.for %[[ARG6:.+]] = %[[C0]] to %[[C56]] step %[[C1]]
// CONF2: %[[SLICE:.+]] = tensor.extract_slice
// CONF2-SAME:  %[[ARG0]][0, 0, %[[ARG6]], 0, 0] [1, 2, 1, 56, 32] [1, 1, 1, 1, 1]
// CONF2-SAME:  : tensor<1x2x56x56x32xi64> to tensor<2x56x32xi64>
// CONF2: %[[SLICE1:.+]] = tensor.extract_slice
// CONF2-SAME:  %[[ARG1]][%[[ARG4]], 0, 0, 0, 0, 0] [1, 2, 1, 1, 32, 32] [1, 1, 1, 1, 1, 1]
// CONF2-SAME:  : tensor<2x2x1x1x32x32xi64> to tensor<2x32x32xi64>
// CONF2: %[[SLICE2:.+]] = tensor.extract_slice
// CONF2-SAME:  %[[ARG2]][0, %[[ARG4]], %[[ARG6]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1]
// CONF2-SAME:  : tensor<1x2x56x56x32xi64> to tensor<56x32xi64>
// CONF2: %[[CONV:.+]] = linalg.generic
// CONF2-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CONF2-SAME:  iterator_types = ["parallel", "parallel", "reduction", "reduction"]
// CONF2-SAME:  ins(%[[SLICE]], %[[SLICE1]]
// CONF2-SAME:  outs(%[[SLICE2]]
// CONF2: %[[SLICE3:.+]] = tensor.extract_slice
// CONF2-SAME:  %[[ARG3]][%[[ARG4]], 0] [1, 32] [1, 1]
// CONF2-SAME:  : tensor<2x32xi64> to tensor<32xi64>
// CONF2: %[[SLICE4:.+]] = tensor.extract_slice
// CONF2-SAME:  %[[EMPTY_OUT_ADD]][0, %[[ARG4]], %[[ARG6]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1]
// CONF2-SAME:  : tensor<1x2x56x56x32xi64> to tensor<56x32xi64>
// CONF2: %[[BIAS:.+]] = linalg.generic
// CONF2-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]]]
// CONF2-SAME:  iterator_types = ["parallel", "parallel"]
// CONF2-SAME:  ins(%[[SLICE3]]
// CONF2-SAME:  outs(%[[SLICE4]]
// CONF2: %[[SLICE4:.+]] = tensor.extract_slice
// CONF2-SAME:  %{{.+}}[0, %[[ARG4]], %[[ARG6]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1]
// CONF2-SAME:  : tensor<1x2x56x56x32xi64> to tensor<56x32xi64>
// CONF2: %[[ADD:.+]] = linalg.generic
// CONF2-SAME:  indexing_maps = [#[[MAP4]], #[[MAP4]], #[[MAP4]]]
// CONF2-SAME:  iterator_types = ["parallel", "parallel"]
// CONF2-SAME:  ins(%[[CONV]], %[[BIAS]]
// CONF2-SAME:  outs(%[[SLICE4]]
// CONF2: %[[INSERT:.+]] = tensor.insert_slice
// CONF2-SAME: %[[ADD]] into %{{.+}}[0, %[[ARG4]], %[[ARG6]], 0, 0] [1, 1, 1, 56, 32] [1, 1, 1, 1, 1]
// CONF2-SAME:  : tensor<56x32xi64> into tensor<1x2x56x56x32xi64>
