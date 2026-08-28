// RUN: tpp-opt %s -split-input-file -tile-consumer-and-fuse-producers="tile-sizes=2,2" -cse | FileCheck %s

func.func @matmul_sequence_only_tiling_fusion(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x64xf32>, %arg4: tensor<32x64xf32>,
    %arg5: tensor<64x32xf32>, %arg6: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x64xf32>)
    outs(%arg4 : tensor<32x64xf32>) -> tensor<32x64xf32> // [M, N1] * [N1, N2]
  %2 = linalg.matmul ins(%1, %arg5 : tensor<32x64xf32>, tensor<64x32xf32>)
    outs(%arg6 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N2] * [N2, N3]
  return %2 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @matmul_sequence_only_tiling_fusion
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 64) step (2, 2)
// CHECK: linalg.matmul
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_eletwise_matmul_and_relu(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
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

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_eletwise_matmul_and_relu
// CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]} outs({{.+}} : tensor<2x2xf32>)

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @matmul_eletwise_blk_matmul(%arg0: tensor<4x4x32x32xf32>, %arg1: tensor<4x4x32x32xf32>,
    %arg2: tensor<4x4x32x32xf32>) -> tensor<4x4x32x32xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
      ins(%arg0, %arg1 : tensor<4x4x32x32xf32>, tensor<4x4x32x32xf32>)
      outs(%arg2 : tensor<4x4x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_2 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<4x4x32x32xf32>
  %3 = linalg.generic {
      indexing_maps = [#map3],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      outs(%0 : tensor<4x4x32x32xf32>) {
    ^bb0(%out: f32):
      %4 = arith.maximumf %out, %c0 : f32
      linalg.yield %4 : f32
  } -> tensor<4x4x32x32xf32>
  return %3 : tensor<4x4x32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func.func @matmul_eletwise_blk_matmul(
// CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (4, 4) step (2, 2)
// CHECK-COUNT-2: linalg.generic

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_sequence_fusion_with_relu(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
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

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_sequence_fusion_with_relu
// CHECK-COUNT-2: linalg.matmul
// CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]]],
// CHECK-SAME:  iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  outs({{.+}} : tensor<2x2xf32>)

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_sequence_fusion_with_eltwise(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>,
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

// CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: matmul_sequence_fusion_with_eltwise
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 64) step (2, 2)
// CHECK: linalg.matmul
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]]], iterator_types = ["parallel", "parallel"]
// CHECK: ^bb0(
// CHECK-NEXT: arith.maximumf
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]]], iterator_types = ["parallel", "parallel"]
// CHECK: ^bb0(
// CHECK-NEXT: arith.maximumf
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]
// CHECK: ^bb0(
// CHECK-NEXT: arith.addf

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Note that %2 has two users, and we cannot fuse across fusion groups. Thus
// we expect not to fuse and only tile each matmul.
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

// CHECK-LABEL: func.func @matmul_sequence_fusion(
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 64) step (2, 2)
// CHECK: linalg.matmul
// CHECK: %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @matmul_sequence_fusion_1(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>,
    %arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>, %arg4: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N0] * [N0, N1]
  %1 = linalg.matmul ins(%0, %arg3 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%arg4 : tensor<32x32xf32>) -> tensor<32x32xf32> // [M, N1] * [N1, N2]
  %2 = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
    ins(%0: tensor<32x32xf32>) outs(%1: tensor<32x32xf32>) {
      ^bb0(%in_0: f32, %out_0: f32):
        %3 = arith.addf %in_0, %out_0 : f32
        linalg.yield %3: f32
  } -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_sequence_fusion_1(
// CHECK: linalg.matmul
// CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (32, 32) step (2, 2)
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:  ins({{.+}}: tensor<2x2xf32>) outs({{.+}} : tensor<2x2xf32>)
// CHECK: ^bb0(
// CHECK-NEXT: arith.addf

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>

// CHECK-LABEL: func.func @mlp
func.func @mlp(%arg0: tensor<8x112x32x32xbf16>, %arg1: tensor<112x112x32x32xbf16>,
    %arg2: tensor<3584xbf16>, %arg3: tensor<8x112x32x32xbf16>, %arg4: tensor<112x112x32x32xbf16>,
    %arg5: tensor<3584xbf16>, %arg6: tensor<8x112x32x32xbf16>, %arg7: tensor<112x112x32x32xbf16>,
    %arg8: tensor<3584xbf16>, %arg9: tensor<8x112x32x32xbf16> , %arg10: tensor<112x112x32x32xbf16>,
    %arg11: tensor<3584xbf16>, %arg12: tensor<8x112x32x32xbf16> , %arg13: tensor<112x112x32x32xbf16>,
    %arg14: tensor<3584xbf16>, %arg15: tensor<8x112x32x32xbf16> , %arg16: tensor<112x112x32x32xbf16>,
    %arg17: tensor<3584xbf16>, %arg18: tensor<8x112x32x32xbf16>, %arg19: tensor<112x112x32x32xbf16>,
    %arg20: tensor<3584xbf16>, %arg21: tensor<8x112x32x32xbf16>  , %arg22: tensor<112x112x32x32xbf16>,
    %arg23: tensor<3584xbf16>, %arg24: tensor<8x112x32x32xbf16>  , %arg25: tensor<112x112x32x32xbf16>,
    %arg26: tensor<3584xbf16>, %arg27: tensor<8x112x32x32xbf16> , %arg28: tensor<112x112x32x32xbf16>,
    %arg29: tensor<3584xbf16>, %arg30: tensor<8x112x32x32xbf16>  ) -> tensor<8x112x32x32xbf16> {

  %cst = arith.constant 0.000000e+00 : bf16
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg3 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded = tensor.expand_shape %arg2 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %1 = tensor.empty() : tensor<8x112x32x32xbf16>
  %2 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %expanded : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%1 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %3 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<8x112x32x32xbf16>) outs(%1 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %4 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%3, %arg4 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg6 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded2 = tensor.expand_shape %arg5 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %5 = tensor.empty() : tensor<8x112x32x32xbf16>
  %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%4, %expanded2 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%5 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : tensor<8x112x32x32xbf16>) outs(%5 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%7, %arg7 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg9 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded3 = tensor.expand_shape %arg8 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %9 = tensor.empty() : tensor<8x112x32x32xbf16>
  %10 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%8, %expanded3 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%9 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %11 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10 : tensor<8x112x32x32xbf16>) outs(%9 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%11, %arg10 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg12 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded4 = tensor.expand_shape %arg11 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %13 = tensor.empty() : tensor<8x112x32x32xbf16>
  %14 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%12, %expanded4 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%13 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %15 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%14 : tensor<8x112x32x32xbf16>) outs(%13 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %16 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%15, %arg13 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg15 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded5 = tensor.expand_shape %arg14 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %17 = tensor.empty() : tensor<8x112x32x32xbf16>
  %18 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%16, %expanded5 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%17 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %19 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%18 : tensor<8x112x32x32xbf16>) outs(%17 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
   %20 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%19, %arg16 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg18 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded6 = tensor.expand_shape %arg17 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %21 = tensor.empty() : tensor<8x112x32x32xbf16>
  %22 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %expanded6 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%21 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %23 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22 : tensor<8x112x32x32xbf16>) outs(%21 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %24 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%23, %arg19 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg21 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded7 = tensor.expand_shape %arg20 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %25 = tensor.empty() : tensor<8x112x32x32xbf16>
  %26 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%24, %expanded7 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%25 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %27 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26 : tensor<8x112x32x32xbf16>) outs(%25 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %28 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%27, %arg22 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg24 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded8 = tensor.expand_shape %arg23 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %29 = tensor.empty() : tensor<8x112x32x32xbf16>
  %30 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %expanded8 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%29 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %31 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<8x112x32x32xbf16>) outs(%29 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %32 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%31, %arg25 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg27 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded9 = tensor.expand_shape %arg26 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %33 = tensor.empty() : tensor<8x112x32x32xbf16>
  %34 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %expanded9 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%33 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %35 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%34 : tensor<8x112x32x32xbf16>) outs(%33 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %36 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%35, %arg28 : tensor<8x112x32x32xbf16>, tensor<112x112x32x32xbf16>) outs(%arg30 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %mul = arith.mulf %in, %in_0 : bf16
      %add = arith.addf %out, %mul : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %expanded10 = tensor.expand_shape %arg29 [[0, 1]] output_shape [112, 32] : tensor<3584xbf16> into tensor<112x32xbf16>
  %37 = tensor.empty() : tensor<8x112x32x32xbf16>
  %38 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%36, %expanded10 : tensor<8x112x32x32xbf16>, tensor<112x32xbf16>) outs(%37 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %add = arith.addf %in, %in_0 : bf16
      linalg.yield %add : bf16
  } -> tensor<8x112x32x32xbf16>
  %39 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38 : tensor<8x112x32x32xbf16>) outs(%37 : tensor<8x112x32x32xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %max = arith.maximumf %in, %cst : bf16
      linalg.yield %max : bf16
  } -> tensor<8x112x32x32xbf16>
  // CHECK: %{{.+}} = scf.forall (%[[I:.+]], %[[J:.+]]) = (0, 0) to (8, 112) step (2, 2)
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.mulf
  // CHECK-NEXT:    arith.addf
  // CHECK: %{{.+}} = linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP4]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.addf
  // CHECK: linalg.generic
  // CHECK-SAME:  indexing_maps = [#[[MAP3]], #[[MAP3]]]
  // CHECK-SAME:  iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  // CHECK:       ^bb0(
  // CHECK-NEXT:    arith.maximumf
  // CHECK: scf.forall.in_parallel
  %c0 = arith.constant 0: index
  %d1 = arith.constant -1.0 : bf16
  %subview_3 = tensor.extract_slice %39[%c0, %c0, 0, 0][1, 1, 4, 4][1, 1, 1, 1] : tensor<8x112x32x32xbf16> to tensor<4x4xbf16>
  %v0 = vector.transfer_read %subview_3[%c0, %c0], %d1 : tensor<4x4xbf16>, vector<4x4xbf16>
  %f1 = arith.extf %v0 : vector<4x4xbf16> to vector<4x4xf32>
  vector.print %f1 : vector<4x4xf32>
  return %39 : tensor<8x112x32x32xbf16>
}
