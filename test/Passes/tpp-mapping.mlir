// RUN: tpp-opt %s -tpp-mapping -split-input-file | FileCheck %s
// RUN: tpp-sched --bundle=tpp-mapping %s --split-input-file | FileCheck %s

func.func @pack_vnni(%arg0: tensor<32x4x4xbf16>, %arg1: tensor<32x4x4xbf16>, %arg2: tensor<4x4xbf16>) -> tensor<4x4xbf16>{
  %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1:tensor<32x4x4xbf16>, tensor<32x4x4xbf16>) outs(%arg2:tensor<4x4xbf16>) -> tensor<4x4xbf16>
  return %0: tensor<4x4xbf16>
}

// CHECK-LABEL: func.func @pack_vnni(
// CHECK-NOT: linalg.batch_reduce_matmul
// CHECK: linalg.generic

// -----

func.func @pack_matmul(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: pack_matmul
// CHECK-COUNT-2: linalg.pack
// Packed matmul
// CHECK:    %{{.+}} = scf.forall (%{{.+}}, %{{.+}}) in (4, 4)
// CHECK:     %{{.+}} = linalg.batch_reduce_matmul ins(%{{.+}}, %{{.+}} : tensor<4x32x32xf32>, tensor<4x32x32xf32>)
// CHECK-SAME:          outs(%{{.+}} : tensor<32x32xf32>) -> tensor<32x32xf32>

// -----

func.func @fold_const_pack() ->  tensor<8x2x1x1x32x32xi64> {
  %cst = arith.constant dense<1> : tensor<1x1x64x256xi64>
  %0 = tensor.empty() : tensor<8x2x1x1x32x32xi64>
  %pack = linalg.pack %cst outer_dims_perm = [3, 2, 0, 1] inner_dims_pos = [2, 3] inner_tiles = [32, 32] into %0 : tensor<1x1x64x256xi64> -> tensor<8x2x1x1x32x32xi64>
  return  %pack : tensor<8x2x1x1x32x32xi64>
}

// CHECK-LABEL: func.func @fold_const_pack(
// CHECK-NOT: linalg.pack
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
  %pack = linalg.pack %arg0 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %0 : tensor<128x512xf32> -> tensor<4x16x32x32xf32>
  %1 = tensor.empty() : tensor<8x16x32x32xf32>
  %pack_0 = linalg.pack %arg1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %1 : tensor<512x256xf32> -> tensor<8x16x32x32xf32>
  %2 = tensor.empty() : tensor<4x8x32x32xf32>
  %pack_1 = linalg.pack %arg2 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %2 : tensor<128x256xf32> -> tensor<4x8x32x32xf32>
  %3 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>) outs(%pack_1 : tensor<4x8x32x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %5 = arith.mulf %in, %in_2 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
  } -> tensor<4x8x32x32xf32>
  %unpack = linalg.unpack %3 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %arg2 : tensor<4x8x32x32xf32> -> tensor<128x256xf32>
  %4 = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel"]} outs(%unpack : tensor<128x256xf32>) {
    ^bb0(%out: f32):
      %5 = arith.maximumf %out, %cst : f32
      linalg.yield %5 : f32
  } -> tensor<128x256xf32>
  return %4 : tensor<128x256xf32>
}

// CHECK-LABEL: propagate_pack_unpack
// CHECK: linalg.batch_reduce_matmul
// CHECK-NOT: linalg.unpack
// CHECK: linalg.generic

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
        %2 = arith.maximumf %out, %c0 : f32
        linalg.yield %2 : f32
    } -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}

// CHECK-LABEL: tile_and_fuse(
// CHECK-COUNT-2: linalg.pack
// Fused matmul and relu
// CHECK: scf.forall
// CHECK: linalg.batch_reduce_matmul{{.*}}ins(%{{.+}}, %{{.+}} : tensor<2x32x32xf32>, tensor<2x32x32xf32>)
// CHECK-SAME:{{.*}}outs(%{{.+}} : tensor<32x32xf32>)
// CHECK: linalg.generic{{.*}}outs(%{{.+}} : tensor<32x32xf32>)
// CHECK:   arith.maximumf
