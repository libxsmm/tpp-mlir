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
// CHECK:         tensor.extract_slice{{[^:]+}}: tensor<1x1x3x8xf32> to tensor<3x8xf32>
// CHECK:         linalg.matmul{{.*}} -> tensor<3x8xf32>
// CHECK:         tensor.insert_slice{{[^:]+}}: tensor<3x8xf32> into tensor<1x1x3x8xf32>
// CHECK:         tensor.insert_slice{{[^:]+}}: tensor<1x1x3x8xf32> into tensor<1x3x3x8xf32>
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
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[C111:.+]] = arith.constant 111 : index
// Conv as matmul
// CHECK-COUNT-3: linalg.pack
// CHECK: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C111]] step %[[C1]]
// CHECK-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK:   linalg.matmul
// CHECK: linalg.unpack

// -----

func.func @conv_2d_nchw_fchw(%i: tensor<14x512x28x28xf32>, %f: tensor<1024x512x1x1xf32>,
                %o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32> {
  %0 = linalg.conv_2d_nchw_fchw ins(%i, %f: tensor<14x512x28x28xf32>, tensor<1024x512x1x1xf32>) outs(%o: tensor<14x1024x28x28xf32>) -> tensor<14x1024x28x28xf32>
  return %0: tensor<14x1024x28x28xf32>
}

// CHECK-LABEL: conv_2d_nchw_fchw
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C14:.+]] = arith.constant 14 : index
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[C28:.+]] = arith.constant 28 : index
// CHECK-COUNT-3: linalg.pack
// CHECK: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C14]] step %[[C1]]
// CHECK-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C28]] step %[[C1]]
// CHECK-NEXT: %{{.+}} = scf.for %{{.+}} = %[[C0]] to %[[C16]] step %[[C1]]
// CHECK: linalg.matmul
// CHECK: linalg.unpack

// -----

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
