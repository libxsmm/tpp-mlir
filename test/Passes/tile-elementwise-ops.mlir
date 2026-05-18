// RUN: tpp-opt %s -tile-elementwise-ops -split-input-file | FileCheck %s
// RUN: tpp-opt %s -tile-elementwise-ops="tile-sizes=4,8" -split-input-file | FileCheck %s --check-prefix=CUSTOM

// -----
// Test 1: Basic 2D elementwise op gets tiled with default sizes [1, N]
// where N is the inner dimension of the output shape.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @tile_basic_elementwise(%arg0: tensor<32x64xf32>,
                                   %arg1: tensor<32x64xf32>,
                                   %arg2: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<32x64xf32>)
    outs(%arg2 : tensor<32x64xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %r = arith.addf %m, %c : f32
    linalg.yield %r : f32
  } -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// CHECK-LABEL: func.func @tile_basic_elementwise
// CHECK:         scf.for %{{.*}} = %c0 to %c32 step %c1
// CHECK:           tensor.extract_slice {{.*}} [1, 64]
// CHECK:           linalg.generic
// CHECK-SAME:        ins({{.*}} : tensor<1x64xf32>, tensor<1x64xf32>)
// CHECK-SAME:        outs({{.*}} : tensor<1x64xf32>)
// CHECK:             arith.mulf
// CHECK:             arith.addf
// CHECK:           tensor.insert_slice

// CUSTOM-LABEL: func.func @tile_basic_elementwise
// CUSTOM:         scf.for %{{.*}} step %c4
// CUSTOM:           scf.for %{{.*}} step %c8
// CUSTOM:             linalg.generic
// CUSTOM-SAME:          ins({{.*}} : tensor<4x8xf32>, tensor<4x8xf32>)

// -----

// Test 2: Single-op body (e.g., just yield of an arg) should NOT be tiled.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @no_tile_single_op_body(%arg0: tensor<32x64xf32>,
                                   %arg1: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<32x64xf32>)
    outs(%arg1 : tensor<32x64xf32>) {
  ^bb0(%a: f32, %c: f32):
    linalg.yield %a : f32
  } -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// CHECK-LABEL: func.func @no_tile_single_op_body
// CHECK-NOT:     scf.for
// CHECK:         linalg.generic
// CHECK-SAME:      ins({{.*}} : tensor<32x64xf32>) outs({{.*}} : tensor<32x64xf32>)

// -----

// Test 3: Op with reduction iterator should NOT be tiled (not elementwise).

#mapA = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapB = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapC = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @no_tile_reduction(%arg0: tensor<32x64xf32>,
                              %arg1: tensor<64x16xf32>,
                              %arg2: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %0 = linalg.generic {
    indexing_maps = [#mapA, #mapB, #mapC],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x16xf32>)
    outs(%arg2 : tensor<32x16xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %m = arith.mulf %a, %b : f32
    %r = arith.addf %m, %c : f32
    linalg.yield %r : f32
  } -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// CHECK-LABEL: func.func @no_tile_reduction
// CHECK-NOT:     scf.for
// CHECK:         linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction"]

// -----

// Test 4: Op without inputs should NOT be tiled.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @no_tile_no_inputs(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %cst = arith.constant 1.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel"]
  } outs(%arg0 : tensor<32x64xf32>) {
  ^bb0(%c: f32):
    linalg.yield %cst : f32
  } -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// CHECK-LABEL: func.func @no_tile_no_inputs
// CHECK-NOT:     scf.for
// CHECK:         linalg.generic

// -----

// Test 5: 3D elementwise op should NOT be tiled (rank != 2).

#map3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @no_tile_3d(%arg0: tensor<4x32x64xf32>,
                       %arg1: tensor<4x32x64xf32>) -> tensor<4x32x64xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map3d, #map3d],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<4x32x64xf32>)
    outs(%arg1 : tensor<4x32x64xf32>) {
  ^bb0(%a: f32, %c: f32):
    %r = math.exp %a : f32
    linalg.yield %r : f32
  } -> tensor<4x32x64xf32>
  return %0 : tensor<4x32x64xf32>
}

// CHECK-LABEL: func.func @no_tile_3d
// CHECK-NOT:     scf.for
// CHECK:         linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "parallel"]

// -----

// Test 6: Dequantization-like elementwise pattern (multi-op body) should be tiled.
// Mimics the i8 dequant pattern: combined_scale * sitofp(dot)

#map = affine_map<(d0, d1) -> (d0, d1)>
#mapR = affine_map<(d0, d1) -> (d0)>
#mapC = affine_map<(d0, d1) -> (d1)>

func.func @tile_dequant_eltwise(%dot: tensor<128x768xi32>,
                                 %iScale: tensor<128xf32>,
                                 %wScale: tensor<768xf32>,
                                 %out: tensor<128x768xf32>) -> tensor<128x768xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #mapR, #mapC, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%dot, %iScale, %wScale
        : tensor<128x768xi32>, tensor<128xf32>, tensor<768xf32>)
    outs(%out : tensor<128x768xf32>) {
  ^bb0(%d: i32, %is: f32, %ws: f32, %o: f32):
    %c = arith.mulf %is, %ws : f32
    %f = arith.sitofp %d : i32 to f32
    %r = arith.mulf %f, %c : f32
    linalg.yield %r : f32
  } -> tensor<128x768xf32>
  return %0 : tensor<128x768xf32>
}

// CHECK-LABEL: func.func @tile_dequant_eltwise
// CHECK:         scf.for %{{.*}} = %c0 to %c128 step %c1
// CHECK:           linalg.generic
// CHECK-SAME:        outs({{.*}} : tensor<1x768xf32>)
// CHECK:             arith.mulf
// CHECK:             arith.sitofp
// CHECK:             arith.mulf

// CUSTOM-LABEL: func.func @tile_dequant_eltwise
// CUSTOM:         scf.for %{{.*}} step %c4
// CUSTOM:           scf.for %{{.*}} step %c8
// CUSTOM:             linalg.generic
// CUSTOM-SAME:          outs({{.*}} : tensor<4x8xf32>)

