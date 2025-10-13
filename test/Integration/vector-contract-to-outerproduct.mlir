// RUN: tpp-opt %s  | tpp-run -e entry --entry-point-result=void -seed 123 -print > %t.1
// RUN: tpp-opt %s  --vector-contract-to-outerproduct  | tpp-run -e entry --entry-point-result=void -seed 123 -print > %t.2
// RUN: fpcmp -r 0.0001 %t.1 %t.2 | FileCheck %s --check-prefix=DIFF --allow-empty

// RUN: tpp-opt %s  | tpp-run -e permA --entry-point-result=void -seed 123 -print > %t.1
// RUN: tpp-opt %s  --vector-contract-to-outerproduct  | tpp-run -e permA --entry-point-result=void -seed 123 -print > %t.2
// RUN: fpcmp -r 0.0001 %t.1 %t.2 | FileCheck %s --check-prefix=DIFF-PERMA --allow-empty

// RUN: tpp-opt %s  | tpp-run -e permB --entry-point-result=void -seed 123 -print > %t.1
// RUN: tpp-opt %s  --vector-contract-to-outerproduct  | tpp-run -e permB --entry-point-result=void -seed 123 -print > %t.2
// RUN: fpcmp -r 0.0001 %t.1 %t.2 | FileCheck %s --check-prefix=DIFF-PERMA --allow-empty

// RUN: tpp-opt %s  | tpp-run -e permAB --entry-point-result=void -seed 123 -print > %t.1
// RUN: tpp-opt %s  --vector-contract-to-outerproduct  | tpp-run -e permAB --entry-point-result=void -seed 123 -print > %t.2
// RUN: fpcmp -r 0.0001 %t.1 %t.2 | FileCheck %s --check-prefix=DIFF-PERMAB --allow-empty

// DIFF-NOT: {{.}}
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

  func.func @entry(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<16x16xf32>, vector<16x16xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<16x16xf32>, vector<16x16xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<16x16xf32>, vector<16x16xf32>
    %3 = vector.contract {indexing_maps = [#map, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>} %0, %1, %2
      : vector<16x16xf32>, vector<16x16xf32> into vector<16x16xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, tensor<16x16xf32>
    return %4 : tensor<16x16xf32>
  }

// -----

// DIFF-PERMA-NOT: {{.}}
#permA0 = affine_map<(d0, d1, d2) -> (d2, d0)>
#permA1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#permA2 = affine_map<(d0, d1, d2) -> (d0, d1)>

  func.func @permA(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %3 = vector.contract {indexing_maps = [#permA0, #permA1, #permA2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>} %0, %1, %2
      : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }

// -----

// DIFF-PERMB-NOT: {{.}}
#permB0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#permB1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#permB2 = affine_map<(d0, d1, d2) -> (d0, d1)>

  func.func @permB(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %3 = vector.contract {indexing_maps = [#permB0, #permB1, #permB2],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>} %0, %1, %2
      : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }

// -----

// DIFF-PERMAB-NOT: {{.}}
#permAB0 = affine_map<(d0, d1, d2) -> (d2, d0)>
#permAB1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#permAB2 = affine_map<(d0, d1, d2) -> (d0, d1)>

  func.func @permAB(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<4x4xf32>, vector<4x4xf32>
    %3 = vector.contract {indexing_maps = [#permAB0, #permAB1, #permAB2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %0, %1, %2 : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x4xf32>, tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }

// -----
