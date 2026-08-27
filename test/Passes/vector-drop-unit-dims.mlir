// RUN: tpp-opt %s -vector-drop-unit-dims -split-input-file | FileCheck %s

// Check a few relevant rewrites.
// Core transforms are driven by upstream patterns.

func.func @drop_unit_dims_read_write(
    %arg0: tensor<512x128xf32>, %arg1: tensor<512x128xf32>,
    %idx: index) -> tensor<512x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%idx, %idx], %c0
    {in_bounds = [true, true]} : tensor<512x128xf32>, vector<1x64xf32>
  %1 = vector.transfer_write %0, %arg1[%idx, %idx]
    {in_bounds = [true, true]} : vector<1x64xf32>, tensor<512x128xf32>
  return %1 : tensor<512x128xf32>
}

// CHECK-LABEL: @drop_unit_dims_read_write
// CHECK: vector.transfer_read{{.*}}vector<64xf32>
// CHECK: vector.transfer_write{{.*}}vector<64xf32>

// -----

func.func @fold_unit_dim_read_shape_cast(
    %arg0: tensor<512x128xf32>, %idx: index
    ) -> vector<64xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%idx, %idx], %c0
    {in_bounds = [true, true]} : tensor<512x128xf32>, vector<1x1xf32>
  %1 = vector.shape_cast %0 : vector<1x1xf32> to vector<1xf32>
  %2 = vector.broadcast %1 : vector<1xf32> to vector<64xf32>
  return %2 : vector<64xf32>
}

// CHECK-LABEL: @fold_unit_dim_read_shape_cast
// CHECK: vector.transfer_read{{.*}}vector<1xf32>
// CHECK-NOT: vector.shape_cast
// CHECK: vector.broadcast

// -----

func.func @fold_unit_dim_shape_cast_insert_slice(
    %arg0: vector<64xf32>, %arg1: vector<4x64xf32>
    ) -> vector<4x64xf32> {
  %0 = vector.shape_cast %arg0 : vector<64xf32> to vector<1x64xf32>
  %1 = vector.insert_strided_slice %0, %arg1
    {offsets = [0, 0], strides = [1, 1]}
    : vector<1x64xf32> into vector<4x64xf32>
  return %1 : vector<4x64xf32>
}

// CHECK-LABEL: @fold_unit_dim_shape_cast_insert_slice
// CHECK-NOT: vector.shape_cast
// CHECK: vector.insert_strided_slice

// -----

func.func @fold_unit_dims_eltwise(
    %arg0 : vector<8xi32>, %arg1 : vector<1x8xi32>
    ) -> vector<8xi32> {
  %0 = vector.shape_cast %arg0 : vector<8xi32> to vector<1x8xi32>
  %1 = arith.addi %0, %arg1 : vector<1x8xi32>
  %2 = vector.shape_cast %1 : vector<1x8xi32> to vector<8xi32>
  return %2 : vector<8xi32>
}

// CHECK-LABEL: @fold_unit_dims_eltwise
// CHECK: vector.shape_cast{{.*}}to vector<8xi32>
// CHECK: arith.addi{{.*}}vector<8xi32>
// CHECK-NOT: vector.shape_cast
