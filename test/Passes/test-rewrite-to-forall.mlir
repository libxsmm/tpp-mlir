// RUN: tpp-opt %s -mlir-disable-threading=true -pass-pipeline="builtin.module(func.func(test-scf-for-rewrite))" | FileCheck %s

// CHECK-LABEL: test
func.func @test() {
  return
}

// CHECK-LABEL: missing_attribute
func.func @missing_attribute(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: scf.forall
  %outer = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg2 = %arg1) -> (tensor<3x3xf32>) {
    %inner = scf.for %j = %c0 to %c3 step %c1 iter_args(%arg3 = %arg2) -> (tensor<3x3xf32>) {
      %source = tensor.extract %arg0[%i, %j] : tensor<3x3xf32>
      %dest = tensor.insert %source into %arg3[%i, %j] : tensor<3x3xf32>
      scf.yield %dest : tensor<3x3xf32>
    }
    scf.yield %inner : tensor<3x3xf32>
  }
  return %outer : tensor<3x3xf32>
}

// CHECK-LABEL: no_extract_slice
func.func @no_extract_slice(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: scf.forall
  %outer = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg2 = %arg1) -> (tensor<3x3xf32>) {
    %inner = scf.for %j = %c0 to %c3 step %c1 iter_args(%arg3 = %arg2) -> (tensor<3x3xf32>) {
      %source = tensor.extract %arg0[%i, %j] : tensor<3x3xf32>
      %dest = tensor.insert %source into %arg3[%i, %j] : tensor<3x3xf32>
      scf.yield %dest : tensor<3x3xf32>
    }
    scf.yield %inner : tensor<3x3xf32>
  } {parallel = "root"}
  return %outer : tensor<3x3xf32>
}

// CHECK-LABEL: expect_to_convert_3d
func.func @expect_to_convert_3d(%arg0: tensor<3x3x3xf32>,
                                            %arg1: tensor<3x3x3xf32>) -> tensor<3x3x3xf32> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.forall
  %outer = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg2 = %arg1) -> (tensor<3x3x3xf32>) {
    %middle = scf.for %j = %c0 to %c3 step %c1 iter_args(%arg3 = %arg2) -> (tensor<3x3x3xf32>) {
      %inner = scf.for %k = %c0 to %c3 step %c1 iter_args(%arg4 = %arg3) -> (tensor<3x3x3xf32>) {
        %source = tensor.extract_slice %arg0[%i, %j, %k][1, 1, 1][1, 1, 1] : tensor<3x3x3xf32> to tensor<f32>
        %dest = tensor.insert_slice %source into %arg4[%i, %j, %k][1, 1, 1][1, 1, 1] : tensor<f32> into tensor<3x3x3xf32>
        scf.yield %dest : tensor<3x3x3xf32>
      }
      scf.yield %inner : tensor<3x3x3xf32>
    }
    scf.yield %middle : tensor<3x3x3xf32>
  } {parallel = "root"}
  return %outer : tensor<3x3x3xf32>
}

// CHECK-LABEL: expect_to_convert_2d
func.func @expect_to_convert_2d(%arg0: tensor<3x3xf32>,
                                            %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.forall
  %outer = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg2 = %arg1) -> (tensor<3x3xf32>) {
    %inner = scf.for %j = %c0 to %c3 step %c1 iter_args(%arg3 = %arg2) -> (tensor<3x3xf32>) {
      %source = tensor.extract_slice %arg0[%i, %j][1, 1][1, 1] : tensor<3x3xf32> to tensor<f32>
      %dest = tensor.insert_slice %source into %arg3[%i, %j][1, 1][1, 1] : tensor<f32> into tensor<3x3xf32>
      scf.yield %dest : tensor<3x3xf32>
    }
    scf.yield %inner : tensor<3x3xf32>
  } {parallel = "root"}
  return %outer : tensor<3x3xf32>
}

// CHECK-LABEL: expect_to_convert_1d
func.func @expect_to_convert_1d(%arg0: tensor<3xf32>,
                                   %arg1: tensor<3xf32>) -> tensor<3xf32> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.forall
  %outer = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg2 = %arg1) -> (tensor<3xf32>) {
    %source = tensor.extract_slice %arg0[%i][1][1] : tensor<3xf32> to tensor<f32>
    %dest = tensor.insert_slice %source into %arg2[%i][1][1] : tensor<f32> into tensor<3xf32>
    scf.yield %dest : tensor<3xf32>
  } {parallel = "root"}
  return %outer : tensor<3xf32>
}

// CHECK-LABEL: no_top_level
func.func @no_top_level(%arg0: tensor<3x3xf32>,
                        %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  // CHECK: scf.for
  %outer = scf.for %i = %c0 to %c3 step %c1 iter_args(%arg2 = %arg1) -> (tensor<3x3xf32>) {
    // CHECK: scf.forall
    %inner = scf.for %j = %c0 to %c3 step %c1 iter_args(%arg3 = %arg2) -> (tensor<3x3xf32>) {
      %source = tensor.extract_slice %arg0[%i, %j][1, 1][1, 1] : tensor<3x3xf32> to tensor<f32>
      %dest = tensor.insert_slice %source into %arg3[%i, %j][1, 1][1, 1] : tensor<f32> into tensor<3x3xf32>
      scf.yield %dest : tensor<3x3xf32>
    } {parallel = "root"}
    scf.yield %inner : tensor<3x3xf32>
  }
  return %outer : tensor<3x3xf32>
}
