// RUN: tpp-opt --vector-to-xsmm  %s --split-input-file | FileCheck %s

func.func @transpose_op_2d_f32(%arg0: memref<3x5xf32>, %arg1: memref<5x3xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<3x5xf32>, vector<3x5xf32>
    %1 = vector.transpose %0, [1, 0] : vector<3x5xf32> to vector<5x3xf32>
    vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<5x3xf32>, memref<5x3xf32>
    return
}

// CHECK-LABEL: func.func @transpose_op_2d_f32(
// CHECK: %[[arg0:.*]]: memref<3x5xf32>, %[[arg1:.*]]: memref<5x3xf32>) {
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c29_i64:.*]] = arith.constant 29 : i64
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c3_i64:.*]] = arith.constant 3 : i64
// CHECK-DAG: %[[c5_i64:.*]] = arith.constant 5 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK: %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c29_i64]], %[[c1_i64]], %[[c3_i64]], %[[c5_i64]], %[[c5_i64]], %[[c3_i64]], %[[c0_i64]])
// CHECK-DAG:  %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]]
// CHECK-NEXT: %[[indexCast1:.*]] = arith.index_cast %[[intptr]]
// CHECK-NEXT: %[[inttoptr:.*]] = llvm.inttoptr %[[indexCast1]]
// CHECK-DAG:  %[[intptr0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK-NEXT: %[[indexCast2:.*]] = arith.index_cast %[[intptr0]]
// CHECK-NEXT: %[[inttoptr2:.*]] = llvm.inttoptr %[[indexCast2]]
// CHECK:   call @xsmm_unary_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr2]], %[[c0]])

// -----
func.func @transpose_op_3d_f32(%arg0: memref<5x3x5xf32>, %arg1: memref<5x5x3xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<5x3x5xf32>, vector<5x3x5xf32>
    %1 = vector.transpose %0, [0, 2, 1] : vector<5x3x5xf32> to vector<5x5x3xf32>
    vector.transfer_write %1, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<5x5x3xf32>, memref<5x5x3xf32>
    return
}
// CHECK-LABEL: func.func @transpose_op_3d_f32(
// CHECK: %[[arg0:.*]]: memref<5x3x5xf32>, %[[arg1:.*]]: memref<5x5x3xf32>) {
// CHECK-NOT: call @xsmm_unary_dispatch
// CHECK-NOT: call @xsmm_unary_invoke

// -----
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @vnni_packing_2d_bf16(%arg0: memref<32x32xbf16, strided<[512, 1], offset: ?>>, %arg1: memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]] output_shape [16, 2, 32] : memref<32x32xbf16, strided<[512, 1], offset: ?>> into memref<16x2x32xbf16, strided<[1024, 512, 1], offset: ?>>
      %0 = vector.transfer_read %expand_shape[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<16x2x32xbf16, strided<[1024, 512, 1], offset: ?>>, vector<16x2x32xbf16>
      %1 = vector.transpose %0, [0, 2, 1] : vector<16x2x32xbf16> to vector<16x32x2xbf16>
      vector.transfer_write %1, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<16x32x2xbf16>, memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>
      return
  }
}

// CHECK-LABEL: func.func @vnni_packing_2d_bf16(
// CHECK: %[[arg0:.*]]: memref<32x32xbf16, strided<[512, 1], offset: ?>>, %[[arg1:.*]]: memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>) {
// CHECK-DAG: %[[c28_i64:.*]] = arith.constant 28 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK: %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c28_i64]], %[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c512_i64]], %[[c32_i64]], %[[c0_i64]])
// CHECK-DAG: %[[expand_shape:.*]] = memref.expand_shape %[[arg0]] {{\[}}[0, 1], [2]] output_shape [16, 2, 32]
// CHECK-NEXT: %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %expand_shape
// CHECK-NEXT:  %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[expand_shape]]
// CHECK-NEXT: %[[indexCast1:.*]] = arith.index_cast %[[intptr]]
// CHECK-NEXT: %[[inttoptr:.*]] = llvm.inttoptr %[[indexCast1]]
// CHECK-DAG: %[[base_buffer_0:.*]], %[[offset_1:.*]], %[[sizes_2:.*]]:3, %[[strides_3:.*]]:3 = memref.extract_strided_metadata %[[arg1]]
// CHECK-NEXT: %[[intptr0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK-NEXT: %[[indexCast2:.*]] = arith.index_cast %[[intptr0]]
// CHECK-NEXT: %[[inttoptr2:.*]] = llvm.inttoptr %[[indexCast2]]
// CHECK:   call @xsmm_unary_invoke(%[[c2_i64]], %[[dispatch]], %[[inttoptr]], %[[offset]], %[[inttoptr2]], %[[offset_1]])

// -----
func.func @not_vnni_packing_2d_f32(%arg0: memref<32x32xf32, strided<[512, 1], offset: ?>>, %arg1: memref<16x32x2xf32, strided<[64, 2, 1], offset: ?>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]] output_shape [16, 2, 32] : memref<32x32xf32, strided<[512, 1], offset: ?>> into memref<16x2x32xf32, strided<[1024, 512, 1], offset: ?>>
    %0 = vector.transfer_read %expand_shape[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<16x2x32xf32, strided<[1024, 512, 1], offset: ?>>, vector<16x2x32xf32>
    %1 = vector.transpose %0, [0, 2, 1] : vector<16x2x32xf32> to vector<16x32x2xf32>
    vector.transfer_write %1, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<16x32x2xf32>, memref<16x32x2xf32, strided<[64, 2, 1], offset: ?>>
    return
}
// CHECK-LABEL: func.func @not_vnni_packing_2d_f32(
// CHECK: %[[arg0:.*]]: memref<32x32xf32, strided<[512, 1], offset: ?>>, %[[arg1:.*]]: memref<16x32x2xf32, strided<[64, 2, 1], offset: ?>>) {
// CHECK-NOT: call @xsmm_unary_dispatch
// CHECK-NOT: call @xsmm_unary_invoke

// -----
#map = affine_map<(d0) -> (d0 * 32)>
module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  func.func @vnni_packing_2d_bf16_forall(%arg0: memref<128x128xbf16>, %arg1: memref<4x4x16x32x2xbf16>) {
      %cst = arith.constant 0.000000e+00 : bf16
      %c0 = arith.constant 0 : index
      scf.forall (%arg2, %arg3) in (4, 4) {
        %0 = affine.apply #map(%arg3)
        %1 = affine.apply #map(%arg2)
        %subview = memref.subview %arg0[%0, %1] [32, 32] [1, 1] : memref<128x128xbf16> to memref<32x32xbf16, strided<[128, 1], offset: ?>>
        %subview_0 = memref.subview %arg1[%arg2, %arg3, 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1] : memref<4x4x16x32x2xbf16> to memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>
        %expand_shape = memref.expand_shape %subview [[0, 1], [2]] output_shape [16, 2, 32] : memref<32x32xbf16, strided<[128, 1], offset: ?>> into memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>>
        %2 = vector.transfer_read %expand_shape[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<16x2x32xbf16, strided<[256, 128, 1], offset: ?>>, vector<16x2x32xbf16>
        %3 = vector.transpose %2, [0, 2, 1] : vector<16x2x32xbf16> to vector<16x32x2xbf16>
        vector.transfer_write %3, %subview_0[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<16x32x2xbf16>, memref<16x32x2xbf16, strided<[64, 2, 1], offset: ?>>
      }
      return
  }
}

// CHECK-LABEL: func.func @vnni_packing_2d_bf16_forall(
// CHECK: %[[arg0:.*]]: memref<128x128xbf16>, %[[arg1:.*]]: memref<4x4x16x32x2xbf16>) {
// CHECK-DAG: %[[c28_i64:.*]] = arith.constant 28 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK-DAG: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[c128_i64:.*]] = arith.constant 128 : i64
// CHECK-DAG: %[[c0_i64:.*]] = arith.constant 0 : i64
// CHECK: %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c28_i64]], %[[c2_i64]], %[[c32_i64]], %[[c32_i64]], %[[c128_i64]], %[[c32_i64]], %[[c0_i64]])
// CHECK: scf.forall (%[[arg2:.*]], %[[arg3:.*]]) in (4, 4) {
// CHECK-DAG:      %[[par1:.*]] = affine.apply #map(%[[arg3]])
// CHECK-DAG:      %[[par2:.*]] = affine.apply #map(%[[arg2]])
// CHECK-NEXT:      %[[subview:.*]] = memref.subview %[[arg0]][%[[par1]], %[[par2]]] [32, 32] [1, 1]
// CHECK-DAG:      %[[subview_0:.*]] = memref.subview %[[arg1]][%[[arg2]], %[[arg3]], 0, 0, 0] [1, 1, 16, 32, 2] [1, 1, 1, 1, 1]
// CHECK:   %[[expand_shape:.*]] = memref.expand_shape %[[subview]] {{\[}}[0, 1], [2]] output_shape [16, 2, 32]
// CHECK-DAG:  %[[base_buffer:.*]], %[[offset:.*]], %[[sizes:.*]]:3, %[[strides:.*]]:3 = memref.extract_strided_metadata %[[expand_shape]]
// CHECK-DAG:  %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[expand_shape]]
// CHECK-NEXT: %[[indexCast1:.*]] = arith.index_cast %[[intptr]]
// CHECK-NEXT: %[[inttoptr:.*]] = llvm.inttoptr %[[indexCast1]]
// CHECK-DAG: %[[base_buffer_0:.*]], %[[offset_1:.*]], %[[sizes_2:.*]]:3, %[[strides_3:.*]]:3 = memref.extract_strided_metadata %[[subview_0]]
// CHECK-NEXT:  %[[intptr0:.*]] = memref.extract_aligned_pointer_as_index %[[subview_0]]
// CHECK-NEXT: %[[indexCast2:.*]] = arith.index_cast %[[intptr0]]
// CHECK-NEXT: %[[inttoptr2:.*]] = llvm.inttoptr %[[indexCast2]]
// CHECK:   func.call @xsmm_unary_invoke(%[[c2_i64]], %[[dispatch]], %[[inttoptr]], %[[offset]], %[[inttoptr2]], %[[offset_1]])
