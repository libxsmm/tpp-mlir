// RUN: tpp-opt --vector-to-xsmm  %s --split-input-file | FileCheck %s
// RUN: tpp-sched --bundle=vector-to-xsmm  %s --split-input-file | FileCheck %s

func.func @identity(%arg0: memref<512xf32>, %arg1: memref<128x512xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<512xf32>, vector<512xf32>
  %1 = vector.broadcast %0 : vector<512xf32> to vector<128x512xf32>
  vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<128x512xf32>, memref<128x512xf32>
  return
}

// CHECK-LABEL:  func.func @identity(
// CHECK: %[[arg0:.*]]: memref<512xf32>, %[[arg1:.*]]: memref<128x512xf32>) {
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c128_i64:.*]] = arith.constant 128 : i64
// CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK:      %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c1_i64]], %[[c1_i64]], %[[c128_i64]], %[[c512_i64]], %[[c512_i64]], %[[c512_i64]], %[[c4_i64]])
// CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]]
// CHECK:      %[[indexcast:.*]] = arith.index_cast %[[intptr]]
// CHECK:      %[[inttoptr:.*]] = llvm.inttoptr %[[indexcast]]
// CHECK:      %[[intptr_0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK:      %[[indexcast3:.*]] = arith.index_cast %[[intptr_0]]
// CHECK:      %[[inttoptr4:.*]] = llvm.inttoptr %[[indexcast3]]
// CHECK:      call @xsmm_unary_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr4]], %[[c0]])

// -----

func.func @identity_2d(%arg0: memref<512xf32>, %arg1: memref<128x4x512xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<512xf32>, vector<512xf32>
  %1 = vector.broadcast %0 : vector<512xf32> to vector<128x4x512xf32>
  vector.transfer_write %1, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<128x4x512xf32>, memref<128x4x512xf32>
  return
}
// CHECK-LABEL:  func.func @identity_2d(
// CHECK: %[[arg0:.*]]: memref<512xf32>, %[[arg1:.*]]: memref<128x4x512xf32>) {
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c2048_i64:.*]] = arith.constant 2048 : i64
// CHECK:      %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c1_i64]], %[[c1_i64]], %[[c512_i64]], %[[c512_i64]], %[[c2048_i64]], %[[c512_i64]], %[[c4_i64]])
// CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]]
// CHECK:      %[[indexcast:.*]] = arith.index_cast %[[intptr]]
// CHECK:      %[[inttoptr:.*]] = llvm.inttoptr %[[indexcast]]
// CHECK:      %[[intptr_0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK:      %[[indexcast3:.*]] = arith.index_cast %[[intptr_0]]
// CHECK:      %[[inttoptr4:.*]] = llvm.inttoptr %[[indexcast3]]
// CHECK:      call @xsmm_unary_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr4]], %[[c0]])

// -----

func.func @identity_subview_copy(%arg0: memref<128x1xf32>, %arg1: memref<512x128xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %subview = memref.subview %arg0[0, 0] [128, 1] [1, 1] : memref<128x1xf32> to memref<128xf32, strided<[1]>>
  %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<128xf32, strided<[1]>>, vector<128xf32>
  %1 = vector.broadcast %0 : vector<128xf32> to vector<512x128xf32>
  vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<512x128xf32>, memref<512x128xf32>
  return
}

// CHECK-LABEL:  func.func @identity_subview_copy(
// CHECK: %[[arg0:.*]]: memref<128x1xf32>, %[[arg1:.*]]: memref<512x128xf32>) {
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c128_i64:.*]] = arith.constant 128 : i64
// CHECK-DAG: %[[c512_i64:.*]] = arith.constant 512 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK:      %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c1_i64]], %[[c1_i64]], %[[c512_i64]], %[[c128_i64]], %[[c128_i64]], %[[c128_i64]], %[[c4_i64]])
// CHECK:      %[[subview:.*]] = memref.subview %[[arg0]][0, 0] [128, 1] [1, 1]
// CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[subview]]
// CHECK:      %[[indexcast:.*]] = arith.index_cast %[[intptr]]
// CHECK:      %[[inttoptr:.*]] = llvm.inttoptr %[[indexcast]]
// CHECK:      %[[intptr_0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK:      %[[indexcast3:.*]] = arith.index_cast %[[intptr_0]]
// CHECK:      %[[inttoptr4:.*]] = llvm.inttoptr %[[indexcast3]]
// CHECK:      call @xsmm_unary_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr4]], %[[c0]])

// -----

func.func @identity_2d_bcast_to_3d(%arg0: memref<128x256xf32>, %arg1: memref<512x128x256xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<128x256xf32>, vector<128x256xf32>
  %1 = vector.broadcast %0 : vector<128x256xf32> to vector<512x128x256xf32>
  vector.transfer_write %1, %arg1[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<512x128x256xf32>, memref<512x128x256xf32>
  return
}

// CHECK-LABEL:  func.func @identity_2d_bcast_to_3d(
// CHECK: %[[arg0:.*]]: memref<128x256xf32>, %[[arg1:.*]]: memref<512x128x256xf32>) {
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c65536_i64:.*]] = arith.constant 65536 : i64
// CHECK-DAG: %[[c256_i64:.*]] = arith.constant 256 : i64
// CHECK-DAG: %[[c32768_i64:.*]] = arith.constant 32768 : i64
// CHECK-DAG: %[[c128_i64:.*]] = arith.constant 128 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK:      %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c1_i64]], %[[c1_i64]], %[[c65536_i64]], %[[c256_i64]], %[[c32768_i64]], %[[c128_i64]], %[[c4_i64]])
// CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]]
// CHECK:      %[[indexcast:.*]] = arith.index_cast %[[intptr]]
// CHECK:      %[[inttoptr:.*]] = llvm.inttoptr %[[indexcast]]
// CHECK:      %[[intptr_0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK:      %[[indexcast3:.*]] = arith.index_cast %[[intptr_0]]
// CHECK:      %[[inttoptr4:.*]] = llvm.inttoptr %[[indexcast3]]
// CHECK:      call @xsmm_unary_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr4]], %[[c0]])

// -----

func.func @identity_broadcast_exact_dim_match(%arg0: memref<4x1xf32>, %arg1: memref<4x2xf32>) -> memref<4x2xf32> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x1xf32>, vector<4x1xf32>
  %1 = vector.broadcast %0 : vector<4x1xf32> to vector<4x2xf32>
  vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]}
    : vector<4x2xf32>, memref<4x2xf32>
  return %arg1 : memref<4x2xf32>
}

// CHECK-LABEL:  func.func @identity_broadcast_exact_dim_match(
// CHECK: %[[arg0:.*]]: memref<4x1xf32>, %[[arg1:.*]]: memref<4x2xf32>) -> memref<4x2xf32> {
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c4_i64:.*]] = arith.constant 4 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK:      %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c1_i64]], %[[c1_i64]], %[[c4_i64]], %[[c2_i64]], %[[c1_i64]], %[[c2_i64]], %[[c2_i64]])
// CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]]
// CHECK:      %[[indexcast:.*]] = arith.index_cast %[[intptr]]
// CHECK:      %[[inttoptr:.*]] = llvm.inttoptr %[[indexcast]]
// CHECK:      %[[intptr_0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK:      %[[indexcast3:.*]] = arith.index_cast %[[intptr_0]]
// CHECK:      %[[inttoptr4:.*]] = llvm.inttoptr %[[indexcast3]]
// CHECK:      call @xsmm_unary_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr4]], %[[c0]])

// -----

func.func @identity_broadcast_same_rank(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<256xf32>, vector<256xf32>
  %1 = vector.broadcast %0 : vector<256xf32> to vector<256xf32>
  vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<256xf32>, memref<256xf32>
  return
}

// CHECK-LABEL:  func.func @identity_broadcast_same_rank(
// CHECK: %[[arg0:.*]]: memref<256xf32>, %[[arg1:.*]]: memref<256xf32>) {
// CHECK-NOT: call @xsmm_unary_dispatch
// CHECK-NOT: call @xsmm_unary_invoke

// -----

func.func @identity_empty_buffer(%arg0: memref<f32, strided<[]>>, %arg1: memref<6x9xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[], %cst {in_bounds = []} : memref<f32, strided<[]>>, vector<f32>
  %1 = vector.broadcast %0 : vector<f32> to vector<6x9xf32>
  vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<6x9xf32>, memref<6x9xf32>
  return
}

// CHECK-LABEL:  func.func @identity_empty_buffer(
// CHECK: %[[arg0:.*]]: memref<f32, strided<[]>>, %[[arg1:.*]]: memref<6x9xf32>) {
// CHECK-NOT: call @xsmm_unary_dispatch
// CHECK-NOT: call @xsmm_unary_invoke

// -----

func.func @identity_strided_buffer(%arg0: memref<6x1xf32, strided<[6, 1]>>, %arg1: memref<6x9xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst{in_bounds=[true, true]} : memref<6x1xf32, strided<[6, 1]>>, vector<6x1xf32>
  %1 = vector.broadcast %0 : vector<6x1xf32> to vector<6x9xf32>
  vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<6x9xf32>, memref<6x9xf32>
  return
}
// CHECK-LABEL:  func.func @identity_strided_buffer(
// CHECK: %[[arg0:.*]]: memref<6x1xf32, strided<[6, 1]>>, %[[arg1:.*]]: memref<6x9xf32>) {
// CHECK-DAG: %[[c6_i64:.*]] = arith.constant 6 : i64
// CHECK-DAG: %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[c9_i64:.*]] = arith.constant 9 : i64
// CHECK-DAG: %[[c2_i64:.*]] = arith.constant 2 : i64
// CHECK:      %[[dispatch:.*]] = call @xsmm_unary_dispatch(%[[c1_i64]], %[[c1_i64]], %[[c6_i64]], %[[c9_i64]], %[[c1_i64]], %[[c9_i64]], %[[c2_i64]])
// CHECK:      %[[intptr:.*]] = memref.extract_aligned_pointer_as_index %[[arg0]]
// CHECK:      %[[indexcast:.*]] = arith.index_cast %[[intptr]]
// CHECK:      %[[inttoptr:.*]] = llvm.inttoptr %[[indexcast]]
// CHECK:      %[[intptr_0:.*]] = memref.extract_aligned_pointer_as_index %[[arg1]]
// CHECK:      %[[indexcast3:.*]] = arith.index_cast %[[intptr_0]]
// CHECK:      %[[inttoptr4:.*]] = llvm.inttoptr %[[indexcast3]]
// CHECK:      call @xsmm_unary_invoke(%[[c1_i64]], %[[dispatch]], %[[inttoptr]], %[[c0]], %[[inttoptr4]], %[[c0]])

