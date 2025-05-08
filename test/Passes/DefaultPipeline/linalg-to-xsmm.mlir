// RUN: tpp-opt %s -default-tpp-passes -split-input-file | FileCheck %s
// RUN: tpp-sched --bundle=default-tpp-passes %s --split-input-file | FileCheck %s

func.func @fill_op(%arg0: memref<3x3xf32>) {
  %cst = arith.constant 0.0 : f32
  linalg.fill ins(%cst : f32) outs(%arg0 : memref<3x3xf32>)
  return
}

// CHECK-LABEL: fill_op
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : i64
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[DIS:.+]] = call @xsmm_unary_dispatch(%[[C2]], %[[C1]], %[[C3]], %[[C3]], %[[C1]], %[[C3]], %[[C8]])
// CHECK: %[[PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]] : memref<3x3xf32> -> index
// CHECK: %[[PTR_TO_INT:.+]] = arith.index_cast %[[PTR]] : index to i64
// CHECK: %[[LLVM_PTR:.+]] = llvm.inttoptr %[[PTR_TO_INT]] : i64 to !llvm.ptr
// CHECK: call @xsmm_unary_scalar_invoke(%[[C1]], %[[DIS]], %[[CST]], %[[LLVM_PTR]], %[[C0]])

// -----

func.func @fill_op_i32(%arg0: memref<3x3xi32>) {
  %cst = arith.constant 0 : i32
  linalg.fill ins(%cst : i32) outs(%arg0 : memref<3x3xi32>)
  return
}

// CHECK-LABEL: fill_op_i32
// CHECK-NOT: xsmm
// CHECK: linalg.fill

// -----

func.func @gemm_with_zero(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<3x3xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
  %mul = linalg.matmul ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>)
                       outs(%fill: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %mul : tensor<3x3xf32>
}

// CHECK-LABEL: gemm_with_zero
// CHECK-SAME: %[[ARG0:.+]]: memref<3x3xf32>, %[[ARG1:.+]]: memref<3x3xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-NOT: xsmm_unary_dispatch
// CHECK: %[[DIS:.+]] = call @xsmm_gemm_dispatch(%[[C1]], %[[C3]], %[[C3]], %[[C3]], %[[C3]], %[[C3]], %[[C3]], %[[C4]])
// CHECK: %[[INT_PTR_ARG0:.+]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[CAST_ARG0:.+]] = arith.index_cast %[[INT_PTR_ARG0]] : index to i64
// CHECK: %[[LLVM_PTR_ARG0:.+]] = llvm.inttoptr %[[CAST_ARG0]] : i64 to !llvm.ptr
// CHECK: %[[INT_PTR_ARG1:.+]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[CAST_ARG1:.+]] = arith.index_cast %[[INT_PTR_ARG1]] : index to i64
// CHECK: %[[LLVM_PTR_ARG1:.+]] = llvm.inttoptr %[[CAST_ARG1]] : i64 to !llvm.ptr
// CHECK: %[[INT_PTR_ALLOC:.+]] = memref.extract_aligned_pointer_as_index
// CHECK: %[[CAST_ALLOC:.+]] = arith.index_cast %[[INT_PTR_ALLOC]] : index to i64
// CHECK: %[[LLVM_PTR_ALLOC:.+]] = llvm.inttoptr %[[CAST_ALLOC]] : i64 to !llvm.ptr
// CHECK: call @xsmm_gemm_invoke(%[[C1]], %[[DIS]], %[[LLVM_PTR_ARG0]], %[[C0]], %[[LLVM_PTR_ARG1]], %[[C0]], %[[LLVM_PTR_ALLOC]], %[[C0]])
