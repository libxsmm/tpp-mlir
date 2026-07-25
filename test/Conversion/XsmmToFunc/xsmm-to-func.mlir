// RUN: tpp-opt %s -convert-xsmm-to-func -split-input-file | FileCheck %s

// CHECK-LABEL: dispatch_unary
func.func @dispatch_unary() -> i64 {
  %0 = xsmm.unary.dispatch identity [5, 6, 5, 6] flags = (bcast_row) data_type = f32
  return %0: i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: call @xsmm_unary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C2]])

// -----

// CHECK-LABEL: dispatch_brgemm
func.func @dispatch_brgemm() -> i64 {
  %0 = xsmm.brgemm.dispatch [5, 5, 4, 4, 5, 5, 5, 5] flags = (none) data_type = f32
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: call @xsmm_brgemm_dispatch(%[[C1]], %[[C5]], %[[C5]], %[[C4]], %[[C4]], %[[C5]], %[[C5]], %[[C5]], %[[C5]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (none) data_type = f32
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C1]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a, vnni_b) data_type = bf16
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// Or between 256 and 512 (see enum for GemmFlags)
// CHECK-DAG: %[[C768:.+]] = arith.constant 768 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C2]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C768]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a, vnni_b, vnni_c) data_type = bf16
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// Or between 256 and 512 and 1024 (see enum for GemmFlags)
// CHECK-DAG: %[[C1792:.+]] = arith.constant 1792 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C2]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C1792]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a) data_type = bf16
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// LIBXSMM is col-major check we swap the flag for A and B (see enum for GemmFlags)
// CHECK-DAG: %[[C512:.+]] = arith.constant 512 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C2]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C512]])

// -----

// CHECK-LABEL: dispatch_gemm
func.func @dispatch_gemm() -> i64 {
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_b) data_type = bf16
  return %0 : i64
}

// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// LIBXSMM is col-major check we swap the flag for A and B (see enum for GemmFlags)
// CHECK-DAG: %[[C256:.+]] = arith.constant 256 : i64
// CHECK: call @xsmm_gemm_dispatch(%[[C2]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C256]])

// -----

func.func @invoke_brgemm(%arg0: memref<2x5x4xf32>, %arg1: memref<2x4x5xf32>,
                           %arg2: memref<4x4xf32>) -> memref<4x4xf32> {
  %0 = xsmm.brgemm.dispatch [5, 5, 4, 4, 5, 5, 5, 5] flags = (none) data_type = f32
  %c2_i64 = arith.constant 2 : i64
  xsmm.brgemm(data_type = f32, %0, %arg0, %arg1, %arg2, %c2_i64)
    : (i64, memref<2x5x4xf32>, memref<2x4x5xf32>, memref<4x4xf32>, i64) -> ()
  return %arg2 : memref<4x4xf32>
}

// CHECK-LABEL: invoke_brgemm
// CHECK-SAME: %[[ARG0:.+]]: memref<2x5x4xf32>, %[[ARG1:.+]]: memref<2x4x5xf32>, %[[ARG2:.+]]: memref<4x4xf32>
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: %[[ADDR:.+]] = call @xsmm_brgemm_dispatch
// CHECK: %[[PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
// CHECK-NEXT: %[[CST_PTR:.+]] = arith.index_cast %[[PTR]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR:.+]] = llvm.inttoptr %[[CST_PTR]] : i64 to !llvm.ptr
// CHECK: %[[PTR1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
// CHECK-NEXT: %[[CST_PTR1:.+]] = arith.index_cast %[[PTR1]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR1:.+]] = llvm.inttoptr %[[CST_PTR1]] : i64 to !llvm.ptr
// CHECK: %[[PTR2:.+]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
// CHECK-NEXT: %[[CST_PTR2:.+]] = arith.index_cast %[[PTR2]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR2:.+]] = llvm.inttoptr %[[CST_PTR2]] : i64 to !llvm.ptr
// CHECK: xsmm_brgemm_invoke(%[[C1]], %[[ADDR]], %[[LLVM_PTR]], %[[C0]], %[[LLVM_PTR1]], %[[C0]], %[[LLVM_PTR2]], %[[C0]], %[[C2]])

// -----

func.func @invoke_unary(%arg0: memref<512xbf16>, %arg1: memref<128x512xbf16>) {
  %0 = xsmm.unary.dispatch identity [128, 512, 512, 512]  flags = (bcast_col) data_type = bf16
  xsmm.unary identity(data_type = bf16, %0, %arg0, %arg1) : (i64, memref<512xbf16>, memref<128x512xbf16>) -> ()
  return
}

// CHECK-LABEL: invoke_unary
// CHECK-SAME: %[[ARG0:.+]]: memref<512xbf16>, %[[ARG1:.+]]: memref<128x512xbf16>
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: %[[ADDR:.+]] = call @xsmm_unary_dispatch
// CHECK: %[[PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
// CHECK-NEXT: %[[PTR_CST:.+]] = arith.index_cast %[[PTR]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR:.+]] = llvm.inttoptr %[[PTR_CST]] : i64 to !llvm.ptr
// CHECK: %[[PTR1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
// CHECK-NEXT: %[[PTR_CST1:.+]] = arith.index_cast %[[PTR1]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR1:.+]] = llvm.inttoptr %[[PTR_CST1]] : i64 to !llvm.ptr
// CHECK: call @xsmm_unary_invoke(%[[C2]], %[[ADDR]], %[[LLVM_PTR]], %[[C0]], %[[LLVM_PTR1]], %[[C0]])

// -----

func.func @invoke_gemm_vnni(%arg0: memref<128x256xbf16>, %arg1: memref<128x512x2xbf16>, %arg2: memref<128x512xbf16>) {
  %0 = xsmm.gemm.dispatch [128, 512, 256, 256, 512, 512]  flags = (vnni_b) data_type = bf16
  xsmm.gemm(data_type = bf16, %0, %arg0, %arg1, %arg2)
    : (i64, memref<128x256xbf16>, memref<128x512x2xbf16>, memref<128x512xbf16>) -> ()
  return
}

// CHECK-LABEL: invoke_gemm_vnni
// CHECK-SAME: %[[ARG0:.+]]: memref<128x256xbf16>, %[[ARG1:.+]]: memref<128x512x2xbf16>, %[[ARG2:.+]]: memref<128x512xbf16>
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: %[[ADDR:.+]] = call @xsmm_gemm_dispatch
// CHECK: %[[PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
// CHECK-NEXT: %[[PTR_CST:.+]] = arith.index_cast %[[PTR]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR:.+]] = llvm.inttoptr %[[PTR_CST]] : i64 to !llvm.ptr
// CHECK: %[[PTR1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG1]]
// CHECK-NEXT: %[[PTR_CST1:.+]] = arith.index_cast %[[PTR1]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR1:.+]] = llvm.inttoptr %[[PTR_CST1]] : i64 to !llvm.ptr
// CHECK: %[[PTR2:.+]] = memref.extract_aligned_pointer_as_index %[[ARG2]]
// CHECK-NEXT: %[[PTR_CST2:.+]] = arith.index_cast %[[PTR2]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR2:.+]] = llvm.inttoptr %[[PTR_CST2]] : i64 to !llvm.ptr
// CHECK: call @xsmm_gemm_invoke(%[[C2]], %[[ADDR]], %[[LLVM_PTR]], %[[C0]], %[[LLVM_PTR1]], %[[C0]], %[[LLVM_PTR2]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_gemm_vnni
func.func @dispatch_gemm_vnni() -> i64 {
  // CHECK: %[[C2:.+]] = arith.constant 2 : i64
  // CHECK-DAG: %[[C128:.+]] = arith.constant 128 : i64
  // CHECK-DAG: %[[C512:.+]] = arith.constant 512 : i64
  // vnni_b is swapped to vnni_a (256) for col-major, which is CSE'd with the k dim constant
  // CHECK-DAG: %[[C256:.+]] = arith.constant 256 : i64
  %0 = xsmm.gemm.dispatch [128, 512, 256, 256, 512, 512]  flags = (vnni_b) data_type = bf16
  return %0 : i64
}

// -----

func.func @invoke_inplace_relu(%arg0: memref<128x512xbf16>) {
  %0 = xsmm.unary.dispatch relu [128, 512, 512, 512]  flags = (none) data_type = bf16
  xsmm.unary relu(data_type = bf16, %0, %arg0, %arg0) : (i64, memref<128x512xbf16>, memref<128x512xbf16>) -> ()
  return
}

// CHECK-LABEL: invoke_inplace_relu
// CHECK-SAME: %[[ARG0:.+]]: memref<128x512xbf16>
// CHECK: %[[C0]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
// CHECK: %[[ADDR:.+]] = call @xsmm_unary_dispatch
// CHECK: %[[PTR:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
// CHECK-NEXT: %[[PTR_CAST:.+]] = arith.index_cast %[[PTR]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR:.+]] = llvm.inttoptr %[[PTR_CAST]] : i64 to !llvm.ptr
// CHECK: %[[PTR1:.+]] = memref.extract_aligned_pointer_as_index %[[ARG0]]
// CHECK-NEXT: %[[PTR_CAST1:.+]] = arith.index_cast %[[PTR1]] : index to i64
// CHECK-NEXT: %[[LLVM_PTR1:.+]] = llvm.inttoptr %[[PTR_CAST1]] : i64 to !llvm.ptr
// CHECK: call @xsmm_unary_invoke(%[[C2]], %[[ADDR]], %[[LLVM_PTR]], %[[C0]], %[[LLVM_PTR1]], %[[C0]])

// -----

func.func @dispatch_fused_brgemm() -> i64 {
  %0 = xsmm.fused_brgemm.dispatch [13, 13, 13, 13, 13, 13, 13, 13] [add, relu]
    flags = (vnni_a) binary_flags = (bcast_col_in0) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// CHECK-LABEL: dispatch_fused_brgemm
// CHECK: %[[DATA_TYPE:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[DIM:.+]] = arith.constant 13 : i64
// CHECK-DAG: %[[GEMM_FLAGS:.+]] = arith.constant 512 : i64
// CHECK-DAG: %[[UNARY_FLAGS:.+]] = arith.constant 0 : i64
// CHECK-DAG: %[[UNARY_KIND:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[BINARY_FLAGS:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[BINARY_KIND:.+]] = arith.constant 1 : i64
// CHECK: %{{.+}} = call @xsmm_fused_brgemm_dispatch(%[[DATA_TYPE]], %[[DIM]], %[[DIM]], %[[DIM]], %[[DIM]], %[[DIM]], %[[DIM]], %[[DIM]], %[[DIM]], %[[GEMM_FLAGS]], %[[UNARY_FLAGS]], %[[UNARY_KIND]], %[[BINARY_FLAGS]], %[[BINARY_KIND]])

// -----

// Current limitation in LIBXSMM we can use only bcast_col_in0 as flag for binary.
// see: https://github.com/libxsmm/libxsmm/issues/766
// CHECK-LABEL: dispatch_fused_brgemm
func.func @dispatch_fused_brgemm() -> i64 {
  // CHECK-NOT: xsmm_fused_brgemm.dispatch
  %0 = xsmm.fused_brgemm.dispatch [13, 13, 13, 13, 13, 13, 13, 13] [add, relu]
    flags = (vnni_a) binary_flags = (none) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// -----

func.func @dispatch_fused_brgemm() -> i64 {
  %0 = xsmm.fused_brgemm.dispatch [13, 13, 13, 13, 13, 13, 13, 13] [add, none]
    flags = (vnni_a) binary_flags = (bcast_col_in0) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// CHECK-LABEL: dispatch_fused_brgemm
// CHECK: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C13:.+]] = arith.constant 13 : i64
// CHECK-DAG: %[[C512:.+]] = arith.constant 512 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: %{{.+}} = call @xsmm_fused_brgemm_dispatch(%[[C2]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C512]], %[[C0]], %[[C0]], %[[C4]], %[[C1]])

// -----

func.func @multiple_gemm_flags_fused_brgemm() -> i64 {
  %0 = xsmm.fused_brgemm.dispatch [13, 13, 13, 13, 13, 13, 13, 13] [add, none]
    flags = (vnni_a, vnni_b, vnni_c) binary_flags = (bcast_col_in0) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// CHECK-LABEL: multiple_gemm_flags_fused_brgemm
// CHECK: %[[C2:.+]] = arith.constant 2 : i64
// CHECK-DAG: %[[C13:.+]] = arith.constant 13 : i64
// Or between 256 and 512 and 1024 (see enum for GemmFlags)
// CHECK-DAG: %[[C1792:.+]] = arith.constant 1792 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK: %{{.+}} = call @xsmm_fused_brgemm_dispatch(%[[C2]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C13]], %[[C1792]], %[[C0]], %[[C0]], %[[C4]], %[[C1]])

// -----

// CHECK-LABEL: transpose
func.func @transpose() -> i64 {
  // CHECK-DAG: %[[C29:.+]] = arith.constant 29 : i64
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
  // CHECK-DAG: %[[C3:.+]] = arith.constant 3 : i64
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i64
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
  // CHECK: %{{.+}} = call @xsmm_unary_dispatch(%[[C29]], %[[C1]], %[[C3]], %[[C2]], %[[C3]], %[[C2]], %[[C0]])
  %0 = xsmm.unary.dispatch transpose [3, 2, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch add [5, 6, 5, 6, 5] flags = (none) data_type = f32
  return %0: i64
}

// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch sub [5, 6, 5, 6, 5] flags = (none) data_type = f32
  return %0: i64
}

// CHECK: %[[C3:.+]] = arith.constant 3 : i64
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C3]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch div [5, 6, 5, 6, 5] flags = (none) data_type = f32
  return %0: i64
}

// CHECK: %[[C4:.+]] = arith.constant 4 : i64
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C4]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C0]])

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch add [5, 6, 5, 6, 5] flags = (bcast_col_in0) data_type = f32
  return %0 : i64
}

// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C4:.+]] = arith.constant 4 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C4]])

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch add [5, 6, 5, 6, 5] flags = (bcast_col_in1) data_type = f32
  return %0 : i64
}

// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C8]])

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch add [5, 6, 5, 6, 5] flags = (bcast_col_in0, bcast_col_in1) data_type = f32
  return %0 : i64
}

// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C12:.+]] = arith.constant 12 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C12]])

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch add [5, 6, 5, 6, 5] flags = (bcast_scalar_in0, bcast_col_in1) data_type = f32
  return %0 : i64
}

// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C24:.+]] = arith.constant 24 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C24]])

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch add [5, 6, 5, 6, 5] flags = (bcast_scalar_in0) data_type = f32
  return %0 : i64
}

// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C16]])

// -----

// CHECK-LABEL: dispatch_binary
func.func @dispatch_binary() -> i64 {
  %0 = xsmm.binary.dispatch add [5, 6, 5, 6, 5] flags = (bcast_scalar_in1) data_type = f32
  return %0 : i64
}

// CHECK: %[[C1:.+]] = arith.constant 1 : i64
// CHECK-DAG: %[[C5:.+]] = arith.constant 5 : i64
// CHECK-DAG: %[[C6:.+]] = arith.constant 6 : i64
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i64
// CHECK: %{{.+}} = call @xsmm_binary_dispatch(%[[C1]], %[[C1]], %[[C5]], %[[C6]], %[[C5]], %[[C6]], %[[C5]], %[[C32]])

