// RUN: tpp-opt -split-input-file -verify-diagnostics %s

func.func @gemm_dispatch() -> i64 {
  // m, n, k, lda, ldb, ldc
  // expected-error@+1 {{expect lda to be >= of dimension k}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 1, 5, 6] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // m, n, k, lda, ldb, ldc
  // expected-error@+1 {{expect ldb to be >= of dimension n}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 1, 6] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // m, n, k, lda, ldb, ldc
  // expected-error@+1 {{expect ldc to be >= of dimension n}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 1] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_b) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_c) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a, vnni_c) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{expected flags to be unique}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (vnni_a, vnni_a) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{'none' flags conflicts with others}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5, 6] flags = (none, vnni_a) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{expect 6 args but got: 5}}
  %0 = xsmm.gemm.dispatch [1, 2, 3, 4, 5] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @gemm_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.gemm.dispatch [-3, 2, 1, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] flags = (vnni_a) data_type = f32
  return %0 : i64
}

// -----

func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] flags = (vnni_b) data_type = f32
  return %0 : i64
}

// -----

func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] flags = (vnni_c) data_type = f32
  return %0 : i64
}

// -----

func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{VNNI flags but type is not bf16}}
  %0 = xsmm.brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] flags = (vnni_a, vnni_c) data_type = f32
  return %0 : i64
}

// -----

func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{expected flags to be unique}}
  %0 = xsmm.brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] flags = (vnni_a, vnni_a) data_type = f32
  return %0 : i64
}

// -----

func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{'none' flags conflicts with others}}
  %0 = xsmm.brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] flags = (none, vnni_a) data_type = f32
  return %0 : i64
}

// -----

func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{expect 8 args but got: 5}}
  %0 = xsmm.brgemm.dispatch [1, 2, 3, 4, 5] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @brgemm_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.brgemm.dispatch [3, 2, -1, 3, 2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @unary_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.unary.dispatch relu [3, 2, 1, -3] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @unary_dispatch() -> i64 {
  // expected-error@+1 {{op expect 4 args but got: 3}}
  %0 = xsmm.unary.dispatch relu [3, 2, 1] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @binary_dispatch() -> i64 {
  // expected-error@+1 {{failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  %0 = xsmm.binary.dispatch add [3, 2, 1, 3, -2] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @binary_dispatch() -> i64 {
  // expected-error@+1 {{op expect 5 args but got: 3}}
  %0 = xsmm.binary.dispatch add [3, 2, 1] flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @fused_dispatch() -> i64 {
  // expected-error@+1 {{op expect 8 args but got: 3}}
  %0 = xsmm.fused_brgemm.dispatch [3, 2, 1] [add, relu]
    flags = (none) binary_flags = (none) unary_flags = (none) data_type = f32
  return %0 : i64
}

// -----

func.func @fused_dispatch() -> i64 {
  // expected-error@+1 {{op expected flags to be unique}}
  %0 = xsmm.fused_brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] [add, relu]
    flags = (vnni_a, vnni_a) binary_flags = (none) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// -----

func.func @fused_dispatch() -> i64 {
  // expected-error@+1 {{op expected binary_flags to be unique}}
  %0 = xsmm.fused_brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] [add, relu]
    flags = (vnni_a) binary_flags = (none, none) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// -----

func.func @fused_dispatch() -> i64 {
  // expected-error@+1 {{op expected unary_flags to be unique}}
  %0 = xsmm.fused_brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] [add, relu]
    flags = (vnni_a) binary_flags = (none) unary_flags = (none, none) data_type = bf16
  return %0 : i64
}

// -----

func.func @fused_brgemm_none_kind_with_flags() -> i64 {
  // expected-error@+1 {{invalid binary flags for kind none}}
   %0 = xsmm.fused_brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] [none, relu]
    flags = (vnni_a) binary_flags = (bcast_col_in0) unary_flags = (none) data_type = bf16
  return %0 : i64
}

// -----

func.func @fused_brgemm_none_kind_with_flags() -> i64 {
  // expected-error@+1 {{invalid unary flags for kind none}}
   %0 = xsmm.fused_brgemm.dispatch [1, 2, 3, 4, 5, 6, 1, 1] [none, none]
    flags = (vnni_a) binary_flags = (none) unary_flags = (bcast_scalar) data_type = bf16
  return %0 : i64
}

// -----

func.func @gemm_invoke(%arg0: i64, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>,
                       %arg3: memref<3x3xf32>) {
  // expected-error@+1 {{expect bf16 but got: 'f32' for operand at index: 1}}
  xsmm.gemm(data_type = bf16, %arg0, %arg1, %arg2, %arg3)
    : (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @gemm_invoke(%arg0: i64, %arg1: memref<3x3xbf16>, %arg2: memref<3x3xbf16>,
                       %arg3: memref<3x3xbf16>) {
  // expected-error@+1 {{expect f32 but got: 'bf16' for operand at index: 1}}
  xsmm.gemm(data_type = f32, %arg0, %arg1, %arg2, %arg3)
    : (i64, memref<3x3xbf16>, memref<3x3xbf16>, memref<3x3xbf16>) -> ()
  return
}

// -----

func.func @gemm_invoke(%arg0: memref<3x3xf32>, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>,
                       %arg3: memref<3x3xf32>) {
  // expected-error@+1 {{expect an i64 but got 'memref<3x3xf32>' for operand 0 (dispatch)}}
  xsmm.gemm(data_type = f32, %arg0, %arg1, %arg2, %arg3)
    : (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @gemm_invoke(%arg0: f32, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>,
                       %arg3: memref<3x3xf32>) {
  // expected-error@+1 {{op operand #0 must be variadic of 2D/3D static memref of 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type values or 64-bit signless integer, but got 'f32'}}
  xsmm.gemm(data_type = f32, %arg0, %arg1, %arg2, %arg3)
    : (f32, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @gemm_invoke(%arg0: i64, %arg1: memref<1x1x3x3xf32>, %arg2: memref<3x3xf32>,
                       %arg3: memref<3x3xf32>) {
  // expected-error@+1 {{op operand #1 must be variadic of 2D/3D static memref of 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type values or 64-bit signless integer, but got 'memref<1x1x3x3xf32>'}}
  xsmm.gemm(data_type = f32, %arg0, %arg1, %arg2, %arg3)
    : (i64, memref<1x1x3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @gemm_invoke(%arg0: i64, %arg1: memref<1x3x3xf32>, %arg2: memref<3x3xf32>,
                       %arg3: memref<3x3xf32>) {
  // expected-error@+1 {{expect VNNI layout for operand: 1}}
  xsmm.gemm(data_type = f32, %arg0, %arg1, %arg2, %arg3)
    : (i64, memref<1x3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @gemm_invoke(%arg0: i64, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  // expected-error@+1 {{expect 4 inputs but got 3}}
  xsmm.gemm(data_type = f32, %arg0, %arg1, %arg2)
    : (i64, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @brgemm_invoke(%arg0: i64, %arg1: memref<3x3xf32>, %arg2: memref<3x3xf32>) {
  // expected-error@+1 {{expect 5 inputs but got 3}}
  xsmm.brgemm(data_type = f32, %arg0, %arg1, %arg2)
    : (i64, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @brgemm_invoke(%arg0: i64, %arg1: memref<3x3xf32>, %arg2: memref<2x3x3xf32>,
                         %arg3: memref<3x3xf32>, %arg4: memref<2xf32>) {
  // expected-error@+1 {{operand #4 must be variadic of 2D/3D/4D static memref of 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type values or 64-bit signless integer, but got 'memref<2xf32>'}}
  xsmm.brgemm(data_type = f32, %arg0, %arg1, %arg2, %arg3, %arg4)
    : (i64, memref<3x3xf32>, memref<2x3x3xf32>, memref<3x3xf32>, memref<2xf32>) -> ()
  return
}

// -----

func.func @brgemm_invoke(%arg0: i64, %arg1: memref<2x3x3xf32>, %arg2: memref<2x3x3xf32>, %arg3: memref<2x3x3xf32>) {
  // expected-error@+1 {{expect a 2d or 3d VNNI layout for operand: 3}}
  xsmm.brgemm(data_type = f32, %arg0, %arg1, %arg2, %arg3, %arg0)
    : (i64, memref<2x3x3xf32>, memref<2x3x3xf32>, memref<2x3x3xf32>, i64) -> ()
  return
}

// -----

func.func @gemm_invoke(%arg0: i64, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
  // expected-error@+1 {{operand #1 must be variadic of 2D/3D static memref of 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type values or 64-bit signless integer, but got 'memref<?x?xf32>'}}
  xsmm.gemm(data_type = f32, %arg0, %arg1, %arg2, %arg2)
    : (i64, memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}

// -----

func.func @gemm_invoke(%arg0: i64, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?xf32>) {
  // expected-error@+1 {{operand #1 must be variadic of 2D/3D/4D static memref of 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type values or 64-bit signless integer, but got 'memref<?x?x?xf32>'}}
  xsmm.brgemm(data_type = f32, %arg0, %arg1, %arg2, %arg3, %arg0)
    : (i64, memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?xf32>, i64) -> ()
  return
}

// -----

func.func @unary_invoke(%arg0: memref<?x?xf32>, %arg1: memref<3x3xf32>, %disp: i64) {
  // expected-error@+1 {{operand #1 must be variadic of 1D/2D/3D/4D static memref of 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type values or 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type or 64-bit signless integer, but got 'memref<?x?xf32>'}}
  xsmm.unary relu(data_type = f32, %disp, %arg0, %arg1) : (i64, memref<?x?xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @unary_invoke(%arg0: memref<?x?xf32>, %arg1: memref<3x3xf32>, %disp: i64) {
  // expected-error@+1 {{operand #1 must be variadic of 1D/2D/3D/4D static memref of 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type values or 32-bit float or bfloat16 type or f8E5M2 type or f8E4M3FN type or 64-bit signless integer, but got 'memref<?x?xf32>'}}
  xsmm.binary add(data_type = f32, %disp, %arg0, %arg0, %arg1)
    : (i64, memref<?x?xf32>, memref<?x?xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @brgemm_invoke(%arg0: i64, %arg1: memref<2x3x3xf32>, %arg2: memref<2x3x3xf32>, %arg3: memref<3x3xf32>) {
  // expected-error@+1 {{expect an i64 but got 'memref<3x3xf32>' for last operand (batch)}}
  xsmm.brgemm(data_type = f32, %arg0, %arg1, %arg2, %arg3, %arg3)
    : (i64, memref<2x3x3xf32>, memref<2x3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @brgemm_invoke(%arg0: i64, %arg1: memref<2x3x3xf32>, %arg2: memref<2x3x3xf32>, %arg3: memref<3x3xf32>) {
  // expected-error@+1 {{expect an i64 but got 'memref<3x3xf32>' for last operand (batch)}}
  xsmm.fused_brgemm(data_type = f32, %arg0, %arg1, %arg2, %arg3, %arg3, %arg3)
    : (i64, memref<2x3x3xf32>, memref<2x3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @unary_invoke(%arg0: i64, %arg1: memref<3x3xf32>) {
  // expected-error@+1 {{expect 3 inputs but got 5}}
  xsmm.unary relu(data_type = f32, %arg0, %arg1, %arg1, %arg1, %arg1)
    : (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @unary_invoke(%arg1: memref<3x3xf32>) {
  // expected-error@+1 {{expect an i64 but got 'memref<3x3xf32>' for operand 0 (dispatch)}}
  xsmm.unary relu(data_type = f32, %arg1, %arg1, %arg1)
    : (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @binary_invoke(%arg1: memref<3x3xf32>) {
  // expected-error@+1 {{expect an i64 but got 'memref<3x3xf32>' for operand 0 (dispatch)}}
  xsmm.binary add(data_type = f32, %arg1, %arg1, %arg1, %arg1)
    : (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}

// -----

func.func @binary_invoke(%arg0: i64, %arg1: memref<3x3xf32>) {
  // expected-error@+1 {{custom op 'xsmm.binary' number of operands and types do not match: got 6 operands and 5 types}}
  xsmm.binary add(data_type = f32, %arg0, %arg1, %arg1, %arg1, %arg1, %arg1)
    : (i64, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>) -> ()
  return
}
