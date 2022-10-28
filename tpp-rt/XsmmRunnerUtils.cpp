//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "XsmmRunnerUtils.h"
#include "libxsmm.h" // NOLINT [build/include_subdir]

extern "C" void _mlir_ciface_xsmm_matmul_invoke_f32(
    int64_t funcAddr, UnrankedMemRefType<float> *A,
    UnrankedMemRefType<float> *B, UnrankedMemRefType<float> *C) {

  //   std::cout << "matrix A: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<float>(*A));
  //    std::cout << "\n";
  //   std::cout << "matrix B: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<float>(*B));
  //   std::cout << "\n";
  //   std::cout << "matrix C: \n";
  //   printMemRefMetaData(std::cout, DynamicMemRefType<float>(*C));
  //   std::cout << "\n";
  //   std::cout << "funcAddr: " << funcAddr << "\n";

  DynamicMemRefType<float> matrixA = DynamicMemRefType<float>(*A);
  DynamicMemRefType<float> matrixB = DynamicMemRefType<float>(*B);
  DynamicMemRefType<float> matrixC = DynamicMemRefType<float>(*C);

  float *addr_a = matrixA.data + matrixA.offset;
  float *addr_b = matrixB.data + matrixB.offset;
  float *addr_c = matrixC.data + matrixC.offset;

  //    int64_t M = matrixC.sizes[0];
  //    int64_t N = matrixC.sizes[1];
  //    int64_t K = matrixA.sizes[1];
  //
  //    for (int i = 0; i < M; i++) {
  //      for (int j = 0; j < N; j++) {
  //        for (int k = 0; k < K; k++) {
  //          float *curr_addr_a = i * matrixA.strides[0] + addr_a + k;
  //          float *curr_addr_b = k * matrixB.strides[0] + addr_b + j;
  //          float *curr_addr_c = i * matrixC.strides[0] + addr_c + j;
  //          *curr_addr_c += (*curr_addr_a) * (*curr_addr_b);
  //        }
  //      }
  //    }

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *)addr_b;
  gemm_param.b.primary = (void *)addr_a;
  gemm_param.c.primary = (void *)addr_c;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  sgemm.gemm(&gemm_param);
}

extern "C" void _mlir_ciface_xsmm_matmul_invoke_bf16(
    int64_t funcAddr, UnrankedMemRefType<bf16> *A, UnrankedMemRefType<bf16> *B,
    UnrankedMemRefType<bf16> *C) {

  // std::cout << "matrix A: \n";
  //    printMemRefMetaData(std::cout, DynamicMemRefType<bf16>(*A));
  //    std::cout << "\n";
  //    std::cout << "matrix B: \n";
  //    printMemRefMetaData(std::cout, DynamicMemRefType<bf16>(*B));
  //    std::cout << "\n";
  //    std::cout << "matrix C: \n";
  //    printMemRefMetaData(std::cout, DynamicMemRefType<bf16>(*C));
  //    std::cout << "\n";
  //    std::cout << "funcAddr: " << funcAddr << "\n";

  DynamicMemRefType<bf16> matrixA = DynamicMemRefType<bf16>(*A);
  DynamicMemRefType<bf16> matrixB = DynamicMemRefType<bf16>(*B);
  DynamicMemRefType<bf16> matrixC = DynamicMemRefType<bf16>(*C);

  bf16 *addr_a = matrixA.data + matrixA.offset;
  bf16 *addr_b = matrixB.data + matrixB.offset;
  bf16 *addr_c = matrixC.data + matrixC.offset;

  //    int64_t M = matrixC.sizes[0];
  //    int64_t N = matrixC.sizes[1];
  //    int64_t K = matrixA.sizes[1];
  //
  //    for (int i = 0; i < M; i++) {
  //      for (int j = 0; j < N; j++) {
  //        for (int k = 0; k < K; k++) {
  //          float *curr_addr_a = i * matrixA.strides[0] + addr_a + k;
  //          float *curr_addr_b = k * matrixB.strides[0] + addr_b + j;
  //          float *curr_addr_c = i * matrixC.strides[0] + addr_c + j;
  //          *curr_addr_c += (*curr_addr_a) * (*curr_addr_b);
  //        }
  //      }
  //    }

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *)addr_b;
  gemm_param.b.primary = (void *)addr_a;
  gemm_param.c.primary = (void *)addr_c;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t
_mlir_ciface_xsmm_matmul_dispatch( const libxsmm_datatype dtype, int64_t m, int64_t n, int64_t k,
                                       int64_t lda, int64_t ldb, int64_t ldc) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "ldb: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";

  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;

  // See:
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication
  // LIBXSMM col-major change m with n.
  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb;
  l_shape.ldb = lda;
  l_shape.ldc = ldc;
  l_shape.a_in_type = dtype;
  l_shape.b_in_type = dtype;
  l_shape.out_type = dtype;
  l_shape.comp_type = dtype;

  auto sgemm = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch_flags);
  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t _mlir_ciface_xsmm_unary_dispatch_f32(int64_t m, int64_t n,
                                                        int64_t ldi,
                                                        int64_t ldo,
                                                        int64_t type,
                                                        int64_t bcast_type) {

  // std::cout << "ldi: " << ldi << "\n";
  // std::cout << "ldo: " << ldo << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "type: " << type << "\n";
  // std::cout << "bcast_type: " << bcast_type << "\n";

  libxsmm_meltw_unary_flags unary_flags =
      static_cast<libxsmm_meltw_unary_flags>(bcast_type);

  libxsmm_meltw_unary_shape unary_shape;

  // Row major to col major swap m with n.
  unary_shape.m = static_cast<libxsmm_blasint>(n);
  unary_shape.n = static_cast<libxsmm_blasint>(m);
  unary_shape.in0_type = LIBXSMM_DATATYPE_F32;
  unary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  unary_shape.out_type = LIBXSMM_DATATYPE_F32;
  unary_shape.ldi = static_cast<libxsmm_blasint>(ldi);
  unary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary_v2(
      static_cast<libxsmm_meltw_unary_type>(type), unary_shape,
      static_cast<libxsmm_bitfield>(unary_flags));

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" int64_t _mlir_ciface_xsmm_unary_dispatch_bf16(int64_t m, int64_t n,
                                                         int64_t ldi,
                                                         int64_t ldo,
                                                         int64_t type,
                                                         int64_t bcast_type) {

  // std::cout << "ldi: " << ldi << "\n";
  // std::cout << "ldo: " << ldo << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "type: " << type << "\n";
  // std::cout << "bcast_type: " << bcast_type << "\n";

  libxsmm_meltw_unary_flags unary_flags =
      static_cast<libxsmm_meltw_unary_flags>(bcast_type);

  libxsmm_meltw_unary_shape unary_shape;

  // Row major to col major swap m with n.
  unary_shape.m = static_cast<libxsmm_blasint>(n);
  unary_shape.n = static_cast<libxsmm_blasint>(m);
  unary_shape.in0_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.comp_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.out_type = LIBXSMM_DATATYPE_BF16;
  unary_shape.ldi = static_cast<libxsmm_blasint>(ldi);
  unary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary_v2(
      static_cast<libxsmm_meltw_unary_type>(type), unary_shape,
      static_cast<libxsmm_bitfield>(unary_flags));

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" int64_t _mlir_ciface_xsmm_binary_dispatch(int64_t m, int64_t n,
                                                     int64_t ldiLhs,
                                                     int64_t ldiRhs,
                                                     int64_t ldo, int64_t type,
                                                     int64_t bcast_type) {

  libxsmm_meltw_binary_flags binary_flags =
      static_cast<libxsmm_meltw_binary_flags>(bcast_type);

  libxsmm_meltw_binary_shape binary_shape;

  // Row major to col major swap m with n.
  binary_shape.m = static_cast<libxsmm_blasint>(n);
  binary_shape.n = static_cast<libxsmm_blasint>(m);
  binary_shape.in0_type = LIBXSMM_DATATYPE_F32;
  binary_shape.in1_type = LIBXSMM_DATATYPE_F32;
  binary_shape.comp_type = LIBXSMM_DATATYPE_F32;
  binary_shape.out_type = LIBXSMM_DATATYPE_F32;
  binary_shape.ldi = static_cast<libxsmm_blasint>(ldiLhs);
  binary_shape.ldi2 = static_cast<libxsmm_blasint>(ldiRhs);
  binary_shape.ldo = static_cast<libxsmm_blasint>(ldo);

  libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary_v2(
      static_cast<libxsmm_meltw_binary_type>(type), binary_shape,
      static_cast<libxsmm_bitfield>(binary_flags));

  return reinterpret_cast<int64_t>(kernel);
}

extern "C" void
_mlir_ciface_xsmm_unary_invoke_f32(int64_t addr,
                                   UnrankedMemRefType<float> *input,
                                   UnrankedMemRefType<float> *output) {
  // std::cout << "tensor input: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*input));
  // std::cout << "tensor output: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*output));

  DynamicMemRefType<float> tensorA = DynamicMemRefType<float>(*input);
  DynamicMemRefType<float> tensorB = DynamicMemRefType<float>(*output);

  float *addr_a = tensorA.data + tensorA.offset;
  float *addr_b = tensorB.data + tensorB.offset;

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)addr_a;
  param.out.primary = (void *)addr_b;
  kernel(&param);
}

extern "C" void
_mlir_ciface_xsmm_unary_invoke_bf16(int64_t addr,
                                    UnrankedMemRefType<bf16> *input,
                                    UnrankedMemRefType<bf16> *output) {
  // std::cout << "tensor input: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<bf16>(*input));
  // std::cout << "tensor output: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<bf16>(*output));

  DynamicMemRefType<bf16> tensorA = DynamicMemRefType<bf16>(*input);
  DynamicMemRefType<bf16> tensorB = DynamicMemRefType<bf16>(*output);

  bf16 *addr_a = tensorA.data + tensorA.offset;
  bf16 *addr_b = tensorB.data + tensorB.offset;

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)addr_a;
  param.out.primary = (void *)addr_b;
  kernel(&param);
}

extern "C" void
_mlir_ciface_xsmm_binary_invoke(int64_t addr, UnrankedMemRefType<float> *lhs,
                                UnrankedMemRefType<float> *rhs) {

  DynamicMemRefType<float> tensorLhs = DynamicMemRefType<float>(*lhs);
  DynamicMemRefType<float> tensorRhs = DynamicMemRefType<float>(*rhs);

  float *addr_tensor_lhs = tensorLhs.data + tensorLhs.offset;
  float *addr_tensor_rhs = tensorRhs.data + tensorRhs.offset;

  libxsmm_meltwfunction_binary kernel =
      reinterpret_cast<libxsmm_meltwfunction_binary>(addr);
  libxsmm_meltw_binary_param param;
  // TODO: check if we need to swap also here.
  param.in0.primary = (void *)addr_tensor_lhs;
  param.in1.primary = (void *)addr_tensor_rhs;
  param.out.primary = (void *)addr_tensor_rhs;
  kernel(&param);
}

extern "C" void
_mlir_ciface_xsmm_unary_scalar_invoke_f32(int64_t addr, float input,
                                          UnrankedMemRefType<float> *output) {

  DynamicMemRefType<float> tensorB = DynamicMemRefType<float>(*output);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)&input;
  param.out.primary = (void *)tensorB.data;
  kernel(&param);
}

extern "C" void
_mlir_ciface_xsmm_unary_scalar_invoke_bf16(int64_t addr, bf16 input,
                                           UnrankedMemRefType<bf16> *output) {

  DynamicMemRefType<bf16> tensorB = DynamicMemRefType<bf16>(*output);

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)&input;
  param.out.primary = (void *)tensorB.data;
  kernel(&param);
}

LIBXSMM_INLINE void matrix_copy_NC_to_NCNC(float *src, float *dst, int T, int N,
                                           int C, int bn, int bc) {
  int t, n1, n2, c1, c2;
  int nBlocks = N / bn;
  int cBlocks = C / bc;
  LIBXSMM_VLA_DECL(3, float, real_src, src, N, C);
  LIBXSMM_VLA_DECL(5, float, real_dst, dst, nBlocks, cBlocks, bn, bc);

  for (t = 0; t < T; t++) {
    for (n1 = 0; n1 < nBlocks; n1++) {
      for (c1 = 0; c1 < cBlocks; c1++) {
        for (n2 = 0; n2 < bn; n2++) {
          for (c2 = 0; c2 < bc; c2++) {
            LIBXSMM_VLA_ACCESS(5, real_dst, t, n1, c1, n2, c2, nBlocks, cBlocks,
                               bn, bc) =
                LIBXSMM_VLA_ACCESS(3, real_src, t, n1 * bn + n2, c1 * bc + c2,
                                   N, C);
          }
        }
      }
    }
  }
}

extern "C" void _mlir_ciface_matrix_copy_NC_to_NCNC(
    UnrankedMemRefType<float> *input, UnrankedMemRefType<float> *output,
    int64_t N, int64_t C, int64_t n, int64_t c) {
  // std::cout << "\n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*input));
  // std::cout << "\n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*output));
  // std::cout << "\n";
  DynamicMemRefType<float> tensorInput = DynamicMemRefType<float>(*input);
  DynamicMemRefType<float> tensorOutput = DynamicMemRefType<float>(*output);

  float *addr_input = tensorInput.data + tensorInput.offset;
  float *addr_output = tensorOutput.data + tensorOutput.offset;

  matrix_copy_NC_to_NCNC(addr_input, addr_output, 1, C, N, c, n);
}

extern "C" void _mlir_ciface_xsmm_brgemm_invoke_f32(
    int64_t addr, UnrankedMemRefType<float> *A, UnrankedMemRefType<float> *B,
    UnrankedMemRefType<float> *C, int64_t numBatches) {
  // std::cout << "numBatch: " << numBatches << "\n";
  // std::cout << "\n A: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*A));
  // std::cout << "\n B: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*B));
  // std::cout << "\n C: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<float>(*C));
  // std::cout << "\n";

  DynamicMemRefType<float> tensorA = DynamicMemRefType<float>(*A);
  DynamicMemRefType<float> tensorB = DynamicMemRefType<float>(*B);
  DynamicMemRefType<float> tensorC = DynamicMemRefType<float>(*C);
  float *addr_tensorA = tensorA.data + tensorA.offset;
  float *addr_tensorB = tensorB.data + tensorB.offset;
  float *addr_tensorC = tensorC.data + tensorC.offset;

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(addr);
  unsigned long long numBatchesVar = numBatches;
  gemm_param.a.primary = (void *)addr_tensorB;
  gemm_param.b.primary = (void *)addr_tensorA;
  gemm_param.c.primary = (void *)addr_tensorC;
  gemm_param.op.tertiary = (void *)&numBatchesVar;
  sgemm.gemm(&gemm_param);
}

extern "C" void _mlir_ciface_xsmm_brgemm_invoke_bf16(
    int64_t addr, UnrankedMemRefType<bf16> *A, UnrankedMemRefType<bf16> *B,
    UnrankedMemRefType<bf16> *C, int64_t numBatches) {
  // std::cout << "numBatch: " << numBatches << "\n";
  // std::cout << "\n A: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<bf16>(*A));
  // std::cout << "\n B: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<bf16>(*B));
  // std::cout << "\n C: \n";
  // printMemRefMetaData(std::cout, DynamicMemRefType<bf16>(*C));
  // std::cout << "\n";

  DynamicMemRefType<bf16> tensorA = DynamicMemRefType<bf16>(*A);
  DynamicMemRefType<bf16> tensorB = DynamicMemRefType<bf16>(*B);
  DynamicMemRefType<bf16> tensorC = DynamicMemRefType<bf16>(*C);
  bf16 *addr_tensorA = tensorA.data + tensorA.offset;
  bf16 *addr_tensorB = tensorB.data + tensorB.offset;
  bf16 *addr_tensorC = tensorC.data + tensorC.offset;

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(addr);
  unsigned long long numBatchesVar = numBatches;
  gemm_param.a.primary = (void *)addr_tensorB;
  gemm_param.b.primary = (void *)addr_tensorA;
  gemm_param.c.primary = (void *)addr_tensorC;
  gemm_param.op.tertiary = (void *)&numBatchesVar;
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t _mlir_ciface_xsmm_brgemm_dispatch_f32(int64_t m, int64_t n,
                                                         int64_t k, int64_t lda,
                                                         int64_t ldb,
                                                         int64_t ldc) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "lbd: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  // TODO: move stride computation to dispatch
  // operation as in: https://github.com/plaidml/plaidml/pull/1983
  libxsmm_blasint stride_a = lda * m * sizeof(float);
  libxsmm_blasint stride_b = ldb * k * sizeof(float);

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  auto sgemm = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch_flags,
                                          l_brconfig);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t
_mlir_ciface_xsmm_brgemm_dispatch_bf16(int64_t m, int64_t n, int64_t k,
                                       int64_t lda, int64_t ldb, int64_t ldc) {
  // std::cout << "lda: " << lda << "\n";
  // std::cout << "lbd: " << ldb << "\n";
  // std::cout << "ldc: " << ldc << "\n";
  // std::cout << "m: " << m << "\n";
  // std::cout << "n: " << n << "\n";
  // std::cout << "k: " << k << "\n";

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  // TODO: move stride computation to dispatch
  // operation as in: https://github.com/plaidml/plaidml/pull/1983
  libxsmm_blasint stride_a = lda * m * sizeof(bf16);
  libxsmm_blasint stride_b = ldb * k * sizeof(bf16);

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_BF16;
  l_shape.b_in_type = LIBXSMM_DATATYPE_BF16;
  l_shape.out_type = LIBXSMM_DATATYPE_BF16;
  l_shape.comp_type = LIBXSMM_DATATYPE_BF16;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  auto sgemm = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch_flags,
                                          l_brconfig);

  return reinterpret_cast<int64_t>(sgemm);
}

//----------------------------------------------------------------------------//
// BRGEMM connection on the IREE side.
//----------------------------------------------------------------------------//

extern "C" int iree_xsmm_brgemm_dispatch_f32(void *context, void *params,
                                             void *reserved) {
  typedef struct {
    int64_t res;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
  } xsmm_brgemm_dispatch_f32_t;
  xsmm_brgemm_dispatch_f32_t *p = (xsmm_brgemm_dispatch_f32_t *)params;
  p->res = _mlir_ciface_xsmm_brgemm_dispatch_f32(p->m, p->n, p->k, p->lda,
                                                 p->ldb, p->ldc);
  return 0;
}

extern "C" int iree_xsmm_matmul_dispatch_f32(void *context, void *params,
                                             void *reserved) {
  typedef struct {
    int64_t res;
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
  } xsmm_matmul_dispatch_f32_t;
  xsmm_matmul_dispatch_f32_t *p = (xsmm_matmul_dispatch_f32_t *)params;
  p->res = _mlir_ciface_xsmm_matmul_dispatch(LIBXSMM_DATATYPE_F32, p->m, p->n, p->k, p->lda,
                                                 p->ldb, p->ldc);
  return 0;
}

extern "C" int iree_xsmm_unary_dispatch(void *context, void *params,
                                        void *reserved) {
  typedef struct {
    int64_t res;
    int64_t m;
    int64_t n;
    int64_t ldi;
    int64_t ldo;
    int64_t type;
    int64_t bcast_type;
  } xsmm_unary_dispatch;
  xsmm_unary_dispatch *p = (xsmm_unary_dispatch *)params;
  p->res = _mlir_ciface_xsmm_unary_dispatch_f32(p->m, p->n, p->ldi, p->ldo,
                                                p->type, p->bcast_type);
  return 0;
}

// TODO: struct slicing. BRGEMM struct is the same as the GEMM one plus the
// batch parameter.
extern "C" int iree_xsmm_brgemm_invoke_f32(void *context, void *params,
                                           void *reserved) {
  typedef struct {
    int64_t addr;
    float *pA;
    int64_t offA;
    float *pB;
    int64_t offB;
    float *pC;
    int64_t offC;
    int64_t numBatches;
  } xsmm_brgemm_invoke_f32_t;
  xsmm_brgemm_invoke_f32_t *p = (xsmm_brgemm_invoke_f32_t *)params;

  float *addr_tensorA = p->pA + p->offA;
  float *addr_tensorB = p->pB + p->offB;
  float *addr_tensorC = p->pC + p->offC;

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(p->addr);
  unsigned long long numBatchesVar = p->numBatches;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *)addr_tensorB;
  gemm_param.b.primary = (void *)addr_tensorA;
  gemm_param.c.primary = (void *)addr_tensorC;
  gemm_param.op.tertiary = (void *)&numBatchesVar;
  sgemm.gemm(&gemm_param);

  return 0;
}

extern "C" int iree_xsmm_matmul_invoke_f32(void *context, void *params,
                                           void *reserved) {
  typedef struct {
    int64_t addr;
    float *pA;
    int64_t offA;
    float *pB;
    int64_t offB;
    float *pC;
    int64_t offC;
  } xsmm_matmul_invoke_f32_t;
  xsmm_matmul_invoke_f32_t *p = (xsmm_matmul_invoke_f32_t *)params;

  float *addr_tensorA = p->pA + p->offA;
  float *addr_tensorB = p->pB + p->offB;
  float *addr_tensorC = p->pC + p->offC;

  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  // LIBXSMM col-major change A with B.
  gemm_param.a.primary = (void *)addr_tensorB;
  gemm_param.b.primary = (void *)addr_tensorA;
  gemm_param.c.primary = (void *)addr_tensorC;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(p->addr);
  sgemm.gemm(&gemm_param);

  return 0;
}

extern "C" int iree_xsmm_unary_invoke(void *context, void *params,
                                      void *reserved) {
  typedef struct {
    int64_t addr;
    float *pA;
    int64_t offA;
    float *pB;
    int64_t offB;
  } xsmm_unary_invoke;
  xsmm_unary_invoke *p = (xsmm_unary_invoke *)params;

  float *addr_a = p->pA + p->offA;
  float *addr_b = p->pB + p->offB;

  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(p->addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = (void *)addr_a;
  param.out.primary = (void *)addr_b;
  kernel(&param);

  return 0;
}
