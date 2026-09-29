// Structure (unit) tests for the PyTorch `emit_brgemm.py` generator. Each RUN
// line invokes the generator through the Lighthouse `uv` environment (the
// `emit-brgemm` substitution) and FileCheck verifies the emitted IR. Gated on
// the 'lighthouse' feature (submodule + `uv` present).
//
// REQUIRES: lighthouse

// Direct path: comp type == C type (f32), so a single linalg.contract is
// emitted with no fill/epilogue.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --br_count 2 2>&1 | FileCheck %s --check-prefix=F32
// F32-DAG: #[[$MA:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// F32-DAG: #[[$MB:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// F32-DAG: #[[$MC:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// F32-LABEL: func.func @entry(
// F32-SAME: %[[A:.+]]: tensor<2x8x8xf32>, %[[B:.+]]: tensor<2x8x8xf32>, %[[C:.+]]: tensor<8x8xf32>) -> tensor<8x8xf32>
// F32: %[[R:.+]] = linalg.contract indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]] ins(%[[A]], %[[B]] : tensor<2x8x8xf32>, tensor<2x8x8xf32>) outs(%[[C]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// F32-NOT: linalg.generic
// F32: return %[[R]] : tensor<8x8xf32>

// Narrow C (bf16): accumulate in f32 (fill + contract into f32), then a generic
// epilogue extends C to f32, adds, and truncates the result back to bf16.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 2>&1 | FileCheck %s --check-prefix=BF16
// BF16-LABEL: func.func @entry(
// BF16-SAME: %[[A:.+]]: tensor<2x8x8xbf16>, %[[B:.+]]: tensor<2x8x8xbf16>, %[[C:.+]]: tensor<8x8xbf16>) -> tensor<8x8xbf16>
// BF16: %[[Z:.+]] = arith.constant 0.000000e+00 : f32
// BF16: %[[E:.+]] = tensor.empty() : tensor<8x8xf32>
// BF16: %[[F:.+]] = linalg.fill ins(%[[Z]] : f32) outs(%[[E]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// BF16: %[[ACC:.+]] = linalg.contract {{.*}} outs(%[[F]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// BF16: linalg.generic {{.*}} ins(%[[ACC]], %[[C]] : tensor<8x8xf32>, tensor<8x8xbf16>) outs(%[[C]] : tensor<8x8xbf16>)
// BF16: ^bb0(%[[IN:.+]]: f32, %[[INC:.+]]: bf16, %[[OUT:.+]]: bf16):
// BF16: %[[EXT:.+]] = arith.extf %[[INC]] : bf16 to f32
// BF16: %[[ADD:.+]] = arith.addf %[[IN]], %[[EXT]] : f32
// BF16: %[[TR:.+]] = arith.truncf %[[ADD]] : f32 to bf16
// BF16: linalg.yield %[[TR]] : bf16

// Mixed precision bf16 inputs, f32 C: comp type == C type (f32) so the direct
// path emits a single linalg.contract (bf16 ins, f32 out) with no epilogue.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType f32 --br_count 2 2>&1 | FileCheck %s --check-prefix=BF16F32
// BF16F32-LABEL: func.func @entry(
// BF16F32-SAME: %[[A:.+]]: tensor<2x8x8xbf16>, %[[B:.+]]: tensor<2x8x8xbf16>, %[[C:.+]]: tensor<8x8xf32>) -> tensor<8x8xf32>
// BF16F32: %[[R:.+]] = linalg.contract {{.*}} ins(%[[A]], %[[B]] : tensor<2x8x8xbf16>, tensor<2x8x8xbf16>) outs(%[[C]] : tensor<8x8xf32>) -> tensor<8x8xf32>
// BF16F32-NOT: linalg.generic
// BF16F32: return %[[R]] : tensor<8x8xf32>

// bf8 alias -> f8E5M2 element type (never emits the literal "bf8").
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 2>&1 | FileCheck %s --check-prefix=BF8
// BF8-NOT: bf8
// BF8-LABEL: func.func @entry(
// BF8-SAME: %[[A:.+]]: tensor<2x8x8xf8E5M2>, %[[B:.+]]: tensor<2x8x8xf8E5M2>, %[[C:.+]]: tensor<8x8xf32>) -> tensor<8x8xf32>
// BF8: linalg.contract {{.*}} ins(%[[A]], %[[B]] : tensor<2x8x8xf8E5M2>, tensor<2x8x8xf8E5M2>)

// hf8 alias -> f8E4M3FN element type (never emits the literal "hf8").
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 2>&1 | FileCheck %s --check-prefix=HF8
// HF8-NOT: hf8
// HF8-LABEL: func.func @entry(
// HF8-SAME: %[[A:.+]]: tensor<2x8x8xf8E4M3FN>, %[[B:.+]]: tensor<2x8x8xf8E4M3FN>, %[[C:.+]]: tensor<8x8xf32>) -> tensor<8x8xf32>
// HF8: linalg.contract {{.*}} ins(%[[A]], %[[B]] : tensor<2x8x8xf8E4M3FN>, tensor<2x8x8xf8E4M3FN>)

// Narrow integer C (i8): accumulate in i32, epilogue uses extsi/addi/trunci.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType i8 --bType i8 --cType i8 --br_count 2 2>&1 | FileCheck %s --check-prefix=I8
// I8-LABEL: func.func @entry(
// I8-SAME: %[[A:.+]]: tensor<2x8x8xi8>, %[[B:.+]]: tensor<2x8x8xi8>, %[[C:.+]]: tensor<8x8xi8>) -> tensor<8x8xi8>
// I8: %[[Z:.+]] = arith.constant 0 : i32
// I8: linalg.fill ins(%[[Z]] : i32) {{.*}} -> tensor<8x8xi32>
// I8: %[[ACC:.+]] = linalg.contract {{.*}} -> tensor<8x8xi32>
// I8: linalg.generic {{.*}} ins(%[[ACC]], %[[C]] : tensor<8x8xi32>, tensor<8x8xi8>) outs(%[[C]] : tensor<8x8xi8>)
// I8: ^bb0(%[[IN:.+]]: i32, %[[INC:.+]]: i8, %[[OUT:.+]]: i8):
// I8: %[[EXT:.+]] = arith.extsi %[[INC]] : i8 to i32
// I8: %[[ADD:.+]] = arith.addi %[[IN]], %[[EXT]] : i32
// I8: %[[TR:.+]] = arith.trunci %[[ADD]] : i32 to i8
// I8: linalg.yield %[[TR]] : i8

// Explicit comp type wider than C (f64 comp, f32 C): accumulate in f64, epilogue
// extends C to f64 and truncates the result back to f32.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --compType f64 --br_count 2 2>&1 | FileCheck %s --check-prefix=COMP64
// COMP64: %[[Z:.+]] = arith.constant 0.000000e+00 : f64
// COMP64: linalg.fill ins(%[[Z]] : f64) {{.*}} -> tensor<8x8xf64>
// COMP64: linalg.contract {{.*}} -> tensor<8x8xf64>
// COMP64: ^bb0(%[[IN:.+]]: f64, %[[INC:.+]]: f32, %[[OUT:.+]]: f32):
// COMP64: arith.extf %[[INC]] : f32 to f64
// COMP64: arith.addf {{.*}} : f64
// COMP64: arith.truncf {{.*}} : f64 to f32

// alpha/beta scaling: epilogue multiplies the accumulator by alpha and C by
// beta before adding.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --alpha 2.0 --beta 3.0 --br_count 2 2>&1 | FileCheck %s --check-prefix=SCALE
// SCALE: ^bb0(%[[IN:.+]]: f32, %[[INC:.+]]: f32, %[[OUT:.+]]: f32):
// SCALE: %[[CA:.+]] = arith.constant 2.000000e+00 : f32
// SCALE: %[[MA:.+]] = arith.mulf %[[CA]], %[[IN]] : f32
// SCALE: %[[CB:.+]] = arith.constant 3.000000e+00 : f32
// SCALE: %[[MB:.+]] = arith.mulf %[[CB]], %[[INC]] : f32
// SCALE: arith.addf %[[MA]], %[[MB]] : f32

// Transpose A and B without VNNI packing: A is stored K x M, B is stored N x K,
// and the contract's indexing maps swap the corresponding dims (A -> (batch, K,
// M), B -> (batch, N, K)). comp type == C type (f32) so a single linalg.contract
// is emitted with no epilogue.
// RUN: emit-brgemm brgemm --M 8 --N 16 --K 32 --aType f32 --bType f32 --cType f32 --transA --transB --br_count 2 2>&1 | FileCheck %s --check-prefix=TRANS
// TRANS-DAG: #[[$MA:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// TRANS-DAG: #[[$MB:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// TRANS-DAG: #[[$MC:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// TRANS-LABEL: func.func @entry(
// TRANS-SAME: %[[A:.+]]: tensor<2x32x8xf32>, %[[B:.+]]: tensor<2x16x32xf32>, %[[C:.+]]: tensor<8x16xf32>) -> tensor<8x16xf32>
// TRANS: %[[R:.+]] = linalg.contract indexing_maps = [#[[$MA]], #[[$MB]], #[[$MC]]] ins(%[[A]], %[[B]] : tensor<2x32x8xf32>, tensor<2x16x32xf32>) outs(%[[C]] : tensor<8x16xf32>) -> tensor<8x16xf32>
// TRANS-NOT: linalg.generic
// TRANS: return %[[R]] : tensor<8x16xf32>

// VNNI layout on A and B (with A transposed): the contract uses 5-D indexing
// maps and the K dimension is split into an inner VNNI factor (K=8 -> 4x2 for
// bf16).
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType f32 --vnniA --vnniB --transA --br_count 2 2>&1 | FileCheck %s --check-prefix=VNNI
// VNNI-DAG: #[[$MA:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
// VNNI-DAG: #[[$MB:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
// VNNI-LABEL: func.func @entry(
// VNNI-SAME: %[[A:.+]]: tensor<2x4x8x2xbf16>, %[[B:.+]]: tensor<2x4x8x2xbf16>, %[[C:.+]]: tensor<8x8xf32>) -> tensor<8x8xf32>
// VNNI: linalg.contract {{.*}} ins(%[[A]], %[[B]] : tensor<2x4x8x2xbf16>, tensor<2x4x8x2xbf16>)

// VNNI on B only: A stays non-VNNI and is expanded into VNNI (K -> K/vf x vf)
// via tensor.expand_shape before the contract.
// RUN: emit-brgemm brgemm --M 64 --N 64 --K 64 --aType bf16 --bType bf16 --cType f32 --vnniB --br_count 8 2>&1 | FileCheck %s --check-prefix=EXPANDB
// EXPANDB-LABEL: func.func @entry(
// EXPANDB-SAME: %[[A:.+]]: tensor<8x64x64xbf16>, %[[B:.+]]: tensor<8x32x64x2xbf16>, %[[C:.+]]: tensor<64x64xf32>) -> tensor<64x64xf32>
// EXPANDB: %[[E:.+]] = tensor.expand_shape %[[A]] {{\[}}[0], [1], [2, 3]] output_shape [8, 64, 32, 2] : tensor<8x64x64xbf16> into tensor<8x64x32x2xbf16>
// EXPANDB: linalg.contract {{.*}} ins(%[[E]], %[[B]] : tensor<8x64x32x2xbf16>, tensor<8x32x64x2xbf16>)

// VNNI + transposing B is rejected: only A may be transposed under VNNI.
// RUN: not emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --vnniA --vnniB --transB 2>&1 | FileCheck %s --check-prefix=VNNI-ERR
// VNNI-ERR: VNNI does not support transposing B (only A may be transposed)
