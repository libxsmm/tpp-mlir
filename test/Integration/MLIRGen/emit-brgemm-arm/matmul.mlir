// Structure (unit) tests for the PyTorch `emit_brgemm.py` generator on ARM. The
// generator auto-detects the host architecture (aarch64/arm64) and selects the
// ARM-specific bf16 VNNI factor of 4 (vs 2 on x86), and rejects any A/B/C
// element type other than f32/bf16. Each RUN line invokes the generator through
// the Lighthouse `uv` environment (the `emit-brgemm` substitution) and
// FileCheck verifies the emitted IR. The directory-level lit.local.cfg gates
// these on the 'lighthouse' feature and an ARM host.

// Direct path: comp type == C type (f32), so a single linalg.contract is
// emitted with no fill/epilogue (arch-independent; f32 has no VNNI).
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

// VNNI layout on A and B (with A transposed): the contract uses 5-D indexing
// maps and the K dimension is split into an inner VNNI factor. On ARM the bf16
// VNNI factor is 4, so K=8 -> 2x4 (vs x86's 4x2).
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType f32 --vnniA --vnniB --transA --br_count 2 2>&1 | FileCheck %s --check-prefix=VNNI
// VNNI-DAG: #[[$MA:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d4)>
// VNNI-DAG: #[[$MB:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
// VNNI-LABEL: func.func @entry(
// VNNI-SAME: %[[A:.+]]: tensor<2x2x8x4xbf16>, %[[B:.+]]: tensor<2x2x8x4xbf16>, %[[C:.+]]: tensor<8x8xf32>) -> tensor<8x8xf32>
// VNNI: linalg.contract {{.*}} ins(%[[A]], %[[B]] : tensor<2x2x8x4xbf16>, tensor<2x2x8x4xbf16>)

// VNNI on B only: A stays non-VNNI and is expanded into VNNI (K -> K/vf x vf)
// via tensor.expand_shape before the contract. ARM bf16 factor is 4, so K=64 ->
// 16x4.
// RUN: emit-brgemm brgemm --M 64 --N 64 --K 64 --aType bf16 --bType bf16 --cType f32 --vnniB --br_count 8 2>&1 | FileCheck %s --check-prefix=EXPANDB
// EXPANDB-LABEL: func.func @entry(
// EXPANDB-SAME: %[[A:.+]]: tensor<8x64x64xbf16>, %[[B:.+]]: tensor<8x16x64x4xbf16>, %[[C:.+]]: tensor<64x64xf32>) -> tensor<64x64xf32>
// EXPANDB: %[[E:.+]] = tensor.expand_shape %[[A]] {{\[}}[0], [1], [2, 3]] output_shape [8, 64, 16, 4] : tensor<8x64x64xbf16> into tensor<8x64x16x4xbf16>
// EXPANDB: linalg.contract {{.*}} ins(%[[E]], %[[B]] : tensor<8x64x16x4xbf16>, tensor<8x16x64x4xbf16>)

// ARM rejects every A/B/C element type other than f32/bf16.
// RUN: not emit-brgemm brgemm --M 8 --N 8 --K 8 --aType i8 --bType i8 --cType f32 --vnniA --vnniB 2>&1 | FileCheck %s --check-prefix=ARM-I8
// ARM-I8: ARM only supports bf16, f32 types, got: i8

// RUN: not emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --vnniA --vnniB 2>&1 | FileCheck %s --check-prefix=ARM-BF8
// ARM-BF8: ARM only supports bf16, f32 types, got: bf8

// RUN: not emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --vnniA --vnniB 2>&1 | FileCheck %s --check-prefix=ARM-HF8
// ARM-HF8: ARM only supports bf16, f32 types, got: hf8

// RUN: not emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f16 --bType f16 --cType f16 --vnniA --vnniB 2>&1 | FileCheck %s --check-prefix=ARM-F16
// ARM-F16: ARM only supports bf16, f32 types, got: f16

// RUN: not emit-brgemm brgemm --M 8 --N 8 --K 8 --aType i16 --bType i16 --cType f32 --vnniA --vnniB 2>&1 | FileCheck %s --check-prefix=ARM-I16
// ARM-I16: ARM only supports bf16, f32 types, got: i16
