// Execution (integration) tests for the PyTorch `emit_brgemm.py` generator. Each
// RUN line generates a batch-reduce matmul through the Lighthouse `uv`
// environment (the `emit-brgemm` substitution), then runs it with tpp-run and
// FileCheck verifies the printed result. Inputs are auto-initialised to all 1.0
// (splat), so for br_count=B, K=8 the accumulator is B*8 and the output is
// B*8 + C(1.0). Only the backend-executable element types (f32, bf16, bf8, hf8)
// are covered here. Gated on the 'lighthouse' feature.
//
// REQUIRES: lighthouse

// f32: 2*8 + 1 = 17 (exact).
// RUN: emit-brgemm matmul --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --br_count 2 -o %t.f32.mlir
// RUN: tpp-run %t.f32.mlir -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=F32
// F32: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// bf16 output: 17 is exactly representable in bf16.
// RUN: emit-brgemm matmul --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 -o %t.bf16.mlir
// RUN: tpp-run %t.bf16.mlir -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16
// BF16: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// bf8 (f8E5M2) inputs accumulated into f32: exact 17.
// RUN: emit-brgemm matmul --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 -o %t.bf8.mlir
// RUN: tpp-run %t.bf8.mlir -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF8
// BF8: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// bf8 output: narrowing 17 to f8E5M2 rounds to 16 (2-bit mantissa; 17 is not
// representable, nearest even is 16).
// RUN: emit-brgemm matmul --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType bf8 --br_count 2 -o %t.bf8out.mlir
// RUN: tpp-run %t.bf8out.mlir -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF8OUT
// BF8OUT: ( 16, 16, 16, 16, 16, 16, 16, 16 )

// hf8 (f8E4M3FN) inputs accumulated into f32: exact 17.
// RUN: emit-brgemm matmul --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 -o %t.hf8.mlir
// RUN: tpp-run %t.hf8.mlir -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=HF8
// HF8: ( 17, 17, 17, 17, 17, 17, 17, 17 )
