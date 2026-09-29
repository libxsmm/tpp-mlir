// Execution (integration) tests for `emit_brgemm.py` generator. Each
// RUN line generates a batch-reduce matmul through the Lighthouse `uv`
// environment (the `emit-brgemm` substitution), then runs it with tpp-run and
// FileCheck verifies the printed result. 
//
// REQUIRES: lighthouse

// f32: 2*8 + 1 = 17 (exact).
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --br_count 2 | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=F32
// F32: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// bf16 output: 17 is exactly representable in bf16.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16
// BF16: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// Larger bf16 case (K=64): 2*64 + 1 = 129 (exactly representable in bf16).
// RUN: emit-brgemm brgemm --M 64 --N 64 --K 64 --aType bf16 --bType bf16 --cType bf16 --br_count 2 | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16BIG
// BF16BIG: ( 129, 129, 129, 129, 129, 129, 129, 129

// bf8 (f8E5M2) inputs accumulated into f32: exact 17.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF8
// BF8: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// bf8 output: narrowing 17 to f8E5M2 rounds to 16 (2-bit mantissa; 17 is not
// representable, nearest even is 16).
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType bf8 --br_count 2 | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF8OUT
// BF8OUT: ( 16, 16, 16, 16, 16, 16, 16, 16 )

// hf8 (f8E4M3FN) inputs accumulated into f32: exact 17.
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=HF8
// HF8: ( 17, 17, 17, 17, 17, 17, 17, 17 )
