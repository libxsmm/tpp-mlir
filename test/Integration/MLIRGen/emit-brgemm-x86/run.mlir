// Execution (integration) tests for `emit_brgemm.py` generator. Each RUN line
// generates a batch-reduce matmul through the Lighthouse `uv` environment (the
// `emit-brgemm` substitution), runs it with tpp-run and FileCheck verifies the
// printed result.

// -----------------------------------------------------------------------------
// f32 (no VNNI): transpose combinations.
// -----------------------------------------------------------------------------
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --br_count 2 | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=F32
// F32: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --br_count 2 --transA | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=F32_TA
// F32_TA: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --br_count 2 --transB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=F32_TB
// F32_TB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f32 --bType f32 --cType f32 --br_count 2 --transA --transB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=F32_TAB
// F32_TAB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// -----------------------------------------------------------------------------
// bf16: transpose combinations plus VNNI.
// -----------------------------------------------------------------------------
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16
// BF16: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 --transA | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16_TA
// BF16_TA: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 --transB | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16_TB
// BF16_TB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 --transA --transB | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16_TAB
// BF16_TAB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 --vnniA --vnniB | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16_VNNI
// BF16_VNNI: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf16 --bType bf16 --cType bf16 --br_count 2 --vnniB | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF16_VNNIB
// BF16_VNNIB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// Larger bf16 VNNI case (K=64): 2*64 + 1 = 129 (exactly representable in bf16).
// RUN: emit-brgemm brgemm --M 64 --N 64 --K 64 --aType bf16 --bType bf16 --cType bf16 --vnniA --vnniB --br_count 2 | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=BF16BIG
// BF16BIG: ( 129, 129, 129, 129, 129, 129, 129, 129

// -----------------------------------------------------------------------------
// f16: VNNI only (plain f16 is not accepted by the xsmm brgemm operand).
// -----------------------------------------------------------------------------
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f16 --bType f16 --cType f16 --br_count 2 --vnniA --vnniB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=F16_VNNI
// F16_VNNI: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType f16 --bType f16 --cType f16 --br_count 2 --vnniB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=F16_VNNIB
// F16_VNNIB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// -----------------------------------------------------------------------------
// bf8 (f8E5M2), accumulated into f32: transpose combinations plus VNNI.
// -----------------------------------------------------------------------------
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF8
// BF8: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 --transA | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF8_TA
// BF8_TA: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 --transB | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF8_TB
// BF8_TB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 --transA --transB | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=BF8_TAB
// BF8_TAB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 --vnniA --vnniB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=BF8_VNNI
// BF8_VNNI: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType bf8 --bType bf8 --cType f32 --br_count 2 --vnniB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=BF8_VNNIB
// BF8_VNNIB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// -----------------------------------------------------------------------------
// hf8 (f8E4M3FN), accumulated into f32: transpose combinations plus VNNI.
// -----------------------------------------------------------------------------
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=HF8
// HF8: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 --transA | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=HF8_TA
// HF8_TA: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 --transB | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=HF8_TB
// HF8_TB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 --transA --transB | tpp-run - -e entry --entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=HF8_TAB
// HF8_TAB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 --vnniA --vnniB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=HF8_VNNI
// HF8_VNNI: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType hf8 --bType hf8 --cType f32 --br_count 2 --vnniB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=HF8_VNNIB
// HF8_VNNIB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// -----------------------------------------------------------------------------
// i8, accumulated into i32: VNNI only (plain i8 is not accepted by xsmm brgemm).
// -----------------------------------------------------------------------------
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType i8 --bType i8 --cType i32 --br_count 2 --vnniA --vnniB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=I8_VNNI
// I8_VNNI: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType i8 --bType i8 --cType i32 --br_count 2 --vnniB | tpp-run - -e entry --entry-point-result=void -print | FileCheck %s --check-prefix=I8_VNNIB
// I8_VNNIB: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// -----------------------------------------------------------------------------
// i16, accumulated into i32: VNNI only.
// -----------------------------------------------------------------------------
// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType i16 --bType i16 --cType i32 --br_count 2 --vnniA --vnniB | tpp-run - -e entry --entry-point-result=void  -print | FileCheck %s --check-prefix=I16_VNNI
// I16_VNNI: ( 17, 17, 17, 17, 17, 17, 17, 17 )

// RUN: emit-brgemm brgemm --M 8 --N 8 --K 8 --aType i16 --bType i16 --cType i32 --br_count 2 --vnniB | tpp-run - -e entry --entry-point-result=void  -print | FileCheck %s --check-prefix=I16_VNNIB
// I16_VNNIB: ( 17, 17, 17, 17, 17, 17, 17, 17 )
