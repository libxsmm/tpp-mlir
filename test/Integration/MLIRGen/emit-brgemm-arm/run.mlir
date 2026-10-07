// Execution (integration) tests for `emit_brgemm.py` generator on ARM. Each RUN
// line generates a batch-reduce matmul through the Lighthouse `uv` environment
// (the `emit-brgemm` substitution), runs it with tpp-run and FileCheck verifies
// the printed result. ARM only supports f32 and bf16 operands, and the bf16
// VNNI factor is 4. The directory-level lit.local.cfg gates these on an ARM
// host.

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
// bf16: transpose combinations plus VNNI (factor 4 on ARM).
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
