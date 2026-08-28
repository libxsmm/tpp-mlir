// RUN: torch-import %S/Inputs/linear.py --splat | FileCheck %s --check-prefix=SINGLE_GEMM
// RUN: torch-import %S/Inputs/linear.py --splat --M=16 --N=32 --K=128 | FileCheck %s --check-prefix=SINGLE_GEMM_OVERRIDE
// RUN: torch-import %S/Inputs/linear.py --splat --layers=2 | FileCheck %s --check-prefix=DOUBLE_GEMM
// RUN: torch-import %S/Inputs/linear.py --splat --bias --relu | FileCheck %s --check-prefix=MLP
// RUN: torch-import %S/Inputs/linear.py --splat --trans_a | FileCheck %s --check-prefix=TRANS_A
// RUN: torch-import %S/Inputs/linear.py --trans_b | FileCheck %s --check-prefix=TRANS_B

// RUN: torch-import %S/Inputs/linear.py 2>/dev/null | \
// RUN:   tpp-opt -default-tpp-passes | FileCheck %s --check-prefix=XSMM
// RUN: torch-import %S/Inputs/linear.py 2>/dev/null | \
// RUN:   tpp-opt -default-tpp-passes="nano-kernel" | FileCheck %s --check-prefix=NANO

// REQUIRES: lighthouse

// SINGLE_GEMM-LABEL: func.func @main(
// SINGLE_GEMM: linalg.matmul {{.*}} tensor<64x64xf32>, tensor<64x64xf32>

// SINGLE_GEMM_OVERRIDE-LABEL: func.func @main(
// SINGLE_GEMM_OVERRIDE: linalg.matmul {{.*}} tensor<16x128xf32>, tensor<128x32xf32>

// DOUBLE_GEMM-LABEL: func.func @main(
// DOUBLE_GEMM-COUNT-2: linalg.matmul

// MLP-LABEL: func.func @main(
// MLP: linalg.matmul
// MLP: linalg.generic
// MLP: arith.addf
// MLP: linalg.generic
// MLP: arith.cmpf
// MLP: arith.select

// TRANS_A-LABEL: func.func @main(
// TRANS_A: linalg.transpose ins(%arg0
// TRANS_A: linalg.matmul

// TRANS_B-LABEL: func.func @main(
// TRANS_B: linalg.transpose ins(%cst_0
// TRANS_B: linalg.matmul

// XSMM-LABEL: func.func @main(
// XSMM-SAME:    %[[ARG0:.+]]: memref<64x64xf32>) -> memref<64x64xf32>
// XSMM-DAG:     memref.get_global @__constant_64x64xf32
// XSMM:         call @xsmm_brgemm_dispatch
// XSMM:         call @xsmm_brgemm_invoke
// XSMM:         return

// NANO-LABEL: func.func @main(
// NANO-SAME:    %[[ARG0:.+]]: memref<64x64xf32>) -> memref<64x64xf32>
// NANO:       scf.parallel
// NANO:       vector.transfer_read
// NANO:       vector.transfer_read
// NANO:       vector.contract
// NANO:       vector.transfer_write
