// MLP with Softmax version
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 --softmax | tpp-run -e entry -entry-point-result=void
// RUN: not --crash mlir-gen --output=named --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 --softmax 2>&1 | FileCheck %s --check-prefix=SOFTMAX-TODO
// SOFTMAX-TODO: Linalg named ops for softmax not implemented yet
// SOFTMAX-TODO: UNREACHABLE executed

// MLP without softmax
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 | tpp-run -e entry -entry-point-result=void

// Matmul only
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 --output=generic | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 --output=contract | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 --output=named | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 --output=named --keep-generic-matmul | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL

// Constant values
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 --output=generic | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 --output=contract | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 --output=named | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 --output=named --keep-generic-matmul | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT

// Kernel - matmul
// RUN: mlir-gen --kernel=args --seed=123 --float-type=f32 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL

// Kernel - fc
// RUN: mlir-gen --kernel=args --bias --relu --seed=123 --float-type=f32 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-FC

// Packed versions
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF

// MATMUL:( 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 )

// CONSTANT:( 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 )

// GEN-MATMUL: ( 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 )

// GEN-FC: ( 12, 12, 12, 12, 12, 12, 12, 12, 12, 12 )

// PERF:    {{[0-9]+}}{{.?}}{{[0-9e-]+}}
