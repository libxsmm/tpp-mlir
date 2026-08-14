// MLP with Softmax version
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 --softmax | tpp-run -e entry -entry-point-result=void

// MLP without softmax
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 | tpp-run -e entry -entry-point-result=void

// MLP with identity
// RUN: mlir-gen --kernel=const --identity --seed=0 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=IDENTITY_CONST
// RUN: mlir-gen --kernel=args --seed=0 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print -identity=1 | FileCheck %s --check-prefix=IDENTITY_ARGS

// Matmul only
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 --output=generic | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 --output=contract | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-gen --kernel=const --batch=10 --layers=10,10 --output=named | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=MATMUL

// Constant values
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 --output=generic | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 --output=contract | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT
// RUN: mlir-gen --kernel=const --bias --relu --batch=10 --layers=10,10 --output=named | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=CONSTANT

// Kernel - matmul
// RUN: mlir-gen --kernel=args --seed=123 --data-type=f32 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL

// Kernel - fc
// RUN: mlir-gen --kernel=args --bias --relu --seed=123 --data-type=f32 --batch=10 --layers=10,10 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-FC

// Packed versions
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=10 --layers=10,10,10 --tiles=2,2,2 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF

// Use two Kernels, one matmul and other quantize-dequantize. That is, the result of matmul would be quantized and then dequantized.
// RUN: mlir-gen --kernel=const --seed=0 --data-type=f32 --batch=3 --layers=3,3 --identity | tpp-run -e entry -entry-point-result=void -print --splat-to-random --init-type normal  -seed 123 > %t.1
// RUN: mlir-gen --identity --kernel=const --seed=0 --data-type=f32-i8 --batch=3 --layers=3,3 --test-quant | tpp-run -e entry -entry-point-result=void -print --splat-to-random --init-type normal  -seed 123 > %t.2

// Comparison is done between the result of matmul and the result of quantize-dequantize kernels.
// RUN: fpcmp -a 0.01 -r 0.01 %t.1 %t.2


// Implements C = A*B, with A=1, B=ID
// IDENTITY_CONST:( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 )

// Implements C += A*B, with A=1, B=ID, C=1
// IDENTITY_ARGS:( 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 )

// MATMUL:( 10, 10, 10, 10, 10, 10, 10, 10, 10, 10 )

// CONSTANT:( 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 )

// GEN-MATMUL: ( 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 )

// GEN-FC: ( 12, 12, 12, 12, 12, 12, 12, 12, 12, 12 )

// PERF:    {{[0-9]+}}{{.?}}{{[0-9e-]+}}
