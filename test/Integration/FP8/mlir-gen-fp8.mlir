// FP8 code generation and execution. libxsmm names E5M2 as BF8 and E4M3 as HF8.
// FP8 uses a VNNI blocking factor of 4.

// Kernel - matmul. Inputs are auto-initialized to 1.0 and the reduction size is
// 16, so every output element is 16 (accumulation happens in F32).
// RUN: mlir-gen --kernel=args --seed=123 --data-type=bf8 --batch=16 --layers=16,16 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8
// RUN: mlir-gen --kernel=args --seed=123 --data-type=hf8 --batch=16 --layers=16,16 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8
// RUN: mlir-gen --kernel=args --seed=123 --data-type=hf8 --batch=16 --layers=16,16 --tiles=8,8,8 -vnni=4 --transpose-a=1 --transpose-b=0 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8-VAR
// RUN: mlir-gen --kernel=args --seed=123 --data-type=hf8 --batch=16 --layers=16,16 --tiles=8,8,8 -vnni=0 --transpose-a=1 --transpose-b=1 | tpp-run -e entry -entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8-VAR
// RUN: mlir-gen --kernel=args --seed=123 --data-type=hf8 --batch=16 --layers=16,16 --tiles=8,8,8 -vnni=0 --transpose-a=1 --transpose-b=0 | tpp-run -e entry -entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8-VAR
// RUN: mlir-gen --kernel=args --seed=123 --data-type=hf8 --batch=16 --layers=16,16 --tiles=8,8,8 -vnni=0 --transpose-a=0 --transpose-b=1 | tpp-run -e entry -entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8-VAR
// RUN: mlir-gen --kernel=args --seed=123 --data-type=bf8 --batch=16 --layers=16,16 --tiles=8,8,8 -vnni=4 --transpose-a=1 --transpose-b=0 | tpp-run -e entry -entry-point-result=void -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8-VAR
// RUN: mlir-gen --kernel=args --seed=123 --data-type=bf8 --batch=16 --layers=16,16 --tiles=8,8,8 -vnni=0 --transpose-a=1 --transpose-b=1 | tpp-run -e entry -entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8-VAR
// RUN: mlir-gen --kernel=args --seed=123 --data-type=bf8 --batch=16 --layers=16,16 --tiles=8,8,8 -vnni=0 --transpose-a=1 --transpose-b=0 | tpp-run -e entry -entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8-VAR
// RUN: mlir-gen --kernel=args --seed=123 --data-type=bf8 --batch=16 --layers=16,16 --tiles=8,8,8 -vnni=0 --transpose-a=0 --transpose-b=1 | tpp-run -e entry -entry-point-result=void --disable-vnni-packing -print | FileCheck %s --check-prefix=GEN-MATMUL-FP8-VAR


// FP8/VNNI execution
// RUN: mlir-gen --kernel=const --seed=123 --batch=16 --layers=16,16 --tiles=8,8,8 --data-type=bf8 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --kernel=const --seed=123 --batch=16 --layers=16,16 --tiles=8,8,8 --data-type=hf8 | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --kernel=const --seed=123 --batch=16 --layers=16,16 --tiles=8,8,8 --data-type=bf8 | tpp-opt --pack-vnni | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF
// RUN: mlir-gen --kernel=const --seed=123 --batch=16 --layers=16,16 --tiles=8,8,8 --data-type=hf8 | tpp-opt --pack-vnni | tpp-run -e entry -entry-point-result=void -n 10 | FileCheck %s --check-prefix=PERF

// GEN-MATMUL-FP8: ( 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16 )
// GEN-MATMUL-FP8-VAR: ( 16, 16, 16, 16, 16, 16, 16, 16 )

// PERF: {{[0-9]+}}{{.?}}{{[0-9e-]+}}
