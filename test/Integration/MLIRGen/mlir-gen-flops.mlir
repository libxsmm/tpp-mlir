// Unit sizes
// RUN: mlir-gen --kernel=args --seed=0 --float-type=f32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=MATMUL-UNIT
// RUN: mlir-gen --output=named --kernel=args --seed=0 --float-type=f32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=MATMUL-UNIT-NAMED
// RUN: mlir-gen --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=FC-UNIT
// RUN: mlir-gen --output=named --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=FC-UNIT-NAMED
// RUN: mlir-gen --kernel=const --bias --relu --seed=0 --float-type=f32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=MLP-UNIT
// RUN: mlir-gen --output=named --kernel=const --bias --relu --seed=0 --float-type=f32 --batch=1 --layers=1,1 2>&1 | FileCheck %s --check-prefix=MLP-UNIT-NAMED
// Small sizes
// RUN: mlir-gen --kernel=args --seed=0 --float-type=f32 --batch=8 --layers=4,16 2>&1 | FileCheck %s --check-prefix=MATMUL-SMALL
// RUN: mlir-gen --output=named --kernel=args --seed=0 --float-type=f32 --batch=8 --layers=4,16 2>&1 | FileCheck %s --check-prefix=MATMUL-SMALL-NAMED
// RUN: mlir-gen --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=8 --layers=4,16 2>&1 | FileCheck %s --check-prefix=FC-SMALL
// RUN: mlir-gen --output=named --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=8 --layers=4,16 2>&1 | FileCheck %s --check-prefix=FC-SMALL-NAMED
// RUN: mlir-gen --kernel=const --bias --relu --seed=0 --float-type=f32 --batch=8 --layers=4,8,16 2>&1 | FileCheck %s --check-prefix=MLP-SMALL
// RUN: mlir-gen --output=named --kernel=const --bias --relu --seed=0 --float-type=f32 --batch=8 --layers=4,8,16 2>&1 | FileCheck %s --check-prefix=MLP-SMALL-NAMED
// Large sizes + no tiling
// RUN: mlir-gen --kernel=args --seed=0 --float-type=f32 --batch=128 --layers=1024,4096 2>&1 | FileCheck %s --check-prefix=MATMUL-LARGE
// RUN: mlir-gen --output=named --kernel=args --seed=0 --float-type=f32 --batch=128 --layers=1024,4096 2>&1 | FileCheck %s --check-prefix=MATMUL-LARGE-NAMED
// RUN: mlir-gen --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=1024,4096 2>&1 | FileCheck %s --check-prefix=FC-LARGE
// RUN: mlir-gen --output=named --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=1024,4096 2>&1 | FileCheck %s --check-prefix=FC-LARGE-NAMED
// RUN: mlir-gen --kernel=const --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=1024,1024,1024 2>&1 | FileCheck %s --check-prefix=MLP-LARGE
// RUN: mlir-gen --output=named --kernel=const --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=1024,1024,1024 2>&1 | FileCheck %s --check-prefix=MLP-LARGE-NAMED
// Large sizes + tiling
// RUN: mlir-gen --kernel=args --seed=0 --float-type=f32 --batch=128 --layers=1024,4096 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=MATMUL-LARGE
// RUN: mlir-gen --output=named --kernel=args --seed=0 --float-type=f32 --batch=128 --layers=1024,4096 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=MATMUL-LARGE-NAMED
// RUN: mlir-gen --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=1024,4096 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=FC-LARGE
// RUN: mlir-gen --output=named --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=1024,4096 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=FC-LARGE-NAMED
// RUN: mlir-gen --kernel=const --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=1024,1024,1024 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=MLP-LARGE
// RUN: mlir-gen --output=named --kernel=const --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=1024,1024,1024 --tiles=64,64,64 2>&1 | FileCheck %s --check-prefix=MLP-LARGE-NAMED

// Validate that flops are computed correctly
// MATMUL-UNIT: // BENCH_TOTAL_FLOPS: 2
// MATMUL-UNIT-NAMED: // BENCH_TOTAL_FLOPS: 2
// FC-UNIT: // BENCH_TOTAL_FLOPS: 4
// FC-UNIT-NAMED: // BENCH_TOTAL_FLOPS: 4
// MLP-UNIT: // BENCH_TOTAL_FLOPS: 4
// MLP-UNIT-NAMED: // BENCH_TOTAL_FLOPS: 4

// MATMUL-SMALL: // BENCH_TOTAL_FLOPS: 1024
// MATMUL-SMALL-NAMED: // BENCH_TOTAL_FLOPS: 1024
// FC-SMALL: // BENCH_TOTAL_FLOPS: 1280
// FC-SMALL-NAMED: // BENCH_TOTAL_FLOPS: 1280
// MLP-SMALL: // BENCH_TOTAL_FLOPS: 2944
// MLP-SMALL-NAMED: // BENCH_TOTAL_FLOPS: 2944

// MATMUL-LARGE: // BENCH_TOTAL_FLOPS: 1073741824
// MATMUL-LARGE-NAMED: // BENCH_TOTAL_FLOPS: 1073741824
// FC-LARGE: // BENCH_TOTAL_FLOPS: 1074790400
// FC-LARGE-NAMED: // BENCH_TOTAL_FLOPS: 1074790400
// MLP-LARGE: // BENCH_TOTAL_FLOPS: 537395200
// MLP-LARGE-NAMED: // BENCH_TOTAL_FLOPS: 537395200
