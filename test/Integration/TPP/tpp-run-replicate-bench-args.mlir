// RUN: tpp-run %s -e entry -entry-point-result=void -n 2 -bench-replication-gb=0.00001 --nano-kernels --registerBlocking=4,4,8 -print-mlir=late 2>&1 | FileCheck %s
// RUN: tpp-run %s -e entry -entry-point-result=void -n 2 -bench-replication-gb=0.00001 --nano-kernels --registerBlocking=4,4,8 -splat-to-random -seed 123 -print-mlir=late 2>&1 | FileCheck %s --check-prefix=RANDOM

func.func @entry(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x4xf32>) outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// One flat i8 global per kernel argument, zero initialized.
// CHECK: memref.global "private" @__bench_replica_0 : memref<{{[0-9]+}}xi8> = dense<0>
// CHECK: memref.global "private" @__bench_replica_1 : memref<{{[0-9]+}}xi8> = dense<0>
// CHECK: memref.global "private" @__bench_replica_2 : memref<{{[0-9]+}}xi8> = dense<0>

// CHECK-LABEL: func.func @entry
// The replication attribute is consumed by the pass.
// CHECK-NOT: tpp.bench_replication_factor

// Pre-bench fill loop writes the constant 1.0 into a typed view of each buffer.
// CHECK: %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[FILL0:.+]] = memref.get_global @__bench_replica_0
// CHECK: %[[FV0:.+]] = memref.view %[[FILL0]]
// CHECK: scf.for
// CHECK:   memref.store %[[ONE]], %[[FV0]]

// Replication loop feeds distinct memref.view slices to the kernel call.
// CHECK: scf.for
// CHECK:   %[[OFF0:.+]] = arith.muli
// CHECK:   %[[V0:.+]] = memref.view %{{.+}}[%[[OFF0]]][] : memref<{{[0-9]+}}xi8> to memref<4x8xf32>
// CHECK:   memref.view %{{.+}}[%{{.+}}][] : memref<{{[0-9]+}}xi8> to memref<8x4xf32>
// CHECK:   memref.view %{{.+}}[%{{.+}}][] : memref<{{[0-9]+}}xi8> to memref<4x4xf32>
// CHECK:   func.call @_entry(%[[V0]]

// With -splat-to-random the fill loop emits a counter-based PRNG instead of 1.0.
// RANDOM: memref.global "private" @__bench_replica_0 : memref<{{[0-9]+}}xi8> = dense<0>
// RANDOM-LABEL: func.func @entry
// RANDOM: scf.for
// RANDOM:   arith.index_cast %{{.+}} : index to i32
// RANDOM:   arith.muli
// RANDOM:   arith.xori
// RANDOM:   arith.bitcast %{{.+}} : i32 to f32
// RANDOM:   memref.store
