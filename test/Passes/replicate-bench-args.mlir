// RUN: tpp-opt %s -replicate-bench-args="replication-factor=2" | FileCheck %s
// RUN: tpp-opt %s -replicate-bench-args="replication-factor=2 random-init=true" | FileCheck %s --check-prefix=RANDOM

// Replicate the kernel arguments of a (bufferized) benchmark wrapper so that
// the timed kernel call iterates over distinct buffers. Each argument is backed
// by a flat, zero-initialized i8 global holding `factor` contiguous copies
// (2 * 4 * 4 * 4 bytes = 128 bytes), and every iteration feeds the kernel a
// `memref.view` into that buffer.

// CHECK: memref.global "private" @__bench_replica_0 : memref<128xi8> = dense<0>
// CHECK: memref.global "private" @__bench_replica_1 : memref<128xi8> = dense<0>
// CHECK: memref.global "private" @__bench_replica_2 : memref<128xi8> = dense<0>

// The kernel signature is preserved (identity-layout contiguous memrefs).
// CHECK-LABEL: func.func @_entry
// CHECK-SAME: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>
func.func @_entry(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
                outs(%c : memref<4x4xf32>)
  return
}

// By default the float buffers are filled once, before perf.bench, with the
// constant 1.0 through a whole-buffer memref.view + scf.for.
// CHECK-LABEL: func.func @entry
// CHECK: memref.view %{{.*}}[%{{.*}}][] : memref<128xi8> to memref<32xf32>
// CHECK: scf.for
// CHECK: arith.constant 1.000000e+00 : f32
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<32xf32>
// CHECK: perf.bench
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: memref.view %{{.*}}[%{{.*}}][] : memref<128xi8> to memref<4x4xf32>
// CHECK: func.call @_entry
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy

// With random-init the float buffers are filled once, before perf.bench, with
// a PRNG value in [1, 2) through a whole-buffer memref.view + scf.for.
// RANDOM-LABEL: func.func @entry
// RANDOM: memref.view %{{.*}}[%{{.*}}][] : memref<128xi8> to memref<32xf32>
// RANDOM: scf.for
// RANDOM: arith.bitcast %{{.*}} : i32 to f32
// RANDOM: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<32xf32>
// RANDOM: perf.bench
func.func @entry() {
  %c10 = arith.constant 10 : i64
  %0 = memref.get_global @g0 : memref<4x4xf32>
  %1 = memref.get_global @g1 : memref<4x4xf32>
  %2 = memref.get_global @g2 : memref<4x4xf32>
  %t = perf.bench(%c10 : i64) -> f64 {
    func.call @_entry(%0, %1, %2)
        : (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>) -> ()
  }
  return
}
memref.global "private" @g0 : memref<4x4xf32> = dense<1.000000e+00>
memref.global "private" @g1 : memref<4x4xf32> = dense<1.000000e+00>
memref.global "private" @g2 : memref<4x4xf32> = dense<1.000000e+00>
