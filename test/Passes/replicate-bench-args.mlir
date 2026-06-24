// RUN: tpp-opt %s -replicate-bench-args="replication-factor=2" | FileCheck %s

// Replicate the kernel arguments of a (bufferized) benchmark wrapper so that
// the timed kernel call iterates over distinct, value-initialized buffers.

// CHECK: memref.global "private" @__bench_replica_0 : memref<2x4x4xf32> = dense<1.000000e+00>
// CHECK: memref.global "private" @__bench_replica_1 : memref<2x4x4xf32> = dense<1.000000e+00>
// CHECK: memref.global "private" @__bench_replica_2 : memref<2x4x4xf32> = dense<1.000000e+00>

// The kernel signature is relaxed to accept strided replica slices.
// CHECK-LABEL: func.func @_entry
// CHECK-SAME: memref<4x4xf32, strided<[4, 1], offset: ?>>
func.func @_entry(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
                outs(%c : memref<4x4xf32>)
  return
}

// CHECK-LABEL: func.func @entry
// CHECK: perf.bench
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: memref.subview
// CHECK: func.call @_entry
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.copy
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
