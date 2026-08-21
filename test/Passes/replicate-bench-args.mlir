// RUN: tpp-opt %s -replicate-bench-args -split-input-file | FileCheck %s
// RUN: tpp-opt %s -replicate-bench-args="random-init=true" -split-input-file | FileCheck %s --check-prefix=RANDOM

// The replication factor comes from the module attribute stamped by the
// benchmark producer. Each kernel argument gets a flat i8 global, the float
// buffers are filled once before the timed region, and the kernel call is
// wrapped in a replication loop feeding distinct memref.view slices.

// CHECK: memref.global "private" @__bench_replica_0 : memref<512xi8> = dense<0>
// CHECK: memref.global "private" @__bench_replica_1 : memref<512xi8> = dense<0>
// CHECK: memref.global "private" @__bench_replica_2 : memref<256xi8> = dense<0>

// CHECK-LABEL: func.func @entry
// CHECK-NOT: tpp.bench_replication_factor

// Pre-bench fill loop writes the constant 1.0 into every float buffer.
// CHECK: %[[FILL0:.+]] = memref.get_global @__bench_replica_0
// CHECK: %[[FILLVIEW0:.+]] = memref.view %[[FILL0]]
// CHECK: scf.for
// CHECK:   %[[ONE:.+]] = arith.constant 1.000000e+00 : f32
// CHECK:   memref.store %[[ONE]], %[[FILLVIEW0]]

// Replication loop wraps the kernel call with memref.view slices.
// CHECK: perf.bench
// CHECK:   %[[G0:.+]] = memref.get_global @__bench_replica_0
// CHECK:   %[[G1:.+]] = memref.get_global @__bench_replica_1
// CHECK:   %[[G2:.+]] = memref.get_global @__bench_replica_2
// CHECK:   %[[UB:.+]] = arith.constant 4 : index
// CHECK:   scf.for %[[IV:.+]] = %{{.+}} to %[[UB]]
// CHECK:     %[[OFF0:.+]] = arith.muli %[[IV]], %{{.+}}
// CHECK:     %[[V0:.+]] = memref.view %[[G0]][%[[OFF0]]][] : memref<512xi8> to memref<4x8xf32>
// CHECK:     %[[V1:.+]] = memref.view %[[G1]]
// CHECK:     %[[V2:.+]] = memref.view %[[G2]]
// CHECK:     func.call @kernel(%[[V0]], %[[V1]], %[[V2]])

// With random-init the fill loop emits a counter-based PRNG instead of 1.0.
// RANDOM-LABEL: func.func @entry
// RANDOM: scf.for
// RANDOM:   arith.index_cast %{{.+}} : index to i32
// RANDOM:   arith.muli
// RANDOM:   arith.xori
// RANDOM:   arith.bitcast
module attributes {tpp.bench_replication_factor = 4 : i64} {
  func.func @kernel(%A: memref<4x8xf32>, %B: memref<8x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>) outs(%C : memref<4x4xf32>)
    return
  }
  func.func @entry(%A: memref<4x8xf32>, %B: memref<8x4xf32>, %C: memref<4x4xf32>, %n: i64) -> f64 {
    %stat = perf.bench (%n : i64) -> f64 {
      func.call @kernel(%A, %B, %C) : (memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) -> ()
      perf.yield
    }
    return %stat : f64
  }
}

// -----

// Without the replication attribute the pass is a no-op: no globals, no
// replication loop, the kernel call stays as-is.

// CHECK-NOT: memref.global
// CHECK-LABEL: func.func @entry
// CHECK: perf.bench
// CHECK:   func.call @kernel(%{{.+}}, %{{.+}}, %{{.+}})
// CHECK-NOT: scf.for
module {
  func.func @kernel(%A: memref<4x8xf32>, %B: memref<8x4xf32>, %C: memref<4x4xf32>) {
    linalg.matmul ins(%A, %B : memref<4x8xf32>, memref<8x4xf32>) outs(%C : memref<4x4xf32>)
    return
  }
  func.func @entry(%A: memref<4x8xf32>, %B: memref<8x4xf32>, %C: memref<4x4xf32>, %n: i64) -> f64 {
    %stat = perf.bench (%n : i64) -> f64 {
      func.call @kernel(%A, %B, %C) : (memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>) -> ()
      perf.yield
    }
    return %stat : f64
  }
}
