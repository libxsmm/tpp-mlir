// RUN: tpp-opt %s -replicate-bench-args="replication-factor=2" | FileCheck %s

// Replicate the kernel arguments of a (bufferized) benchmark wrapper so that
// the timed kernel call iterates over distinct buffers. In the current
// lowering, each argument gets a stack-like temporary `memref<128xi8>`
// allocation populated with two `memref.copy` operations (for factor=2), and
// each benchmark iteration feeds `_entry` with per-iteration `memref.view`s.

// The kernel signature is preserved (identity-layout contiguous memrefs).
// CHECK-LABEL: func.func @_entry
// CHECK-SAME: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>, %{{.*}}: memref<4x4xf32>
func.func @_entry(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>)
                outs(%c : memref<4x4xf32>)
  return
}

// By default the argument buffers are initialized by copying from globals into
// each per-replica view before entering perf.bench.
// CHECK-LABEL: func.func @entry
// CHECK: %[[A0:.*]] = memref.alloc() {alignment = 128 : i64} : memref<128xi8>
// CHECK: %[[A1:.*]] = memref.alloc() {alignment = 128 : i64} : memref<128xi8>
// CHECK: %[[A2:.*]] = memref.alloc() {alignment = 128 : i64} : memref<128xi8>
// CHECK: scf.for
// CHECK: memref.view %[[A0]][%{{.*}}][] : memref<128xi8> to memref<4x4xf32>
// CHECK: memref.copy
// CHECK: scf.for
// CHECK: memref.view %[[A1]][%{{.*}}][] : memref<128xi8> to memref<4x4xf32>
// CHECK: memref.copy
// CHECK: scf.for
// CHECK: memref.view %[[A2]][%{{.*}}][] : memref<128xi8> to memref<4x4xf32>
// CHECK: memref.copy
// CHECK: perf.bench
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK: memref.view %[[A0]][%{{.*}}][] : memref<128xi8> to memref<4x4xf32>
// CHECK: memref.view %[[A1]][%{{.*}}][] : memref<128xi8> to memref<4x4xf32>
// CHECK: memref.view %[[A2]][%{{.*}}][] : memref<128xi8> to memref<4x4xf32>
// CHECK: func.call @_entry
// CHECK: memref.dealloc %[[A0]] : memref<128xi8>
// CHECK: memref.dealloc %[[A1]] : memref<128xi8>
// CHECK: memref.dealloc %[[A2]] : memref<128xi8>

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
