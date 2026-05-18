// RUN: tpp-opt %s -convert-perf-to-loops -canonicalize -split-input-file | FileCheck %s

// CHECK-LABEL: @perf_single_op
func.func @perf_single_op(%a: i32, %b: i32, %n: i64) -> f64 {
  // CHECK-DAG: %[[lb:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[step:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
  // CHECK: %[[ub:.*]] = arith.index_cast %arg2 : i64 to index
  // CHECK: %[[acc:.*]] = memref.alloca() : memref<f64>
  // CHECK: memref.store %[[zero]], %[[acc]][] : memref<f64>
  // CHECK: scf.for %{{.*}} = %[[lb]] to %[[ub]] step %[[step]] {
  // CHECK:   %[[timer:.*]] = perf.start_timer : !perf.timer
  // CHECK:   arith.addi
  // CHECK:   perf.sink
  // CHECK:   %[[delta:.*]] = perf.stop_timer(%[[timer]] : !perf.timer) : f64
  // CHECK:   %[[curr:.*]] = memref.load %[[acc]][] : memref<f64>
  // CHECK:   %[[new:.*]] = arith.addf %[[curr]], %[[delta]] : f64
  // CHECK:   memref.store %[[new]], %[[acc]][] : memref<f64>
  // CHECK: }
  // CHECK: %[[stat:.*]] = memref.load %[[acc]][] : memref<f64>
  // CHECK: return %[[stat]]
  %stat = perf.bench (%n : i64) -> f64 {
    %c = arith.addi %a, %b : i32
    perf.sink(%c) : i32
    perf.yield
  }
  return %stat : f64
}
