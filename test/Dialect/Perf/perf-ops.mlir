// RUN: tpp-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: @perf_timer
func.func @perf_timer(%a: i32, %b: i32, %n: i64) -> f64 {
  // CHECK: perf.start_timer
  %t = perf.start_timer : !perf.timer
  // CHECK: perf.stop_timer
  %s = perf.stop_timer(%t : !perf.timer) : f64

  return %s : f64
}

// -----

/// CHECK-LABEL: @perf_matmul_bench
func.func @perf_matmul_bench(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: i64) -> f64 {
  // CHECK: perf.bench
  %stat = perf.bench (%n : i64) -> f64 {
    // CHECK: linalg.matmul
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: perf.sink
    perf.sink(%D) : tensor<4x4xf32>
    perf.yield
  }

  return %stat : f64
}

// -----

// CHECK-LABEL: @perf_matmul_loops
func.func @perf_matmul_loops(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: index) -> f64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: perf.start_timer
  %t = perf.start_timer : !perf.timer
  scf.for %arg0 = %c0 to %n step %c1 {
    // CHECK: linalg.matmul
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    // CHECK: perf.sink
    perf.sink(%D) : tensor<4x4xf32>
  }
  // CHECK: perf.stop_timer
  %del = perf.stop_timer(%t : !perf.timer) : f64

  return %del : f64
}

// -----

// CHECK-LABEL: @perf_yield_empty
func.func @perf_yield_empty(%a: i32, %b: i32, %n: i64) -> f64 {
  // CHECK: perf.bench
  %stat = perf.bench (%n : i64) -> f64 {
    // CHECK: arith.addi
    %c = arith.addi %a, %b : i32
    // CHECK: perf.sink
    perf.sink(%c) : i32
    perf.yield
  }
  return %stat : f64
}

// -----

// CHECK-LABEL: @perf_yield_result
func.func @perf_yield_result(%a: i32, %b: i32, %n: i64) -> i32 {
  %bench_res = arith.constant 0 : i32

  // CHECK: %[[out:[0-9]+]]:2 = perf.bench
  %stat, %out = perf.bench (%n : i64) iter_args(%arg0 = %bench_res) -> (f64, i32) {
    // CHECK: %[[c:.*]] = arith.addi
    %c = arith.addi %a, %b : i32
    // CHECK: perf.yield %[[c]]
    perf.yield %c : i32
  }
  // Unused iter_args, %out = %bench_res

  // CHECK: return %[[out]]#1
  return %out : i32
}

// -----

// An example of perf dialect usage.
// CHECK-LABEL: @perf_example
func.func @perf_example(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: i64) -> (f64, i64) {
  %output = arith.constant 0 : i64

  // CHECK: %[[res:[0-9]+]]:2 = perf.bench
  %stat, %res = perf.bench (%n : i64) iter_args(%arg0 = %output) -> (f64, i64) {
    // CHECK: %[[mulres:.*]] = linalg.matmul
    // CHECK: perf.sink(%[[mulres]])
    %D = linalg.matmul ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>)
                       outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    perf.sink(%D) : tensor<4x4xf32>

    // CHECK: %[[sum:.*]] = arith.addi
    %sum = arith.addi %arg0, %n : i64

    // CHECK: perf.yield %[[sum]]
    perf.yield %sum : i64
  }
  // Used iter_args, %res = %output + sum(n)

  // CHECK: return %[[res]]#0, %[[res]]#1
  return %stat, %res : f64, i64
}
