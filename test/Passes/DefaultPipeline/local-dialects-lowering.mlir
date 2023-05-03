// RUN: tpp-opt %s -bufferize -lower-local-dialects -split-input-file | FileCheck %s

func.func @check_dialect() {
 %b = arith.constant dense<[
     [ 1.1, 2.1, 3.1, 4.1 ],
     [ 1.2, 2.2, 3.2, 4.2 ],
     [ 1.3, 2.3, 3.3, 4.3 ],
     [ 1.4, 2.4, 3.4, 4.4 ]
    ]> : tensor<4x4xf32>
 %c =  arith.constant dense<[
     [ 1.1, 2.1, 3.1, 4.1 ],
     [ 1.2, 2.2, 3.2, 4.2 ],
     [ 1.3, 2.3, 3.3, 4.3 ],
     [ 1.4, 2.4, 3.4, 4.35 ]
    ]> : tensor<4x4xf32>

  %threshold = arith.constant 0.1: f32
  check.expect_almost_eq(%b, %c, %threshold):tensor<4x4xf32>, tensor<4x4xf32>, f32
  return
}

// CHECK-LABEL: func.func @check_dialect(
// CHECK-NOT: check.expect_almost_eq
// CHECK: scf.for

// -----

func.func @perf_dialect(%A: tensor<4x8xf32>,
          %B: tensor<8x4xf32>, %C: tensor<4x4xf32>, %n: i64) -> (f64, f64, i64) {
  %size = arith.index_cast %n : i64 to index
  %deltas = memref.alloc(%size) : memref<?xf64>
  %output = arith.constant 0 : i64

  %res = perf.bench (%n, %deltas : i64, memref<?xf64>) iter_args(%output : i64) {
    %sum = arith.addi %n, %n : i64
    perf.yield %sum : i64
  } -> i64

  %mean = perf.mean(%deltas : memref<?xf64>) : f64
  %stdev = perf.stdev(%deltas : memref<?xf64>, %mean : f64) : f64

  memref.dealloc %deltas : memref<?xf64>
  
  return %mean, %stdev, %res : f64, f64, i64
}

// CHECK-LABEL: func.func @perf_dialect(
// CHECK-NOT: perf.bench
// CHECK: scf.for
// CHECK:   {{[\w]*\.?}}call
// CHECK-NOT: perf.mean
// CHECK-NOT: perf.stdev
// CHECK: {{[\w]*\.?}}call
// CHECK: {{[\w]*\.?}}call

// -----

func.func @xsmm_dialect(%arg0: memref<32x256xf32>, %arg1: memref<1x8x32x32xf32>) -> i64 {
  %0 = xsmm.unary.dispatch identity [5, 6, 5, 6] flags = (bcast_row) data_type = f32
  %1 = xsmm.gemm.dispatch [3, 3, 3, 3, 3, 3] flags = (none) data_type = f32
  %2 = arith.addi %0, %1 : i64
  return %2: i64
}

// CHECK-DAG: func.func private
// CHECK-DAG: func.func private
// CHECK-LABEL: func.func @xsmm_dialect(
// CHECK-NOT: xsmm.unary.dispatch
// CHECK-NOT: xsmm.ternary.dispatch
// CHECK-DAG: {{[\w]*\.?}}call
// CHECK-DAG: {{[\w]*\.?}}call
