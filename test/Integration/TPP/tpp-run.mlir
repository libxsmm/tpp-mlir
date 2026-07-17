// Print mlir
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early 2>&1 | FileCheck %s --check-prefix=EARLY
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=mid 2>&1 | FileCheck %s --check-prefix=MID
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=late  2>&1 | FileCheck %s --check-prefix=LATE
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=llvm  2>&1 | FileCheck %s --check-prefix=LLVM

// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=late -linalg-to-loops 2>&1 | FileCheck %s --check-prefix=LOOPS

// Benchmark options
// RUN: tpp-run %s -e entry -entry-point-result=void -print -print-mlir=early 2>&1 | FileCheck %s --check-prefix=BENCH_PRINT
// RUN: tpp-run %s -e entry -entry-point-result=void -n 2 -print-mlir=late 2>&1 | FileCheck %s --check-prefix=BENCH_STATS_2
// RUN: tpp-run %s -e entry -entry-point-result=void -n 10 -print-mlir=late 2>&1 | FileCheck %s --check-prefix=BENCH_STATS_10

// CPU target-feature propagation test
// FIXME: This test fails on Arm, as the message is not printed
// FIXME: tpp-run %s -e entry -entry-point-result=void -print-mlir=llvm --target-feature=testfeature 2>&1 | FileCheck %s --check-prefix=CPU_FEATURE

// CPU options can't be tested as even the LLVM IR is identical
// Splat and init options in tpp-run-splat-* tests

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> ()>
#map4 = affine_map<(d0, d1) -> (d0, d1)>

func.func @entry(%A: tensor<4x8xf32>, %B: tensor<8x4xf32>,
                 %C: tensor<4x4xf32>, %arg0: tensor<f32>) -> tensor<4x4xf32> {
  // Weight is defined locally as a dense
  %D = linalg.generic {indexing_maps = [#map0, #map1, #map2],
                       iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%A, %B: tensor<4x8xf32>, tensor<8x4xf32>) outs(%C: tensor<4x4xf32>) {
              ^bb0(%a: f32, %b: f32, %c: f32):
                %0 = arith.mulf %a, %b : f32
                %1 = arith.addf %c, %0 : f32
                linalg.yield %1 : f32
       } -> tensor<4x4xf32>

  // Generic unmapped to TPP
  %res = linalg.generic {
    indexing_maps = [#map3, #map4],
    iterator_types = ["parallel", "parallel"]} ins(%arg0: tensor<f32>)
    outs(%D: tensor<4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  } -> tensor<4x4xf32>

  return %res : tensor<4x4xf32>
}

// EARLY-DAG: memref.global "private" @__wrapper{{.*}} : memref<f32> = dense<1.000000e+00> {alignment = 128 : i64}
// EARLY-DAG: memref.global "private" @__wrapper{{.*}} : memref<4x4xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// EARLY-DAG: memref.global "private" @__wrapper{{.*}} : memref<4x8xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// EARLY-DAG: memref.global "private" @__wrapper{{.*}} : memref<8x4xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// EARLY-LABEL: @_entry
// EARLY: linalg.generic
// EARLY:   arith.mulf
// EARLY:   arith.addf
// EARLY: linalg.generic
// EARLY:   arith.addf
// EARLY-LABEL: @entry
// EARLY: memref.get_global @__wrapper{{.*}}
// EARLY: memref.get_global @__wrapper{{.*}}
// EARLY: memref.get_global @__wrapper{{.*}}
// EARLY: memref.get_global @__wrapper{{.*}}
// EARLY: call @_entry({{.*}}) : (tensor<4x8xf32>, tensor<8x4xf32>, tensor<4x4xf32>, tensor<f32>)


// MID-DAG: memref.global "private" @__wrapper{{.*}} : memref<f32> = dense<1.000000e+00> {alignment = 128 : i64}
// MID-DAG: memref.global "private" @__wrapper{{.*}} : memref<4x4xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// MID-DAG: memref.global "private" @__wrapper{{.*}} : memref<4x8xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// MID-DAG: memref.global "private" @__wrapper{{.*}} : memref<8x4xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// MID-LABEL: @_entry
// MID: call @xsmm_gemm_dispatch
// MID: call @xsmm_gemm_invoke
// MID: linalg.generic
// MID:   arith.addf
// MID-DAG: @xsmm_gemm_invoke
// MID-DAG: @xsmm_gemm_dispatch
// MID-LABEL: @entry
// MID-DAG: memref.get_global @__wrapper{{.*}} : memref<4x8xf32>
// MID-DAG: memref.get_global @__wrapper{{.*}} : memref<4x4xf32>
// MID-DAG: memref.get_global @__wrapper{{.*}} : memref<f32>
// MID: call @_entry({{.*}}) : (memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>, memref<f32>)


// LATE-DAG: memref.global "private" @__wrapper{{.*}} : memref<f32> = dense<1.000000e+00> {alignment = 128 : i64}
// LATE-DAG: memref.global "private" @__wrapper{{.*}} : memref<4x4xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// LATE-DAG: memref.global "private" @__wrapper{{.*}} : memref<4x8xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// LATE-DAG: memref.global "private" @__wrapper{{.*}} : memref<8x4xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// LATE-LABEL: @_entry
// LATE:   call @xsmm_gemm_dispatch
// LATE:   call @xsmm_gemm_invoke
// LATE:   scf.for
// LATE:     memref.load
// LATE:     arith.addf
// LATE-DAG: @xsmm_gemm_invoke
// LATE-DAG: @xsmm_gemm_dispatch
// LATE-LABEL: @entry
// LATE-DAG: memref.get_global @__wrapper{{.*}} : memref<4x8xf32>
// LATE-DAG: memref.get_global @__wrapper{{.*}} : memref<4x4xf32>
// LATE-DAG: memref.get_global @__wrapper{{.*}} : memref<8x4xf32>
// LATE-DAG: memref.get_global @__wrapper{{.*}} : memref<f32>
// LATE:   call @_entry({{.*}}) : (memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>, memref<f32>)


// LLVM-DAG: private @__wrapper{{.*}}(1.000000e+00 : f32) {addr_space = 0 : i32, alignment = 128 : i64} : f32
// LLVM-DAG: private @__wrapper{{.*}}(dense<1.000000e+00> : tensor<4x4xf32>) {addr_space = 0 : i32, alignment = 128 : i64} : !llvm.array<4 x array<4 x f32>>
// LLVM-DAG: private @__wrapper{{.*}}(dense<1.000000e+00> : tensor<4x8xf32>) {addr_space = 0 : i32, alignment = 128 : i64} : !llvm.array<4 x array<8 x f32>>
// LLVM-DAG: private @__wrapper{{.*}}(dense<1.000000e+00> : tensor<8x4xf32>) {addr_space = 0 : i32, alignment = 128 : i64} : !llvm.array<8 x array<4 x f32>>
// LLVM-LABEL: @_entry
// LLVM:   llvm.call @xsmm_gemm_dispatch
// LLVM:   llvm.call @xsmm_gemm_invoke
// LLVM:   llvm.cond_br %{{.*}}, [[BODY:.*]], [[LATCH:.*]]
// LLVM:  [[BODY]]:
// LLVM:    llvm.mul
// LLVM:    llvm.add
// LLVM:    llvm.load
// LLVM:    llvm.fadd
// LLVM:  [[LATCH]]:
// LLVM-LABEL: @entry
// LLVM:   llvm.call @_entry

// LOOPS-DAG: memref.global "private" @__wrapper{{.*}} : memref<f32> = dense<1.000000e+00> {alignment = 128 : i64}
// LOOPS-DAG: memref.global "private" @__wrapper{{.*}} : memref<4x4xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// LOOPS-DAG: memref.global "private" @__wrapper{{.*}} : memref<4x8xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// LOOPS-DAG: memref.global "private" @__wrapper{{.*}} : memref<8x4xf32> = dense<1.000000e+00> {alignment = 128 : i64}
// LOOPS-LABEL: @_entry
// LOOPS:   scf.for
// LOOPS:     memref.load
// LOOPS:     memref.load
// LOOPS:     memref.load
// LOOPS:     arith.mulf
// LOOPS:     arith.addf
// LOOPS:     memref.store
// LOOPS:   scf.for
// LOOPS:     memref.load
// LOOPS:     memref.load
// LOOPS:     arith.addf
// LOOPS:     memref.store
// LOOPS-LABEL: @entry
// LOOPS:   call @_entry({{.*}}) : (memref<4x8xf32>, memref<8x4xf32>, memref<4x4xf32>, memref<f32>)

// BENCH_PRINT-LABEL: @entry
// BENCH_PRINT: call @_entry
// BENCH_PRINT: print
// BENCH_PRINT: ( 10, 10, 10, 10 )
// BENCH_PRINT: ( 10, 10, 10, 10 )
// BENCH_PRINT: ( 10, 10, 10, 10 )
// BENCH_PRINT: ( 10, 10, 10, 10 )

// BENCH_STATS_2-LABEL: @entry
// BENCH_STATS_2-DAG: %[[c2:.+]] = arith.constant 2 : index
// BENCH_STATS_2: scf.for{{.*}}to %[[c2]]
// BENCH_STATS_2: call @_entry
// BENCH_STATS_2-NOT: call @_entry
// BENCH_STATS_2: {{[0-9]+}}{{.?}}{{[0-9e-]+}}

// BENCH_STATS_10-LABEL: @entry
// BENCH_STATS_10-DAG: %[[c10:.+]] = arith.constant 10 : index
// BENCH_STATS_10: scf.for{{.*}}to %[[c10]]
// BENCH_STATS_10: call @_entry
// BENCH_STATS_10-NOT: call @_entry
// BENCH_STATS_10: {{[0-9]+}}{{.?}}{{[0-9e-]+}}

// FIXME: This test fails on Arm, as the message is not printed
// CPU_FEATURE-LABLE: @entry
// CPU_FEATURE: '+testfeadture' is not a recognized feature for this target (ignoring feature)
