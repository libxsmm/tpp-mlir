// RUN: mlir-gen --output=named --kernel=args --bias --relu --seed=0 --float-type=f32 --batch=128 --layers=2304,768 2>&1 | FileCheck %s

// CHECK: // RUN{{.*}}tpp-run %s -n {{\d*}}
// CHECK: // RUN{{.*}}-e entry -entry-point-result=void
// CHECK: // BENCH_TOTAL_FLOPS: 453181440
// CHECK:     func.func @entry(
// CHECK-SAME:  %[[ARG0:.+]]: tensor<128x2304xf32>, %[[ARG1:.+]]: tensor<2304x768xf32>,
// CHECK-SAME:  %[[ARG2:.+]]: tensor<768xf32>, %[[ARG3:.+]]: tensor<128x768xf32>) -> tensor<128x768xf32>
// CHECK:     %[[MATMUL:.+]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] :{{.*}}) outs(%[[ARG3]] :{{.*}})
// CHECK:     %[[EMPTY_BIAS:.+]] = tensor.empty() : tensor<128x768xf32>
// CHECK:     %[[BROADCAST:.+]] = linalg.broadcast ins(%[[ARG2]] :{{.*}}) outs(%[[EMPTY_BIAS]] :{{.*}})
// CHECK:     %[[ADD:.+]] = linalg.add ins(%[[BROADCAST]], %[[MATMUL]] :{{.*}}) outs(%[[EMPTY_BIAS]] :{{.*}})
// CHECK:     %[[C0:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:     %[[EMPTY_RELU:.+]] = tensor.empty() : tensor<128x768xf32>
// CHECK:     %[[FILL:.+]] = linalg.fill ins(%[[C0]] :{{.*}}) outs(%[[EMPTY_RELU]] :{{.*}})
// CHECK:     %[[MAX:.+]] = linalg.max ins(%[[ADD]], %[[FILL]] :{{.*}}) outs(%[[EMPTY_RELU]] :{{.*}})
// CHECK:     return %[[MAX]]
