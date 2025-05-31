// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s

// RUN: tpp-opt -default-tpp-passes %s | FileCheck %s -check-prefix=IR
// RUN: tpp-sched --bundle=default-tpp-passes %s | FileCheck %s -check-prefix=IR

// IR-LABEL: entry
func.func @entry(%A: tensor<48x96xf32>, %B: tensor<96x64xf32>,
                  %C: tensor<48x64xf32>) -> tensor<48x64xf32> {
  // IR: xsmm_gemm_invoke
  %D = linalg.matmul ins(%A, %B: tensor<48x96xf32>, tensor<96x64xf32>) outs(%C: tensor<48x64xf32>) -> tensor<48x64xf32>
  return %D : tensor<48x64xf32>
}

// CHECK-COUNT-48: ( 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97 )
