// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s

// RUN: tpp-run %s -linalg-to-xsmm \
// RUN:  -e entry -entry-point-result=void -print | \
// RUN: FileCheck %s

// RUN: tpp-opt %s -default-tpp-passes="linalg-to-xsmm" | \
// RUN: FileCheck %s -check-prefix=IR

// RUN: tpp-opt %s -default-tpp-passes | \
// RUN: FileCheck %s -check-prefix=IR

func.func @entry(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>,
                  %C: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // IR: xsmm_brgemm_invoke
  %D = linalg.matmul ins(%A, %B: tensor<64x64xf32>, tensor<64x64xf32>) outs(%C: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %D : tensor<64x64xf32>
}

// CHECK-COUNT-64: ( 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65 )
