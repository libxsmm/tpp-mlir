// RUN: tpp-run %s -e entry -entry-point-result=void -mlir-print-ir-before=bufferize 2>&1 | FileCheck %s --check-prefix=BEFORE
// RUN: tpp-run %s -e entry -entry-point-result=void -mlir-print-ir-after=bufferize 2>&1 | FileCheck %s --check-prefix=AFTER

func.func @entry(%arg0: tensor<128x512xf32>, %arg1: tensor<512x256xf32>, %arg2: tensor<128x256xf32>) 
  -> tensor<128x256xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1: tensor<128x512xf32>, tensor<512x256xf32>)
    outs(%arg2: tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// BEFORE: IR Dump Before Bufferize (bufferize)
// BEFORE-LABEL: @_entry(
// BEFORE: linalg.batch_reduce_matmul{{.*}}tensor<

// AFTER: IR Dump After Bufferize (bufferize)
// AFTER-LABEL: @_entry(
// AFTER: linalg.batch_reduce_matmul{{.*}}memref<
