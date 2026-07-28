// RUN: tpp-run -e optimal_register_packing --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-run -e optimal_register_packing --entry-point-result=void  --nano-kernels --registerBlocking=6,64,1 --gemm-unroll=1,32,1 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.2
// RUN: tpp-run -e optimal_register_packing --entry-point-result=void  --nano-kernels --registerBlocking=3,32,1 --gemm-unroll=1,16,1 -print  --splat-to-random --init-type normal  -seed 123  %s > %t.3
// RUN: fpcmp -r 0.001 %t.1 %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.3


func.func @optimal_register_packing(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) -> memref<24x64xf32> {
    linalg.batch_reduce_matmul ins(%arg0, %arg1 : memref<32x24x32xf32>, memref<32x32x64xf32>) outs(%arg2 : memref<24x64xf32>)
  return %arg2 : memref<24x64xf32>
}
