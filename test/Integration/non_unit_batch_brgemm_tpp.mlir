// RUN: standalone-opt %s -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-loops -convert-linalg-to-loops -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// RUN: standalone-opt %s -map-linalg-to-tpp -pre-bufferization -one-shot-bufferize="bufferize-function-boundaries allow-return-allocs function-boundary-type-conversion=identity-layout-map"  -canonicalize -drop-equivalent-buffer-results -finalizing-bufferize -convert-linalg-to-tpp -convert-tpp-to-xsmm -convert-xsmm-to-func -convert-vector-to-scf -convert-scf-to-cf -convert-vector-to-llvm -convert-func-to-llvm -convert-memref-to-llvm -canonicalize -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN: -shared-libs=%llvmlirdir/libmlir_c_runner_utils%shlibext,%standalonelibdir/libstandalone_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//

// TODO when batch > 1 we don't get the same result from xsmm.
// XFAIL: *

module {

  func.func @brgemmtpp(%A: tensor<2x4x8xf32>,
                     %B: tensor<2x8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {llvm.emit_c_interface} {
    %D = linalg.reduce_batch_matmul ins(%A, %B: tensor<2x4x8xf32>, tensor<2x8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
    return %D: tensor<4x4xf32>
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %d1 = arith.constant -1.0 : f32
    
    // Initialize various matrices, dense for stress testing,
    // and sparse to verify correct nonzero structure.
    %da = arith.constant dense<[[
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ]
      ],
      [
        [ 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1 ],
        [ 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2 ],
        [ 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3 ],
        [ 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4 ]
      ]]> : tensor<2x4x8xf32>
    %db = arith.constant dense<[[
        [ 10.1, 11.1, 12.1, 13.1 ],
        [ 10.2, 11.2, 12.2, 13.2 ],
        [ 10.3, 11.3, 12.3, 13.3 ],
        [ 10.4, 11.4, 12.4, 13.4 ],
        [ 10.5, 11.5, 12.5, 13.5 ],
        [ 10.6, 11.6, 12.6, 13.6 ],
        [ 10.7, 11.7, 12.7, 13.7 ],
        [ 10.8, 11.8, 12.8, 13.8 ]
    ],
    [
        [ 10.1, 11.1, 12.1, 13.1 ],
        [ 10.2, 11.2, 12.2, 13.2 ],
        [ 10.3, 11.3, 12.3, 13.3 ],
        [ 10.4, 11.4, 12.4, 13.4 ],
        [ 10.5, 11.5, 12.5, 13.5 ],
        [ 10.6, 11.6, 12.6, 13.6 ],
        [ 10.7, 11.7, 12.7, 13.7 ],
        [ 10.8, 11.8, 12.8, 13.8 ]
    ]]> : tensor<2x8x4xf32>

    // Call kernel.
    %C = arith.constant dense<0.0> : tensor<4x4xf32>
    %0 = call @brgemmtpp(%da, %db, %C)
      : (tensor<2x4x8xf32>, tensor<2x8x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  
    //
    // CHECK:    ( ( 777.52, 851.12, 924.72, 998.32 ),
    // CHECK-SAME: ( 794.24, 869.44, 944.64, 1019.84 ),
    // CHECK-SAME: ( 810.96, 887.76, 964.56, 1041.36 ),
    // CHECK-SAME: ( 827.68, 906.08, 984.48, 1062.88 ) )
    //
    %m0 = bufferization.to_memref %0 : memref<4x4xf32>
    %v0 = vector.transfer_read %m0[%c0, %c0], %d1 : memref<4x4xf32>, vector<4x4xf32>
    vector.print %v0 : vector<4x4xf32>

    return 
  }
}
