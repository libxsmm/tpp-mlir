// RUN: tpp-run -e entry --entry-point-result=void  -print %s > %t.1
// RUN: tpp-opt %s --loop-invariant-code-motion  --vectorization-pass --loop-invariant-code-motion --hoist-vector-transfer | tpp-run -e entry --entry-point-result=void  -print > %t.2
// RUN: diff -q %t.1 %t.2
// RUN: rm %t.1 %t.2

module {
  memref.global "private" constant @__constant_24x64x64xf32 : memref<24x64x64xf32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @entry(%arg0: memref<8x24x32x64xf32>) -> memref<8x24x32x64xf32> {
    %c1 = arith.constant 1 : index
    %c24 = arith.constant 24 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_24x64x64xf32 : memref<24x64x64xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x24x32x64xf32>
    scf.forall (%arg1, %arg2) in (8, 24) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<32x64xf32, strided<[64, 1], offset: ?>>
      linalg.fill ins(%cst : f32) outs(%subview : memref<32x64xf32, strided<[64, 1], offset: ?>>)
      %subview_0 = memref.subview %arg0[%arg1, 0, 0, 0] [1, 24, 32, 64] [1, 1, 1, 1] : memref<8x24x32x64xf32> to memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>>
      scf.for %arg3 = %c0 to %c32 step %c4 {
        scf.for %arg4 = %c0 to %c64 step %c64 {
          %subview_1 = memref.subview %subview[%arg3, %arg4] [4, 64] [1, 1] : memref<32x64xf32, strided<[64, 1], offset: ?>> to memref<4x64xf32, strided<[64, 1], offset: ?>>
          scf.for %arg5 = %c0 to %c24 step %c1 {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              %subview_2 = memref.subview %subview_0[%arg5, %arg3, %arg6] [1, 4, 1] [1, 1, 1] : memref<24x32x64xf32, strided<[2048, 64, 1], offset: ?>> to memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>
              %subview_3 = memref.subview %0[%arg5, %arg6, %arg4] [1, 1, 64] [1, 1, 1] : memref<24x64x64xf32> to memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>
              linalg.batch_reduce_matmul ins(%subview_2, %subview_3 : memref<1x4x1xf32, strided<[2048, 64, 1], offset: ?>>, memref<1x1x64xf32, strided<[4096, 64, 1], offset: ?>>) outs(%subview_1 : memref<4x64xf32, strided<[64, 1], offset: ?>>)
            }
          }
        }
      }
    }
    return %alloc : memref<8x24x32x64xf32>
  }
}

// -----
