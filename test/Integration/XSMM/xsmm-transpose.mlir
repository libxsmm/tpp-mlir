// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @xsmm_transpose(%arg0: memref<4x8xf32>, %arg1: memref<8x4xf32>) {
  %0 = xsmm.unary.dispatch transpose [4, 8, 8, 4] flags = (none) data_type = f32
  xsmm.unary transpose(data_type = f32, %0, %arg0, %arg1) : (i64, memref<4x8xf32>, memref<8x4xf32>) -> ()
  return
}

memref.global "private" constant @__constant_4x8xf32 : memref<4x8xf32> =
  dense<[[1.100000e+00, 2.100000e+00, 3.100000e+00, 4.100000e+00, 5.100000e+00, 6.100000e+00, 7.100000e+00, 8.100000e+00],
         [1.200000e+00, 2.200000e+00, 3.200000e+00, 4.200000e+00, 5.200000e+00, 6.200000e+00, 7.200000e+00, 8.200000e+00],
         [1.300000e+00, 2.300000e+00, 3.300000e+00, 4.300000e+00, 5.300000e+00, 6.300000e+00, 7.300000e+00, 8.300000e+00],
         [1.400000e+00, 2.400000e+00, 3.400000e+00, 4.400000e+00, 5.400000e+00, 6.400000e+00, 7.400000e+00, 8.400000e+00]]>
  {alignment = 64 : i64}

func.func @entry() {
  %0 = memref.get_global @__constant_4x8xf32 : memref<4x8xf32>
  %alloc = memref.alloc() : memref<8x4xf32>

  call @xsmm_transpose(%0, %alloc) :
    (memref<4x8xf32>, memref<8x4xf32>) -> ()

  %cst = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32
  %v0 = vector.transfer_read %alloc[%cst, %cst], %d1 : memref<8x4xf32>, vector<8x4xf32>
  vector.print %v0 : vector<8x4xf32>

  //
  // CHECK: (     ( 1.1, 1.2, 1.3, 1.4 ),
  // CHECK-SAME:  ( 2.1, 2.2, 2.3, 2.4 ),
  // CHECK-SAME:  ( 3.1, 3.2, 3.3, 3.4 ),
  // CHECK-SAME:  ( 4.1, 4.2, 4.3, 4.4 ),
  // CHECK-SAME:  ( 5.1, 5.2, 5.3, 5.4 ),
  // CHECK-SAME:  ( 6.1, 6.2, 6.3, 6.4 ),
  // CHECK-SAME:  ( 7.1, 7.2, 7.3, 7.4 ),
  // CHECK-SAME:  ( 8.1, 8.2, 8.3, 8.4 ) )
  //
  return
}
