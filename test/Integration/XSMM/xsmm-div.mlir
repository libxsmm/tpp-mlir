// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

func.func @xsmm_div(%arg0: memref<4x8xf32>, %arg1: memref<4x8xf32>, %arg2: memref<4x8xf32>) {
  %0 = xsmm.binary.dispatch div [4, 8, 8, 8, 8] flags = (none) data_type = f32
  xsmm.binary div(data_type = f32, %0, %arg0, %arg1, %arg2)
    : (i64, memref<4x8xf32>, memref<4x8xf32>, memref<4x8xf32>) -> ()
  return
}

func.func @xsmm_div_col_bcast(%arg0: memref<4x8xf32>, %arg1: memref<1x8xf32>, %arg2: memref<4x8xf32>) {
  %0 = xsmm.binary.dispatch div [4, 8, 8, 8, 8] flags = (bcast_col_in1) data_type = f32
  xsmm.binary div(data_type = f32, %0, %arg0, %arg1, %arg2)
    : (i64, memref<4x8xf32>, memref<1x8xf32>, memref<4x8xf32>) -> ()
  return
}

func.func @xsmm_div_row_bcast(%arg0: memref<4x8xf32>, %arg1: memref<4x1xf32>, %arg2: memref<4x8xf32>) {
  %0 = xsmm.binary.dispatch div [4, 8, 8, 1, 8] flags = (bcast_row_in1) data_type = f32
  xsmm.binary div(data_type = f32, %0, %arg0, %arg1, %arg2)
    : (i64, memref<4x8xf32>, memref<4x1xf32>, memref<4x8xf32>) -> ()
  return
}

func.func @xsmm_div_scalar_bcast(%arg0: memref<4x8xf32>, %arg1: memref<1xf32>, %arg2: memref<4x8xf32>) {
  %0 = xsmm.binary.dispatch div [4, 8, 8, 1, 8] flags = (bcast_scalar_in1) data_type = f32
  xsmm.binary div(data_type = f32, %0, %arg0, %arg1, %arg2)
    : (i64, memref<4x8xf32>, memref<1xf32>, memref<4x8xf32>) -> ()
  return
}

memref.global "private" constant @__constant_4x8xf32 : memref<4x8xf32> =
  dense<[[2.000000e+00, 4.000000e+00, 6.000000e+00, 8.000000e+00, 10.000000e+00, 12.000000e+00, 14.000000e+00, 16.000000e+00],
         [18.000000e+00, 20.000000e+00, 22.000000e+00, 24.000000e+00, 26.000000e+00, 28.000000e+00, 30.000000e+00, 32.000000e+00],
         [34.000000e+00, 36.000000e+00, 38.000000e+00, 40.000000e+00, 42.000000e+00, 44.000000e+00, 46.000000e+00, 48.000000e+00],
         [50.000000e+00, 52.000000e+00, 54.000000e+00, 56.000000e+00, 58.000000e+00, 60.000000e+00, 62.000000e+00, 64.000000e+00]]>
  {alignment = 64 : i64}

memref.global "private" constant @__constant_1_4x8xf32 : memref<4x8xf32> =
  dense<[[2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00],
         [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00],
         [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00],
         [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]>
  {alignment = 64 : i64}

memref.global "private" constant @__constant_1x8xf32 : memref<1x8xf32> =
  dense<[[2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00]]>
  {alignment = 64 : i64}

memref.global "private" constant @__constant_4x1xf32 : memref<4x1xf32> =
  dense<[[2.000000e+00],
         [2.000000e+00],
         [2.000000e+00],
         [2.000000e+00]]>
  {alignment = 64 : i64}

memref.global "private" constant @__constant_1xf32 : memref<1xf32> =
  dense<[2.000000e+00]>
  {alignment = 64 : i64}



func.func @entry() {
  %0 = memref.get_global @__constant_4x8xf32 : memref<4x8xf32>
  %1 = memref.get_global @__constant_1_4x8xf32 : memref<4x8xf32>
  %alloc = memref.alloc() : memref<4x8xf32>

  %cst = arith.constant 0 : index
  %d1 = arith.constant -1.0 : f32

  call @xsmm_div(%0, %1, %alloc) :
    (memref<4x8xf32>, memref<4x8xf32>, memref<4x8xf32>) -> ()

  %v0 = vector.transfer_read %alloc[%cst, %cst], %d1 : memref<4x8xf32>, vector<4x8xf32>
  vector.print %v0 : vector<4x8xf32>

  //
  // CHECK: (     ( 1, 2, 3, 4, 5, 6, 7, 8 )
  // CHECK-SAME:  ( 9, 10, 11, 12, 13, 14, 15, 16 )
  // CHECK-SAME:  ( 17, 18, 19, 20, 21, 22, 23, 24 )
  // CHECK-SAME:  ( 25, 26, 27, 28, 29, 30, 31, 32 ) )
  //

  %2 = memref.get_global @__constant_1x8xf32 : memref<1x8xf32>
  call @xsmm_div_col_bcast(%0, %2, %alloc) :
    (memref<4x8xf32>, memref<1x8xf32>, memref<4x8xf32>) -> ()

  %v1 = vector.transfer_read %alloc[%cst, %cst], %d1 : memref<4x8xf32>, vector<4x8xf32>
  vector.print %v1 : vector<4x8xf32>

  //
  // CHECK: (     ( 1, 2, 3, 4, 5, 6, 7, 8 )
  // CHECK-SAME:  ( 9, 10, 11, 12, 13, 14, 15, 16 )
  // CHECK-SAME:  ( 17, 18, 19, 20, 21, 22, 23, 24 )
  // CHECK-SAME:  ( 25, 26, 27, 28, 29, 30, 31, 32 ) )
  //

  %3 = memref.get_global @__constant_4x1xf32 : memref<4x1xf32>
  call @xsmm_div_row_bcast(%0, %3, %alloc) :
    (memref<4x8xf32>, memref<4x1xf32>, memref<4x8xf32>) -> ()

  %v2 = vector.transfer_read %alloc[%cst, %cst], %d1 : memref<4x8xf32>, vector<4x8xf32>
  vector.print %v2 : vector<4x8xf32>

  //
  // CHECK: (     ( 1, 2, 3, 4, 5, 6, 7, 8 )
  // CHECK-SAME:  ( 9, 10, 11, 12, 13, 14, 15, 16 )
  // CHECK-SAME:  ( 17, 18, 19, 20, 21, 22, 23, 24 )
  // CHECK-SAME:  ( 25, 26, 27, 28, 29, 30, 31, 32 ) )
  //

  %4 = memref.get_global @__constant_1xf32 : memref<1xf32>
  call @xsmm_div_scalar_bcast(%0, %4, %alloc) :
    (memref<4x8xf32>, memref<1xf32>, memref<4x8xf32>) -> ()

  //
  // CHECK: (     ( 1, 2, 3, 4, 5, 6, 7, 8 )
  // CHECK-SAME:  ( 9, 10, 11, 12, 13, 14, 15, 16 )
  // CHECK-SAME:  ( 17, 18, 19, 20, 21, 22, 23, 24 )
  // CHECK-SAME:  ( 25, 26, 27, 28, 29, 30, 31, 32 ) )
  //

  %v3 = vector.transfer_read %alloc[%cst, %cst], %d1 : memref<4x8xf32>, vector<4x8xf32>
  vector.print %v3 : vector<4x8xf32>

  return
}
