//RUN: tpp-opt -verify-xsmm-calls  --combine-xsmm-op-optimization  -verify-xsmm-calls %s --split-input-file | FileCheck %s

memref.global "private" constant @__constant_4x32x32xf32 : memref<4x32x32xf32> = dense<1.000000e+00> alignment = 128
memref.global "private" constant @__constant_8x32x32xf32 :  memref<8x32x32xf32> = dense<1.000000e+00> alignment = 128
memref.global "private" constant @__constant_32xf32:  memref<32xf32, strided<[32], offset:?>> = dense<1.000000e+00> alignment = 128

func.func @bcast_col_in0_on_binary_add(%arg0: memref<256x128xf32>) -> memref<256x512xf32>  {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_4x32x32xf32 : memref<4x32x32xf32>
  %1 = memref.get_global @__constant_8x32x32xf32 : memref<8x32x32xf32>
  %2 = memref.get_global @__constant_32xf32 : memref<32xf32, strided<[32], offset:?>>
  %alloc = memref.alloc() alignment = 64 : memref<8x4x32x32xf32>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = f32
  %alloc_0 = memref.alloc() alignment = 64 : memref<8x8x32x32xf32>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0) data_type = f32
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in0) data_type = f32
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
  %alloc_1 = memref.alloc() alignment = 64 : memref<256x512xf32>
  scf.parallel (%arg3, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
	  %subview = memref.subview %alloc_0[%arg3, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
	  %subview_2 = memref.subview %alloc[%arg3, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xf32> to memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>
	  xsmm.brgemm(data_type = f32, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<4x32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
	  xsmm.binary add(data_type = f32, %5, %2, %subview, %subview) : (i64, memref<32xf32, strided<[32], offset:?>>, memref<32x32xf32, strided<[32,1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
	  xsmm.unary relu(data_type = f32, %6, %subview, %subview) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
	  scf.reduce
  }
  return %alloc_1 : memref<256x512xf32>
 }

// CHECK-LABEL: func.func @bcast_col_in0_on_binary_add(
// CHECK: %[[ARG0:.*]]: memref<256x128xf32>) -> memref<256x512xf32> {
// CHECK: %[[BIAS:.*]] = memref.get_global @__constant_32xf32 : memref<32xf32, strided<[32], offset: ?>>
// CHECK-NOT: xsmm.brgemm.dispatch
// CHECK-NOT: xsmm.unary.dispatch
// CHECK-NOT: xsmm.binary.dispatch
// CHECK: %[[DISPATCH:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (beta_0)  binary_flags = (bcast_col_in0)  unary_flags = (none) data_type = f32
// CHECK-NOT: xsmm.brgemm(
// CHECK-NOT: xsmm.binary add
// CHECK-NOT: xsmm.unary relu
// CHECK: xsmm.fused_brgemm(data_type = f32, %[[DISPATCH]], %{{.*}}, %{{.*}}, %{{.*}}, %[[BIAS]], %{{.*}})


// -----

memref.global "private" constant @__constant_4x32x32xf32 : memref<4x32x32xf32> = dense<1.000000e+00> alignment = 128
memref.global "private" constant @__constant_8x32x32xf32 :  memref<8x32x32xf32> = dense<1.000000e+00> alignment = 128
memref.global "private" constant @__constant_32xf32:  memref<32xf32, strided<[32], offset:?>> = dense<1.000000e+00> alignment = 128

func.func @bcast_col_in1_on_binary_add(%arg0: memref<256x128xf32>) -> memref<256x512xf32>  {
 %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_4x32x32xf32 : memref<4x32x32xf32>
  %1 = memref.get_global @__constant_8x32x32xf32 : memref<8x32x32xf32>
  %2 = memref.get_global @__constant_32xf32 : memref<32xf32, strided<[32], offset:?>>
  %alloc = memref.alloc() alignment = 64 : memref<8x4x32x32xf32>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = f32
  %alloc_0 = memref.alloc() alignment = 64 : memref<8x8x32x32xf32>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0) data_type = f32
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in1) data_type = f32
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
  %alloc_1 = memref.alloc() alignment = 64 : memref<256x512xf32>
  scf.parallel (%arg3, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
	  %subview = memref.subview %alloc_0[%arg3, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
	  %subview_2 = memref.subview %alloc[%arg3, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xf32> to memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>
	  xsmm.brgemm(data_type = f32, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<4x32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
	  xsmm.binary add(data_type = f32, %5,  %subview, %2, %subview) : (i64, memref<32x32xf32, strided<[32,1], offset: ?>>,  memref<32xf32, strided<[32], offset:?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
	  xsmm.unary relu(data_type = f32, %6, %subview, %subview) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
	  scf.reduce
  }
  return %alloc_1 : memref<256x512xf32>
 }

// CHECK-LABEL: func.func @bcast_col_in1_on_binary_add(
// CHECK: %[[ARG0:.*]]: memref<256x128xf32>) -> memref<256x512xf32> {
// CHECK: %[[BIAS:.*]] = memref.get_global @__constant_32xf32 : memref<32xf32, strided<[32], offset: ?>>
// CHECK-NOT: %[[DISPATCH:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (beta_0)  binary_flags = (bcast_col_in1)  unary_flags = (none) data_type = f32
// CHECK-NOT: xsmm.fused_brgemm(data_type = f32, %[[DISPATCH]], %{{.*}}, %{{.*}}, %{{.*}}, %[[BIAS]], %{{.*}})

// -----

memref.global "private" constant @__constant_4x32x32xf32 : memref<4x32x32xf32> = dense<1.000000e+00> alignment = 128
memref.global "private" constant @__constant_8x32x32xf32 :  memref<8x32x32xf32> = dense<1.000000e+00> alignment = 128
memref.global "private" constant @__constant_32x32xf32:  memref<32x32xf32> = dense<1.000000e+00> alignment = 128

func.func @none_on_binary_add(%arg0: memref<256x128xf32>) -> memref<256x512xf32>  {
 %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c4_i64 = arith.constant 4 : i64
  %c8_i64 = arith.constant 8 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_4x32x32xf32 : memref<4x32x32xf32>
  %1 = memref.get_global @__constant_8x32x32xf32 : memref<8x32x32xf32>
  %2 = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
  %alloc = memref.alloc() alignment = 64 : memref<8x4x32x32xf32>
  %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = f32
  %alloc_0 = memref.alloc() alignment = 64 : memref<8x8x32x32xf32>
  %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (beta_0) data_type = f32
  %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (none) data_type = f32
  %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
  %alloc_1 = memref.alloc() alignment = 64 : memref<256x512xf32>
  scf.parallel (%arg3, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
    %subview = memref.subview %alloc_0[%arg3, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    %subview_2 = memref.subview %alloc[%arg3, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xf32> to memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    xsmm.brgemm(data_type = f32, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<4x32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
    xsmm.binary add(data_type = f32, %5, %subview, %2,  %subview) : (i64, memref<32x32xf32, strided<[32,1], offset: ?>>,  memref<32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    xsmm.unary relu(data_type = f32, %6, %subview, %subview) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    scf.reduce
  }
  return %alloc_1 : memref<256x512xf32>
 }

// CHECK-LABEL: func.func @none_on_binary_add(
// CHECK: %[[ARG0:.*]]: memref<256x128xf32>) -> memref<256x512xf32> {
// CHECK: %[[BIAS:.*]] = memref.get_global @__constant_32x32xf32 : memref<32x32xf32>
// CHECK-NOT: %[[DISPATCH:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (beta_0)  binary_flags = (none)  unary_flags = (none) data_type = f32
// CHECK-NOT: xsmm.fused_brgemm(data_type = f32, %[[DISPATCH]], %{{.*}}, %{{.*}}, %{{.*}}, %[[BIAS]], %{{.*}})

// -----

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  memref.global "private" constant @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16> = dense<1.000000e+00> alignment = 128
  memref.global "private" constant @__constant_8x16x32x2xbf16 :  memref<8x16x32x2xbf16> = dense<1.000000e+00> alignment = 128
  memref.global "private" constant @__constant_32xbf16:  memref<32xbf16, strided<[32], offset:?>> = dense<1.000000e+00> alignment = 128

  // Bcast_col_in0 flag set on binary add
  func.func @bcast_col_in0_on_binary_add_bf16(%arg0: memref<256x128xbf16>) -> memref<256x512xbf16>  {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16>
    %1 = memref.get_global @__constant_8x16x32x2xbf16 : memref<8x16x32x2xbf16>
    %2 = memref.get_global @__constant_32xbf16 : memref<32xbf16, strided<[32], offset:?>>
    %alloc = memref.alloc() alignment = 64 : memref<8x4x32x32xbf16>
    %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = bf16
    %alloc_0 = memref.alloc() alignment = 64 : memref<8x8x32x32xbf16>
    %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0) data_type = bf16
    %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in0) data_type = bf16
    %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = bf16
    %alloc_1 = memref.alloc() alignment = 64 : memref<256x512xbf16>
    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xbf16> to memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
      xsmm.brgemm(data_type = bf16, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<4x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
      xsmm.binary add(data_type = bf16, %5, %2, %subview, %subview) : (i64, memref<32xbf16, strided<[32], offset:?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
      xsmm.unary relu(data_type = bf16, %6, %subview, %subview) : (i64, memref<32x32xbf16, strided<[32, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
      scf.reduce
    }
    return %alloc_1 : memref<256x512xbf16>
  }
}

// CHECK-LABEL: func.func @bcast_col_in0_on_binary_add_bf16(
// CHECK: %[[ARG0:.*]]: memref<256x128xbf16>) -> memref<256x512xbf16> {
// CHECK: %[[BIAS:.*]] = memref.get_global @__constant_32xbf16 : memref<32xbf16, strided<[32], offset: ?>>
// CHECK: %[[DISPATCH:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (vnni_b, beta_0)  binary_flags = (bcast_col_in0)  unary_flags = (none) data_type = bf16
// CHECK: xsmm.fused_brgemm(data_type = bf16, %[[DISPATCH]], %{{.*}}, %{{.*}}, %{{.*}}, %[[BIAS]], %{{.*}})

// -----

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  memref.global "private" constant @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16> = dense<1.000000e+00> alignment = 128
  memref.global "private" constant @__constant_8x16x32x2xbf16 :  memref<8x16x32x2xbf16> = dense<1.000000e+00> alignment = 128
  memref.global "private" constant @__constant_32xbf16:  memref<32xbf16, strided<[32], offset:?>> = dense<1.000000e+00> alignment = 128

  // Bcast_col_in1 flag set on binary add
  func.func @bcast_col_in1_on_binary_add_bf16(%arg0: memref<256x128xbf16>) -> memref<256x512xbf16>  {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16>
    %1 = memref.get_global @__constant_8x16x32x2xbf16 : memref<8x16x32x2xbf16>
    %2 = memref.get_global @__constant_32xbf16 : memref<32xbf16, strided<[32], offset:?>>
    %alloc = memref.alloc() alignment = 64 : memref<8x4x32x32xbf16>
    %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = bf16
    %alloc_0 = memref.alloc() alignment = 64 : memref<8x8x32x32xbf16>
    %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0) data_type = bf16
    %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in1) data_type = bf16
    %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = bf16
    %alloc_1 = memref.alloc() alignment = 64 : memref<256x512xbf16>
    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xbf16> to memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
      xsmm.brgemm(data_type = bf16, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<4x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
      xsmm.binary add(data_type = bf16, %5, %subview, %2,  %subview) : (i64 , memref<32x32xbf16, strided<[32, 1], offset: ?>>,memref<32xbf16, strided<[32], offset:?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
      xsmm.unary relu(data_type = bf16, %6, %subview, %subview) : (i64, memref<32x32xbf16, strided<[32, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
      scf.reduce
    }
    return %alloc_1 : memref<256x512xbf16>
  }
}

// CHECK-LABEL: func.func @bcast_col_in1_on_binary_add_bf16(
// CHECK: %[[ARG0:.*]]: memref<256x128xbf16>) -> memref<256x512xbf16> {
// CHECK: %[[BIAS:.*]] = memref.get_global @__constant_32xbf16 : memref<32xbf16, strided<[32], offset: ?>>
// CHECK-NOT: %[[DISPATCH:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (vnni_b, beta_0)  binary_flags = (bcast_col_in1)  unary_flags = (none) data_type = bf16
// CHECK-NOT: xsmm.fused_brgemm(data_type = bf16, %[[DISPATCH]] , %{{.*}}, %{{.*}}, %{{.*}}, %[[BIAS]], %{{.*}})


// -----

module attributes {
  "#dlti.sys_spec" = #dlti.target_system_spec<"CPU"
    = #dlti.target_device_spec<"vnni" = 2 : i32>>
} {
  memref.global "private" constant @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16> = dense<1.000000e+00> alignment = 128
  memref.global "private" constant @__constant_8x16x32x2xbf16 :  memref<8x16x32x2xbf16> = dense<1.000000e+00> alignment = 128
  memref.global "private" constant @__constant_32x32xbf16:  memref<32x32xbf16> = dense<1.000000e+00> alignment = 128

  // None flag set on binary add
  func.func @none_on_binary_add_bf16(%arg0: memref<256x128xbf16>) -> memref<256x512xbf16>  {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_4x16x32x2xbf16 : memref<4x16x32x2xbf16>
    %1 = memref.get_global @__constant_8x16x32x2xbf16 : memref<8x16x32x2xbf16>
    %2 = memref.get_global @__constant_32x32xbf16 : memref<32x32xbf16>
    %alloc = memref.alloc() alignment = 64 : memref<8x4x32x32xbf16>
    %3 = xsmm.unary.dispatch identity [32, 32, 128, 32] flags = (none) data_type = bf16
    %alloc_0 = memref.alloc() alignment = 64 : memref<8x8x32x32xbf16>
    %4 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0) data_type = bf16
    %5 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (none) data_type = bf16
    %6 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = bf16
    %alloc_1 = memref.alloc() alignment = 64 : memref<256x512xbf16>
    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
      %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x8x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
      %subview_2 = memref.subview %alloc[%arg1, 0, 0, 0] [1, 4, 32, 32] [1, 1, 1, 1] : memref<8x4x32x32xbf16> to memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
      xsmm.brgemm(data_type = bf16, %4, %subview_2, %0, %subview, %c4_i64) : (i64, memref<4x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<4x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
      xsmm.binary add(data_type = bf16, %5, %subview, %2,  %subview) : (i64 , memref<32x32xbf16, strided<[32, 1], offset: ?>>,memref<32x32xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
      xsmm.unary relu(data_type = bf16, %6, %subview, %subview) : (i64, memref<32x32xbf16, strided<[32, 1], offset: ?>>, memref<32x32xbf16, strided<[32, 1], offset: ?>>) -> ()
      scf.reduce
    }
    return %alloc_1 : memref<256x512xbf16>
  }
}

// CHECK-LABEL: func.func @none_on_binary_add_bf16(
// CHECK: %[[ARG0:.*]]: memref<256x128xbf16>) -> memref<256x512xbf16> {
// CHECK: %[[BIAS:.*]] = memref.get_global @__constant_32x32xbf16 : memref<32x32xbf16>
// CHECK-NOT: %[[DISPATCH:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (vnni_b, beta_0)  binary_flags = (none)  unary_flags = (none) data_type = bf16
// CHECK-NOT: xsmm.fused_brgemm(data_type = bf16, %[[DISPATCH]] , %{{.*}}, %{{.*}}, %{{.*}}, %[[BIAS]], %{{.*}})

// -----
 memref.global "private" constant @__constant_32x32x32xf32_1 : memref<32x32x32xf32> = dense<1.600000e+00> alignment = 64
 memref.global "private" constant @__constant_32xf32_1 : memref<32xf32> = dense<1.300000e+00> alignment = 64
 memref.global "private" constant @__constant_32x32x32xf32_0 : memref<32x32x32xf32> = dense<1.500000e+00> alignment = 64
 memref.global "private" constant @__constant_32xf32_0 : memref<32xf32> = dense<1.200000e+00> alignment = 64
 memref.global "private" constant @__constant_32x32x32xf32 : memref<32x32x32xf32> = dense<1.400000e+00> alignment = 64
 memref.global "private" constant @__constant_32xf32 : memref<32xf32> = dense<1.100000e+00> alignment = 64

func.func @forward(%arg0: memref<256x1024xf32>) -> memref<256x1024xf32> {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @__constant_32xf32 : memref<32xf32>
  %1 = memref.get_global @__constant_32x32x32xf32 : memref<32x32x32xf32>
  %2 = memref.get_global @__constant_32x32x32xf32_0 : memref<32x32x32xf32>
  %3 = memref.get_global @__constant_32xf32_0 : memref<32xf32>
  %4 = memref.get_global @__constant_32x32x32xf32_1 : memref<32x32x32xf32>
  %alloc = memref.alloc() alignment = 64 : memref<256x1024xf32>
  %alloc_0 = memref.alloc() alignment = 64 : memref<8x32x32x32xf32>
  %alloc_1 = memref.alloc() alignment = 64 : memref<8x32x32x32xf32>
  scf.forall (%arg1, %arg2) in (8, 32) {
    %subview = memref.subview %alloc_0[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
    %5 = xsmm.unary.dispatch zero [32, 32, 1, 32] flags = (bcast_scalar) data_type = f32
    xsmm.unary zero(data_type = f32, %5, %cst, %subview) : (i64, f32, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    %subview_3 = memref.subview %alloc_1[%arg1, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
    %6 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (none) data_type = f32
    xsmm.brgemm(data_type = f32, %6, %subview_3, %4, %subview, %c32_i64) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, i64) -> ()
    %7 = xsmm.binary.dispatch add [32, 32, 32, 32, 32] flags = (bcast_col_in0) data_type = f32
    xsmm.binary add(data_type = f32, %7, %3, %subview, %subview) : (i64, memref<32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
    %8 = xsmm.unary.dispatch relu [32, 32, 32, 32] flags = (none) data_type = f32
    xsmm.unary relu(data_type = f32, %8, %subview, %subview) : (i64, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32x32xf32, strided<[32, 1], offset: ?>>) -> ()
  }
  return %alloc : memref<256x1024xf32>
}

// CHECK-LABEL:func.func @forward(
// CHECK: %[[ARG0:.*]]: memref<256x1024xf32>) -> memref<256x1024xf32> {
// CHECK-DAG: %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK:  scf.forall (%[[arg1:.*]], %[[arg2:.*]]) in (8, 32) {
// CHECK:      %[[subview:.*]] = memref.subview %{{.*}}[%[[arg1]], %[[arg2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
// CHECK:      %[[subview_2:.*]] = memref.subview %{{.*}}[%[[arg1]], 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xf32> to memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>
// CHECK:      %[[temp2:.*]] = xsmm.fused_brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024][add,relu]  flags = (beta_0)  binary_flags = (bcast_col_in0)  unary_flags = (none) data_type = f32
// CHECK:      xsmm.fused_brgemm(data_type = f32, %[[temp2]], %[[subview_2]], %{{.*}}, %[[subview]], %{{.*}} %[[c32_i64]]) : (i64, memref<32x32x32xf32, strided<[1024, 32, 1], offset: ?>>, memref<32x32x32xf32>, memref<32x32xf32, strided<[32, 1], offset: ?>>, memref<32xf32>, i64) -> ()
// CHECK:    }
// CHECK:    return %{{.*}} : memref<256x1024xf32>
