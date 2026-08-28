// RUN: tpp-opt %s --intel-amx-tile-config-insertion-pass | FileCheck %s

module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> alignment = 64
  func.func @entry(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c32_i64 = arith.constant 32 : i64
    %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
    %alloc = memref.alloc() alignment = 64 : memref<8x32x32x32xbf16>
    %1 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0) data_type = bf16
    %c0_0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c8_1 = arith.constant 8 : index
    %2 = arith.muli %c1, %c2 : index
    %3 = arith.muli %c1, %c8_1 : index
    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c32) step (%2, %3) {
      scf.for %arg3 = %c0_0 to %2 step %c1 {
        scf.for %arg4 = %c0_0 to %3 step %c1 {
          %8 = arith.addi %arg3, %arg1 : index
          %9 = arith.addi %arg4, %arg2 : index
          %subview = memref.subview %alloc[%8, %9, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
          %subview_9 = memref.subview %arg0[%8, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
          xsmm.brgemm(data_type = bf16, %1, %subview_9, %0, %subview, %c32_i64) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
        }
      }
      scf.reduce
    }
    %alloc_2 = memref.alloc() alignment = 64 : memref<8x32x32x32xbf16>
    %c0_3 = arith.constant 0 : index
    %c2_4 = arith.constant 2 : index
    %c8_5 = arith.constant 8 : index
    %4 = arith.muli %c1, %c2_4 : index
    %5 = arith.muli %c1, %c8_5 : index
    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c32) step (%4, %5) {
      scf.for %arg3 = %c0_3 to %4 step %c1 {
        scf.for %arg4 = %c0_3 to %5 step %c1 {
          %8 = arith.addi %arg3, %arg1 : index
          %9 = arith.addi %arg4, %arg2 : index
          %subview = memref.subview %alloc_2[%8, %9, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
          %subview_9 = memref.subview %alloc[%8, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
          xsmm.brgemm(data_type = bf16, %1, %subview_9, %0, %subview, %c32_i64) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
        }
      }
      scf.reduce
    }
    %c0_6 = arith.constant 0 : index
    %c2_7 = arith.constant 2 : index
    %c8_8 = arith.constant 8 : index
    %6 = arith.muli %c1, %c2_7 : index
    %7 = arith.muli %c1, %c8_8 : index
    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c32) step (%6, %7) {
      scf.for %arg3 = %c0_6 to %6 step %c1 {
        scf.for %arg4 = %c0_6 to %7 step %c1 {
          %8 = arith.addi %arg3, %arg1 : index
          %9 = arith.addi %arg4, %arg2 : index
          %subview = memref.subview %alloc[%8, %9, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
          %subview_9 = memref.subview %alloc_2[%8, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
          xsmm.brgemm(data_type = bf16, %1, %subview_9, %0, %subview, %c32_i64) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
        }
      }
      scf.reduce
    }
    memref.dealloc %alloc_2 : memref<8x32x32x32xbf16>
    return %alloc : memref<8x32x32x32xbf16>
  }
}

// CHECK:func.func @entry(%[[ARG0:.*]]: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
// CHECK-DAG:  %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG:  %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[c32:.*]] = arith.constant 32 : index
// CHECK-DAG:  %[[c8:.*]] = arith.constant 8 : index
// CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK:  scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c8]]) {
// CHECK:    scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:      scf.for %[[ARG4:.*]] = %[[c0]] to %[[c8]] step %[[c1]] {
// CHECK:        %[[temp1:.*]] = arith.addi %[[ARG3]], %[[ARG1]] : index
// CHECK:        %[[temp2:.*]] = arith.addi %[[ARG4]], %[[ARG2]] : index
// CHECK:        %[[temp3:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_reset_tileconfig) data_type = bf16
// CHECK:        %[[temp4:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_setup_tileconfig) data_type = bf16
// CHECK:        %[[temp5:.*]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_reset_tileconfig, no_setup_tileconfig) data_type = bf16
// CHECK:        %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// CHECK:        "xsmm.IntelAMXtileConfig"(%[[temp3]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:        xsmm.brgemm(data_type = bf16, %[[temp5]], %{{.*}}, %{{.*}}, %{{.*}}, %[[c32_i64]]) 
// CHECK:        "xsmm.IntelAMXtileConfig"(%[[temp4]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:  scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c8]]) {
// CHECK:    scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:      scf.for %[[ARG4:.*]] = %[[c0]] to %[[c8]] step %[[c1]] {
// CHECK:        %[[temp1:.*]] = arith.addi %[[ARG3]], %[[ARG1]] : index
// CHECK:        %[[temp2:.*]] = arith.addi %[[ARG4]], %[[ARG2]] : index
// CHECK:        %[[temp3:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_reset_tileconfig) data_type = bf16
// CHECK:        %[[temp4:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_setup_tileconfig) data_type = bf16
// CHECK:        %[[temp5:.*]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_reset_tileconfig, no_setup_tileconfig) data_type = bf16
// CHECK:        %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// CHECK:        "xsmm.IntelAMXtileConfig"(%[[temp3]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:        xsmm.brgemm(data_type = bf16, %[[temp5]], %{{.*}}, %{{.*}}, %{{.*}}, %[[c32_i64]])
// CHECK:        "xsmm.IntelAMXtileConfig"(%[[temp4]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:  scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c8]]) {
// CHECK:    scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:      scf.for %[[ARG4:.*]] = %[[c0]] to %[[c8]] step %[[c1]] {
// CHECK:        %[[temp1:.*]] = arith.addi %[[ARG3]], %[[ARG1]] : index
// CHECK:        %[[temp2:.*]] = arith.addi %[[ARG4]], %[[ARG2]] : index
// CHECK:        %[[temp3:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_reset_tileconfig) data_type = bf16
// CHECK:        %[[temp4:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_setup_tileconfig) data_type = bf16
// CHECK:        %[[temp5:.*]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (vnni_b, beta_0, no_reset_tileconfig, no_setup_tileconfig) data_type = bf16
// CHECK:        %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// CHECK:        "xsmm.IntelAMXtileConfig"(%[[temp3]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:        xsmm.brgemm(data_type = bf16, %[[temp5]], %{{.*}}, %{{.*}}, %{{.*}}, %[[c32_i64]])
// CHECK:        "xsmm.IntelAMXtileConfig"(%[[temp4]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
