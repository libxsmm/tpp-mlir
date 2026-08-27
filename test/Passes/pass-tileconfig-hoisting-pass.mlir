// RUN: tpp-opt %s --intel-amx-tile-config-hoisting-pass | FileCheck %s

module{

memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> alignment = 64

func.func @entry(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c8 = arith.constant 8 : index
  %c0 = arith.constant 0 : index
  %c32_i64 = arith.constant 32 : i64
  %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
  %alloc = memref.alloc() alignment = 64 : memref<8x32x32x32xbf16>
  %1 = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, vnni_b, beta_0) data_type = bf16
  %2 = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
  %3 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c32) step (%c2, %c8) {
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %10 = arith.addi %arg3, %arg1 : index
      %subview = memref.subview %arg0[%10, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg4 = %c0 to %c8 step %c1 {
        %11 = arith.addi %arg4, %arg2 : index
        %subview_1 = memref.subview %alloc[%10, %11, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
        %alloca = memref.alloca() : memref<64xi8>
        "xsmm.IntelAMXtileConfig"(%1, %alloca) : (i64, memref<64xi8>) -> ()
        xsmm.brgemm(data_type = bf16, %3, %subview, %0, %subview_1, %c32_i64) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
        "xsmm.IntelAMXtileConfig"(%2, %alloca) : (i64, memref<64xi8>) -> ()
      }
    }
    scf.reduce
  }
  %alloc_0 = memref.alloc() alignment = 64 : memref<8x32x32x32xbf16>
  %4 = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, vnni_b, beta_0) data_type = bf16
  %5 = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
  %6 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c32) step (%c2, %c8) {
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %10 = arith.addi %arg3, %arg1 : index
      %subview = memref.subview %alloc[%10, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg4 = %c0 to %c8 step %c1 {
        %11 = arith.addi %arg4, %arg2 : index
        %subview_1 = memref.subview %alloc_0[%10, %11, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
        %alloca = memref.alloca() : memref<64xi8>
        "xsmm.IntelAMXtileConfig"(%4, %alloca) : (i64, memref<64xi8>) -> ()
        xsmm.brgemm(data_type = bf16, %6, %subview, %0, %subview_1, %c32_i64) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
        "xsmm.IntelAMXtileConfig"(%5, %alloca) : (i64, memref<64xi8>) -> ()
      }
    }
    scf.reduce
  }
  %7 = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, vnni_b, beta_0) data_type = bf16
  %8 = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
  %9 = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
  scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c8, %c32) step (%c2, %c8) {
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %10 = arith.addi %arg3, %arg1 : index
      %subview = memref.subview %alloc_0[%10, 0, 0, 0] [1, 32, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>
      scf.for %arg4 = %c0 to %c8 step %c1 {
        %11 = arith.addi %arg4, %arg2 : index
        %subview_1 = memref.subview %alloc[%10, %11, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
        %alloca = memref.alloca() : memref<64xi8>
        "xsmm.IntelAMXtileConfig"(%7, %alloca) : (i64, memref<64xi8>) -> ()
        xsmm.brgemm(data_type = bf16, %9, %subview, %0, %subview_1, %c32_i64) : (i64, memref<32x32x32xbf16, strided<[1024, 32, 1], offset: ?>>, memref<32x16x32x2xbf16>, memref<32x32xbf16, strided<[32, 1], offset: ?>>, i64) -> ()
        "xsmm.IntelAMXtileConfig"(%8, %alloca) : (i64, memref<64xi8>) -> ()
      }
    }
    scf.reduce
  }
  memref.dealloc %alloc_0 : memref<8x32x32x32xbf16>
  return %alloc : memref<8x32x32x32xbf16>
}
}

// CHECK-LABEL: func.func @entry(
// CHECK: %[[ARG0:.*]]: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
// CHECK-DAG:  %[[c2:.*]] = arith.constant 2 : index
// CHECK-DAG:  %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:  %[[c32:.*]] = arith.constant 32 : index
// CHECK-DAG:  %[[c8:.*]] = arith.constant 8 : index
// CHECK-DAG:  %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:  %[[c32_i64:.*]] = arith.constant 32 : i64
// CHECK:  %[[dispatch1:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  %[[dispatch2:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  %[[brgemmdispatch:.*]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c8]]) {
// CHECK:    %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// CHECK:    "xsmm.IntelAMXtileConfig"(%[[dispatch1]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:    scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:      %[[temp10:.*]] = arith.addi %[[ARG3]], %[[ARG1]] : index
// CHECK:      scf.for %[[ARG4:.*]] = %c0 to %c8 step %c1 {
// CHECK:        %[[temp11:.*]] = arith.addi %[[ARG4]], %[[ARG2]] : index
// CHECK:        xsmm.brgemm(data_type = bf16, %[[brgemmdispatch]], %{{.*}}, %{{.*}}, %{{.*}}, %[[c32_i64]]) 
// CHECK:      }
// CHECK:    }
// CHECK:    "xsmm.IntelAMXtileConfig"(%[[dispatch2]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:    scf.reduce
// CHECK:  }
// CHECK:  %[[dispatch3:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  %[[dispatch4:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  %[[brgemmdispatch2:.*]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c8]]) {
// CHECK:    %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// CHECK:    "xsmm.IntelAMXtileConfig"(%[[dispatch3]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:    scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:      %[[temp10:.*]] = arith.addi %[[ARG3]], %[[ARG1]] : index
// CHECK:      scf.for %[[ARG4:.*]] = %c0 to %c8 step %c1 {
// CHECK:        %[[temp11:.*]] = arith.addi %[[ARG4]], %[[ARG2]] : index
// CHECK:        xsmm.brgemm(data_type = bf16, %[[brgemmdispatch2]], %{{.*}}, %{{.*}}, %{{.*}}, %[[c32_i64]])
// CHECK:      }
// CHECK:    }
// CHECK:    "xsmm.IntelAMXtileConfig"(%[[dispatch4]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:    scf.reduce
// CHECK:  }
// CHECK:  %[[dispatch5:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  %[[dispatch6:.*]] = xsmm.IntelAMXtileConfig.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  %[[brgemmdispatch3:.*]] = xsmm.brgemm.dispatch [32, 32, 32, 32, 32, 32, 1024, 1024] flags = (no_reset_tileconfig, no_setup_tileconfig, vnni_b, beta_0) data_type = bf16
// CHECK:  scf.parallel (%[[ARG1:.*]], %[[ARG2:.*]]) = (%[[c0]], %[[c0]]) to (%[[c8]], %[[c32]]) step (%[[c2]], %[[c8]]) {
// CHECK:    %[[alloca:.*]] = memref.alloca() : memref<64xi8>
// CHECK:    "xsmm.IntelAMXtileConfig"(%[[dispatch5]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:    scf.for %[[ARG3:.*]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:      %[[temp10:.*]] = arith.addi %[[ARG3]], %[[ARG1]] : index
// CHECK:      scf.for %[[ARG4:.*]] = %c0 to %c8 step %c1 {
// CHECK:        %[[temp11:.*]] = arith.addi %[[ARG4]], %[[ARG2]] : index
// CHECK:        xsmm.brgemm(data_type = bf16, %[[brgemmdispatch3]], %{{.*}}, %{{.*}}, %{{.*}}, %[[c32_i64]])
// CHECK:      }
// CHECK:    }
// CHECK:    "xsmm.IntelAMXtileConfig"(%[[dispatch6]], %[[alloca]]) : (i64, memref<64xi8>) -> ()
// CHECK:    scf.reduce
// CHECK:  }

