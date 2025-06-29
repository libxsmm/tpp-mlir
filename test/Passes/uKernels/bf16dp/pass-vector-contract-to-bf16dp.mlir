// RUN: tpp-opt %s --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK %s


#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @optimal_register_allocation(%arg0: memref<2x24x16x2xbf16>) -> memref<24x128xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant dense<0.000000e+00> : vector<24x128xbf16>
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c6 = arith.constant 6 : index
    %c24 = arith.constant 24 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_2x16x128x2xbf16 : memref<2x16x128x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<24x128xbf16>
    vector.transfer_write %cst_0, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<24x128xbf16>, memref<24x128xbf16>
    scf.for %arg1 = %c0 to %c24 step %c6 {
      scf.for %arg2 = %c0 to %c128 step %c64 {
        %subview = memref.subview %alloc[%arg1, %arg2] [6, 64] [1, 1] : memref<24x128xbf16> to memref<6x64xbf16, strided<[128, 1], offset: ?>>
        %1 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<6x64xbf16, strided<[128, 1], offset: ?>>, vector<6x64xbf16>
        %2 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %1) -> (vector<6x64xbf16>) {
          %3 = scf.for %arg5 = %c0 to %c16 step %c1 iter_args(%arg6 = %arg4) -> (vector<6x64xbf16>) {
            %subview_1 = memref.subview %arg0[%arg3, %arg1, %arg5, 0] [1, 6, 1, 2] [1, 1, 1, 1] : memref<2x24x16x2xbf16> to memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>
            %subview_2 = memref.subview %0[%arg3, %arg5, %arg2, 0] [1, 1, 64, 2] [1, 1, 1, 1] : memref<2x16x128x2xbf16> to memref<1x1x64x2xbf16, strided<[4096, 256, 2, 1], offset: ?>>
            %4 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x6x1x2xbf16, strided<[768, 32, 2, 1], offset: ?>>, vector<1x6x1x2xbf16>
            %5 = vector.transfer_read %subview_2[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x1x64x2xbf16, strided<[4096, 256, 2, 1], offset: ?>>, vector<1x1x64x2xbf16>
            %6 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg6 : vector<1x6x1x2xbf16>, vector<1x1x64x2xbf16> into vector<6x64xbf16>
            scf.yield %6 : vector<6x64xbf16>
          }
          scf.yield %3 : vector<6x64xbf16>
        }
        vector.transfer_write %2, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<6x64xbf16>, memref<6x64xbf16, strided<[128, 1], offset: ?>>
      }
    }
    return %alloc : memref<24x128xbf16>
  }
}

// CHECK-LABEL:   func.func @optimal_register_allocation
// CHECK: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.bitcast
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot

// -----

#mlp_map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#mlp_map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#mlp_map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_32xbf16 : memref<32xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @mlp_bf16(%arg0: memref<4x32x16x2xbf16>, %arg1: memref<4x16x32x2xbf16>, %arg2: memref<32x32xbf16>) {
    %cst = arith.constant dense<0.000000e+00> : vector<32x32xbf16>
    %cst_0 = arith.constant 0.000000e+00 : bf16
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_32xbf16 : memref<32xbf16>
    scf.for %arg3 = %c0 to %c32 step %c2 {
      scf.for %arg4 = %c0 to %c32 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [2, 32] [1, 1] : memref<32x32xbf16> to memref<2x32xbf16, strided<[32, 1], offset: ?>>
        %7 = vector.transfer_read %subview[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<2x32xbf16, strided<[32, 1], offset: ?>>, vector<2x32xbf16>
        %8 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %7) -> (vector<2x32xbf16>) {
          %9 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %arg6) -> (vector<2x32xbf16>) {
            %subview_1 = memref.subview %arg0[%arg5, %arg3, %arg7, 0] [1, 2, 1, 2] [1, 1, 1, 1] : memref<4x32x16x2xbf16> to memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
            %subview_2 = memref.subview %arg1[%arg5, %arg7, %arg4, 0] [1, 1, 32, 2] [1, 1, 1, 1] : memref<4x16x32x2xbf16> to memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>
            %10 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : memref<1x2x1x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, vector<1x2x1x2xbf16>
            %11 = vector.transfer_read %subview_2[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true]} : memref<1x1x32x2xbf16, strided<[1024, 64, 2, 1], offset: ?>>, vector<1x1x32x2xbf16>
            %12 = vector.contract {indexing_maps = [#mlp_map, #mlp_map1, #mlp_map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %10, %11, %arg8 : vector<1x2x1x2xbf16>, vector<1x1x32x2xbf16> into vector<2x32xbf16>
            scf.yield %12 : vector<2x32xbf16>
          }
          scf.yield %9 : vector<2x32xbf16>
        }
        vector.transfer_write %8, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x32xbf16>, memref<2x32xbf16, strided<[32, 1], offset: ?>>
      }
    }
    %1 = vector.transfer_read %0[%c0], %cst_0 {in_bounds = [true]} : memref<32xbf16>, vector<32xbf16>
    %2 = vector.broadcast %1 : vector<32xbf16> to vector<32x32xbf16>
    %3 = vector.transfer_read %arg2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xbf16>, vector<32x32xbf16>
    %4 = arith.addf %2, %3 : vector<32x32xbf16>
    vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16>
    %5 = vector.transfer_read %arg2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<32x32xbf16>, vector<32x32xbf16>
    %6 = arith.maximumf %5, %cst : vector<32x32xbf16>
    vector.transfer_write %6, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16>
    return
  }
}

// CHECK-LABEL: func.func @mlp_bf16
// CHECK-COUNT-4: x86vector.avx512.dot
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write
