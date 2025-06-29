// RUN: tpp-opt %s --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK %s


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @optimal_register_allocation_gemm(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c24 step %c3 {
      scf.for %arg4 = %c0 to %c64 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [3, 32] [1, 1] : memref<24x64xf32> to memref<3x32xf32, strided<[64, 1], offset: ?>>
        %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<3x32xf32>
        %1 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %0) -> (vector<3x32xf32>) {
          %2 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (vector<3x32xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 3, 1] [1, 1, 1] : memref<32x24x32xf32> to memref<1x3x1xf32, strided<[768, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 1, 32] [1, 1, 1] : memref<32x32x64xf32> to memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>
            %3 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x3x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1x3x1xf32>
            %4 = vector.transfer_read %subview_1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x1x32xf32>
            %5 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg8 : vector<1x3x1xf32>, vector<1x1x32xf32> into vector<3x32xf32>
            scf.yield %5 : vector<3x32xf32>
          }
          scf.yield %2 : vector<3x32xf32>
        }
        vector.transfer_write %1, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<3x32xf32>, memref<3x32xf32, strided<[64, 1], offset: ?>>
      }
    }
    return
  }
}

// CHECK-LABEL:   func.func @optimal_register_allocation_gemm
// CHECK: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>

// -----

#no_map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#no_map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#no_map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @no_lowering_k_4(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) -> memref<24x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %arg3 = %c0 to %c24 step %c3 {
      scf.for %arg4 = %c0 to %c64 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [3, 32] [1, 1] : memref<24x64xf32> to memref<3x32xf32, strided<[64, 1], offset: ?>>
        %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<3x32xf32>
        %1 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %0) -> (vector<3x32xf32>) {
          %2 = scf.for %arg7 = %c0 to %c32 step %c4 iter_args(%arg8 = %arg6) -> (vector<3x32xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 3, 4] [1, 1, 1] : memref<32x24x32xf32> to memref<1x3x4xf32, strided<[768, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 4, 32] [1, 1, 1] : memref<32x32x64xf32> to memref<1x4x32xf32, strided<[2048, 64, 1], offset: ?>>
            %3 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x3x4xf32, strided<[768, 32, 1], offset: ?>>, vector<1x3x4xf32>
            %4 = vector.transfer_read %subview_1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x4x32xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x4x32xf32>
            %5 = vector.contract {indexing_maps = [#no_map, #no_map1, #no_map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg8 : vector<1x3x4xf32>, vector<1x4x32xf32> into vector<3x32xf32>
            scf.yield %5 : vector<3x32xf32>
          }
          scf.yield %2 : vector<3x32xf32>
        }
        vector.transfer_write %1, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<3x32xf32>, memref<3x32xf32, strided<[64, 1], offset: ?>>
      }
    }
    return %arg2 : memref<24x64xf32>
  }
}

// CHECK-LABEL: func.func @no_lowering_k_4
// CHECK-NOT: vector.fma
// CHECK: vector.contract

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
// CHECK-COUNT-16: vector.fma{{.*}}vector<8xf32>
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write
