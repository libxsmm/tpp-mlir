// RUN: tpp-opt %s --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK %s


#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @optimal_register_allocation_gemm(%arg0: memref<32x24x32xf32>, %arg1: memref<32x32x64xf32>, %arg2: memref<24x64xf32>) -> memref<24x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c6 = arith.constant 6 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c24 step %c6 {
      scf.for %arg4 = %c0 to %c64 step %c64 {
        %subview = memref.subview %arg2[%arg3, %arg4] [6, 64] [1, 1] : memref<24x64xf32> to memref<6x64xf32, strided<[64, 1], offset: ?>>
        %0 = vector.transfer_read %subview[%c0, %c0], %cst {in_bounds = [true, true]} : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<6x64xf32>
        %1 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %0) -> (vector<6x64xf32>) {
          %2 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (vector<6x64xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 6, 1] [1, 1, 1] : memref<32x24x32xf32> to memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 1, 64] [1, 1, 1] : memref<32x32x64xf32> to memref<1x1x64xf32, strided<[2048, 64, 1], offset: ?>>
            %3 = vector.transfer_read %subview_0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x6x1xf32, strided<[768, 32, 1], offset: ?>>, vector<1x6x1xf32>
            %4 = vector.transfer_read %subview_1[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x64xf32, strided<[2048, 64, 1], offset: ?>>, vector<1x1x64xf32>
            %5 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg8 : vector<1x6x1xf32>, vector<1x1x64xf32> into vector<6x64xf32>
            scf.yield %5 : vector<6x64xf32>
          }
          scf.yield %2 : vector<6x64xf32>
        }
        vector.transfer_write %1, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<6x64xf32>, memref<6x64xf32, strided<[64, 1], offset: ?>>
      }
    }
    return %arg2 : memref<24x64xf32>
  }
}

// CHECK-LABEL:   func.func @optimal_register_allocation_gemm
// CHECK: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<16xf32>

// -----

