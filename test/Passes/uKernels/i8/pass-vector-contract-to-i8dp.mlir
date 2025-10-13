// RUN: tpp-opt %s --vector-contract-to-micro-kernels --split-input-file  | FileCheck -check-prefix=CHECK %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  func.func @optimal_register_allocation_i8(%arg0: memref<2x24x8x4xi8>, %arg1: memref<2x8x128x4xi8>, %arg2: memref<24x128xi32>) -> memref<24x128xi32> {
    %0 = ub.poison : i32
    %1 = ub.poison : i8
    %c0 = arith.constant 0 : index
    %c24 = arith.constant 24 : index
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    scf.for %arg3 = %c0 to %c24 step %c3 {
      scf.for %arg4 = %c0 to %c128 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [3, 32] [1, 1] : memref<24x128xi32> to memref<3x32xi32, strided<[128, 1], offset: ?>>
        %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<3x32xi32, strided<[128, 1], offset: ?>>, vector<3x32xi32>
        %3 = scf.for %arg5 = %c0 to %c2 step %c1 iter_args(%arg6 = %2) -> (vector<3x32xi32>) {
          %4 = scf.for %arg7 = %c0 to %c8 step %c1 iter_args(%arg8 = %arg6) -> (vector<3x32xi32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7, 0] [1, 3, 1, 4] [1, 1, 1, 1] : memref<2x24x8x4xi8> to memref<1x3x1x4xi8, strided<[768, 32, 4, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4, 0] [1, 1, 32, 4] [1, 1, 1, 1] : memref<2x8x128x4xi8> to memref<1x1x32x4xi8, strided<[4096, 512, 4, 1], offset: ?>>
            %5 = vector.transfer_read %subview_0[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x3x1x4xi8, strided<[768, 32, 4, 1], offset: ?>>, vector<1x3x1x4xi8>
            %6 = vector.transfer_read %subview_1[%c0, %c0, %c0, %c0], %1 {in_bounds = [true, true, true, true]} : memref<1x1x32x4xi8, strided<[4096, 512, 4, 1], offset: ?>>, vector<1x1x32x4xi8>
            %7 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %arg8 : vector<1x3x1x4xi8>, vector<1x1x32x4xi8> into vector<3x32xi32>
            scf.yield %7 : vector<3x32xi32>
          }
          scf.yield %4 : vector<3x32xi32>
        }
        vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<3x32xi32>, memref<3x32xi32, strided<[128, 1], offset: ?>>
      }
    }
    return %arg2 : memref<24x128xi32>
  }
}


// CHECK-LABEL:   func.func @optimal_register_allocation_i8
// CHECK: x86vector.avx.dot.i8
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: vector.load
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: vector.load
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: vector.load
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: x86vector.avx.dot.i8
// CHECK-NEXT: x86vector.avx.dot.i8
