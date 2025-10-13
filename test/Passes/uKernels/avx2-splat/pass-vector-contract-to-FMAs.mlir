// RUN: tpp-opt %s --vector-contract-to-micro-kernels="target-feature=avx2"  --split-input-file  | FileCheck -check-prefix=CHECK %s

module {
  func.func @gemm_splat(%arg0: memref<1x3x32xbf16>, %arg1: memref<1x32x32xbf16>, %arg2: memref<3x32xbf16>) -> memref<3x32xbf16> {
    %0 = ub.poison : bf16
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c3 step %c3 {
      scf.for %arg4 = %c0 to %c32 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [3, 32] [1, 1] : memref<3x32xbf16> to memref<3x32xbf16, strided<[32, 1], offset: ?>>
        %1 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<3x32xbf16, strided<[32, 1], offset: ?>>, vector<3x32xbf16>
        %2 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %1) -> (vector<3x32xbf16>) {
          %3 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %arg6) -> (vector<3x32xbf16>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 3, 1] [1, 1, 1] : memref<1x3x32xbf16> to memref<1x3x1xbf16, strided<[96, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 1, 32] [1, 1, 1] : memref<1x32x32xbf16> to memref<1x1x32xbf16, strided<[1024, 32, 1], offset: ?>>
            %4 = vector.transfer_read %subview_0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x3x1xbf16, strided<[96, 32, 1], offset: ?>>, vector<1x3x1xbf16>
            %5 = vector.transfer_read %subview_1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x1x32xbf16, strided<[1024, 32, 1], offset: ?>>, vector<1x1x32xbf16>
            %6 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x3x1xbf16>, vector<1x1x32xbf16> into vector<3x32xbf16>
            scf.yield %6 : vector<3x32xbf16>
          }
          scf.yield %3 : vector<3x32xbf16>
        }
        vector.transfer_write %2, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<3x32xbf16>, memref<3x32xbf16, strided<[32, 1], offset: ?>>
      }
    }
    return %arg2 : memref<3x32xbf16>
  }
}

// CHECK-LABEL:   func.func @gemm_splat
// We leverage the ARL nature and do even->odd loads
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK-NEXT: vector.fma{{.*}}vector<8xf32>
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// CHECK: x86vector.avx.cvt.packed.even.indexed_to_f32
// CHECK: x86vector.avx.cvt.packed.odd.indexed_to_f32
// The final accumulated value has to be shuffled as the load is even+odd
// CHECK: vector.shuffle{{.*}}[0, 8, 1, 9, 2, 10, 3, 11] : vector<8xf32>, vector<8xf32>
// CHECK: vector.shuffle{{.*}}[4, 12, 5, 13, 6, 14, 7, 15] : vector<8xf32>, vector<8xf32>

// -----
