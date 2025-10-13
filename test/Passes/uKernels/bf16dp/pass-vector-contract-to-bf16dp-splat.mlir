// RUN: tpp-opt %s --vector-contract-to-micro-kernels  --split-input-file  | FileCheck -check-prefix=CHECK %s

module {
  func.func @register_2x3(%arg0: memref<1x2x32xbf16>, %arg1: memref<1x32x48xbf16>, %arg2: memref<2x48xf32>) -> memref<2x48xf32> {
    %0 = ub.poison : f32
    %1 = ub.poison : bf16
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c48 = arith.constant 48 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    scf.for %arg3 = %c0 to %c2 step %c2 {
      scf.for %arg4 = %c0 to %c48 step %c48 {
        %subview = memref.subview %arg2[%arg3, %arg4] [2, 48] [1, 1] : memref<2x48xf32> to memref<2x48xf32, strided<[48, 1], offset: ?>>
        %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<2x48xf32, strided<[48, 1], offset: ?>>, vector<2x48xf32>
        %3 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %2) -> (vector<2x48xf32>) {
          %4 = scf.for %arg7 = %c0 to %c32 step %c2 iter_args(%arg8 = %arg6) -> (vector<2x48xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 2, 2] [1, 1, 1] : memref<1x2x32xbf16> to memref<1x2x2xbf16, strided<[64, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 2, 48] [1, 1, 1] : memref<1x32x48xbf16> to memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>
            %5 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<1x2x2xbf16, strided<[64, 32, 1], offset: ?>>, vector<1x2x2xbf16>
            %6 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>, vector<1x2x48xbf16>
            %7 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %arg8 : vector<1x2x2xbf16>, vector<1x2x48xbf16> into vector<2x48xf32>
            scf.yield %7 : vector<2x48xf32>
          }
          scf.yield %4 : vector<2x48xf32>
        }
        vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<2x48xf32>, memref<2x48xf32, strided<[48, 1], offset: ?>>
      }
    }
    return %arg2 : memref<2x48xf32>
  }
}

// CHECK-LABEL:   func.func @register_2x3
// In order to make splat layout look a like vnni between two vector<32xbf16>, we shuffle them like below to get vnni format + aline them in 128 bit packing.
// CHECK: vector.shuffle{{.*}}[0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK: vector.shuffle{{.*}}[4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// CHECK-NEXT: x86vector.avx512.dot
// For cases where we have one vector<32xbf16>, we extract as ymm i.e two vector<16xbf16> and interleave them to get one vector<32xbf16> in vnni format.
// CHECK: vector.shuffle{{.*}}[0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31] : vector<16xbf16>, vector<16xbf16>
// The final accumulated value has to be shuffled with respect to the earlier input matrix shuffle as below.
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>

// -----

module {
  func.func @opt_register_9x3_splat(%arg0: memref<1x9x32xbf16>, %arg1: memref<1x32x48xbf16>, %arg2: memref<9x48xbf16>) -> memref<9x48xbf16> {
    %0 = ub.poison : bf16
    %c0 = arith.constant 0 : index
    %c9 = arith.constant 9 : index
    %c48 = arith.constant 48 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0 to %c9 step %c9 {
      scf.for %arg4 = %c0 to %c48 step %c48 {
        %subview = memref.subview %arg2[%arg3, %arg4] [9, 48] [1, 1] : memref<9x48xbf16> to memref<9x48xbf16, strided<[48, 1], offset: ?>>
        %1 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<9x48xbf16, strided<[48, 1], offset: ?>>, vector<9x48xbf16>
        %2 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %1) -> (vector<9x48xbf16>) {
          %3 = scf.for %arg7 = %c0 to %c32 step %c2 iter_args(%arg8 = %arg6) -> (vector<9x48xbf16>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 9, 2] [1, 1, 1] : memref<1x9x32xbf16> to memref<1x9x2xbf16, strided<[288, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 2, 48] [1, 1, 1] : memref<1x32x48xbf16> to memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>
            %4 = vector.transfer_read %subview_0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x9x2xbf16, strided<[288, 32, 1], offset: ?>>, vector<1x9x2xbf16>
            %5 = vector.transfer_read %subview_1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x2x48xbf16, strided<[1536, 48, 1], offset: ?>>, vector<1x2x48xbf16>
            %6 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x9x2xbf16>, vector<1x2x48xbf16> into vector<9x48xbf16>
            scf.yield %6 : vector<9x48xbf16>
          }
          scf.yield %3 : vector<9x48xbf16>
        }
        vector.transfer_write %2, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<9x48xbf16>, memref<9x48xbf16, strided<[48, 1], offset: ?>>
      }
    }
    return %arg2 : memref<9x48xbf16>
  }
}

// CHECK-LABEL:   func.func @opt_register_9x3_splat
// CHECK-COUNT-3: vector.shuffle
// CHECK-COUNT-27: x86vector.avx512.dot

// -----

module {
  func.func @opt_register_6x4_splat(%arg0: memref<1x6x32xbf16>, %arg1: memref<1x32x64xbf16>, %arg2: memref<6x64xf32>) -> memref<6x64xf32> {
    %0 = ub.poison : f32
    %1 = ub.poison : bf16
    %c0 = arith.constant 0 : index
    %c6 = arith.constant 6 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    scf.for %arg3 = %c0 to %c6 step %c6 {
      scf.for %arg4 = %c0 to %c64 step %c64 {
        %subview = memref.subview %arg2[%arg3, %arg4] [6, 64] [1, 1] : memref<6x64xf32> to memref<6x64xf32, strided<[64, 1], offset: ?>>
        %2 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<6x64xf32, strided<[64, 1], offset: ?>>, vector<6x64xf32>
        %3 = scf.for %arg5 = %c0 to %c1 step %c1 iter_args(%arg6 = %2) -> (vector<6x64xf32>) {
          %4 = scf.for %arg7 = %c0 to %c32 step %c2 iter_args(%arg8 = %arg6) -> (vector<6x64xf32>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 6, 2] [1, 1, 1] : memref<1x6x32xbf16> to memref<1x6x2xbf16, strided<[192, 32, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 2, 64] [1, 1, 1] : memref<1x32x64xbf16> to memref<1x2x64xbf16, strided<[2048, 64, 1], offset: ?>>
            %5 = vector.transfer_read %subview_0[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<1x6x2xbf16, strided<[192, 32, 1], offset: ?>>, vector<1x6x2xbf16>
            %6 = vector.transfer_read %subview_1[%c0, %c0, %c0], %1 {in_bounds = [true, true, true]} : memref<1x2x64xbf16, strided<[2048, 64, 1], offset: ?>>, vector<1x2x64xbf16>
            %7 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %6, %arg8 : vector<1x6x2xbf16>, vector<1x2x64xbf16> into vector<6x64xf32>
            scf.yield %7 : vector<6x64xf32>
          }
          scf.yield %4 : vector<6x64xf32>
        }
        vector.transfer_write %3, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<6x64xf32>, memref<6x64xf32, strided<[64, 1], offset: ?>>
      }
    }
    return %arg2 : memref<6x64xf32>
  }
}

// CHECK-LABEL:   func.func @opt_register_6x4_splat
// CHECK-COUNT-3: vector.shuffle
// CHECK-COUNT-24: x86vector.avx512.dot
