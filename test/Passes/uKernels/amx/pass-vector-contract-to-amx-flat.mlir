// RUN: tpp-opt %s --vector-contract-to-micro-kernels-amx --split-input-file  | FileCheck -check-prefix=CHECK %s

module {
  func.func @gemm_64(%arg0: memref<4x64x64xbf16>, %arg1: memref<4x64x64xbf16>, %arg2: memref<64x64xbf16>) -> memref<64x64xbf16> {
    %0 = ub.poison : bf16
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c64 step %c32 {
      scf.for %arg4 = %c0 to %c64 step %c32 {
        %subview = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1] : memref<64x64xbf16> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
        %1 = vector.transfer_read %subview[%c0, %c0], %0 {in_bounds = [true, true]} : memref<32x32xbf16, strided<[64, 1], offset: ?>>, vector<32x32xbf16>
        %2 = scf.for %arg5 = %c0 to %c4 step %c1 iter_args(%arg6 = %1) -> (vector<32x32xbf16>) {
          %3 = scf.for %arg7 = %c0 to %c64 step %c32 iter_args(%arg8 = %arg6) -> (vector<32x32xbf16>) {
            %subview_0 = memref.subview %arg0[%arg5, %arg3, %arg7] [1, 32, 32] [1, 1, 1] : memref<4x64x64xbf16> to memref<1x32x32xbf16, strided<[4096, 64, 1], offset: ?>>
            %subview_1 = memref.subview %arg1[%arg5, %arg7, %arg4] [1, 32, 32] [1, 1, 1] : memref<4x64x64xbf16> to memref<1x32x32xbf16, strided<[4096, 64, 1], offset: ?>>
            %4 = vector.transfer_read %subview_0[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x32x32xbf16, strided<[4096, 64, 1], offset: ?>>, vector<1x32x32xbf16>
            %5 = vector.transfer_read %subview_1[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x32x32xbf16, strided<[4096, 64, 1], offset: ?>>, vector<1x32x32xbf16>
            %6 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %4, %5, %arg8 : vector<1x32x32xbf16>, vector<1x32x32xbf16> into vector<32x32xbf16>
            scf.yield %6 : vector<32x32xbf16>
          }
          scf.yield %3 : vector<32x32xbf16>
        }
        vector.transfer_write %2, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[64, 1], offset: ?>>
      }
    }
    return %arg2 : memref<64x64xbf16>
  }
}


// CHECK-LABEL:   func.func @gemm_64
// In order to make splat layout look a like vnni between two vector<32xbf16>, we shuffle them like below to get vnni format + aline them in 128 bit packing.
// CHECK: vector.shuffle{{.*}}[0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK: vector.shuffle{{.*}}[4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// 4 set of 4 tile_mulf from loops: (a) 0 to br-2 (b) br-1 (c) 0 to k-2 and (d) k-1
// CHECK-COUNT-16: amx.tile_mulf
// The final accumulated value has to be shuffled with respect to the earlier input matrix shuffle as below.
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>

// -----

module {
 func.func @mlp_64(%arg0: memref<8x16x64x64xbf16>, %arg1: memref<16x16x64x64xbf16>, %arg2: memref<16x64xbf16>, %arg3: memref<8x16x64x64xbf16>) {
   %cst = arith.constant dense<0.000000e+00> : vector<64x64xbf16>
   %0 = ub.poison : bf16
   %c1 = arith.constant 1 : index
   %c16 = arith.constant 16 : index
   %c32 = arith.constant 32 : index
   %c64 = arith.constant 64 : index
   %c0 = arith.constant 0 : index
   scf.forall (%arg4, %arg5) in (8, 16) {
     %subview = memref.subview %arg0[%arg4, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<8x16x64x64xbf16> to memref<16x64x64xbf16, strided<[4096, 64, 1], offset: ?>>
     %subview_0 = memref.subview %arg1[%arg5, 0, 0, 0] [1, 16, 64, 64] [1, 1, 1, 1] : memref<16x16x64x64xbf16> to memref<16x64x64xbf16, strided<[4096, 64, 1], offset: ?>>
     %subview_1 = memref.subview %arg3[%arg4, %arg5, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<8x16x64x64xbf16> to memref<64x64xbf16, strided<[64, 1], offset: ?>>
     scf.for %arg6 = %c0 to %c64 step %c32 {
       scf.for %arg7 = %c0 to %c64 step %c32 {
         %subview_3 = memref.subview %subview_1[%arg6, %arg7] [32, 32] [1, 1] : memref<64x64xbf16, strided<[64, 1], offset: ?>> to memref<32x32xbf16, strided<[64, 1], offset: ?>>
         %7 = vector.transfer_read %subview_3[%c0, %c0], %0 {in_bounds = [true, true]} : memref<32x32xbf16, strided<[64, 1], offset: ?>>, vector<32x32xbf16>
         %8 = scf.for %arg8 = %c0 to %c16 step %c1 iter_args(%arg9 = %7) -> (vector<32x32xbf16>) {
           %9 = scf.for %arg10 = %c0 to %c64 step %c32 iter_args(%arg11 = %arg9) -> (vector<32x32xbf16>) {
             %subview_4 = memref.subview %subview[%arg8, %arg6, %arg10] [1, 32, 32] [1, 1, 1] : memref<16x64x64xbf16, strided<[4096, 64, 1], offset: ?>> to memref<1x32x32xbf16, strided<[4096, 64, 1], offset: ?>>
             %subview_5 = memref.subview %subview_0[%arg8, %arg10, %arg7] [1, 32, 32] [1, 1, 1] : memref<16x64x64xbf16, strided<[4096, 64, 1], offset: ?>> to memref<1x32x32xbf16, strided<[4096, 64, 1], offset: ?>>
             %10 = vector.transfer_read %subview_4[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x32x32xbf16, strided<[4096, 64, 1], offset: ?>>, vector<1x32x32xbf16>
             %11 = vector.transfer_read %subview_5[%c0, %c0, %c0], %0 {in_bounds = [true, true, true]} : memref<1x32x32xbf16, strided<[4096, 64, 1], offset: ?>>, vector<1x32x32xbf16>
             %12 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %10, %11, %arg11 : vector<1x32x32xbf16>, vector<1x32x32xbf16> into vector<32x32xbf16>
             scf.yield %12 : vector<32x32xbf16>
 	  }
           scf.yield %9 : vector<32x32xbf16>
         }
         vector.transfer_write %8, %subview_3[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<32x32xbf16, strided<[64, 1], offset: ?>>
       }
     }
     %subview_2 = memref.subview %arg2[%arg5, 0] [1, 64] [1, 1] : memref<16x64xbf16> to memref<64xbf16, strided<[1], offset: ?>>
     %1 = vector.transfer_read %subview_2[%c0], %0 {in_bounds = [true]} : memref<64xbf16, strided<[1], offset: ?>>, vector<64xbf16>
     %2 = vector.broadcast %1 : vector<64xbf16> to vector<64x64xbf16>
     %3 = vector.transfer_read %subview_1[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xbf16, strided<[64, 1], offset: ?>>, vector<64x64xbf16>
     %4 = arith.addf %2, %3 : vector<64x64xbf16>
     vector.transfer_write %4, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xbf16>, memref<64x64xbf16, strided<[64, 1], offset: ?>>
     %5 = vector.transfer_read %subview_1[%c0, %c0], %0 {in_bounds = [true, true]} : memref<64x64xbf16, strided<[64, 1], offset: ?>>, vector<64x64xbf16>
     %6 = arith.maximumf %5, %cst : vector<64x64xbf16>
     vector.transfer_write %6, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xbf16>, memref<64x64xbf16, strided<[64, 1], offset: ?>>
   }
   return
 }
}


// CHECK-LABEL:   func.func @mlp_64
// CHECK: vector.shuffle{{.*}}[0, 32, 1, 33, 2, 34, 3, 35, 8, 40, 9, 41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50, 19, 51, 24, 56, 25, 57, 26, 58, 27, 59] : vector<32xbf16>, vector<32xbf16>
// CHECK: vector.shuffle{{.*}}[4, 36, 5, 37, 6, 38, 7, 39, 12, 44, 13, 45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54, 23, 55, 28, 60, 29, 61, 30, 62, 31, 63] : vector<32xbf16>, vector<32xbf16>
// CHECK-COUNT-16: amx.tile_mulf
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23] : vector<16xf32>, vector<16xf32>
// CHECK: vector.shuffle{{.*}}[8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31] : vector<16xf32>, vector<16xf32>
// For MLP, bias and relu get fused by nanokernels. So no transfer read/write.
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write
