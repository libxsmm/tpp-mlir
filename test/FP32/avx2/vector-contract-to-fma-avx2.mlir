// RUN: tpp-run -e entry --entry-point-result=void -seed 123 -print %s > %t.1
// RUN: tpp-opt %s  --vector-contract-to-fma  | tpp-run -e entry --entry-point-result=void -seed 123 -print > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @entry(%arg0: memref<4x32x24x4xf32>, %arg1: memref<32x32x4x64xf32>, %arg2: memref<4x32x24x64xf32>) -> memref<4x32x24x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c3 = arith.constant 3 : index
  %c24 = arith.constant 24 : index
  %c0 = arith.constant 0 : index
  scf.forall (%arg3, %arg4) in (4, 32) {
    %subview = memref.subview %arg0[%arg3, 0, 0, 0] [1, 32, 24, 4] [1, 1, 1, 1] : memref<4x32x24x4xf32> to memref<32x24x4xf32, strided<[96, 4, 1], offset: ?>>
    %subview_0 = memref.subview %arg1[%arg4, 0, 0, 0] [1, 32, 4, 64] [1, 1, 1, 1] : memref<32x32x4x64xf32> to memref<32x4x64xf32, strided<[256, 64, 1], offset: ?>>
    %subview_1 = memref.subview %arg2[%arg3, %arg4, 0, 0] [1, 1, 24, 64] [1, 1, 1, 1] : memref<4x32x24x64xf32> to memref<24x64xf32, strided<[64, 1], offset: ?>>
    scf.for %arg5 = %c0 to %c24 step %c3 {
      scf.for %arg6 = %c0 to %c64 step %c32 {
        %subview_2 = memref.subview %subview_1[%arg5, %arg6] [3, 32] [1, 1] : memref<24x64xf32, strided<[64, 1], offset: ?>> to memref<3x32xf32, strided<[64, 1], offset: ?>>
        %0 = vector.transfer_read %subview_2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<3x32xf32, strided<[64, 1], offset: ?>>, vector<3x32xf32>
        %1 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %0) -> (vector<3x32xf32>) {
          %2 = scf.for %arg9 = %c0 to %c4 step %c1 iter_args(%arg10 = %arg8) -> (vector<3x32xf32>) {
            %subview_3 = memref.subview %subview[%arg7, %arg5, %arg9] [1, 3, 1] [1, 1, 1] : memref<32x24x4xf32, strided<[96, 4, 1], offset: ?>> to memref<1x3x1xf32, strided<[96, 4, 1], offset: ?>>
            %subview_4 = memref.subview %subview_0[%arg7, %arg9, %arg6] [1, 1, 32] [1, 1, 1] : memref<32x4x64xf32, strided<[256, 64, 1], offset: ?>> to memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>
            %3 = vector.transfer_read %subview_3[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x3x1xf32, strided<[96, 4, 1], offset: ?>>, vector<1x3x1xf32>
            %4 = vector.transfer_read %subview_4[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x32xf32, strided<[256, 64, 1], offset: ?>>, vector<1x1x32xf32>
            %5 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg10 : vector<1x3x1xf32>, vector<1x1x32xf32> into vector<3x32xf32>
            scf.yield %5 : vector<3x32xf32>
          }
          scf.yield %2 : vector<3x32xf32>
        }
        vector.transfer_write %1, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<3x32xf32>, memref<3x32xf32, strided<[64, 1], offset: ?>>
      }
    }
  }
  return %arg2 : memref<4x32x24x64xf32>
}

// -----
