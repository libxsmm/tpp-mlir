// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>

func.func @entry(){
  %f0 = arith.constant 1.0:bf16
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<32x64x4x4xbf16>
  linalg.fill ins(%f0:bf16) outs(%alloc:memref<32x64x4x4xbf16>)
  %wt_alloc = memref.alloc() {alignment = 128 : i64} : memref<128x64x2x4x2xbf16>
  linalg.fill ins(%f0:bf16) outs(%wt_alloc:memref<128x64x2x4x2xbf16>)
  %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<32x128x4x4xbf16>
  linalg.fill ins(%f0:bf16) outs(%alloc_1:memref<32x128x4x4xbf16>)
  scf.for %arg4 = %c0 to %c4 step %c1 {
    %subview = memref.subview %alloc[%arg4, 0, 0, 0] [1, 64, 4, 4] [1, 1, 1, 1] : memref<32x64x4x4xbf16> to memref<64x4x4xbf16, strided<[16, 4, 1], offset: ?>>
    %subview_2 = memref.subview %alloc_1[%arg4, 0, 0, 0] [1, 128, 4, 4] [1, 1, 1, 1] : memref<32x128x4x4xbf16> to memref<128x4x4xbf16, strided<[16, 4, 1], offset: ?>>
    scf.for %arg5 = %c0 to %c16 step %c1 {
      %subview_3 = memref.subview %wt_alloc[%arg5, 0, 0, 0, 0] [1, 64, 2, 4, 2] [1, 1, 1, 1, 1] : memref<128x64x2x4x2xbf16> to memref<64x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>
      %subview_4 = memref.subview %subview_2[%arg5, 0, 0] [1, 4, 4] [1, 1, 1] : memref<128x4x4xbf16, strided<[16, 4, 1], offset: ?>> to memref<4x4xbf16, strided<[4, 1], offset: ?>>
      %expanded = memref.expand_shape %subview [[0], [1], [2, 3]] output_shape [64, 4, 2, 2]
        : memref<64x4x4xbf16, strided<[16, 4, 1], offset: ?>> into memref<64x4x2x2xbf16, strided<[16, 4, 2, 1], offset: ?>>
      linalg.generic {
        indexing_maps = [#map, #map1, #map2],
        iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]}
        ins(%expanded, %subview_3 : memref<64x4x2x2xbf16, strided<[16, 4, 2, 1], offset: ?>>,
                                   memref<64x2x4x2xbf16, strided<[16, 8, 2, 1], offset: ?>>)
      outs(%subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>) {
        ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
          %5 = arith.mulf %in, %in_5 : bf16
          %6 = arith.addf %out, %5 : bf16
        linalg.yield %6 : bf16
      }

      %c0_bf16 = arith.constant 0.0 : bf16
      linalg.generic {
        indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel"]}
        ins(%subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>)
        outs(%subview_4 : memref<4x4xbf16, strided<[4, 1], offset: ?>>) {
          ^bb0(%in: bf16, %out: bf16):
          %2 = arith.maximumf %in, %c0_bf16 : bf16
          linalg.yield %2 : bf16
      }
      %d1 = arith.constant -1.0 : bf16
      %v0 = vector.transfer_read %subview_4[%c0, %c0], %d1 : memref<4x4xbf16, strided<[4,1], offset:?>>, vector<4x4xbf16>
      %f1 = arith.extf %v0:vector<4x4xbf16> to vector<4x4xf32>

      //
      // CHECK:( ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ), ( 256, 256, 256, 256 ) )
      //
      vector.print %f1 : vector<4x4xf32>
    }
  }
  return
}
