// RUN: tpp-run -e register_tile_bf16 --entry-point-result=void -print %s > %t.1
// RUN: tpp-opt %s --tile-brgemm-linalg="registerBlocking=32,32,32" -convert-linalg-to-xsmm | tpp-run -e  register_tile_bf16 --entry-point-result=void -print > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
module {
  memref.global "private" constant @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @register_tile_bf16(%arg0: memref<8x32x32x32xbf16>) -> memref<8x32x32x32xbf16> {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = memref.get_global @__constant_32x16x32x2xbf16 : memref<32x16x32x2xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x32x32x32xbf16>
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [8, 32, 32, 16, 2] : memref<8x32x32x32xbf16> into memref<8x32x32x16x2xbf16>
    scf.forall (%arg1, %arg2) in (8, 32) {
      %subview = memref.subview %alloc[%arg1, %arg2, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<8x32x32x32xbf16> to memref<32x32xbf16, strided<[32, 1], offset: ?>>
      linalg.fill ins(%cst : bf16) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>)
      %subview_0 = memref.subview %expand_shape[%arg1, 0, 0, 0, 0] [1, 32, 32, 16, 2] [1, 1, 1, 1, 1] : memref<8x32x32x16x2xbf16> to memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_0, %0 : memref<32x32x16x2xbf16, strided<[1024, 32, 2, 1], offset: ?>>, memref<32x16x32x2xbf16>) outs(%subview : memref<32x32xbf16, strided<[32, 1], offset: ?>>) {
      ^bb0(%in: bf16, %in_1: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_1 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
      }
    }
    return %alloc : memref<8x32x32x32xbf16>
  }
}

