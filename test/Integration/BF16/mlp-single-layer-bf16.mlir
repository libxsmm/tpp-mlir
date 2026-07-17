// Validate default pipeline
// RUN: tpp-run %s  \
// RUN:  -e entry -entry-point-result=void

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>

func.func @entry(){
  %c0 = arith.constant 0.0 : bf16
  %c1 = arith.constant 1.0 : bf16
  %arg0 = memref.alloc() : memref<128x128x2xbf16>
  linalg.fill ins(%c1 : bf16) outs(%arg0 : memref<128x128x2xbf16>)
  %arg2 = memref.alloc() : memref<512xbf16>
  linalg.fill ins(%c1 : bf16) outs(%arg2 : memref<512xbf16>)
  %arg3 = memref.alloc() : memref<128x512xbf16>
  linalg.fill ins(%c1 : bf16) outs(%arg3 : memref<128x512xbf16>)

  linalg.generic {
    indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg2: memref<512xbf16>) outs(%arg3: memref<128x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
    linalg.yield %in : bf16
  }

  %wt = memref.alloc() : memref<128x512x2xbf16>
  linalg.fill ins(%c1 : bf16) outs(%wt : memref<128x512x2xbf16>)

  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %wt : memref<128x128x2xbf16>, memref<128x512x2xbf16>)
    outs(%arg3 : memref<128x512xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
  }

  linalg.generic {
    indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg3 : memref<128x512xbf16>) outs(%arg3 : memref<128x512xbf16>) {
      ^bb0(%in: bf16, %out: bf16):
        %2 = arith.maximumf %in, %c0 : bf16
        linalg.yield %2 : bf16
  }

  %result = memref.alloc() : memref<128x512xbf16>
  %c256 = arith.constant 256.0 : bf16
  linalg.fill ins(%c256 : bf16) outs(%result : memref<128x512xbf16>)
  %threshold = arith.constant 1.0 : bf16
  check.expect_almost_eq(%arg3, %result, %threshold) : memref<128x512xbf16>, memref<128x512xbf16>, bf16
  return
}
