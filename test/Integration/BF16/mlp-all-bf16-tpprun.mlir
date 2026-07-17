// RUN: tpp-run %s \
// RUN:  -e entry -entry-point-result=void

memref.global "private" constant @arg1 : memref<128x512x2xbf16> = dense<1.00e+00>
memref.global "private" constant @arg3 : memref<256x1024x2xbf16> = dense<1.00e+00>
memref.global "private" constant @arg5 : memref<512x2048x2xbf16> = dense<1.00e+00>
memref.global "private" constant @arg7 : memref<1024x1000x2xbf16> = dense<1.00e+00>

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d1)>

func.func @entry(%arg0: memref<128x256xbf16>, %arg2: memref<512xbf16>, %arg4: memref<1024xbf16>,
    %arg6: memref<2048xbf16>, %arg8: memref<1000xbf16>, %arg9: memref<128x512xbf16>,
    %arg10: memref<128x1024xbf16>, %arg11: memref<128x2048xbf16>, %arg12: memref<128x1000xbf16>) {
  %c0 = arith.constant 0.0 : bf16
  linalg.generic {
    indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg2: memref<512xbf16>) outs(%arg9: memref<128x512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
    linalg.yield %in : bf16
  }

  %e0 = memref.expand_shape %arg0 [[0], [1, 2]] output_shape [128, 128, 2]
    : memref<128x256xbf16> into memref<128x128x2xbf16>
  %relayout_arg0 = memref.get_global @arg1:memref<128x512x2xbf16>
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
    ins(%e0, %relayout_arg0 : memref<128x128x2xbf16>, memref<128x512x2xbf16>)
    outs(%arg9 : memref<128x512xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
  }
  linalg.generic {
    indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg9 : memref<128x512xbf16>) outs(%arg9 : memref<128x512xbf16>) {
      ^bb0(%in: bf16, %out: bf16):
        %2 = arith.maximumf %in, %c0 : bf16
        linalg.yield %2 : bf16
  }

  linalg.generic {
    indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg4: memref<1024xbf16>) outs(%arg10: memref<128x1024xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
    linalg.yield %in : bf16
  }

  %e1 = memref.expand_shape %arg9 [[0], [1, 2]] output_shape [128, 256, 2]
    : memref<128x512xbf16> into memref<128x256x2xbf16>
  %relayout_arg12 = memref.get_global @arg3:memref<256x1024x2xbf16>
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
    ins(%e1, %relayout_arg12 : memref<128x256x2xbf16>, memref<256x1024x2xbf16>)
    outs(%arg10 : memref<128x1024xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
  }
  linalg.generic {
    indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg10 : memref<128x1024xbf16>) outs(%arg10 : memref<128x1024xbf16>) {
      ^bb0(%in: bf16, %out: bf16):
        %2 = arith.maximumf %in, %c0 : bf16
        linalg.yield %2 : bf16
  }

  linalg.generic {
    indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg6: memref<2048xbf16>) outs(%arg11: memref<128x2048xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
    linalg.yield %in : bf16
  }

  %relayout_arg11 = memref.get_global @arg5:memref<512x2048x2xbf16>
  %e2 = memref.expand_shape %arg10 [[0], [1, 2]] output_shape [128, 512, 2]
    : memref<128x1024xbf16> into memref<128x512x2xbf16>
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
    ins(%e2, %relayout_arg11 : memref<128x512x2xbf16>, memref<512x2048x2xbf16>)
    outs(%arg11 : memref<128x2048xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
  }
  linalg.generic {
    indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg11 : memref<128x2048xbf16>) outs(%arg11 : memref<128x2048xbf16>) {
      ^bb0(%in: bf16, %out: bf16):
        %2 = arith.maximumf %in, %c0 : bf16
        linalg.yield %2 : bf16
  }

  linalg.generic {
    indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg8: memref<1000xbf16>) outs(%arg12: memref<128x1000xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
    linalg.yield %in : bf16
  }

  %relayout_arg10 = memref.get_global @arg7:memref<1024x1000x2xbf16>
  %e3 = memref.expand_shape %arg11 [[0], [1, 2]] output_shape [128, 1024, 2]
    : memref<128x2048xbf16> into memref<128x1024x2xbf16>
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
    ins(%e3, %relayout_arg10 : memref<128x1024x2xbf16>, memref<1024x1000x2xbf16>)
    outs(%arg12 : memref<128x1000xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
  }
  linalg.generic {
    indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]}
    ins(%arg12 : memref<128x1000xbf16>) outs(%arg12 : memref<128x1000xbf16>) {
      ^bb0(%in: bf16, %out: bf16):
        %2 = arith.maximumf %in, %c0 : bf16
        linalg.yield %2 : bf16
  }

  %threshold = arith.constant 1.0 : bf16
  %c4 = arith.constant 2.74878e+11: bf16
  %interim4 = memref.alloc(): memref<128x1000xbf16>
  linalg.fill ins(%c4:bf16) outs(%interim4: memref<128x1000xbf16>)
  check.expect_almost_eq(%interim4, %arg12, %threshold): memref<128x1000xbf16>, memref<128x1000xbf16>, bf16
  return
}
