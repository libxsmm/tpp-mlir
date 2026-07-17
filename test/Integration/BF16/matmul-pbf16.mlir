// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3, d2, d0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>

func.func @matmultpp(%A: memref<4x8xbf16>,
          %B: memref<4x4x2xbf16>, %C: memref<4x4xbf16>)  {
  %expanded = memref.expand_shape %A [[0], [1, 2]] output_shape [4, 4, 2]
    : memref<4x8xbf16> into memref<4x4x2xbf16>
  linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
    ins(%expanded, %B : memref<4x4x2xbf16>, memref<4x4x2xbf16>)
    outs(%C : memref<4x4xbf16>) {
      ^bb0(%in: bf16, %in_2: bf16, %out: bf16):
        %1 = arith.mulf %in, %in_2 : bf16
        %2 = arith.addf %out, %1 : bf16
        linalg.yield %2 : bf16
    }
  return
}

func.func @entry() {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 1.0 : bf16
  %da = memref.alloc() :memref<4x8xbf16>
  linalg.fill ins(%f0 : bf16) outs(%da : memref<4x8xbf16>)
  // Call kernel.
  %0 = memref.alloc() : memref<4x4x2xbf16>
  linalg.fill ins(%f0:bf16) outs (%0: memref<4x4x2xbf16>)
  %D = memref.alloc() : memref<4x4xbf16>
  %zero = arith.constant 0.0 : bf16
  linalg.fill ins(%zero : bf16) outs(%D:memref<4x4xbf16>)
  call @matmultpp(%da, %0, %D)
       : (memref<4x8xbf16>, memref<4x4x2xbf16>, memref<4x4xbf16>)->()

  //
  // CHECK:( ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ), ( 8, 8, 8, 8 ) )
  //
  %d1 = arith.constant -1.0 : bf16

  %v0 = vector.transfer_read %D[%c0, %c0], %d1 : memref<4x4xbf16>, vector<4x4xbf16>
  %f1 = arith.extf %v0:vector<4x4xbf16> to vector<4x4xf32>
  vector.print %f1 : vector<4x4xf32>

  return
}
