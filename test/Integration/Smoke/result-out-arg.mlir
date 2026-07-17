// RUN: tpp-run %s -print \
// RUN:  -e entry -entry-point-result=void | \
// RUN: FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @gemm_tpp(%A: memref<2x2xf32>,
          %B: memref<2x2xf32>, %C: memref<2x2xf32>)  {
  linalg.generic {indexing_maps = [#map0, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%A, %B: memref<2x2xf32>, memref<2x2xf32>) outs(%C: memref<2x2xf32>) {
        ^bb0(%a: f32, %b: f32, %c: f32):
          %0 = arith.mulf %a, %b : f32
          %1 = arith.addf %c, %0 : f32
          linalg.yield %1 : f32
  }
  return
}

memref.global "private" constant @__constant_da : memref<2x2xf32> =
  dense<[
        [ 1.0, 2.0 ],
        [ 3.0, 4.0 ]
  ]>

memref.global "private" constant @__constant_db : memref<2x2xf32> =
  dense<[
        [ 2.0, 1.0 ],
        [ 1.0, 2.0 ]
  ]>

func.func @entry(%out : memref<2x2xf32>) -> memref<2x2xf32> {
  %c0 = arith.constant 0.0 : f32
  linalg.fill ins(%c0 : f32) outs(%out : memref<2x2xf32>)

  %A = memref.get_global @__constant_da : memref<2x2xf32>
  %B = memref.get_global @__constant_db : memref<2x2xf32>

  // Call kernel.
  call @gemm_tpp(%A, %B, %out) : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()

  return %out : memref<2x2xf32>
}

// CHECK: ( 4, 5 )
// CHECK: ( 10, 11 )
