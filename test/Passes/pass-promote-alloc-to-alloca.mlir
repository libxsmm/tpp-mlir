// RUN: tpp-opt --bufferize %s | FileCheck %s -check-prefix=IR


// IR: memref.alloca() {alignment = 64 : i64} : memref<32x32xi32>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d1)>

  func.func @entry(%arg0: memref<128x128x32x64xi8>, %arg1: memref<4096xf32>, %arg2: memref<128x128x16x32x4xi8>, %arg3: memref<4096xf32>, %arg4: memref<128x128x32x32xf32>) {
    %cst = arith.constant dense<0> : vector<16384xi16>
    %cst_0 = arith.constant dense<0> : vector<16384xi16>
    %c0_i32 = arith.constant 0 : i32
    %expand_shape = memref.expand_shape %arg0 [[0], [1], [2], [3, 4]] output_shape [128, 128, 32, 16, 4] : memref<128x128x32x64xi8> into memref<128x128x32x16x4xi8>
    %expand_shape_1 = memref.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [128, 1, 32, 1] : memref<4096xf32> into memref<128x1x32x1xf32>
    %expand_shape_2 = memref.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [128, 1, 32, 1] : memref<4096xf32> into memref<128x1x32x1xf32>
    scf.forall (%arg5) in (16384) {
      %0 = vector.extract %cst_0[%arg5] : i16 from vector<16384xi16>
      %1 = vector.extract %cst[%arg5] : i16 from vector<16384xi16>
      %2 = arith.index_cast %0 : i16 to index
      %3 = arith.index_cast %1 : i16 to index
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xi32>
      linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<32x32xi32>)
      %subview = memref.subview %expand_shape[%2, 0, 0, 0, 0] [1, 128, 32, 16, 4] [1, 1, 1, 1, 1] : memref<128x128x32x16x4xi8> to memref<128x32x16x4xi8, strided<[2048, 64, 4, 1], offset: ?>>
      %subview_3 = memref.subview %arg2[%3, 0, 0, 0, 0] [1, 128, 16, 32, 4] [1, 1, 1, 1, 1] : memref<128x128x16x32x4xi8> to memref<128x16x32x4xi8, strided<[2048, 128, 4, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%subview, %subview_3 : memref<128x32x16x4xi8, strided<[2048, 64, 4, 1], offset: ?>>, memref<128x16x32x4xi8, strided<[2048, 128, 4, 1], offset: ?>>) outs(%alloc : memref<32x32xi32>) {
      ^bb0(%in: i8, %in_7: i8, %out: i32):
        %4 = arith.extsi %in : i8 to i32
        %5 = arith.extsi %in_7 : i8 to i32
        %6 = arith.muli %4, %5 : i32
        %7 = arith.addi %out, %6 : i32
        linalg.yield %7 : i32
      }
      %subview_4 = memref.subview %expand_shape_1[%2, 0, 0, 0] [1, 1, 32, 1] [1, 1, 1, 1] : memref<128x1x32x1xf32> to memref<32xf32, strided<[1], offset: ?>>
      %subview_5 = memref.subview %expand_shape_2[%3, 0, 0, 0] [1, 1, 32, 1] [1, 1, 1, 1] : memref<128x1x32x1xf32> to memref<32xf32, strided<[1], offset: ?>>
      %subview_6 = memref.subview %arg4[%2, %3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<128x128x32x32xf32> to memref<32x32xf32, strided<[32, 1], offset: ?>>
      linalg.generic {indexing_maps = [#map3, #map4, #map5, #map3], iterator_types = ["parallel", "parallel"]} ins(%alloc, %subview_4, %subview_5 : memref<32x32xi32>, memref<32xf32, strided<[1], offset: ?>>, memref<32xf32, strided<[1], offset: ?>>) outs(%subview_6 : memref<32x32xf32, strided<[32, 1], offset: ?>>) {
      ^bb0(%in: i32, %in_7: f32, %in_8: f32, %out: f32):
        %4 = arith.mulf %in_7, %in_8 : f32
        %5 = arith.sitofp %in : i32 to f32
        %6 = arith.mulf %5, %4 : f32
        linalg.yield %6 : f32
      }
    }
    return
  }
