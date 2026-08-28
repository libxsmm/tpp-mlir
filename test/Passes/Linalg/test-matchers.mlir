// RUN: tpp-opt %s -mlir-disable-threading=true -pass-pipeline="builtin.module(func.func(test-structural-matchers))" -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: test
func.func @test() {
  return
}

// CHECK-LABEL: test_matmul
func.func @test_matmul(%arg0: tensor<32x32xf32>,
                       %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cst = arith.constant 0.0 : f32
  // CHECK: not a match
  %0 = linalg.fill ins(%cst : f32) outs(%arg2: tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: match linalg.matmul
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%0: tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>

// CHECK-LABEL: test_vnni_brgemm
func.func @test_vnni_brgemm(%arg0: tensor<48x32x16x2xbf16>,
                            %arg1: tensor<48x16x32x2xbf16>,
                            %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  // CHECK: match vnni.brgemm
  // CHECK-NOT: not a match
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<48x32x16x2xbf16>, tensor<48x16x32x2xbf16>)
    outs(%arg2 : tensor<32x32xbf16>) {
      ^bb0(%in: bf16, %in_8: bf16, %out: bf16):
        %11 = arith.mulf %in, %in_8 : bf16
        %12 = arith.addf %out, %11 : bf16
        linalg.yield %12 : bf16
  } -> tensor<32x32xbf16>
  return %0: tensor<32x32xbf16>
}

#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<() -> ()>

// CHECK-LABEL: test_tpp_add
func.func @test_tpp_add(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>,
                        %arg2: memref<32x32xf32>, %arg3: memref<1xf32>,
                        %arg4: memref<1xf32>, %arg5: memref<1xf32>,
                        %arg6: memref<f32>, %arg7: memref<f32>,
                        %arg8: memref<f32>,
                        %arg9: memref<32x32x32xf32>,
                        %arg10: memref<32x32x32xf32>,
                        %arg11: memref<32x32x32xf32>,
                        %arg12: memref<4x4xf32>,
                        %arg13: memref<4x4xf32, strided<[4, 1], offset: ?>>) {
  // CHECK-COUNT-4: match tpp.add
  // CHECK-NOT: not a match
  linalg.generic {
    indexing_maps = [#map3, #map3, #map3],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1: memref<32x32xf32>, memref<32x32xf32>)
    outs(%arg2: memref<32x32xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %1 = arith.addf %in, %in_1 : f32
        linalg.yield %1 : f32
  }
  linalg.generic {
    indexing_maps = [#map4, #map4, #map4],
    iterator_types = ["parallel"]}
    ins(%arg3, %arg4: memref<1xf32>, memref<1xf32>)
    outs(%arg5: memref<1xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %1 = arith.addf %in, %in_1 : f32
        linalg.yield %1 : f32
  }
  // Empty map is an identity. Note we don't match
  // this in the linalg to tpp conversion.
  linalg.generic {
    indexing_maps = [#map5, #map5, #map5],
    iterator_types = []}
    ins(%arg6, %arg7: memref<f32>, memref<f32>)
    outs(%arg8: memref<f32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %1 = arith.addf %in, %in_1 : f32
        linalg.yield %1 : f32
  }
  linalg.generic {
    indexing_maps = [#map3, #map3, #map3],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg13, %arg13: memref<4x4xf32, strided<[4, 1], offset: ?>>,
                        memref<4x4xf32, strided<[4, 1], offset: ?>>)
    outs(%arg12: memref<4x4xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %1 = arith.addf %in, %in_1 : f32
        linalg.yield %1 : f32
  }
  return
}

#map6 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0, d1) -> (0, d1)>
#map9 = affine_map<(d0, d1) -> (d0)>
#map10 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: tpp_add_must_not_match
func.func @tpp_add_must_not_match(%arg0: memref<3x3xf32>, %arg1: memref<1x3xf32>,
                                  %arg2: memref<3x3xf32>,
                                  %arg3: memref<32x32x32xf32>,
                                  %arg4: memref<32x32x32xf32>,
                                  %arg5: memref<32x32x32xf32>,
                                  %arg6: memref<3xf32>) {
  // CHECK-NOT: match tpp.add
  // CHECK-COUNT-5: not a match
  linalg.generic {
    indexing_maps = [#map7, #map8, #map7],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1: memref<3x3xf32>, memref<1x3xf32>)
    outs(%arg2: memref<3x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %1 = arith.addf %in, %in_1 : f32
        linalg.yield %1 : f32
  }
  linalg.generic {
    indexing_maps = [#map6, #map6, #map6],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg3, %arg4 : memref<32x32x32xf32>, memref<32x32x32xf32>)
    outs(%arg5: memref<32x32x32xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %1 = arith.addf %in, %in_1 : f32
        linalg.yield %1 : f32
  }
  linalg.generic {
    indexing_maps = [#map7, #map9, #map7],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg6 : memref<3x3xf32>, memref<3xf32>)
    outs(%arg2: memref<3x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %1 = arith.addf %in, %in_1 : f32
        linalg.yield %1 : f32
  }
  linalg.generic {
    indexing_maps = [#map7, #map10, #map7],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0: memref<3x3xf32>, memref<3x3xf32>)
    outs(%arg2: memref<3x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %1 = arith.addf %in, %in_1 : f32
        linalg.yield %1 : f32
  }
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  linalg.generic {
    indexing_maps = [#map7, #map7, #map7],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0: memref<3x3xf32>, memref<3x3xf32>)
    outs(%arg2: memref<3x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %0 = arith.addf %c0, %c1 : f32
        linalg.yield %0 : f32
  }
  return
}

// CHECK-LABEL: test_predicates
func.func @test_predicates(%arg0: memref<3x3xf32>) {
  // CHECK: match op with 1 or 2 inputs
  linalg.generic {
    indexing_maps = [#map7, #map7, #map7],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0: memref<3x3xf32>, memref<3x3xf32>)
    outs(%arg0: memref<3x3xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %0 = arith.addf %in, %in_1 : f32
        linalg.yield %0 : f32
  }
  // CHECK: match op with 1 or 2 inputs
  linalg.generic {
    indexing_maps = [#map7, #map7],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: memref<3x3xf32>)
    outs(%arg0: memref<3x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %in : f32
        linalg.yield %0 : f32
  }
  // CHECK: not a match
  linalg.generic {
    indexing_maps = [#map7],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg0: memref<3x3xf32>) {
      ^bb0(%out: f32):
        %0 = arith.addf %out, %out : f32
        linalg.yield %0 : f32
  }
  return
}

#map11 = affine_map<(d0, d1) -> (d0, d1)>
#map12 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: test_tpp_identity
func.func @test_tpp_identity(%arg0: memref<3xf32>, %arg1: memref<5x3xf32>) {
  // CHECK: match tpp.identity
  // CHECK-NOT: not a match
  linalg.generic {
    indexing_maps = [#map12, #map11],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: memref<3xf32>)
    outs(%arg1: memref<5x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
  }
  // CHECK: match tpp.identity
  // CHECK-NOT: not a match
  linalg.generic {
    indexing_maps = [#map11],
    iterator_types = ["parallel", "parallel"]}
    outs(%arg1: memref<5x3xf32>) {
      ^bb0(%out: f32):
        linalg.yield %out : f32
  }
  return
}

#map13 = affine_map<(d0, d1) -> (0)>
#map14 = affine_map<(d0, d1) -> (d0, d1)>
#map15 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map16 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: test_rank
func.func @test_rank(%arg0: memref<1xf32>,
                     %arg1: memref<4x4xf32>, %arg2: memref<4x4x4xf32>,
                     %arg3: f32) {
  // CHECK: match rank 1
  // CHECK-NOT: match rank 2
  // CHECK-NOT: match scalar
  linalg.generic {
    indexing_maps = [#map13, #map14],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0: memref<1xf32>)
    outs(%arg1: memref<4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  // CHECK: match rank 2
  // CHECK-NOT: match rank 1
  // CHECK-NOT: match scalar
  linalg.generic {
    indexing_maps = [#map14, #map14],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg1: memref<4x4xf32>)
    outs(%arg1: memref<4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  // CHECK: match scalar
  // CHECK-NOT: match rank 2
  // CHECK-NOT: match rank 1
  linalg.generic {
    indexing_maps = [#map16, #map14],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg3: f32)
    outs(%arg1: memref<4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  // CHECK-NOT: match scalar
  // CHECK-NOT: match rank 1
  // CHECK-NOT: match rank 2
  linalg.generic {
    indexing_maps = [#map15, #map15],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg2: memref<4x4x4xf32>)
    outs(%arg2: memref<4x4x4xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.addf %in, %out : f32
        linalg.yield %0 : f32
  }
  return
}

#map17 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map18 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map19 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>

// CHECK-LABEL: test_capture_affine_maps
func.func @test_capture_affine_maps(%arg0: tensor<4x16x32x32xf32>,
                                    %arg1: tensor<8x16x32x32xf32>,
                                    %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK-NOT: not a match
  // CHECK-DAG: match operation with affine map: (d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)
  // CHECK-DAG: match operation with affine map: (d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)
  // CHECK-DAG: match operation with affine map: (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)
  %0 = linalg.generic {
    indexing_maps = [#map17, #map18, #map19],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>)
    outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  } -> tensor<4x8x32x32xf32>
  return %0 : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: test_capture_affine_maps_expect_to_fail
func.func @test_capture_affine_maps_expect_to_fail(
                                    %arg0: tensor<4x16x32x32xf32>,
                                    %arg1: tensor<8x16x32x32xf32>,
                                    %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK-NOT: match operation with affine map: (d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)
  // CHECK-NOT: match operation with affine map: (d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)
  // CHECK-NOT: match operation with affine map: (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)
  // CHECK: not a match
  %0 = linalg.generic {
    indexing_maps = [#map17, #map18, #map19],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>)
    outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  } -> tensor<4x8x32x32xf32>
  return %0 : tensor<4x8x32x32xf32>
}

#map20 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: test_number_of_affine_maps
func.func @test_number_of_affine_maps(%arg0: tensor<4x16x32x32xf32>,
                                    %arg1: tensor<8x16x32x32xf32>,
                                    %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK: match
  %0 = linalg.generic {
    indexing_maps = [#map17, #map18, #map19],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>)
    outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  } -> tensor<4x8x32x32xf32>

  // CHECK: not a match
  %1 = linalg.generic {
    indexing_maps = [#map20, #map20],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%0 : tensor<4x8x32x32xf32>) outs(%0 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg5: f32, %arg6: f32):
      %10 = arith.addf %arg5, %arg6 : f32
      linalg.yield %10 : f32
  } -> tensor<4x8x32x32xf32>

  return %1 : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: test_capture_shape
func.func @test_capture_shape(%arg0: tensor<4x16x32x32xf32>,
                              %arg1: tensor<8x16x32x32xf32>,
                              %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK: Shape: 4, 16, 32, 32
  %0 = linalg.generic {
    indexing_maps = [#map17, #map18, #map19],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>)
    outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  } -> tensor<4x8x32x32xf32>
  return %0 : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: test_strides_tensor
func.func @test_strides_tensor(%arg0: tensor<4x16x32x32xf32>,
                               %arg1: tensor<8x16x32x32xf32>,
                               %arg2: tensor<4x8x32x32xf32>) -> tensor<4x8x32x32xf32> {
  // CHECK: Strides:
  %0 = linalg.generic {
    indexing_maps = [#map17, #map18, #map19],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<4x16x32x32xf32>, tensor<8x16x32x32xf32>)
    outs(%arg2 : tensor<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  } -> tensor<4x8x32x32xf32>
  return %0 : tensor<4x8x32x32xf32>
}

// CHECK-LABEL: test_strides_memref
func.func @test_strides_memref(%arg0: memref<4x16x32x32xf32>,
                               %arg1: memref<8x16x32x32xf32>,
                               %arg2: memref<4x8x32x32xf32>) {
  // CHECK: Strides: 16384, 1024, 32, 1
  linalg.generic {
    indexing_maps = [#map17, #map18, #map19],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : memref<4x16x32x32xf32>, memref<8x16x32x32xf32>)
    outs(%arg2 : memref<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  }
  return
}

// CHECK-LABEL: test_strides_dyn
func.func @test_strides_dyn(%arg0: memref<4x16x32x32xf32, strided<[?, ?, ?, ?], offset: ?>>,
                            %arg1: memref<8x16x32x32xf32>,
                            %arg2: memref<4x8x32x32xf32>) {
  // CHECK-NOT: Strides:
  linalg.generic {
    indexing_maps = [#map17, #map18, #map19],
    iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : memref<4x16x32x32xf32, strided<[?, ?, ?, ?], offset: ?>>, memref<8x16x32x32xf32>)
    outs(%arg2 : memref<4x8x32x32xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.mulf %arg3, %arg4 : f32
      %9 = arith.addf %arg5, %8 : f32
      linalg.yield %9 : f32
  }
  return
}
