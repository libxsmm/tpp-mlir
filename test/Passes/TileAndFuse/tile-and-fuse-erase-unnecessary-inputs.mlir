// RUN: tpp-opt %s -tile-consumer-and-fuse-producers -split-input-file | FileCheck %s

// Inputs of an elementwise linalg.generic that alias its init are erased
// (linalg::populateEraseUnnecessaryInputsPatterns) before tiling and fusion.

// An input that is the same value as the init with the same indexing
// map (bias-add epilogue out = f(acc, out)) is read through the init block
// argument and dropped, so the tiled op updates the output in place.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @erase_input_same_as_init(%acc: tensor<32x64xf32>,
                                    %out: tensor<32x64xbf16>) -> tensor<32x64xbf16> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%acc, %out : tensor<32x64xf32>, tensor<32x64xbf16>)
    outs(%out : tensor<32x64xbf16>) {
  ^bb0(%a: f32, %b: bf16, %o: bf16):
    %e = arith.extf %b : bf16 to f32
    %s = arith.addf %a, %e : f32
    %t = arith.truncf %s : f32 to bf16
    linalg.yield %t : bf16
  } -> tensor<32x64xbf16>
  return %0 : tensor<32x64xbf16>
}

// CHECK-LABEL: func.func @erase_input_same_as_init
// CHECK-SAME:    %[[ACC:.+]]: tensor<32x64xf32>, %[[OUT:.+]]: tensor<32x64xbf16>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[ACC]] : tensor<32x64xf32>) outs(%[[OUT]] : tensor<32x64xbf16>)
// CHECK-NEXT:    ^bb0(%[[A:.+]]: f32, %[[O:.+]]: bf16):
// CHECK-NEXT:      %[[E:.+]] = arith.extf %[[O]] : bf16 to f32
// CHECK-NEXT:      arith.addf %[[A]], %[[E]] : f32

// -----

// Negative: the input reads the init through a transposed map, i.e., a
// different element than the one being updated, so it is kept.

#map = affine_map<(d0, d1) -> (d0, d1)>
#mapT = affine_map<(d0, d1) -> (d1, d0)>

func.func @keep_input_transposed_init(%acc: tensor<64x64xf32>,
                                      %out: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #mapT, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%acc, %out : tensor<64x64xf32>, tensor<64x64xf32>)
    outs(%out : tensor<64x64xf32>) {
  ^bb0(%a: f32, %b: f32, %o: f32):
    %s = arith.addf %a, %b : f32
    linalg.yield %s : f32
  } -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @keep_input_transposed_init
// CHECK-SAME:    %[[ACC:.+]]: tensor<64x64xf32>, %[[OUT:.+]]: tensor<64x64xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[ACC]], %[[OUT]] : tensor<64x64xf32>, tensor<64x64xf32>) outs(%[[OUT]] : tensor<64x64xf32>)
// CHECK-NEXT:    ^bb0(%[[A:.+]]: f32, %[[B:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT:      arith.addf %[[A]], %[[B]] : f32

// -----

// The only input is the init: it is dropped too, leaving an op with no
// inputs that reads the init block argument.

#map = affine_map<(d0, d1) -> (d0, d1)>

func.func @erase_only_input_same_as_init(%out: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel", "parallel"]
  } ins(%out : tensor<32x64xf32>) outs(%out : tensor<32x64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %m = arith.maximumf %in, %cst : f32
    linalg.yield %m : f32
  } -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// CHECK-LABEL: func.func @erase_only_input_same_as_init
// CHECK-SAME:    %[[OUT:.+]]: tensor<32x64xf32>
// CHECK:         linalg.generic
// CHECK-NOT:       ins(
// CHECK-SAME:      outs(%[[OUT]] : tensor<32x64xf32>)
// CHECK-NEXT:    ^bb0(%[[O:.+]]: f32):
// CHECK-NEXT:      arith.maximumf %[[O]], %{{.+}} : f32
