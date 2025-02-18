// RUN: tpp-opt %s -pack-vnni | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
module {
  func.func @mlp(%arg0: tensor<32x64x4x4xbf16>, %arg1: tensor<128x64x4x4xbf16>, %arg2: tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16> {
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = bufferization.alloc_tensor() : tensor<32x128x4x4xbf16>
    %1 = scf.for %arg4 = %c0 to %c32 step %c1 iter_args(%arg5 = %0) -> (tensor<32x128x4x4xbf16>) {
      %2 = scf.for %arg6 = %c0 to %c128 step %c1 iter_args(%arg7 = %arg5) -> (tensor<32x128x4x4xbf16>) {
        %extracted_slice = tensor.extract_slice %arg0[%arg4, 0, 0, 0] [1, 64, 4, 4] [1, 1, 1, 1]
          : tensor<32x64x4x4xbf16> to tensor<64x4x4xbf16>
        %extracted_slice_0 = tensor.extract_slice %arg1[%arg6, 0, 0, 0] [1, 64, 4, 4] [1, 1, 1, 1]
          : tensor<128x64x4x4xbf16> to tensor<64x4x4xbf16>
        %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, %arg6, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1]
          : tensor<32x128x4x4xbf16> to tensor<4x4xbf16>
        %3 = linalg.generic {
          indexing_maps = [#map, #map1, #map2],
          iterator_types = ["reduction", "parallel", "parallel", "reduction"]}
          ins(%extracted_slice, %extracted_slice_0 : tensor<64x4x4xbf16>, tensor<64x4x4xbf16>)
          outs(%extracted_slice_1 : tensor<4x4xbf16>) {
        ^bb0(%in: bf16, %in_4: bf16, %out: bf16):
          %5 = arith.mulf %in, %in_4 : bf16
          %6 = arith.addf %out, %5 : bf16
          linalg.yield %6 : bf16
        } -> tensor<4x4xbf16>
        %inserted_slice = tensor.insert_slice %3 into %arg7[%arg4, %arg6, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<4x4xbf16> into tensor<32x128x4x4xbf16>
        scf.yield %inserted_slice : tensor<32x128x4x4xbf16>
      }
      scf.yield %2 : tensor<32x128x4x4xbf16>
    }
    return %1 : tensor<32x128x4x4xbf16>
  }
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d2, d4)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>

// CHECK: func.func @mlp(
// CHECK: %{{.+}}: tensor<32x64x4x4xbf16>,
// CHECK: %{{.+}}: tensor<128x64x4x4xbf16>,
// CHECK: %{{.+}}: tensor<32x128x4x4xbf16>) -> tensor<32x128x4x4xbf16> {
// CHECK: scf.for
// CHECK: scf.for
// CHECK:       %{{.+}} = linalg.pack
// CHECK:  %{{.+}} = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel", "reduction", "reduction"]
