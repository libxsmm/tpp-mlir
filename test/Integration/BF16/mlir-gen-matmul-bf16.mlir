// RUN: mlir-gen --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --vnni=0 2>&1 | FileCheck %s --check-prefix=BF16
// RUN: mlir-gen --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --vnni=2 2>&1 | FileCheck %s --check-prefix=DP2
// RUN: mlir-gen --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=DP4

// RUN: not --crash mlir-gen --output=named --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --vnni=2 2>&1 | FileCheck %s --check-prefix=VNNI-TODO
// RUN: mlir-gen --output=named --keep-generic-matmul --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --vnni=2 2>&1 | FileCheck %s --check-prefix=GENERIC

// BF16: // RUN{{.*}}tpp-run %s -n {{\d*}}
// BF16: // RUN{{.*}}-e entry -entry-point-result=void
// BF16: // BENCH_TOTAL_FLOPS: 452984832
// BF16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// BF16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// BF16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// BF16:     func.func @entry(%arg0: tensor<2x36x64x64xbf16>, %arg1: tensor<16x36x64x48xbf16>, %arg2: tensor<2x16x64x48xbf16>) -> tensor<2x16x64x48xbf16>
// BF16-NOT: alloc
// BF16:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// BF16:         arith.mulf
// BF16:         arith.addf
// BF16-NOT: dealloc

// DP2: // RUN{{.*}}tpp-run %s -n {{\d*}}
// DP2: // RUN{{.*}}-e entry -entry-point-result=void
// DP2: // BENCH_TOTAL_FLOPS: 452984832
// DP2-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// DP2-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// DP2-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// DP2:     func.func @entry(%arg0: tensor<2x36x64x64xbf16>, %arg1: tensor<16x36x32x48x2xbf16>, %arg2: tensor<2x16x64x48xbf16>) -> tensor<2x16x64x48xbf16>
// DP2-NOT: alloc
// DP2:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]
// DP2:         arith.mulf
// DP2:         arith.addf
// DP2-NOT: dealloc

// DP4: // RUN{{.*}}tpp-run %s -n {{\d*}}
// DP4: // RUN{{.*}}-e entry -entry-point-result=void
// DP4: // BENCH_TOTAL_FLOPS: 452984832
// DP4-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// DP4-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// DP4-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// DP4:     func.func @entry(%arg0: tensor<2x36x64x64xbf16>, %arg1: tensor<16x36x16x48x4xbf16>, %arg2: tensor<2x16x64x48xbf16>) -> tensor<2x16x64x48xbf16>
// DP4-NOT: alloc
// DP4:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]
// DP4:         arith.mulf
// DP4:         arith.addf
// DP4-NOT: dealloc


// GENERIC: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// GENERIC: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// GENERIC: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>

// GENERIC-LABEL:   func.func @entry(
// GENERIC-SAME:                     %[[VAL_0:.*]]: tensor<2x36x64x64xbf16>,
// GENERIC-SAME:                     %[[VAL_1:.*]]: tensor<16x36x32x48x2xbf16>,
// GENERIC-SAME:                     %[[VAL_2:.*]]: tensor<2x16x64x48xbf16>) -> tensor<2x16x64x48xbf16> {
// GENERIC:           %[[VNNI_A:.+]] = tensor.expand_shape %[[VAL_0]] {{\[}}[0], [1], [2], [3, 4]] output_shape [2, 36, 64, 32, 2] : tensor<2x36x64x64xbf16> into tensor<2x36x64x32x2xbf16>
// GENERIC:           %[[VAL_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%[[VNNI_A]], %[[VAL_1]] : tensor<2x36x64x32x2xbf16>, tensor<16x36x32x48x2xbf16>) outs(%[[VAL_2]] : tensor<2x16x64x48xbf16>) {
// GENERIC:           ^bb0(%[[VAL_4:.*]]: bf16, %[[VAL_5:.*]]: bf16, %[[VAL_6:.*]]: bf16):
// GENERIC:             %[[VAL_7:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : bf16
// GENERIC:             %[[VAL_8:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : bf16
// GENERIC:             linalg.yield %[[VAL_8]] : bf16
// GENERIC:           } -> tensor<2x16x64x48xbf16>
// GENERIC:           return %[[VAL_3]] : tensor<2x16x64x48xbf16>
// GENERIC:         }

// VNNI-TODO: Unsupported Lowering for VNNI, Try '--keep-generic-matmul'
// VNNI-TODO: UNREACHABLE executed
