// RUN: mlir-gen --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --vnni=0 2>&1 | FileCheck %s --check-prefix=BF16
// RUN: mlir-gen --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --vnni=2 2>&1 | FileCheck %s --check-prefix=DP2
// RUN: mlir-gen --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=DP4

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
// DP2-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
// DP2-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 2, d5, d3)>
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
// DP4-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
// DP4-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6 floordiv 4, d5, d3)>
// DP4-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// DP4:     func.func @entry(%arg0: tensor<2x36x64x64xbf16>, %arg1: tensor<16x36x16x48x4xbf16>, %arg2: tensor<2x16x64x48xbf16>) -> tensor<2x16x64x48xbf16>
// DP4-NOT: alloc
// DP4:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]
// DP4:         arith.mulf
// DP4:         arith.addf
// DP4-NOT: dealloc
