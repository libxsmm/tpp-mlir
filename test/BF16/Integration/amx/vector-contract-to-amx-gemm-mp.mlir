// RUN: tpp-run -e optimal_blocking --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-opt %s  --tile-brgemm-linalg="registerBlocking=48,16,32"  --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-amx | tpp-run -e optimal_blocking --entry-point-result=void -print  --splat-to-random --init-type normal  -seed 123  > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

func.func @optimal_blocking(%arg0: memref<1x48x16x2xbf16>, %arg1: memref<1x16x16x2xbf16>, %arg2: memref<48x16xf32>) -> memref<48x16xf32> {
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<1x48x16x2xbf16>, memref<1x16x16x2xbf16>) outs(%arg2 : memref<48x16xf32>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: f32):
      %a = arith.extf %in : bf16 to f32
      %b = arith.extf %in_1 : bf16 to f32
      %1 = arith.mulf %a, %b : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
  return %arg2 : memref<48x16xf32>
}


// RUN: tpp-run -e optimal_blocking_1x3 --entry-point-result=void -print --splat-to-random --init-type normal  -seed 123  %s > %t.1
// RUN: tpp-opt %s  --tile-brgemm-linalg="registerBlocking=16,48,32"  --loop-invariant-code-motion --vectorization-pass --hoist-vector-transfer --vector-contract-to-amx | tpp-run -e optimal_blocking_1x3 --entry-point-result=void -print  --splat-to-random --init-type normal  -seed 123  > %t.2
// RUN: fpcmp -r 0.001 %t.1 %t.2

func.func @optimal_blocking_1x3(%arg0: memref<1x16x16x2xbf16>, %arg1: memref<1x16x48x2xbf16>, %arg2: memref<16x48xf32>) -> memref<16x48xf32> {
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3, d1)>, affine_map<(d0, d1, d2, d3, d4) -> (d2, d3)>], iterator_types = ["reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<1x16x16x2xbf16>, memref<1x16x48x2xbf16>) outs(%arg2 : memref<16x48xf32>) {
    ^bb0(%in: bf16, %in_1: bf16, %out: f32):
      %a = arith.extf %in : bf16 to f32
      %b = arith.extf %in_1 : bf16 to f32
      %1 = arith.mulf %a, %b : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    }
  return %arg2 : memref<16x48xf32>
}
