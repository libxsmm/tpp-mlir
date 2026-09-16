// RUN: tpp-opt %s --gemm-k-cache-blocking="k-cache-blocking=32" --split-input-file | FileCheck %s
// RUN: tpp-opt %s --gemm-k-cache-blocking="k-cache-blocking=0" --split-input-file | FileCheck -check-prefix=DISABLED %s

// Cache-block the batch-reduce (K) dimension of a tensor brgemm by 32.
module {
  func.func @brgemm_k_cache_blocking(%arg0: tensor<128x256x512xf32>, %arg1: tensor<128x512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<128x256x512xf32>, tensor<128x512x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}

// CHECK-LABEL: func.func @brgemm_k_cache_blocking
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// The batch-reduce (K) dimension is tiled by 32, threading the accumulator.
// CHECK: scf.for %[[I:.+]] = %[[C0]] to %[[C128]] step %[[C32]] iter_args(%[[ACC:.+]] = %arg2) -> (tensor<256x256xf32>)
// CHECK:   tensor.extract_slice %arg0[%[[I]], 0, 0] [32, 256, 512]
// CHECK:   tensor.extract_slice %arg1[%[[I]], 0, 0] [32, 512, 256]
// CHECK:   %[[MM:.+]] = linalg.batch_reduce_matmul ins({{.*}} : tensor<32x256x512xf32>, tensor<32x512x256xf32>) outs({{.*}} : tensor<256x256xf32>)
// CHECK:   scf.yield

// A tile size of 0 disables the pass; the op is left unchanged.
// DISABLED-LABEL: func.func @brgemm_k_cache_blocking
// DISABLED-NOT: scf.for
// DISABLED: linalg.batch_reduce_matmul ins(%arg0, %arg1

// -----

// The tile must strictly divide the batch-reduce extent; otherwise no tiling.
module {
  func.func @brgemm_no_divide(%arg0: tensor<48x256x512xf32>, %arg1: tensor<48x512x256xf32>, %arg2: tensor<256x256xf32>) -> tensor<256x256xf32> {
    %0 = linalg.batch_reduce_matmul ins(%arg0, %arg1 : tensor<48x256x512xf32>, tensor<48x512x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}

// 32 does not divide the batch-reduce extent 48, so the op is unchanged.
// CHECK-LABEL: func.func @brgemm_no_divide
// CHECK-NOT: scf.for

// -----

// An M-cache-panel (leading parallel dim > 1) combined with a k-cache-block
// that covers all K-blocks must still split the panel into register tiles.
// Previously the pass bailed on the K-block guard, leaving a panelized generic
// the AMX/nano path could not lower due to surviving vector.contract.
#mapA = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5, d2)>
#mapB = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d5, d4, d2)>
#mapC = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d3, d4)>
module {
  func.func @panel_full_kblock(%A: tensor<2x16x32x16x2xbf16>, %B: tensor<16x16x32x2xbf16>, %C: tensor<2x32x32xf32>) -> tensor<2x32x32xf32> {
    %0 = linalg.generic {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]} ins(%A, %B : tensor<2x16x32x16x2xbf16>, tensor<16x16x32x2xbf16>) outs(%C : tensor<2x32x32xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_0 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<2x32x32xf32>
    return %0 : tensor<2x32x32xf32>
  }
}

// The k-cache-block (32, clamped to the 16 K-blocks) yields a single K-loop
// iteration, and the panel dim (2) is split into an inner register-tile loop
// producing a canonical 32x32 tile generic.
// CHECK-LABEL: func.func @panel_full_kblock
// CHECK: scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %arg2) -> (tensor<2x32x32xf32>)
// CHECK:   scf.for %[[P:.+]] = %{{.+}} to %{{.+}} step %{{.+}} iter_args
// CHECK:     tensor.extract_slice %{{.+}}[%[[P]], 0, 0] [1, 32, 32]
// CHECK:     linalg.generic
// CHECK-SAME: outs({{.*}} : tensor<32x32xf32>)
// CHECK:     tensor.insert_slice

// A tile size of 0 disables the pass; the panelized generic is left unchanged.
// DISABLED-LABEL: func.func @panel_full_kblock
// DISABLED-NOT: scf.for
// DISABLED: linalg.generic
