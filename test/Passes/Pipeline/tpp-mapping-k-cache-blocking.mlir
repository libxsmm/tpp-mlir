// RUN: tpp-opt %s -tpp-mapping="k-cache-blocking=2" -split-input-file | FileCheck %s
// RUN: tpp-opt %s -tpp-mapping -split-input-file | FileCheck -check-prefix=OFF %s

// With k-cache-blocking, routing is (M/N cache tile + epilogue fuse) then
// (K cache-block tiling). The K-block scf.for threads the high-precision C
// accumulator across K-blocks and is written once, with the spatial
// register-tile loops nested inside. For plain -tpp-mapping (no --def-parallel)
// the spatial loops are scf.for; with --def-parallel + nano the outer spatial
// loop is an scf.forall over cache blocks and the K-block loop nests inside
// each block's small f32 patch.
func.func @matmul(%A: tensor<128x256xf32>, %B: tensor<256x128xf32>,
                  %C: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %D = linalg.matmul ins(%A, %B: tensor<128x256xf32>, tensor<256x128xf32>)
                     outs(%C: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %D : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @matmul
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
// The K-block loop threads the packed C accumulator across K-blocks (step 2).
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C8]] step %[[C2]] iter_args({{.*}}) -> (tensor<4x4x32x32xf32>)
// A and B are sliced to the K-block extent (2), not the full 8.
// CHECK:   tensor.extract_slice %{{.*}} [4, 2, 32, 32]
// CHECK:   tensor.extract_slice %{{.*}} [4, 2, 32, 32]
// The spatial register-tile loops nest inside the K-block loop.
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       tensor.extract_slice %{{.*}} [1, 2, 32, 32]
// CHECK:       linalg.generic
// CHECK:         arith.mulf
// CHECK:         arith.addf

// Without k-cache-blocking, there is no enclosing reduction scf.for threading C.
// OFF-LABEL: func.func @matmul
// OFF-NOT: scf.for {{.*}}iter_args
