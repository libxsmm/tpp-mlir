// RUN: tpp-opt %s -tile-consumer-and-fuse-producers="use-for-all=false tile-sizes=32,32 min-tile-factor=2" -cse -split-input-file | FileCheck %s

func.func @tile_factor_half(%arg0: tensor<16x16xf32>,
                 %arg1: tensor<16x16xf32>,
                 %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<16x16xf32>, tensor<16x16xf32>)
                     outs(%arg2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @tile_factor_half
// CHECK-NOT: scf.for
// CHECK: linalg.matmul{{.*}}-> tensor<16x16xf32>

// -----

func.func @tile_factor_1(%arg0: tensor<32x32xf32>,
                 %arg1: tensor<32x32xf32>,
                 %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>)
                     outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @tile_factor_1
// CHECK-NOT: scf.for
// CHECK: linalg.matmul{{.*}}-> tensor<32x32xf32>

// -----

func.func @tile_factor_2(%arg0: tensor<64x64xf32>,
                 %arg1: tensor<64x64xf32>,
                 %arg2: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%arg2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// CHECK-LABEL: func.func @tile_factor_2
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[c32:.+]] = arith.constant 32 : index
// CHECK-COUNT-2: scf.for {{.*}} = %[[c0]] to %[[c64]] step %[[c32]]
// CHECK: linalg.matmul ins({{.*}}: tensor<32x64xf32>, tensor<64x32xf32>){{.*}}-> tensor<32x32xf32>

// -----

func.func @tile_factor_4(%arg0: tensor<128x128xf32>,
                 %arg1: tensor<128x128xf32>,
                 %arg2: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2 : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %0 : tensor<128x128xf32>
}

// CHECK-LABEL: func.func @tile_factor_4
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[c32:.+]] = arith.constant 32 : index
// CHECK-COUNT-2: scf.for {{.*}} = %[[c0]] to %[[c128]] step %[[c32]]
// CHECK: linalg.matmul ins({{.*}}: tensor<32x128xf32>, tensor<128x32xf32>){{.*}}-> tensor<32x32xf32>

// -----

func.func @tile_factor_1_by_2(%arg0: tensor<32x64xf32>,
                 %arg1: tensor<64x64xf32>,
                 %arg2: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<32x64xf32>, tensor<64x64xf32>)
                     outs(%arg2 : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %0 : tensor<32x64xf32>
}

// CHECK-LABEL: func.func @tile_factor_1_by_2
// CHECK-NOT: scf.for
// CHECK: linalg.matmul{{.*}}-> tensor<32x64xf32>

// -----

func.func @tile_factor_2_by_4(%arg0: tensor<64x128xf32>,
                 %arg1: tensor<128x128xf32>,
                 %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x128xf32>)
                     outs(%arg2 : tensor<64x128xf32>) -> tensor<64x128xf32>
  return %0 : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @tile_factor_2_by_4
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[c32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[c128:.+]] = arith.constant 128 : index
// CHECK: scf.for {{.*}} = %[[c0]] to %[[c64]] step %[[c32]]
// CHECK: scf.for {{.*}} = %[[c0]] to %[[c128]] step %[[c32]]
// CHECK: linalg.matmul ins({{.*}}: tensor<32x128xf32>, tensor<128x32xf32>){{.*}}-> tensor<32x32xf32>
