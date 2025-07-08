// RUN: tpp-opt %s -x86-vectorizer -split-input-file | FileCheck %s

func.func @gemm_fma(
  %arg0: tensor<256x512xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<256x128xf32>)
    -> tensor<256x128xf32>
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>,
      #dlti.dl_entry<"reg_gemm_unroll", [1, 32, 1]>
  >>}
{
  %0 = linalg.matmul
    ins(%arg0, %arg1: tensor<256x512xf32>, tensor<512x128xf32>)
    outs(%arg2: tensor<256x128xf32>)
    -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// CHECK-LABEL: @gemm_fma(
// CHECK-SAME:  %[[A:.+]]: tensor<256x512xf32>
// CHECK-SAME:  %[[B:.+]]: tensor<512x128xf32>
// CHECK-SAME:  %[[C:.+]]: tensor<256x128xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG: %[[C256:.+]] = arith.constant 256 : index
// CHECK-DAG: %[[C512:.+]] = arith.constant 512 : index
// Matrix C block loops for M and N dims
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C256]] step %[[C2]]
// CHECK:           scf.for %{{.+}} = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK-COUNT-1:     vector.transfer_read
// Reduction loop for K dim
// CHECK:      scf.for %{{.+}} = %[[C0]] to %[[C512]] step %[[C1]]
// CHECK:        %[[eleA0:.+]] = vector.transfer_read %[[A]]
// CHECK-SAME:    : tensor<256x512xf32>, vector<1xf32>
// CHECK:        %[[eleA1:.+]] = vector.transfer_read %[[A]]
// CHECK-SAME:    : tensor<256x512xf32>, vector<1xf32>
// CHECK:        %[[vecB:.+]] = vector.transfer_read %[[B]]
// CHECK-SAME:    : tensor<512x128xf32>, vector<32xf32>
// CHECK:        %[[vecA0:.+]] = vector.broadcast %[[eleA0]]
// CHECK-SAME:    : vector<1xf32> to vector<32xf32>
// CHECK:        %[[fma0:.+]] = vector.fma %[[vecA0]], %[[vecB]]{{.*}}: vector<32xf32>
// CHECK:        %[[vecA1:.+]] = vector.broadcast %[[eleA1]]
// CHECK-SAME:    : vector<1xf32> to vector<32xf32>
// CHECK:        %[[fma1:.+]] = vector.fma %[[vecA1]], %[[vecB]]{{.*}}: vector<32xf32>
// CHECK:        %[[insert0:.+]] = vector.insert_strided_slice %[[fma0]]
// CHECK:        %[[insert1:.+]] = vector.insert_strided_slice %[[fma1]], %[[insert0]]
// CHECK:        scf.yield %[[insert1]] : vector<2x32xf32>
// Store results
// CHECK-COUNT-1: vector.transfer_write
// CHECK-NOT: vector.transfer_read
// CHECK-NOT: vector.transfer_write

// -----

func.func @gemm_fma_remainder_block(
  %arg0: tensor<62x512xf32>, %arg1: tensor<512x128xf32>, %arg2: tensor<62x128xf32>)
    -> tensor<62x128xf32>
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [3, 32, 1]>,
      #dlti.dl_entry<"reg_gemm_unroll", [1, 32, 1]>
  >>}
{
  %0 = linalg.matmul
    ins(%arg0, %arg1: tensor<62x512xf32>, tensor<512x128xf32>)
    outs(%arg2: tensor<62x128xf32>)
    -> tensor<62x128xf32>
  return %0 : tensor<62x128xf32>
}

// CHECK-LABEL: @gemm_fma_remainder_block(
// CHECK-SAME:  %[[A:.+]]: tensor<62x512xf32>
// CHECK-SAME:  %[[B:.+]]: tensor<512x128xf32>
// CHECK-SAME:  %[[C:.+]]: tensor<62x128xf32>
// Main matrix C blocking loops
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
// CHECK-DAG: %[[C60:.+]] = arith.constant 60 : index
// CHECK-DAG: %[[C128:.+]] = arith.constant 128 : index
// Matrix C block loops for M and N dims
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C60]] step %[[C3]]
// CHECK:           scf.for %{{.+}} = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK-COUNT-1:     vector.transfer_read
// Reduction loop for K dim
// CHECK:         scf.for
// CHECK-COUNT-4:   vector.transfer_read
// CHECK-COUNT-3:   vector.fma
// CHECK-COUNT-3:   vector.insert_strided_slice
// CHECK:           scf.yield
// CHEC
// CHECK-COUNT-1: vector.transfer_write
// CHECK-COUNT-2: scf.yield
// Tail block loops
// Peeled M dim block offsets are statically know and don't require a loop.
// Thus, one on for loop for the N dim is present here.
// CHECK: scf.for
// CHECK-COUNT-1: vector.transfer_read
// CHECK:         scf.for
// CHECK-COUNT-3:   vector.transfer_read
// CHECK-COUNT-2:   vector.fma
// CHECK-COUNT-2:   vector.insert_strided_slice
// CHECK:           scf.yield
// CHECK-COUNT-1: vector.transfer_write
// CHECK-COUNT-1: scf.yield

// -----

func.func @batch_gemm_fma(
  %arg0: tensor<3x256x512xf32>, %arg1: tensor<3x512x128xf32>, %arg2: tensor<3x256x128xf32>)
    -> tensor<3x256x128xf32>
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>,
      #dlti.dl_entry<"reg_gemm_unroll", [1, 32, 1]>
  >>}
{
  %0 = linalg.batch_matmul
    ins(%arg0, %arg1: tensor<3x256x512xf32>, tensor<3x512x128xf32>)
    outs(%arg2: tensor<3x256x128xf32>)
    -> tensor<3x256x128xf32>
  return %0 : tensor<3x256x128xf32>
}

// CHECK-LABEL: @batch_gemm_fma(
// CHECK-SAME:  %[[A:.+]]: tensor<3x256x512xf32>
// CHECK-SAME:  %[[B:.+]]: tensor<3x512x128xf32>
// CHECK-SAME:  %[[C:.+]]: tensor<3x256x128xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// Outermost loop for batch dim
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C3]] step %[[C1]]
// Remaining blocking loops
// CHECK-COUNT-2: scf.for
// CHECK-COUNT-1: vector.transfer_read
// CHECK: scf.for
// CHECK-COUNT-3: vector.transfer_read
// CHECK: vector.broadcast
// CHECK: vector.fma
// CHECK: vector.broadcast
// CHECK: vector.fma
// CHECK-COUNT-2: vector.insert_strided_slice
// CHECK scf.yield
// CHECK-COUNT-1: vector.transfer_write

// -----

func.func @brgemm_fma(
  %arg0: tensor<3x256x512xf32>, %arg1: tensor<3x512x128xf32>, %arg2: tensor<256x128xf32>)
    -> tensor<256x128xf32>
  attributes {
    dlti.target_system_spec = #dlti.target_system_spec<
    "CPU" = #dlti.target_device_spec<
      #dlti.dl_entry<"reg_blocks", [2, 32, 1]>,
      #dlti.dl_entry<"reg_gemm_unroll", [1, 32, 1]>
  >>}
{
  %0 = linalg.batch_reduce_matmul
    ins(%arg0, %arg1: tensor<3x256x512xf32>, tensor<3x512x128xf32>)
    outs(%arg2: tensor<256x128xf32>)
    -> tensor<256x128xf32>
  return %0 : tensor<256x128xf32>
}

// CHECK-LABEL: @brgemm_fma(
// CHECK-SAME:  %[[A:.+]]: tensor<3x256x512xf32>
// CHECK-SAME:  %[[B:.+]]: tensor<3x512x128xf32>
// CHECK-SAME:  %[[C:.+]]: tensor<256x128xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// Matrix C blocking loops
// CHECK-COUNT-2: scf.for
// CHECK-COUNT-1: vector.transfer_read
// Reduction loop for batch reduce dim
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C3]] step %[[C1]]
// Reduction loop for K dim
// CHECK: scf.for
// CHECK-COUNT-3: vector.transfer_read
// CHECK: vector.broadcast
// CHECK: vector.fma
// CHECK: vector.broadcast
// CHECK: vector.fma
// CHECK-COUNT-2: vector.insert_strided_slice
// CHECK-COUNT-2: scf.yield
// CHECK-COUNT-1: vector.transfer_write
