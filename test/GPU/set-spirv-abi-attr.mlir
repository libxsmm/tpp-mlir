// RUN: tpp-opt %s -set-spirv-abi-attr=client-api=opencl | \
// RUN: FileCheck %s --check-prefix=OPENCL

module attributes {gpu.container_module} {
  func.func @entry(%arg0: memref<32x32xf32>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c8, %c8, %c1) threads in (%c4, %c4, %c1)
      args(%arg0 : memref<32x32xf32>)
    return
  }
  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: memref<32x32xf32>)
    kernel attributes {known_block_size = array<i32: 4, 4, 1>, known_grid_size = array<i32: 8, 8, 1>} {
      %b0 = gpu.block_id  x
      %b1 = gpu.block_id  y
      %t0 = gpu.thread_id  x
      %t1 = gpu.thread_id  y
      gpu.printf "Block (%lld, %lld, 1) - Thread (%lld, %lld, 1)\n", %b0, %b1, %t0, %t1  : index, index, index, index
      gpu.return
    }
  }
}

// OPENCL: gpu.func @entry_kernel(%arg0: memref<32x32xf32>)
// OPENCL-SAME: kernel attributes {{{.*}}spirv.entry_point_abi = #spirv.entry_point_abi<>
