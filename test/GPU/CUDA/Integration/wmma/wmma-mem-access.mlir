// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda -gpu-wmma -print \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

module attributes {gpu.container_module} {
  memref.global "private" @matrix_vals : memref<2x16x16xf16> = dense<[
    [
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    ],
    [
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    ]
  ]>

  func.func @entry(%arg0: memref<16x16xf16>) -> memref<16x16xf16> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %A = memref.get_global @matrix_vals : memref<2x16x16xf16>
    %B = memref.get_global @matrix_vals : memref<2x16x16xf16>

    %gpuA = gpu.alloc() : memref<2x16x16xf16>
    %gpuB = gpu.alloc() : memref<2x16x16xf16>

    %tOut = gpu.memcpy async %gpuA, %A : memref<2x16x16xf16>, memref<2x16x16xf16>
    gpu.wait [%tOut]
    %tOut1 = gpu.memcpy async %gpuB, %B : memref<2x16x16xf16>, memref<2x16x16xf16>
    gpu.wait [%tOut1]

    gpu.launch_func  @entry_kernel::@entry_kernel blocks in (%c1, %c1, %c1) threads in (%c32, %c1, %c1)
    args(%c1 : index, %c0 : index, %gpuA : memref<2x16x16xf16>, %gpuB : memref<2x16x16xf16>, %arg0 : memref<16x16xf16>)

    gpu.dealloc %gpuA : memref<2x16x16xf16>
    gpu.dealloc %gpuB : memref<2x16x16xf16>

    return %arg0 : memref<16x16xf16>
  }

  gpu.module @entry_kernel {
    gpu.func @entry_kernel(%arg0: index, %arg1: index, %arg2: memref<2x16x16xf16>, %arg3: memref<2x16x16xf16>, %arg4: memref<16x16xf16>) kernel
    attributes {gpu.known_block_size = array<i32: 32, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      %C = gpu.subgroup_mma_load_matrix %arg4[%c0, %c0]
              {leadDimension = 16 : index}
              : memref<16x16xf16>
              -> !gpu.mma_matrix<16x16xf16, "COp">
      %A = gpu.subgroup_mma_load_matrix %arg2[%c1, %c0, %c0]
                {leadDimension = 16 : index}
                : memref<2x16x16xf16>
                -> !gpu.mma_matrix<16x16xf16, "AOp">
      %B = gpu.subgroup_mma_load_matrix %arg3[%c1, %c0, %c0]
                {leadDimension = 16 : index}
                : memref<2x16x16xf16>
                -> !gpu.mma_matrix<16x16xf16, "BOp">

      %R = gpu.subgroup_mma_compute %A, %B, %C
            : !gpu.mma_matrix<16x16xf16, "AOp">,
              !gpu.mma_matrix<16x16xf16, "BOp">
            -> !gpu.mma_matrix<16x16xf16, "COp">

      gpu.subgroup_mma_store_matrix %R, %arg4[%c0, %c0]
        {leadDimension = 16 : index}
        : !gpu.mma_matrix<16x16xf16, "COp">,
          memref<16x16xf16>

      gpu.return
    }
  }
}

// CHECK-COUNT-16: ( 137, 273, 409, 545, 681, 817, 953, 1089, 1225, 1361, 1497, 1633, 1769, 1905, 2041, 2176 )
