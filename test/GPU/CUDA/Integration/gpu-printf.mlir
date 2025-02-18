// RUN: ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:${ASAN_OPTIONS} \
// RUN: tpp-run %s -gpu=cuda \
// RUN:  -entry-point-result=void -e entry 2>&1 | \
// RUN: FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @hello() kernel {
      %0 = gpu.thread_id x
      %csti8 = arith.constant 2 : i8
      %cstf32 = arith.constant 3.0 : f32
      gpu.printf "Hello from %lld, %d, %f\n", %0, %csti8, %cstf32  : index, i8, f32
      gpu.return
    }
  }

  func.func @entry() {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    gpu.launch_func @kernels::@hello
      blocks in (%c1, %c1, %c1)
      threads in (%c2, %c1, %c1)
    return
  }
}

// CHECK: Hello from 0, 2, 3.000000
// CHECK: Hello from 1, 2, 3.000000
