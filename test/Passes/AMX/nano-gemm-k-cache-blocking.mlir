// RUN: tpp-opt %s --nano-gemm-k-cache-blocking="k-cache-tile=2" --split-input-file | FileCheck %s
// RUN: tpp-opt %s --nano-gemm-k-cache-blocking="k-cache-tile=0" --split-input-file | FileCheck -check-prefix=DISABLED %s

// A lowered nano (AMX) GEMM tile: a batch-reduce reduction loop carrying an AMX
// accumulator, stored to an f32 scratch, a bias-add loop, and a truncf store to
// the bf16 output tile. K-cache-blocking by 2 (batch-reduce extent 4) must hoist
// an outer K-block loop around the forall and accumulate into C in place.
func.func @nano_gemm(%A: memref<4x16x32xbf16>, %B: memref<4x16x32xbf16>,
                     %C: memref<2x2x16x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %zero = memref.alloca() : memref<16x16xf32>
  scf.forall (%t) in (4) {
    %csub = memref.subview %C[0, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1]
      : memref<2x2x16x16xbf16> to memref<16x16xbf16, strided<[16, 1]>>
    %z = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
    %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %z)
        -> (!x86.amx.tile<16x16xf32>) {
      %la = x86.amx.tile_load %A[%k, %c0, %c0]
        : memref<4x16x32xbf16> into !x86.amx.tile<16x32xbf16>
      %lb = x86.amx.tile_load %B[%k, %c0, %c0]
        : memref<4x16x32xbf16> into !x86.amx.tile<16x32xbf16>
      %m = x86.amx.tile_mulf %la, %lb, %acc
        : !x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x16xf32>
      scf.yield %m : !x86.amx.tile<16x16xf32>
    }
    %part = memref.alloca() : memref<16x16xf32>
    x86.amx.tile_store %part[%c0, %c0], %r
      : memref<16x16xf32>, !x86.amx.tile<16x16xf32>
    scf.for %row = %c0 to %c16 step %c1 {
      %p = vector.load %part[%row, %c0] : memref<16x16xf32>, vector<16xf32>
      %b = vector.load %zero[%row, %c0] : memref<16x16xf32>, vector<16xf32>
      %s = arith.addf %p, %b : vector<16xf32>
      vector.store %s, %part[%row, %c0] : memref<16x16xf32>, vector<16xf32>
    }
    scf.for %row = %c0 to %c16 step %c1 {
      %p = vector.load %part[%row, %c0] : memref<16x16xf32>, vector<16xf32>
      %tr = arith.truncf %p : vector<16xf32> to vector<16xbf16>
      vector.store %tr, %csub[%row, %c0]
        : memref<16x16xbf16, strided<[16, 1]>>, vector<16xbf16>
    }
  }
  return
}

// The reduction loop runs over the full extent [0, 4); an outer K-block loop
// with step 2 wraps the forall, the reduction is restricted to the block, and
// the (previously zero) bias is re-sourced from C via extf/select (beta=1).
// CHECK-LABEL: func.func @nano_gemm
// CHECK: scf.for %[[KB:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK:   scf.forall
// CHECK:     %[[END:.+]] = arith.addi %[[KB]], %{{.+}}
// CHECK:     scf.for %{{.+}} = %[[KB]] to %[[END]] step
// CHECK:       x86.amx.tile_mulf
// CHECK:     x86.amx.tile_store
// CHECK:     %[[FIRST:.+]] = arith.cmpi eq, %[[KB]], %{{.+}}
// CHECK:     scf.for
// CHECK:       %[[CV:.+]] = vector.load %{{.+}} : memref<16x16xbf16, strided<[16, 1]>>, vector<16xbf16>
// CHECK:       %[[EXT:.+]] = arith.extf %[[CV]]
// CHECK:       arith.select %[[FIRST]], %{{.+}}, %[[EXT]]

// A tile size of 0 disables the pass; no K-block loop or beta=1 rewrite.
// DISABLED-LABEL: func.func @nano_gemm
// DISABLED-NOT: arith.select
// DISABLED-NOT: arith.cmpi

// -----

// The block factor must divide the batch-reduce extent; extent 3 is not
// divisible by 2, so the tile is left unchanged (no beta=1 rewrite).
func.func @nano_gemm_no_divide(%A: memref<3x16x32xbf16>, %B: memref<3x16x32xbf16>,
                               %C: memref<2x2x16x16xbf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c16 = arith.constant 16 : index
  %zero = memref.alloca() : memref<16x16xf32>
  scf.forall (%t) in (4) {
    %csub = memref.subview %C[0, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1]
      : memref<2x2x16x16xbf16> to memref<16x16xbf16, strided<[16, 1]>>
    %z = x86.amx.tile_zero : !x86.amx.tile<16x16xf32>
    %r = scf.for %k = %c0 to %c3 step %c1 iter_args(%acc = %z)
        -> (!x86.amx.tile<16x16xf32>) {
      %la = x86.amx.tile_load %A[%k, %c0, %c0]
        : memref<3x16x32xbf16> into !x86.amx.tile<16x32xbf16>
      %lb = x86.amx.tile_load %B[%k, %c0, %c0]
        : memref<3x16x32xbf16> into !x86.amx.tile<16x32xbf16>
      %m = x86.amx.tile_mulf %la, %lb, %acc
        : !x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x32xbf16>, !x86.amx.tile<16x16xf32>
      scf.yield %m : !x86.amx.tile<16x16xf32>
    }
    %part = memref.alloca() : memref<16x16xf32>
    x86.amx.tile_store %part[%c0, %c0], %r
      : memref<16x16xf32>, !x86.amx.tile<16x16xf32>
    scf.for %row = %c0 to %c16 step %c1 {
      %p = vector.load %part[%row, %c0] : memref<16x16xf32>, vector<16xf32>
      %b = vector.load %zero[%row, %c0] : memref<16x16xf32>, vector<16xf32>
      %s = arith.addf %p, %b : vector<16xf32>
      vector.store %s, %part[%row, %c0] : memref<16x16xf32>, vector<16xf32>
    }
    scf.for %row = %c0 to %c16 step %c1 {
      %p = vector.load %part[%row, %c0] : memref<16x16xf32>, vector<16xf32>
      %tr = arith.truncf %p : vector<16xf32> to vector<16xbf16>
      vector.store %tr, %csub[%row, %c0]
        : memref<16x16xbf16, strided<[16, 1]>>, vector<16xbf16>
    }
  }
  return
}

// CHECK-LABEL: func.func @nano_gemm_no_divide
// CHECK-NOT: arith.select
// CHECK-NOT: arith.cmpi

// -----

// A lowered quantized nano (AMX) GEMM tile: an i8 batch-reduce reduction into a
// wide i32 accumulator, an addi bias-add loop against a zero i32 bias, and a
// requant loop (sitofp, clamp, fptosi) storing the i8 output tile. Because the
// i8 C cannot carry the wide partial and requant is non-linear, K-cache-blocking
// must thread the running sum through a dedicated i32 carrier (beta=1) and let
// the last K-block requantize the full reduction.
func.func @nano_gemm_quant(%A: memref<4x16x64xi8>, %B: memref<4x16x64xi8>,
                           %C: memref<2x2x16x16xi8>,
                           %iScale: memref<16xf32>, %wScale: memref<16xf32>,
                           %oScale: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %lo = arith.constant dense<-1.280000e+02> : vector<16xf32>
  %hi = arith.constant dense<1.270000e+02> : vector<16xf32>
  %zero = memref.alloca() : memref<16x16xi32>
  scf.forall (%t) in (4) {
    %csub = memref.subview %C[0, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1]
      : memref<2x2x16x16xi8> to memref<16x16xi8, strided<[16, 1]>>
    %z = x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
    %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%acc = %z)
        -> (!x86.amx.tile<16x16xi32>) {
      %la = x86.amx.tile_load %A[%k, %c0, %c0]
        : memref<4x16x64xi8> into !x86.amx.tile<16x64xi8>
      %lb = x86.amx.tile_load %B[%k, %c0, %c0]
        : memref<4x16x64xi8> into !x86.amx.tile<16x64xi8>
      %m = x86.amx.tile_muli %la, %lb, %acc
        : !x86.amx.tile<16x64xi8>, !x86.amx.tile<16x64xi8>, !x86.amx.tile<16x16xi32>
      scf.yield %m : !x86.amx.tile<16x16xi32>
    }
    %part = memref.alloca() : memref<16x16xi32>
    x86.amx.tile_store %part[%c0, %c0], %r
      : memref<16x16xi32>, !x86.amx.tile<16x16xi32>
    scf.for %row = %c0 to %c16 step %c1 {
      %p = vector.load %part[%row, %c0] : memref<16x16xi32>, vector<16xi32>
      %b = vector.load %zero[%row, %c0] : memref<16x16xi32>, vector<16xi32>
      %s = arith.addi %p, %b : vector<16xi32>
      vector.store %s, %part[%row, %c0] : memref<16x16xi32>, vector<16xi32>
    }
    scf.for %row = %c0 to %c16 step %c1 {
      %p = vector.load %part[%row, %c0] : memref<16x16xi32>, vector<16xi32>
      %pf = arith.sitofp %p : vector<16xi32> to vector<16xf32>
      %cl = arith.maximumf %pf, %lo : vector<16xf32>
      %ch = arith.minimumf %cl, %hi : vector<16xf32>
      %q = arith.fptosi %ch : vector<16xf32> to vector<16xi8>
      vector.store %q, %csub[%row, %c0]
        : memref<16x16xi8, strided<[16, 1]>>, vector<16xi8>
    }
  }
  return
}

// A full-grid i32 carrier is allocated outside the hoisted K-block loop; the
// reduction runs over the block, the (zero) bias is re-sourced from the carrier
// via select (beta=1), and the summed partial is mirrored back into the carrier.
// CHECK-LABEL: func.func @nano_gemm_quant
// CHECK: %[[ACC:.+]] = memref.alloc() : memref<2x2x16x16xi32>
// CHECK: scf.for %[[KB:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK:   scf.forall
// CHECK:     %[[END:.+]] = arith.addi %[[KB]], %{{.+}}
// CHECK:     scf.for %{{.+}} = %[[KB]] to %[[END]] step
// CHECK:       x86.amx.tile_muli
// CHECK:     x86.amx.tile_store
// CHECK:     %[[FIRST:.+]] = arith.cmpi eq, %[[KB]], %{{.+}}
// CHECK:     %[[ACCSV:.+]] = memref.subview %[[ACC]]
// CHECK:     scf.for
// CHECK:       %[[AV:.+]] = vector.load %[[ACCSV]]
// CHECK:       arith.select %[[FIRST]], %{{.+}}, %[[AV]]
// CHECK:       arith.addi
// CHECK:       vector.store %{{.+}}, %[[ACCSV]]
// CHECK: memref.dealloc %[[ACC]]

// DISABLED-LABEL: func.func @nano_gemm_quant
// DISABLED-NOT: memref.alloc()
// DISABLED-NOT: arith.select

// -----

// The block factor must divide the batch-reduce extent; extent 3 is not
// divisible by 2, so the quantized tile is left unchanged (no carrier/rewrite).
func.func @nano_gemm_quant_no_divide(%A: memref<3x16x64xi8>, %B: memref<3x16x64xi8>,
                                     %C: memref<2x2x16x16xi8>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c16 = arith.constant 16 : index
  %lo = arith.constant dense<-1.280000e+02> : vector<16xf32>
  %hi = arith.constant dense<1.270000e+02> : vector<16xf32>
  %zero = memref.alloca() : memref<16x16xi32>
  scf.forall (%t) in (4) {
    %csub = memref.subview %C[0, 0, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1]
      : memref<2x2x16x16xi8> to memref<16x16xi8, strided<[16, 1]>>
    %z = x86.amx.tile_zero : !x86.amx.tile<16x16xi32>
    %r = scf.for %k = %c0 to %c3 step %c1 iter_args(%acc = %z)
        -> (!x86.amx.tile<16x16xi32>) {
      %la = x86.amx.tile_load %A[%k, %c0, %c0]
        : memref<3x16x64xi8> into !x86.amx.tile<16x64xi8>
      %lb = x86.amx.tile_load %B[%k, %c0, %c0]
        : memref<3x16x64xi8> into !x86.amx.tile<16x64xi8>
      %m = x86.amx.tile_muli %la, %lb, %acc
        : !x86.amx.tile<16x64xi8>, !x86.amx.tile<16x64xi8>, !x86.amx.tile<16x16xi32>
      scf.yield %m : !x86.amx.tile<16x16xi32>
    }
    %part = memref.alloca() : memref<16x16xi32>
    x86.amx.tile_store %part[%c0, %c0], %r
      : memref<16x16xi32>, !x86.amx.tile<16x16xi32>
    scf.for %row = %c0 to %c16 step %c1 {
      %p = vector.load %part[%row, %c0] : memref<16x16xi32>, vector<16xi32>
      %b = vector.load %zero[%row, %c0] : memref<16x16xi32>, vector<16xi32>
      %s = arith.addi %p, %b : vector<16xi32>
      vector.store %s, %part[%row, %c0] : memref<16x16xi32>, vector<16xi32>
    }
    scf.for %row = %c0 to %c16 step %c1 {
      %p = vector.load %part[%row, %c0] : memref<16x16xi32>, vector<16xi32>
      %pf = arith.sitofp %p : vector<16xi32> to vector<16xf32>
      %cl = arith.maximumf %pf, %lo : vector<16xf32>
      %ch = arith.minimumf %cl, %hi : vector<16xf32>
      %q = arith.fptosi %ch : vector<16xf32> to vector<16xi8>
      vector.store %q, %csub[%row, %c0]
        : memref<16x16xi8, strided<[16, 1]>>, vector<16xi8>
    }
  }
  return
}

// CHECK-LABEL: func.func @nano_gemm_quant_no_divide
// CHECK-NOT: memref.alloc()
// CHECK-NOT: arith.select
// CHECK-NOT: arith.cmpi
