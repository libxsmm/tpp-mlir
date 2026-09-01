// RUN: tpp-opt %s --convert-to-streaming-store --split-input-file | FileCheck %s

// A 1-D write into a destination buffer that is never read back must become a
// nontemporal vector.store.

// CHECK-LABEL: func.func @streaming_store
func.func @streaming_store(%v: vector<64xi8>, %c: memref<64x64xi8>) {
  %c0 = arith.constant 0 : index
  %sv = memref.subview %c[0, 0] [1, 64] [1, 1]
    : memref<64x64xi8> to memref<64xi8, strided<[1], offset: 0>>
  // CHECK: vector.store %{{.*}} nontemporal = true : memref<64xi8, strided<[1]>>, vector<64xi8>
  // CHECK-NOT: vector.transfer_write
  vector.transfer_write %v, %sv[%c0] {in_bounds = [true]}
    : vector<64xi8>, memref<64xi8, strided<[1], offset: 0>>
  return
}

// -----

// A destination buffer that is later read back must not be streamed.

// CHECK-LABEL: func.func @read_back
func.func @read_back(%v: vector<64xi8>, %c: memref<64xi8>) -> vector<64xi8> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0 : i8
  // CHECK: vector.transfer_write {{.*}} : vector<64xi8>, memref<64xi8>
  // CHECK-NOT: nontemporal
  vector.transfer_write %v, %c[%c0] {in_bounds = [true]}
    : vector<64xi8>, memref<64xi8>
  %r = vector.transfer_read %c[%c0], %pad {in_bounds = [true]}
    : memref<64xi8>, vector<64xi8>
  return %r : vector<64xi8>
}

// -----

// Multi-dimensional writes are left untouched: only rank-1 stores lower to LLVM.

// CHECK-LABEL: func.func @rank2_write_only
func.func @rank2_write_only(%v: vector<1x64xi8>, %c: memref<1x64xi8>) {
  %c0 = arith.constant 0 : index
  // CHECK: vector.transfer_write {{.*}} : vector<1x64xi8>, memref<1x64xi8>
  // CHECK-NOT: nontemporal
  vector.transfer_write %v, %c[%c0, %c0] {in_bounds = [true, true]}
    : vector<1x64xi8>, memref<1x64xi8>
  return
}
