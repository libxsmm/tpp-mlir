// RUN: mlir-gen   --kernel=const --bias --relu --seed=123 | tpp-run -e entry --entry-point-result=void -print-mlir=mid  2>&1 | FileCheck %s
// CHECK: func.func @_entry(%arg0: memref<256x128xf32>) -> memref<256x512xf32>  {
// CHECK: call @xsmm_fused_brgemm_dispatch
// CHECK: scf.parallel
// CHECK:   func.call @xsmm_fused_brgemm_invoke
// CHECK: scf.parallel
// CHECK:   func.call @xsmm_fused_brgemm_invoke
