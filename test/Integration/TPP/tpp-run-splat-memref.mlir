// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early 2>&1 | \
// RUN: FileCheck %s --check-prefix=SPLAT
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 -splat-to-random 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM-SPLAT

memref.global "private" constant @__constant_2x16xf32 : memref<2x16xf32> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_2x16xf64 : memref<2x16xf64> = dense<1.000000e+00> {alignment = 128 : i64}
memref.global "private" constant @__constant_4x16xf32 : memref<4x16xf32> = dense<2.0> {alignment = 128 : i64}
memref.global "private" constant @__constant_4x8xf32 : memref<4x8xf32> = dense<0.0> {alignment = 128 : i64}
memref.global "private" constant @__constant_non_splat : memref<2x2xf32> = dense<[[0.0, 1.0],[2.0, 3.0]]> {alignment = 128 : i64}
memref.global "private" constant @__constant_zero_i32 : memref<4x8xi32> = dense<0> {alignment = 128 : i64}
memref.global "private" constant @__constant_one_i32 : memref<4x8xi32> = dense<1> {alignment = 128 : i64}
memref.global "private" constant @__constant_one_i64 : memref<4x8xi64> = dense<1> {alignment = 128 : i64}
memref.global "private" constant @__constant_non_splat_i32 : memref<2x2xi32> = dense<[[0, 1],[2, 3]]> {alignment = 128 : i64}
func.func @entry(%arg0: memref<4x2xf32>, %arg1: memref<4x2xi32>) {
  %0 = memref.get_global @__constant_2x16xf32 : memref<2x16xf32>
  %1 = memref.get_global @__constant_4x16xf32 : memref<4x16xf32>
  %2 = memref.get_global @__constant_4x8xf32 : memref<4x8xf32>
  return
}

// SPLAT-DAG: @__wrapper_0 : memref<4x2xf32> = dense<1.000000e+00>
// SPLAT-DAG: @__wrapper_1 : memref<4x2xi32> = dense<1>
// SPLAT: constant @__constant_2x16xf32 : memref<2x16xf32> = dense<1.000000e+00>
// SPLAT: constant @__constant_2x16xf64 : memref<2x16xf64> = dense<1.000000e+00>
// SPLAT: constant @__constant_4x16xf32 : memref<4x16xf32> = dense<2.000000e+00>
// SPLAT: constant @__constant_4x8xf32 : memref<4x8xf32> = dense<0.000000e+00>
// SPLAT: constant @__constant_non_splat : memref<2x2xf32> = dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// SPLAT: constant @__constant_zero_i32 : memref<4x8xi32> = dense<0>
// SPLAT: constant @__constant_one_i32 : memref<4x8xi32> = dense<1>
// SPLAT: constant @__constant_one_i64 : memref<4x8xi64> = dense<1>
// SPLAT: constant @__constant_non_splat_i32 : memref<2x2xi32> = dense<{{\[}}{{\[}}0, 1], [2, 3]]>
// SPLAT-LABEL: @_entry

// RANDOM-NOT: @__wrapper_0 : memref<4x2xf32> = dense<1.000000e+00>
// RANDOM-NOT: @__wrapper_1 : memref<4x2xi32> = dense<1>
// RANDOM: constant @__constant_2x16xf32 : memref<2x16xf32> = dense<1.000000e+00>
// RANDOM: constant @__constant_2x16xf64 : memref<2x16xf64> = dense<1.000000e+00>
// RANDOM: constant @__constant_4x16xf32 : memref<4x16xf32> = dense<2.000000e+00>
// RANDOM: constant @__constant_4x8xf32 : memref<4x8xf32> = dense<0.000000e+00>
// RANDOM: constant @__constant_non_splat : memref<2x2xf32> = dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// RANDOM: constant @__constant_zero_i32 : memref<4x8xi32> = dense<0>
// RANDOM: constant @__constant_one_i32 : memref<4x8xi32> = dense<1>
// RANDOM: constant @__constant_one_i64 : memref<4x8xi64> = dense<1>
// RANDOM: constant @__constant_non_splat_i32 : memref<2x2xi32> = dense<{{\[}}{{\[}}0, 1], [2, 3]]>
// RANDOM-LABEL: @_entry

// RANDOM-SPLAT-NOT: @__wrapper_0 : memref<4x2xf32> = dense<1.000000e+00>
// RANDOM-SPLAT-NOT: @__wrapper_1 : memref<4x2xi32> = dense<1>
// RANDOM-SPLAT-NOT: constant @__constant_2x16xf32 : memref<2x16xf32> = dense<1.000000e+00>
// RANDOM-SPLAT-NOT: constant @__constant_2x16xf64 : memref<2x16xf64> = dense<1.000000e+00>
// RANDOM-SPLAT-NOT: constant @__constant_4x16xf32 : memref<4x16xf32> = dense<2.000000e+00>
// RANDOM-SPLAT: constant @__constant_4x8xf32 : memref<4x8xf32> = dense<0.000000e+00>
// RANDOM-SPLAT: constant @__constant_non_splat : memref<2x2xf32> = dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// RANDOM-SPLAT: constant @__constant_zero_i32 : memref<4x8xi32> = dense<0>
// RANDOM-SPLAT-NOT: constant @__constant_one_i32 : memref<4x8xi32> = dense<1>
// RANDOM-SPLAT-NOT: constant @__constant_one_i64 : memref<4x8xi64> = dense<1>
// RANDOM-SPLAT: constant @__constant_non_splat_i32 : memref<2x2xi32> = dense<{{\[}}{{\[}}0, 1], [2, 3]]>
// RANDOM-SPLAT-LABEL: @_entry
