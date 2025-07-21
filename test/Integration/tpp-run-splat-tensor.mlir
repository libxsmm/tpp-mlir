// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early 2>&1 | \
// RUN: FileCheck %s --check-prefix=SPLAT
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 -splat-to-random 2>&1 | \
// RUN: FileCheck %s --check-prefix=RANDOM-SPLAT

// Options for -init-type
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 -splat-to-random -init-type=const 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-CONST
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 -splat-to-random -init-type=simple 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-SIMPLE
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 -splat-to-random -init-type=cont 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-CONT
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 -splat-to-random -init-type=random 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-RANDOM
// RUN: tpp-run %s -e entry -entry-point-result=void -print-mlir=early -seed 123 -splat-to-random -init-type=normal 2>&1 | \
// RUN: FileCheck %s --check-prefix=OPT-NORMAL

func.func @entry(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xi32>, %arg2: tensor<4x2xi32>, %arg3: tensor<4x2xf16>, %arg4: tensor<4x2xi8>) {
  %0 = arith.constant dense<1.0> : tensor<2x16xf32>
  %5 = arith.constant dense<1.0> : tensor<2x16xf64>
  %1 = arith.constant dense<2.0> : tensor<4x16xf32>
  %10 = arith.constant dense<2.0> : tensor<4x4xf32>
  %2 = arith.constant dense<0.0> : tensor<4x8xf32>
  %3 = arith.constant dense<[[0.0, 1.0],[2.0, 3.0]]> : tensor<2x2xf32>
  %13 = arith.constant dense<1> : tensor<4x2xi8>
  %4 = arith.constant dense<0> : tensor<4x8xi32>
  %6 = arith.constant dense<1> : tensor<4x8xi32>
  %11 = arith.constant dense<1> : tensor<4x8xi32>
  %7 = arith.constant dense<1> : tensor<4x8xi64>
  %8 = arith.constant dense<[[0, 1],[2, 3]]> : tensor<2x2xi32>
  %9 = arith.constant 1.0 : f32
  %12 = arith.constant dense<1.0> : tensor<4x8xf16>
  return
}

// Constants
// SPLAT-DAG: memref.global "private" @__wrapper_0 : memref<4x2xf32> = dense<1.000000e+00>
// SPLAT-DAG: memref.global "private" @__wrapper_1 : memref<4x2xi32> = dense<1>
// SPLAT-DAG: memref.global "private" @__wrapper_2 : memref<4x2xi32> = dense<1>
// SPLAT-DAG: memref.global "private" @__wrapper_3 : memref<4x2xf16> = dense<1.000000e+00>
// SPLAT-DAG: memref.global "private" @__wrapper_4 : memref<4x2xi8> = dense<1>
// SPLAT-LABEL: @_entry
// SPLAT: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// SPLAT: arith.constant dense<1.000000e+00> : tensor<2x16xf64>
// SPLAT: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// SPLAT: arith.constant dense<2.000000e+00> : tensor<4x4xf32>
// SPLAT: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// SPLAT: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}> : tensor<2x2xf32>
// SPLAT: arith.constant dense<1> : tensor<4x2xi8>
// SPLAT: arith.constant dense<0> : tensor<4x8xi32>
// SPLAT: arith.constant dense<1> : tensor<4x8xi32>
// SPLAT: arith.constant dense<1> : tensor<4x8xi32>
// SPLAT: arith.constant dense<1> : tensor<4x8xi64>
// SPLAT: arith.constant dense<{{\[}}{{\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
// SPLAT: arith.constant 1.000000e+00 : f32
// SPLAT: arith.constant dense<1.000000e+00> : tensor<4x8xf16>
// Input
// SPLAT-LABEL: @entry
// SPLAT: memref.get_global @__wrapper_0

// Constants
// RANDOM-DAG: memref.global "private" @__wrapper_0 : memref<4x2xf32> = dense<{{\[}}{{\[}}0.000000e+00, 1.303520e-01], [0.151291341, 0.0106364777]
// RANDOM-DAG: memref.global "private" @__wrapper_1 : memref<4x2xi32> = dense<{{\[}}{{\[}}132, 126], [117, 123], [126, 121], [132, 133]]>
// RANDOM-DAG: memref.global "private" @__wrapper_2 : memref<4x2xi32> = dense<{{\[}}{{\[}}129, 134], [129, 126], [141, 131], [138, 121]]>
// RANDOM-DAG: memref.global "private" @__wrapper_3 : memref<4x2xf16> = dense<{{\[}}{{\[}}0.000000e+00, 1.303710e-01], [1.512450e-01, 1.063540e-02]
// RANDOM-DAG: memref.global "private" @__wrapper_4 : memref<4x2xi8> = dense<{{\[}}{{\[}}67, 63], [56, 60], [62, 59], [60, 71]]>
// RANDOM-LABEL: @_entry
// RANDOM: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// RANDOM: arith.constant dense<1.000000e+00> : tensor<2x16xf64>
// RANDOM: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// RANDOM: arith.constant dense<2.000000e+00> : tensor<4x4xf32>
// RANDOM: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// RANDOM: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}> : tensor<2x2xf32>
// RANDOM: arith.constant dense<1> : tensor<4x2xi8>
// RANDOM: arith.constant dense<0> : tensor<4x8xi32>
// RANDOM: arith.constant dense<1> : tensor<4x8xi32>
// RANDOM: arith.constant dense<1> : tensor<4x8xi32>
// RANDOM: arith.constant dense<1> : tensor<4x8xi64>
// RANDOM: arith.constant dense<{{\[}}{{\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
// RANDOM: arith.constant 1.000000e+00 : f32
// RANDOM: arith.constant dense<1.000000e+00> : tensor<4x8xf16>
// Input
// RANDOM-LABEL: @entry
// RANDOM: memref.get_global @__wrapper_0

// Constants
// RANDOM-SPLAT-NOT: memref.global "private" @__wrapper_0 : memref<4x2xf32> = dense<1.000000e+00>
// RANDOM-SPLAT-NOT: memref.global "private" @__wrapper_1 : memref<4x2xi32> = dense<1>
// RANDOM-SPLAT-NOT: memref.global "private" @__wrapper_2 : memref<4x2xi32> = dense<1>
// RANDOM-SPLAT-NOT: memref.global "private" @__wrapper_3 : memref<4x2xf16> = dense<1.000000e+00>
// RANDOM-SPLAT-NOT: memref.global "private" @__wrapper_4 : memref<4x2xi8> = dense<1>
// RANDOM-SPLAT-LABEL: @_entry
// RANDOM-SPLAT-NOT: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// RANDOM-SPLAT-NOT: arith.constant dense<1.000000e+00> : tensor<2x16xf64>
// RANDOM-SPLAT-NOT: arith.constant dense<2.000000e+00> : tensor<4x16xf32>
// RANDOM-SPLAT-NOT: arith.constant dense<2.000000e+00> : tensor<4x4xf32>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}0.000000e+00, 1.303520e-01, 0.151291341{{.*}}: tensor<2x16xf32>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}0.000000e+00, 0.13035200536251068, 0.15129134058952332{{.*}}: tensor<2x16xf64>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}0.186750099, 0.111865185, 0.388891816{{.*}}: tensor<4x16xf32>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}0.0440550111, 0.221581057, 0.000000e+00{{.*}}: tensor<4x4xf32>
// RANDOM-SPLAT: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// RANDOM-SPLAT: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}> : tensor<2x2xf32>
// RANDOM-SPLAT-NOT: arith.constant dense<1> : tensor<4x2xi8>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}67, 63{{.*}}> : tensor<4x2xi8>
// RANDOM-SPLAT-NOT: arith.constant dense<1> : tensor<4x2xi8>
// RANDOM-SPLAT: arith.constant dense<0> : tensor<4x8xi32>
// RANDOM-SPLAT-NOT: arith.constant dense<1> : tensor<4x8xi32>
// RANDOM-SPLAT-NOT: arith.constant dense<1> : tensor<4x8xi64>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}132, 126, 117{{.*}}> : tensor<4x8xi32>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}130, 121, 126{{.*}}> : tensor<4x8xi32>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}132, 126, 117{{.*}}> : tensor<4x8xi64>
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
// RANDOM-SPLAT: arith.constant 1.000000e+00 : f32
// RANDOM-SPLAT: arith.constant dense<{{\[}}{{\[}}0.000000e+00, 1.303710e-01, 1.512450e-01{{.*}}: tensor<4x8xf16>
// Input
// RANDOM-SPLAT-LABEL: @entry
// RANDOM-SPLAT: memref.get_global @__wrapper_0

// OPT-CONST-LABEL: @_entry
// OPT-CONST: arith.constant dense<1.000000e+00> : tensor<2x16xf32>
// OPT-CONST: arith.constant dense<1.000000e+00> : tensor<2x16xf64>
// OPT-CONST: arith.constant dense<1.000000e+00> : tensor<4x16xf32>
// OPT-CONST: arith.constant dense<1.000000e+00> : tensor<4x4xf32>
// OPT-CONST: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-CONST: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-CONST: arith.constant dense<1> : tensor<4x2xi8>
// OPT-CONST: arith.constant dense<0> : tensor<4x8xi32>
// OPT-CONST: arith.constant dense<1> : tensor<4x8xi32>
// OPT-CONST: arith.constant dense<1> : tensor<4x8xi32>
// OPT-CONST: arith.constant dense<1> : tensor<4x8xi64>
// OPT-CONST: arith.constant dense<{{\[}}{{\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
// OPT-CONST: arith.constant 1.000000e+00 : f32
// OPT-CONST: arith.constant dense<1.000000e+00> : tensor<4x8xf16>

// OPT-SIMPLE-LABEL: @_entry
// OPT-SIMPLE: arith.constant dense<{{.*}}3.000000e-01, 6.000000e-01, 0.899999976, {{.*}}> : tensor<2x16xf32>
// OPT-SIMPLE: arith.constant dense<{{.*}}0.30000001192092896, 0.60000002384185791, 0.89999997615814208, {{.*}}> : tensor<2x16xf64>
// OPT-SIMPLE: arith.constant dense<{{.*}}3.000000e-01, 6.000000e-01, 0.899999976, {{.*}}> : tensor<4x16xf32>
// OPT-SIMPLE: arith.constant dense<{{.*}}3.000000e-01, 6.000000e-01, 0.899999976, {{.*}}> : tensor<4x4xf32>
// OPT-SIMPLE: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-SIMPLE: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-SIMPLE-NOT: arith.constant dense<1> : tensor<4x2xi8>
// OPT-SIMPLE: arith.constant dense<{{\[}}{{\[}}0, 1{{.*}}> : tensor<4x2xi8>
// OPT-SIMPLE: arith.constant dense<0> : tensor<4x8xi32>
// OPT-SIMPLE-NOT: arith.constant dense<1> : tensor<4x8xi32>
// OPT-SIMPLE-NOT: arith.constant dense<1> : tensor<4x8xi64>
// OPT-SIMPLE: arith.constant dense<{{\[}}{{\[}}0, 1, 2, 0, 1, 2{{.*}}> : tensor<4x8xi32>
// OPT-SIMPLE: arith.constant dense<{{\[}}{{\[}}0, 1, 2, 0, 1, 2{{.*}}> : tensor<4x8xi32>
// OPT-SIMPLE: arith.constant dense<{{\[}}{{\[}}0, 1, 2, 0, 1, 2{{.*}}> : tensor<4x8xi64>
// OPT-SIMPLE: arith.constant dense<{{\[}}{{\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
// OPT-SIMPLE: arith.constant 1.000000e+00 : f32
// OPT-SIMPLE: arith.constant dense<{{.*}}3.000490e-01, 6.000980e-01, 8.999020e-01, {{.*}}> : tensor<4x8xf16>

// OPT-CONT-LABEL: @_entry
// OPT-CONT: arith.constant dense<{{.*}}0.000000e+00, 3.125000e-02, 6.250000e-02, {{.*}}> : tensor<2x16xf32>
// OPT-CONT: arith.constant dense<{{.*}}0.000000e+00, 3.125000e-02, 6.250000e-02, {{.*}}> : tensor<2x16xf64>
// OPT-CONT: arith.constant dense<{{.*}}0.000000e+00, 1.562500e-02, 3.125000e-02,  {{.*}}> : tensor<4x16xf32>
// OPT-CONT: arith.constant dense<{{.*}}0.000000e+00, 6.250000e-02, 1.250000e-01,  {{.*}}> : tensor<4x4xf32>
// OPT-CONT: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-CONT: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-CONT-NOT: arith.constant dense<1> : tensor<4x2xi8>
// OPT-CONT: arith.constant dense<{{\[}}{{\[}}0, 15{{.*}}> : tensor<4x2xi8>
// OPT-CONT: arith.constant dense<0> : tensor<4x8xi32>
// OPT-CONT-NOT: arith.constant dense<1> : tensor<4x8xi32>
// OPT-CONT-NOT: arith.constant dense<1> : tensor<4x8xi64>
// OPT-CONT: arith.constant dense<{{\[}}{{\[}}0, 7, 15, 23, 31{{.*}}> : tensor<4x8xi32>
// OPT-CONT: arith.constant dense<{{\[}}{{\[}}0, 7, 15, 23, 31{{.*}}> : tensor<4x8xi32>
// OPT-CONT: arith.constant dense<{{\[}}{{\[}}0, 7, 15, 23, 31{{.*}}> : tensor<4x8xi64>
// OPT-CONT: arith.constant dense<{{\[}}{{\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
// OPT-CONT: arith.constant 1.000000e+00 : f32
// OPT-CONT: arith.constant dense<{{.*}}0.000000e+00, 3.125000e-02, 6.250000e-02, {{.*}}> : tensor<4x8xf16>

// OPT-RANDOM-LABEL: @_entry
// OPT-RANDOM: arith.constant dense<{{.*}}9.62642952E-4, 0.179147944, 0.939454615, {{.*}}> : tensor<2x16xf32>
// OPT-RANDOM: arith.constant dense<{{.*}}9.6264295279979705E-4, 0.17914794385433197, 0.93945461511611938, {{.*}}> : tensor<2x16xf64>
// OPT-RANDOM: arith.constant dense<{{.*}}0.281718224, 0.838135182, 0.538071811, {{.*}}> : tensor<4x16xf32>
// OPT-RANDOM: arith.constant dense<{{.*}}0.685934782, 0.505808651, 0.126024485, {{.*}}> : tensor<4x4xf32>
// OPT-RANDOM: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-RANDOM: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-RANDOM: arith.constant dense<{{\[}}{{\[}}0, 22{{.*}}> : tensor<4x2xi8>
// OPT-RANDOM-NOT: arith.constant dense<1> : tensor<4x2xi8>
// OPT-RANDOM: arith.constant dense<0> : tensor<4x8xi32>
// OPT-RANDOM-NOT: arith.constant dense<1> : tensor<4x8xi32>
// OPT-RANDOM-NOT: arith.constant dense<1> : tensor<4x8xi64>
// OPT-RANDOM: arith.constant dense<{{\[}}{{\[}}0, 45, 240, 105, 135{{.*}}> : tensor<4x8xi32>
// OPT-RANDOM: arith.constant dense<{{\[}}{{\[}}72, 214, 137, 95, 208{{.*}}> : tensor<4x8xi32>
// OPT-RANDOM: arith.constant dense<{{\[}}{{\[}}0, 45, 240, 105, 135{{.*}}> : tensor<4x8xi64>
// OPT-RANDOM: arith.constant dense<{{\[}}{{\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
// OPT-RANDOM: arith.constant 1.000000e+00 : f32
// OPT-RANDOM: arith.constant dense<{{.*}}9.627340e-04, 1.791990e-01, 9.394530e-01, {{.*}}> : tensor<4x8xf16>

// OPT-NORMAL-LABEL: @_entry
// OPT-NORMAL: arith.constant dense<{{.*}}0.000000e+00, 1.303520e-01, 0.151291341, {{.*}}> : tensor<2x16xf32>
// OPT-NORMAL: arith.constant dense<{{.*}}0.000000e+00, 0.13035200536251068, 0.15129134058952332, {{.*}}> : tensor<2x16xf64>
// OPT-NORMAL: arith.constant dense<{{.*}}0.186750099, 0.111865185, 0.388891816, {{.*}}> : tensor<4x16xf32>
// OPT-NORMAL: arith.constant dense<{{.*}}0.0440550111, 0.221581057, 0.000000e+00, {{.*}}> : tensor<4x4xf32>
// OPT-NORMAL: arith.constant dense<0.000000e+00> : tensor<4x8xf32>
// OPT-NORMAL: arith.constant dense<{{.*}}0.000000e+00, 1.000000e+00], [2.000000e+00, 3.000000e+00{{.*}}>
// OPT-NORMAL-NOT: arith.constant dense<1> : tensor<4x2xi8>
// OPT-NORMAL: arith.constant dense<{{\[}}{{\[}}67, 63{{.*}}> : tensor<4x2xi8>
// OPT-NORMAL: arith.constant dense<0> : tensor<4x8xi32>
// OPT-NORMAL-NOT: arith.constant dense<1> : tensor<4x8xi32>
// OPT-NORMAL-NOT: arith.constant dense<1> : tensor<4x8xi64>
// OPT-NORMAL: arith.constant dense<{{\[}}{{\[}}132, 126, 117, 123, 126{{.*}}> : tensor<4x8xi32>
// OPT-NORMAL: arith.constant dense<{{\[}}{{\[}}130, 121, 126, 112, 129{{.*}}> : tensor<4x8xi32>
// OPT-NORMAL: arith.constant dense<{{\[}}{{\[}}132, 126, 117, 123, 126{{.*}}> : tensor<4x8xi64>
// OPT-NORMAL: arith.constant dense<{{\[}}{{\[}}0, 1], [2, 3]]> : tensor<2x2xi32>
// OPT-NORMAL: arith.constant 1.000000e+00 : f32
// OPT-NORMAL: arith.constant dense<{{.*}}0.000000e+00, 1.303710e-01, 1.512450e-01, {{.*}}> : tensor<4x8xf16>
