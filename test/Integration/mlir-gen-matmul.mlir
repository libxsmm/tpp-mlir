// RUN: mlir-gen --kernel=args --seed=0 --float-type=f32 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=FP32
// RUN: mlir-gen --kernel=args --seed=0 --float-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=BF16
// RUN: mlir-gen --kernel=args --seed=0 --float-type=f16 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=FP16

// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 --output=contract 2>&1 | FileCheck %s --check-prefix=MXBF16-CONTRACT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-i8 --batch=128 --layers=2304,768 --tiles=64,48,64 --output=contract 2>&1 | FileCheck %s --check-prefix=MXI8-CONTRACT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-f16 --batch=128 --layers=2304,768 --tiles=64,48,64 --output=contract 2>&1 | FileCheck %s --check-prefix=MXF16-CONTRACT

// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-bf16 --batch=128 --layers=2304,768 --quant-type=dequantize 2>&1 | FileCheck %s --check-prefix=MXBF16-DEQUANT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-i8-f32 --batch=128 --layers=2304,768 --quant-type=dequantize 2>&1 | FileCheck %s --check-prefix=MXI8F32-DEQUANT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-i8-f32 --batch=128 --layers=2304,768 --quant-type=dequantize --scale-type=f8E8M0FNU 2>&1 | FileCheck %s --check-prefix=MXI8-I8SCALE-DEQUANT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-f32-i8 --batch=128 --layers=2304,768 --quant-type=quantize 2>&1 | FileCheck %s --check-prefix=MXF32I8-QUANT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-i8-f32 --batch=4096 --layers=8192,4096 --quant-type=dequantize --output=generic --tiles=32,32,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=MXI8F32-PACKED-DEQUANT
// RUN: mlir-gen --kernel=args --seed=0 --float-type=mx-i8-f32 --batch=128 --layers=2304,768 --quant-type=dequantize --scale-type=f8E8M0FNU --tiles=32,32,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=MXI8-I8SCALE-PACKED-DEQUANT


// FP32: // RUN{{.*}}tpp-run %s -n {{\d*}}
// FP32: // RUN{{.*}}-e entry -entry-point-result=void
// FP32: // BENCH_TOTAL_FLOPS: 452984832
// FP32-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// FP32-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// FP32-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// FP32:     func.func @entry(%arg0: tensor<2x36x64x64xf32>, %arg1: tensor<16x36x64x48xf32>, %arg2: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32>
// FP32-NOT: alloc
// FP32:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// FP32:         arith.mulf
// FP32:         arith.addf
// FP32-NOT: dealloc

// BF16: // RUN{{.*}}tpp-run %s -n {{\d*}}
// BF16: // RUN{{.*}}-e entry -entry-point-result=void
// BF16: // BENCH_TOTAL_FLOPS: 452984832
// BF16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// BF16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// BF16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// BF16:     func.func @entry(%arg0: tensor<2x36x64x64xbf16>, %arg1: tensor<16x36x64x48xbf16>, %arg2: tensor<2x16x64x48xbf16>) -> tensor<2x16x64x48xbf16>
// BF16-NOT: alloc
// BF16:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// BF16:         arith.mulf
// BF16:         arith.addf
// BF16-NOT: dealloc

// FP16: // RUN{{.*}}tpp-run %s -n {{\d*}}
// FP16: // RUN{{.*}}-e entry -entry-point-result=void
// FP16: // BENCH_TOTAL_FLOPS: 452984832
// FP16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// FP16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// FP16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// FP16:     func.func @entry(%arg0: tensor<2x36x64x64xf16>, %arg1: tensor<16x36x64x48xf16>, %arg2: tensor<2x16x64x48xf16>) -> tensor<2x16x64x48xf16>
// FP16-NOT: alloc
// FP16:     linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// FP16:         arith.mulf
// FP16:         arith.addf
// FP16-NOT: dealloc

// MXBF16-GENERIC: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXBF16-GENERIC: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXBF16-GENERIC: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXBF16-GENERIC-LABEL:   func.func @entry(
// MXBF16-GENERIC-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xbf16>,
// MXBF16-GENERIC-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xbf16>,
// MXBF16-GENERIC-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32> {
// MXBF16-GENERIC:           %[[VAL_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xbf16>, tensor<16x36x64x48xbf16>) outs(%[[ARG2]] : tensor<2x16x64x48xf32>) {
// MXBF16-GENERIC:           ^bb0(%[[VAL_1:.*]]: bf16, %[[VAL_2:.*]]: bf16, %[[VAL_3:.*]]: f32):
// MXBF16-GENERIC:             %[[VAL_4:.*]] = arith.extf %[[VAL_1]] : bf16 to f32
// MXBF16-GENERIC:             %[[VAL_5:.*]] = arith.extf %[[VAL_2]] : bf16 to f32
// MXBF16-GENERIC:             %[[VAL_6:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// MXBF16-GENERIC:             %[[VAL_7:.*]] = arith.addf %[[VAL_3]], %[[VAL_6]] : f32
// MXBF16-GENERIC:             linalg.yield %[[VAL_7]] : f32
// MXBF16-GENERIC:           } -> tensor<2x16x64x48xf32>
// MXBF16-GENERIC:           return %[[VAL_0]] : tensor<2x16x64x48xf32>
// MXBF16-GENERIC:         }

// MXI8-GENERIC: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXI8-GENERIC: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXI8-GENERIC: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXI8-GENERIC-LABEL:   func.func @entry(
// MXI8-GENERIC-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xi8>,
// MXI8-GENERIC-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xi8>,
// MXI8-GENERIC-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32> {
// MXI8-GENERIC:           %[[VAL_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xi8>, tensor<16x36x64x48xi8>) outs(%[[ARG2]] : tensor<2x16x64x48xi32>) {
// MXI8-GENERIC:           ^bb0(%[[VAL_1:.*]]: i8, %[[VAL_2:.*]]: i8, %[[VAL_3:.*]]: i32):
// MXI8-GENERIC:             %[[VAL_4:.*]] = arith.extsi %[[VAL_1]] : i8 to i32
// MXI8-GENERIC:             %[[VAL_5:.*]] = arith.extsi %[[VAL_2]] : i8 to i32
// MXI8-GENERIC:             %[[VAL_6:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : i32
// MXI8-GENERIC:             %[[VAL_7:.*]] = arith.addi %[[VAL_3]], %[[VAL_6]] : i32
// MXI8-GENERIC:             linalg.yield %[[VAL_7]] : i32
// MXI8-GENERIC:           } -> tensor<2x16x64x48xi32>
// MXI8-GENERIC:           return %[[VAL_0]] : tensor<2x16x64x48xi32>
// MXI8-GENERIC:         }

// MXBF16-CONTRACT: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXBF16-CONTRACT: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXBF16-CONTRACT: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXBF16-CONTRACT-LABEL:   func.func @entry(
// MXBF16-CONTRACT-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xbf16>,
// MXBF16-CONTRACT-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xbf16>,
// MXBF16-CONTRACT-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32> {
// MXBF16-CONTRACT:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// MXBF16-CONTRACT:           %[[VAL_1:.*]] = tensor.empty() : tensor<2x16x64x48xf32>
// MXBF16-CONTRACT:           %[[VAL_2:.*]] = linalg.fill ins(%[[VAL_0]] : f32) outs(%[[VAL_1]] : tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32>
// MXBF16-CONTRACT:           %[[VAL_3:.*]] = linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xbf16>, tensor<16x36x64x48xbf16>) outs(%[[VAL_2]] : tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32>
// MXBF16-CONTRACT:           return %[[VAL_3]] : tensor<2x16x64x48xf32>
// MXBF16-CONTRACT:         }

// MXI8-CONTRACT: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXI8-CONTRACT: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXI8-CONTRACT: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXI8-CONTRACT-LABEL:   func.func @entry(
// MXI8-CONTRACT-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xi8>,
// MXI8-CONTRACT-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xi8>,
// MXI8-CONTRACT-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32> {
// MXI8-CONTRACT:           %[[VAL_0:.*]] = arith.constant 0 : i32
// MXI8-CONTRACT:           %[[VAL_1:.*]] = tensor.empty() : tensor<2x16x64x48xi32>
// MXI8-CONTRACT:           %[[VAL_2:.*]] = linalg.fill ins(%[[VAL_0]] : i32) outs(%[[VAL_1]] : tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32>
// MXI8-CONTRACT:           %[[VAL_3:.*]] = linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xi8>, tensor<16x36x64x48xi8>) outs(%[[VAL_2]] : tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32>
// MXI8-CONTRACT:           return %[[VAL_3]] : tensor<2x16x64x48xi32>
// MXI8-CONTRACT:         }

// MXF16-GENERIC: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXF16-GENERIC: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXF16-GENERIC: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXF16-GENERIC-LABEL:   func.func @entry(
// MXF16-GENERIC-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xf16>,
// MXF16-GENERIC-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xf16>,
// MXF16-GENERIC-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32> {
// MXF16-GENERIC:           %[[VAL_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xf16>, tensor<16x36x64x48xf16>) outs(%[[ARG2]] : tensor<2x16x64x48xf32>) {
// MXF16-GENERIC:           ^bb0(%[[VAL_1:.*]]: f16, %[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f32):
// MXF16-GENERIC:             %[[VAL_4:.*]] = arith.extf %[[VAL_1]] : f16 to f32
// MXF16-GENERIC:             %[[VAL_5:.*]] = arith.extf %[[VAL_2]] : f16 to f32
// MXF16-GENERIC:             %[[VAL_6:.*]] = arith.mulf %[[VAL_4]], %[[VAL_5]] : f32
// MXF16-GENERIC:             %[[VAL_7:.*]] = arith.addf %[[VAL_3]], %[[VAL_6]] : f32
// MXF16-GENERIC:             linalg.yield %[[VAL_7]] : f32
// MXF16-GENERIC:           } -> tensor<2x16x64x48xf32>
// MXF16-GENERIC:           return %[[VAL_0]] : tensor<2x16x64x48xf32>
// MXF16-GENERIC:         }

// MXF16-CONTRACT: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// MXF16-CONTRACT: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// MXF16-CONTRACT: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// MXF16-CONTRACT-LABEL:   func.func @entry(
// MXF16-CONTRACT-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xf16>,
// MXF16-CONTRACT-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xf16>,
// MXF16-CONTRACT-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32> {
// MXF16-CONTRACT:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f32
// MXF16-CONTRACT:           %[[VAL_1:.*]] = tensor.empty() : tensor<2x16x64x48xf32>
// MXF16-CONTRACT:           %[[VAL_2:.*]] = linalg.fill ins(%[[VAL_0]] : f32) outs(%[[VAL_1]] : tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32>
// MXF16-CONTRACT:           %[[VAL_3:.*]] = linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xf16>, tensor<16x36x64x48xf16>) outs(%[[VAL_2]] : tensor<2x16x64x48xf32>) -> tensor<2x16x64x48xf32>
// MXF16-CONTRACT:           return %[[VAL_3]] : tensor<2x16x64x48xf32>
// MXF16-CONTRACT:         }


// Perform Gemm dequntization using given scales.

// MXBF16-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// MXBF16-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// MXBF16-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// MXBF16-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// MXBF16-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0)>
// MXBF16-DEQUANT-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d1)>
// MXBF16-DEQUANT-LABEL:   func.func @entry(
// MXBF16-DEQUANT-SAME:                     %arg0: tensor<128x2304xbf16>,
// MXBF16-DEQUANT-SAME:                     %arg1: tensor<128xf32>,
// MXBF16-DEQUANT-SAME:                     %arg2: tensor<2304x768xbf16>,
// MXBF16-DEQUANT-SAME:                     %arg3: tensor<768xf32>,
// MXBF16-DEQUANT-SAME:                     %arg4: tensor<128x768xf32>) -> tensor<128x768xf32> {
// MXBF16-DEQUANT:           linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
// MXBF16-DEQUANT:           linalg.generic  {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_4]], #[[$ATTR_5]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel"]}
// MXBF16-DEQUANT:           ^bb0
// MXBF16-DEQUANT-2:           arith.mulf
// MXBF16-DEQUANT:             linalg.yield

// MXI8F32-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// MXI8F32-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// MXI8F32-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// MXI8F32-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// MXI8F32-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0)>
// MXI8F32-DEQUANT-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d1)>
// MXI8F32-DEQUANT-LABEL:   func.func @entry(
// MXI8F32-DEQUANT-SAME:                     %arg0: tensor<128x2304xi8>,
// MXI8F32-DEQUANT-SAME:                     %arg1: tensor<128xf32>,
// MXI8F32-DEQUANT-SAME:                     %arg2: tensor<2304x768xi8>,
// MXI8F32-DEQUANT-SAME:                     %arg3: tensor<768xf32>,
// MXI8F32-DEQUANT-SAME:                     %arg4: tensor<128x768xf32>) -> tensor<128x768xf32> {
// MXI8F32-DEQUANT:           linalg.contract indexing_maps = [#[[$ATTR_0:.+]], #[[$ATTR_1:.+]], #[[$ATTR_2:.+]]]
// MXI8F32-DEQUANT:           linalg.generic  {indexing_maps = [#[[$ATTR_3:.+]], #[[$ATTR_4:.+]], #[[$ATTR_5:.+]], #[[$ATTR_3:.+]]], iterator_types = ["parallel", "parallel"]}
// MXI8F32-DEQUANT:           ^bb0
// MXI8F32-DEQUANT:             arith.mulf
// MXI8F32-DEQUANT:             arith.sitofp
// MXI8F32-DEQUANT:             arith.mulf
// MXI8F32-DEQUANT:             linalg.yield


// Perform Gemm quntization with dynamic scale computation.

// MXF32I8-QUANT: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// MXF32I8-QUANT: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// MXF32I8-QUANT: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// MXF32I8-QUANT: #map3 = affine_map<(d0) -> (d0)>
// MXF32I8-QUANT: #map4 = affine_map<(d0, d1) -> (d0, d1)>
// MXF32I8-QUANT-LABEL:   func.func @entry(
// MXF32I8-QUANT-SAME:                     %[[ARG0:.*]]: tensor<128x2304xf32>,
// MXF32I8-QUANT-SAME:                     %[[ARG1:.*]]: tensor<2304x768xf32>,
// MXF32I8-QUANT-SAME:                     %[[ARG2:.*]]: tensor<128x768xi8>) -> tensor<128x768xi8> {
// MXF32I8-QUANT:           linalg.contract
// MXF32I8-QUANT:           linalg.reduce {{.*}} dimensions = [0]
// MXF32I8-QUANT:               math.absf
// MXF32I8-QUANT:               arith.maximumf
// MXF32I8-QUANT:           linalg.generic  {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel"]}
// MXF32I8-QUANT:               llvm.intr.frexp
// MXF32I8-QUANT:               llvm.extractvalue
// MXF32I8-QUANT:               arith.constant 7
// MXF32I8-QUANT:               arith.subi
// MXF32I8-QUANT:               arith.subi
// MXF32I8-QUANT:               arith.sitofp
// MXF32I8-QUANT:               arith.sitofp
// MXF32I8-QUANT:               math.exp2
// MXF32I8-QUANT:           linalg.broadcast
// MXF32I8-QUANT:           linalg.mul
// MXF32I8-QUANT:           linalg.generic
// MXF32I8-QUANT:               arith.fptosi


// MXI8F32-PACKED-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// MXI8F32-PACKED-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// MXI8F32-PACKED-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// MXI8F32-PACKED-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// MXI8F32-PACKED-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
// MXI8F32-PACKED-DEQUANT-LABEL:   func.func @entry(
// MXI8F32-PACKED-DEQUANT-SAME:                     {{.*}}: tensor<128x128x32x64xi8>,
// MXI8F32-PACKED-DEQUANT-SAME:                     {{.*}}: tensor<4096xf32>, {{.*}}: tensor<128x128x16x32x4xi8>,
// MXI8F32-PACKED-DEQUANT-SAME:                     {{.*}}: tensor<4096xf32>,
// MXI8F32-PACKED-DEQUANT-SAME:                     {{.*}}: tensor<128x128x32x32xf32>) -> tensor<128x128x32x32xf32> {

// MXI8F32-PACKED-DEQUANT:           linalg.fill
// MXI8F32-PACKED-DEQUANT:           tensor.expand_shape
// MXI8F32-PACKED-DEQUANT:           linalg.contract  indexing_maps = [#[[$ATTR_0:.+]], #[[$ATTR_1:.+]], #[[$ATTR_2:.+]]]
// MXI8F32-PACKED-DEQUANT-2:         tensor.expand_shape
// MXI8F32-PACKED-DEQUANT:           linalg.generic {indexing_maps = [#[[$ATTR_3:.+]], #[[$ATTR_4:.+]], #[[$ATTR_4:.+]], #[[$ATTR_3:.+]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// MXI8F32-PACKED-DEQUANT:           ^bb0
// MXI8F32-PACKED-DEQUANT:             arith.mulf
// MXI8F32-PACKED-DEQUANT:             arith.sitofp
// MXI8F32-PACKED-DEQUANT:             arith.mulf
// MXI8F32-PACKED-DEQUANT:             linalg.yield


// MXI8-I8SCALE-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// MXI8-I8SCALE-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// MXI8-I8SCALE-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// MXI8-I8SCALE-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// MXI8-I8SCALE-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0)>
// MXI8-I8SCALE-DEQUANT-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d1)>
// MXI8-I8SCALE-DEQUANT-LABEL:   func.func @entry(
// MXI8-I8SCALE-DEQUANT-SAME:                     %[[ARG0:.*]]: tensor<128x2304xi8>,
// MXI8-I8SCALE-DEQUANT-SAME:                     %[[ARG1:.*]]: tensor<128xf8E8M0FNU>,
// MXI8-I8SCALE-DEQUANT-SAME:                     %[[ARG2:.*]]: tensor<2304x768xi8>,
// MXI8-I8SCALE-DEQUANT-SAME:                     %[[ARG3:.*]]: tensor<768xf8E8M0FNU>,
// MXI8-I8SCALE-DEQUANT-SAME:                     %[[ARG4:.*]]: tensor<128x768xf32>) -> tensor<128x768xf32> {
// MXI8-I8SCALE-DEQUANT:           arith.constant 0 : i32
// MXI8-I8SCALE-DEQUANT:           tensor.empty() : tensor<128x768xi32>
// MXI8-I8SCALE-DEQUANT:           linalg.fill
// MXI8-I8SCALE-DEQUANT:           linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG2]]
// MXI8-I8SCALE-DEQUANT:           linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_4]], #[[$ATTR_5]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel"]}
// MXI8-I8SCALE-DEQUANT:           ^bb0
// MXI8-I8SCALE-DEQUANT-2:             arith.extf {{.*}} fastmath<nnan>
// MXI8-I8SCALE-DEQUANT:               arith.mulf
// MXI8-I8SCALE-DEQUANT:               arith.sitofp
// MXI8-I8SCALE-DEQUANT:               arith.mulf
// MXI8-I8SCALE-DEQUANT:               linalg.yield
// MXI8-I8SCALE-DEQUANT:           } -> tensor<128x768xf32>


// MXI8-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// MXI8-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// MXI8-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// MXI8-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// MXI8-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
// MXI8-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3, 0)>
// MXI8-I8SCALE-PACKED-DEQUANT-LABEL:   func.func @entry(
// MXI8-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG0:.*]]: tensor<4x36x32x64xi8>,
// MXI8-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG1:.*]]: tensor<128xf8E8M0FNU>,
// MXI8-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG2:.*]]: tensor<24x36x16x32x4xi8>,
// MXI8-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG3:.*]]: tensor<768xf8E8M0FNU>,
// MXI8-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG4:.*]]: tensor<4x24x32x32xf32>) -> tensor<4x24x32x32xf32> {
// MXI8-I8SCALE-PACKED-DEQUANT:           arith.constant 0 : i32
// MXI8-I8SCALE-PACKED-DEQUANT:           tensor.empty() : tensor<4x24x32x32xi32>
// MXI8-I8SCALE-PACKED-DEQUANT:           linalg.fill
// MXI8-I8SCALE-PACKED-DEQUANT:           tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 36, 32, 16, 4] : tensor<4x36x32x64xi8> into tensor<4x36x32x16x4xi8>
// MXI8-I8SCALE-PACKED-DEQUANT:           linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
// MXI8-I8SCALE-PACKED-DEQUANT:           tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1, 2, 3]] output_shape [4, 1, 32, 1] : tensor<128xf8E8M0FNU> into tensor<4x1x32x1xf8E8M0FNU>
// MXI8-I8SCALE-PACKED-DEQUANT:           tensor.expand_shape %[[ARG3]] {{\[\[}}0, 1, 2, 3]] output_shape [24, 1, 32, 1] : tensor<768xf8E8M0FNU> into tensor<24x1x32x1xf8E8M0FNU>
// MXI8-I8SCALE-PACKED-DEQUANT:           linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_4]], #[[$ATTR_5]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// MXI8-I8SCALE-PACKED-DEQUANT:           ^bb0
// MXI8-I8SCALE-PACKED-DEQUANT-2:             arith.extf {{.*}} fastmath<nnan>
// MXI8-I8SCALE-PACKED-DEQUANT:               arith.mulf
// MXI8-I8SCALE-PACKED-DEQUANT:               arith.sitofp
// MXI8-I8SCALE-PACKED-DEQUANT:               arith.mulf
// MXI8-I8SCALE-PACKED-DEQUANT:             linalg.yield
// MXI8-I8SCALE-PACKED-DEQUANT:           } -> tensor<4x24x32x32xf32>
