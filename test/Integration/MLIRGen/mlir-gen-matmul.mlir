// RUN: mlir-gen --kernel=args --seed=0 --data-type=f32 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=FP32
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf16 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=BF16
// RUN: mlir-gen --kernel=args --seed=0 --data-type=f16 --batch=128 --layers=2304,768 --tiles=64,48,64 2>&1 | FileCheck %s --check-prefix=FP16

// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=128 --layers=2304,768 --tiles=64,48,64 --output=contract 2>&1 | FileCheck %s --check-prefix=I8-CONTRACT

// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8-f32 --batch=128 --layers=2304,768 --quant 2>&1 | FileCheck %s --check-prefix=I8F32-DEQUANT
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8-f32 --batch=128 --layers=2304,768 --quant --scale-type=f8E8M0FNU 2>&1 | FileCheck %s --check-prefix=I8F32-I8SCALE-DEQUANT
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8-f32 --batch=4096 --layers=8192,4096 --quant --output=generic --tiles=32,32,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=I8F32-PACKED-DEQUANT
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8-f32 --batch=128 --layers=2304,768 --quant --scale-type=f8E8M0FNU --tiles=32,32,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=I8F32-I8SCALE-PACKED-DEQUANT
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=128 --layers=2304,768 --quant 2>&1 | FileCheck %s --check-prefix=I8-REQUANT
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=128 --layers=2304,768 --quant --tiles=32,32,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=I8-REQUANT-PACKED
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=128 --layers=2304,768 --quant --scale-type=f8E8M0FNU 2>&1 | FileCheck %s --check-prefix=I8-REQUANT-I8SCALE
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=128 --layers=2304,768 --quant --scale-type=f8E8M0FNU --tiles=32,32,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=I8-REQUANT-I8SCALE-PACKED


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

// I8-GENERIC: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// I8-GENERIC: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// I8-GENERIC: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// I8-GENERIC-LABEL:   func.func @entry(
// I8-GENERIC-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xi8>,
// I8-GENERIC-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xi8>,
// I8-GENERIC-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32> {
// I8-GENERIC:           %[[VAL_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xi8>, tensor<16x36x64x48xi8>) outs(%[[ARG2]] : tensor<2x16x64x48xi32>) {
// I8-GENERIC:           ^bb0(%[[VAL_1:.*]]: i8, %[[VAL_2:.*]]: i8, %[[VAL_3:.*]]: i32):
// I8-GENERIC:             %[[VAL_4:.*]] = arith.extsi %[[VAL_1]] : i8 to i32
// I8-GENERIC:             %[[VAL_5:.*]] = arith.extsi %[[VAL_2]] : i8 to i32
// I8-GENERIC:             %[[VAL_6:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : i32
// I8-GENERIC:             %[[VAL_7:.*]] = arith.addi %[[VAL_3]], %[[VAL_6]] : i32
// I8-GENERIC:             linalg.yield %[[VAL_7]] : i32
// I8-GENERIC:           } -> tensor<2x16x64x48xi32>
// I8-GENERIC:           return %[[VAL_0]] : tensor<2x16x64x48xi32>
// I8-GENERIC:         }

// I8-CONTRACT: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// I8-CONTRACT: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// I8-CONTRACT: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// I8-CONTRACT-LABEL:   func.func @entry(
// I8-CONTRACT-SAME:                     %[[ARG0:.*]]: tensor<2x36x64x64xi8>,
// I8-CONTRACT-SAME:                     %[[ARG1:.*]]: tensor<16x36x64x48xi8>,
// I8-CONTRACT-SAME:                     %[[ARG2:.*]]: tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32> {
// I8-CONTRACT-NOT:       tensor.empty() : tensor<2x16x64x48xi32>
// I8-CONTRACT:           %[[VAL_3:.*]] = linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG1]] : tensor<2x36x64x64xi8>, tensor<16x36x64x48xi8>) outs(%[[ARG2]] : tensor<2x16x64x48xi32>) -> tensor<2x16x64x48xi32>
// I8-CONTRACT:           return %[[VAL_3]] : tensor<2x16x64x48xi32>
// I8-CONTRACT:         }


// Perform Gemm dequntization using given scales.

// I8F32-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// I8F32-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// I8F32-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// I8F32-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// I8F32-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0)>
// I8F32-DEQUANT-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d1)>
// I8F32-DEQUANT-LABEL:   func.func @entry(
// I8F32-DEQUANT-SAME:                     %arg0: tensor<128x2304xi8>,
// I8F32-DEQUANT-SAME:                     %arg1: tensor<128xf32>,
// I8F32-DEQUANT-SAME:                     %arg2: tensor<2304x768xi8>,
// I8F32-DEQUANT-SAME:                     %arg3: tensor<768xf32>,
// I8F32-DEQUANT-SAME:                     %arg4: tensor<128x768xf32>) -> tensor<128x768xf32> {
// I8F32-DEQUANT:           linalg.contract indexing_maps = [#[[$ATTR_0:.+]], #[[$ATTR_1:.+]], #[[$ATTR_2:.+]]]
// I8F32-DEQUANT:           linalg.generic  {indexing_maps = [#[[$ATTR_3:.+]], #[[$ATTR_4:.+]], #[[$ATTR_5:.+]], #[[$ATTR_3:.+]]], iterator_types = ["parallel", "parallel"]}
// I8F32-DEQUANT:           ^bb0
// I8F32-DEQUANT:             arith.mulf
// I8F32-DEQUANT:             arith.sitofp
// I8F32-DEQUANT:             arith.mulf
// I8F32-DEQUANT:             linalg.yield


// Requantize i8xi8->i32 Gemm output back to i8. The i32 accumulator is
// dequantized with the per-row input scale and per-output-channel weight scale,
// then rescaled with the per-output-channel output scale and saturated to i8.

// I8-REQUANT: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// I8-REQUANT: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// I8-REQUANT: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// I8-REQUANT: #map3 = affine_map<(d0, d1) -> (d0, d1)>
// I8-REQUANT: #map4 = affine_map<(d0, d1) -> (d0)>
// I8-REQUANT: #map5 = affine_map<(d0, d1) -> (d1)>
// I8-REQUANT-LABEL:   func.func @entry(
// I8-REQUANT-SAME:                     %[[ARG0:[a-z0-9_]+]]: tensor<128x2304xi8>,
// I8-REQUANT-SAME:                     %[[ARG1:[a-z0-9_]+]]: tensor<128xf32>,
// I8-REQUANT-SAME:                     %[[ARG2:[a-z0-9_]+]]: tensor<2304x768xi8>,
// I8-REQUANT-SAME:                     %[[ARG3:[a-z0-9_]+]]: tensor<768xf32>,
// I8-REQUANT-SAME:                     %[[ARG4:[a-z0-9_]+]]: tensor<768xf32>,
// I8-REQUANT-SAME:                     %[[ARG5:[a-z0-9_]+]]: tensor<128x768xi8>) -> tensor<128x768xi8> {
// I8-REQUANT:           %[[ZERO:.*]] = arith.constant 0 : i32
// I8-REQUANT:           linalg.fill ins(%[[ZERO]] : i32){{.*}} -> tensor<128x768xi32>
// I8-REQUANT:           linalg.contract indexing_maps = [#map, #map1, #map2] ins(%[[ARG0]], %[[ARG2]] : tensor<128x2304xi8>, tensor<2304x768xi8>) outs({{.*}} : tensor<128x768xi32>) -> tensor<128x768xi32>
// I8-REQUANT:           %[[LOW:.*]] = arith.constant -1.280000e+02 : f32
// I8-REQUANT:           %[[HIGH:.*]] = arith.constant 1.270000e+02 : f32
// I8-REQUANT:           linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5, #map3], iterator_types = ["parallel", "parallel"]} ins({{.*}}, %[[ARG1]], %[[ARG3]], %[[ARG4]] : tensor<128x768xi32>, tensor<128xf32>, tensor<768xf32>, tensor<768xf32>) outs(%[[ARG5]] : tensor<128x768xi8>) {
// I8-REQUANT:           ^bb0(%[[IN:.*]]: i32, %[[INS:.*]]: f32, %[[WS:.*]]: f32, %[[OS:.*]]: f32, %[[OUT:.*]]: i8):
// I8-REQUANT:             %[[F:.*]] = arith.sitofp %[[IN]] : i32 to f32
// I8-REQUANT:             %[[SCALE0:.*]] = arith.mulf %[[INS]], %[[WS]] : f32
// I8-REQUANT:             %[[SCALE1:.*]] = arith.mulf %[[SCALE0]], %[[OS]] : f32
// I8-REQUANT:             %[[MUL:.*]] = arith.mulf %[[F]], %[[SCALE1]] : f32
// I8-REQUANT:             %[[MAX:.*]] = arith.maximumf %[[MUL]], %[[LOW]] : f32
// I8-REQUANT:             %[[MIN:.*]] = arith.minimumf %[[MAX]], %[[HIGH]] : f32
// I8-REQUANT:             %[[I8:.*]] = arith.fptosi %[[MIN]] : f32 to i8
// I8-REQUANT:             linalg.yield %[[I8]] : i8
// I8-REQUANT:           } -> tensor<128x768xi8>


// I8-REQUANT-PACKED: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// I8-REQUANT-PACKED: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// I8-REQUANT-PACKED: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// I8-REQUANT-PACKED: #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// I8-REQUANT-PACKED: #map4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
// I8-REQUANT-PACKED: #map5 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3, 0)>
// I8-REQUANT-PACKED-LABEL:   func.func @entry(
// I8-REQUANT-PACKED-SAME:                     %[[ARG0:[a-z0-9_]+]]: tensor<4x36x32x64xi8>,
// I8-REQUANT-PACKED-SAME:                     %[[ARG1:[a-z0-9_]+]]: tensor<128xf32>,
// I8-REQUANT-PACKED-SAME:                     %[[ARG2:[a-z0-9_]+]]: tensor<24x36x16x32x4xi8>,
// I8-REQUANT-PACKED-SAME:                     %[[ARG3:[a-z0-9_]+]]: tensor<768xf32>,
// I8-REQUANT-PACKED-SAME:                     %[[ARG4:[a-z0-9_]+]]: tensor<768xf32>,
// I8-REQUANT-PACKED-SAME:                     %[[ARG5:[a-z0-9_]+]]: tensor<4x24x32x32xi8>) -> tensor<4x24x32x32xi8> {
// I8-REQUANT-PACKED:           linalg.fill{{.*}} -> tensor<4x24x32x32xi32>
// I8-REQUANT-PACKED:           tensor.expand_shape %[[ARG0]]
// I8-REQUANT-PACKED:           linalg.contract indexing_maps = [#map, #map1, #map2] ins({{.*}}, %[[ARG2]] : tensor<4x36x32x16x4xi8>, tensor<24x36x16x32x4xi8>) outs({{.*}} : tensor<4x24x32x32xi32>) -> tensor<4x24x32x32xi32>
// I8-REQUANT-PACKED:           tensor.expand_shape %[[ARG1]] {{.*}} output_shape [4, 1, 32, 1] : tensor<128xf32> into tensor<4x1x32x1xf32>
// I8-REQUANT-PACKED:           tensor.expand_shape %[[ARG3]] {{.*}} output_shape [24, 1, 32, 1] : tensor<768xf32> into tensor<24x1x32x1xf32>
// I8-REQUANT-PACKED:           tensor.expand_shape %[[ARG4]] {{.*}} output_shape [24, 1, 32, 1] : tensor<768xf32> into tensor<24x1x32x1xf32>
// I8-REQUANT-PACKED:           linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{.*}} : tensor<4x24x32x32xi32>, tensor<4x1x32x1xf32>, tensor<24x1x32x1xf32>, tensor<24x1x32x1xf32>) outs(%[[ARG5]] : tensor<4x24x32x32xi8>) {
// I8-REQUANT-PACKED:             arith.sitofp
// I8-REQUANT-PACKED:             arith.mulf
// I8-REQUANT-PACKED:             arith.mulf
// I8-REQUANT-PACKED:             arith.mulf
// I8-REQUANT-PACKED:             arith.maximumf
// I8-REQUANT-PACKED:             arith.minimumf
// I8-REQUANT-PACKED:             arith.fptosi
// I8-REQUANT-PACKED:             linalg.yield


// Requantize with narrow (f8E8M0FNU) scales. The scales are extended to f32
// before being combined and applied to the i32 accumulator.

// I8-REQUANT-I8SCALE: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// I8-REQUANT-I8SCALE: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// I8-REQUANT-I8SCALE: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// I8-REQUANT-I8SCALE: #map3 = affine_map<(d0, d1) -> (d0, d1)>
// I8-REQUANT-I8SCALE: #map4 = affine_map<(d0, d1) -> (d0)>
// I8-REQUANT-I8SCALE: #map5 = affine_map<(d0, d1) -> (d1)>
// I8-REQUANT-I8SCALE-LABEL:   func.func @entry(
// I8-REQUANT-I8SCALE-SAME:                     %[[ARG0:[a-z0-9_]+]]: tensor<128x2304xi8>,
// I8-REQUANT-I8SCALE-SAME:                     %[[ARG1:[a-z0-9_]+]]: tensor<128xf8E8M0FNU>,
// I8-REQUANT-I8SCALE-SAME:                     %[[ARG2:[a-z0-9_]+]]: tensor<2304x768xi8>,
// I8-REQUANT-I8SCALE-SAME:                     %[[ARG3:[a-z0-9_]+]]: tensor<768xf8E8M0FNU>,
// I8-REQUANT-I8SCALE-SAME:                     %[[ARG4:[a-z0-9_]+]]: tensor<768xf8E8M0FNU>,
// I8-REQUANT-I8SCALE-SAME:                     %[[ARG5:[a-z0-9_]+]]: tensor<128x768xi8>) -> tensor<128x768xi8> {
// I8-REQUANT-I8SCALE:           %[[ZERO:.*]] = arith.constant 0 : i32
// I8-REQUANT-I8SCALE:           linalg.fill ins(%[[ZERO]] : i32){{.*}} -> tensor<128x768xi32>
// I8-REQUANT-I8SCALE:           linalg.contract indexing_maps = [#map, #map1, #map2] ins(%[[ARG0]], %[[ARG2]] : tensor<128x2304xi8>, tensor<2304x768xi8>) outs({{.*}} : tensor<128x768xi32>) -> tensor<128x768xi32>
// I8-REQUANT-I8SCALE:           %[[LOW:.*]] = arith.constant -1.280000e+02 : f32
// I8-REQUANT-I8SCALE:           %[[HIGH:.*]] = arith.constant 1.270000e+02 : f32
// I8-REQUANT-I8SCALE:           linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5, #map3], iterator_types = ["parallel", "parallel"]} ins({{.*}}, %[[ARG1]], %[[ARG3]], %[[ARG4]] : tensor<128x768xi32>, tensor<128xf8E8M0FNU>, tensor<768xf8E8M0FNU>, tensor<768xf8E8M0FNU>) outs(%[[ARG5]] : tensor<128x768xi8>) {
// I8-REQUANT-I8SCALE:           ^bb0(%[[IN:.*]]: i32, %[[INS:.*]]: f8E8M0FNU, %[[WS:.*]]: f8E8M0FNU, %[[OS:.*]]: f8E8M0FNU, %[[OUT:.*]]: i8):
// I8-REQUANT-I8SCALE:             %[[F:.*]] = arith.sitofp %[[IN]] : i32 to f32
// I8-REQUANT-I8SCALE:             %[[EINS:.*]] = arith.extf %[[INS]] : f8E8M0FNU to f32
// I8-REQUANT-I8SCALE:             %[[EWS:.*]] = arith.extf %[[WS]] : f8E8M0FNU to f32
// I8-REQUANT-I8SCALE:             %[[EOS:.*]] = arith.extf %[[OS]] : f8E8M0FNU to f32
// I8-REQUANT-I8SCALE:             %[[SCALE0:.*]] = arith.mulf %[[EINS]], %[[EWS]] : f32
// I8-REQUANT-I8SCALE:             %[[SCALE1:.*]] = arith.mulf %[[SCALE0]], %[[EOS]] : f32
// I8-REQUANT-I8SCALE:             %[[MUL:.*]] = arith.mulf %[[F]], %[[SCALE1]] : f32
// I8-REQUANT-I8SCALE:             %[[MAX:.*]] = arith.maximumf %[[MUL]], %[[LOW]] : f32
// I8-REQUANT-I8SCALE:             %[[MIN:.*]] = arith.minimumf %[[MAX]], %[[HIGH]] : f32
// I8-REQUANT-I8SCALE:             %[[I8:.*]] = arith.fptosi %[[MIN]] : f32 to i8
// I8-REQUANT-I8SCALE:             linalg.yield %[[I8]] : i8
// I8-REQUANT-I8SCALE:           } -> tensor<128x768xi8>


// Transposed A/B operand layouts. The prefix encodes the layout: N = normal,
// T = transposed; first letter is A, second is B (e.g. TN = A transposed, B
// flat). Covers the four flat (packed, non-VNNI) combinations and the VNNI
// (--transpose-a with --vnni) path for bf16, i8, bf8 and hf8.

// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf16 --batch=256 --layers=512,512 --tiles=32,32,32 --transpose-a=0 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=NN-BF16
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf16 --batch=256 --layers=512,512 --tiles=32,32,32 --transpose-a=1 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=TN-BF16
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf16 --batch=256 --layers=512,512 --tiles=32,32,32 --transpose-a=0 --transpose-b=1 2>&1 | FileCheck %s --check-prefix=NT-BF16
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf16 --batch=256 --layers=512,512 --tiles=32,32,32 --transpose-a=1 --transpose-b=1 2>&1 | FileCheck %s --check-prefix=TT-BF16
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=256 --layers=512,512 --tiles=32,32,64 --output=contract --transpose-a=0 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=NN-I8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=256 --layers=512,512 --tiles=32,32,64 --output=contract --transpose-a=1 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=TN-I8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=256 --layers=512,512 --tiles=32,32,64 --output=contract --transpose-a=0 --transpose-b=1 2>&1 | FileCheck %s --check-prefix=NT-I8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=256 --layers=512,512 --tiles=32,32,64 --output=contract --transpose-a=1 --transpose-b=1 2>&1 | FileCheck %s --check-prefix=TT-I8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf8 --batch=256 --layers=512,512 --tiles=32,32,64 --transpose-a=0 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=NN-BF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf8 --batch=256 --layers=512,512 --tiles=32,32,64 --transpose-a=1 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=TN-BF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf8 --batch=256 --layers=512,512 --tiles=32,32,64 --transpose-a=0 --transpose-b=1 2>&1 | FileCheck %s --check-prefix=NT-BF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf8 --batch=256 --layers=512,512 --tiles=32,32,64 --transpose-a=1 --transpose-b=1 2>&1 | FileCheck %s --check-prefix=TT-BF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=hf8 --batch=256 --layers=512,512 --tiles=32,32,64 --transpose-a=0 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=NN-HF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=hf8 --batch=256 --layers=512,512 --tiles=32,32,64 --transpose-a=1 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=TN-HF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=hf8 --batch=256 --layers=512,512 --tiles=32,32,64 --transpose-a=0 --transpose-b=1 2>&1 | FileCheck %s --check-prefix=NT-HF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=hf8 --batch=256 --layers=512,512 --tiles=32,32,64 --transpose-a=1 --transpose-b=1 2>&1 | FileCheck %s --check-prefix=TT-HF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf16 --batch=256 --layers=512,512 --tiles=32,32,32 --vnni=2 --transpose-a=1 2>&1 | FileCheck %s --check-prefix=TA-VNNI-BF16
// RUN: mlir-gen --kernel=args --seed=0 --data-type=i8 --batch=256 --layers=512,512 --tiles=32,32,64 --output=contract --vnni=4 2>&1 | FileCheck %s --check-prefix=VNNI-I8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf8 --batch=256 --layers=512,512 --tiles=32,32,64 --vnni=4 --transpose-a=1 2>&1 | FileCheck %s --check-prefix=TA-VNNI-BF8
// RUN: mlir-gen --kernel=args --seed=0 --data-type=hf8 --batch=256 --layers=512,512 --tiles=32,32,64 --vnni=4 2>&1 | FileCheck %s --check-prefix=VNNI-HF8

// Un-tiled (pytorch-style) flat GEMM stays in the natural NN layout; the
// pytorch path does not support transposed operands.
// RUN: mlir-gen --kernel=args --seed=0 --data-type=bf16 --batch=256 --layers=512,512 --transpose-a=0 --transpose-b=0 2>&1 | FileCheck %s --check-prefix=FLAT-NN-BF16


// ===== bf16 flat combinations =====

// NN-BF16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// NN-BF16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// NN-BF16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// NN-BF16: func.func @entry(%arg0: tensor<8x16x32x32xbf16>, %arg1: tensor<16x16x32x32xbf16>, %arg2: tensor<8x16x32x32xbf16>) -> tensor<8x16x32x32xbf16>
// NN-BF16: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// TN-BF16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// TN-BF16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// TN-BF16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// TN-BF16: func.func @entry(%arg0: tensor<16x8x32x32xbf16>, %arg1: tensor<16x16x32x32xbf16>, %arg2: tensor<8x16x32x32xbf16>) -> tensor<8x16x32x32xbf16>
// TN-BF16: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// NT-BF16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// NT-BF16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// NT-BF16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// NT-BF16: func.func @entry(%arg0: tensor<8x16x32x32xbf16>, %arg1: tensor<16x16x32x32xbf16>, %arg2: tensor<8x16x32x32xbf16>) -> tensor<8x16x32x32xbf16>
// NT-BF16: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// TT-BF16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// TT-BF16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// TT-BF16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// TT-BF16: func.func @entry(%arg0: tensor<16x8x32x32xbf16>, %arg1: tensor<16x16x32x32xbf16>, %arg2: tensor<8x16x32x32xbf16>) -> tensor<8x16x32x32xbf16>
// TT-BF16: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// ===== i8 flat combinations (i8, linalg.contract) =====

// NN-I8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// NN-I8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// NN-I8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// NN-I8: func.func @entry(%arg0: tensor<8x8x32x64xi8>, %arg1: tensor<16x8x64x32xi8>, %arg2: tensor<8x16x32x32xi32>) -> tensor<8x16x32x32xi32>
// NN-I8: linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<8x8x32x64xi8>, tensor<16x8x64x32xi8>) outs(%arg2 : tensor<8x16x32x32xi32>)

// TN-I8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// TN-I8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// TN-I8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// TN-I8: func.func @entry(%arg0: tensor<8x8x64x32xi8>, %arg1: tensor<16x8x64x32xi8>, %arg2: tensor<8x16x32x32xi32>) -> tensor<8x16x32x32xi32>
// TN-I8: linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<8x8x64x32xi8>, tensor<16x8x64x32xi8>) outs(%arg2 : tensor<8x16x32x32xi32>)

// NT-I8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// NT-I8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// NT-I8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// NT-I8: func.func @entry(%arg0: tensor<8x8x32x64xi8>, %arg1: tensor<16x8x32x64xi8>, %arg2: tensor<8x16x32x32xi32>) -> tensor<8x16x32x32xi32>
// NT-I8: linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<8x8x32x64xi8>, tensor<16x8x32x64xi8>) outs(%arg2 : tensor<8x16x32x32xi32>)

// TT-I8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// TT-I8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// TT-I8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// TT-I8: func.func @entry(%arg0: tensor<8x8x64x32xi8>, %arg1: tensor<16x8x32x64xi8>, %arg2: tensor<8x16x32x32xi32>) -> tensor<8x16x32x32xi32>
// TT-I8: linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<8x8x64x32xi8>, tensor<16x8x32x64xi8>) outs(%arg2 : tensor<8x16x32x32xi32>)

// ===== bf8 flat combinations =====

// NN-BF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// NN-BF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// NN-BF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// NN-BF8: func.func @entry(%arg0: tensor<8x8x32x64xf8E5M2>, %arg1: tensor<16x8x64x32xf8E5M2>, %arg2: tensor<8x16x32x32xf8E5M2>) -> tensor<8x16x32x32xf8E5M2>
// NN-BF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// TN-BF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// TN-BF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// TN-BF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// TN-BF8: func.func @entry(%arg0: tensor<8x8x64x32xf8E5M2>, %arg1: tensor<16x8x64x32xf8E5M2>, %arg2: tensor<8x16x32x32xf8E5M2>) -> tensor<8x16x32x32xf8E5M2>
// TN-BF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// NT-BF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// NT-BF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// NT-BF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// NT-BF8: func.func @entry(%arg0: tensor<8x8x32x64xf8E5M2>, %arg1: tensor<16x8x32x64xf8E5M2>, %arg2: tensor<8x16x32x32xf8E5M2>) -> tensor<8x16x32x32xf8E5M2>
// NT-BF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// TT-BF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// TT-BF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// TT-BF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// TT-BF8: func.func @entry(%arg0: tensor<8x8x64x32xf8E5M2>, %arg1: tensor<16x8x32x64xf8E5M2>, %arg2: tensor<8x16x32x32xf8E5M2>) -> tensor<8x16x32x32xf8E5M2>
// TT-BF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// ===== hf8 flat combinations =====

// NN-HF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// NN-HF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// NN-HF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// NN-HF8: func.func @entry(%arg0: tensor<8x8x32x64xf8E4M3FN>, %arg1: tensor<16x8x64x32xf8E4M3FN>, %arg2: tensor<8x16x32x32xf8E4M3FN>) -> tensor<8x16x32x32xf8E4M3FN>
// NN-HF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// TN-HF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// TN-HF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
// TN-HF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// TN-HF8: func.func @entry(%arg0: tensor<8x8x64x32xf8E4M3FN>, %arg1: tensor<16x8x64x32xf8E4M3FN>, %arg2: tensor<8x16x32x32xf8E4M3FN>) -> tensor<8x16x32x32xf8E4M3FN>
// TN-HF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// NT-HF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
// NT-HF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// NT-HF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// NT-HF8: func.func @entry(%arg0: tensor<8x8x32x64xf8E4M3FN>, %arg1: tensor<16x8x32x64xf8E4M3FN>, %arg2: tensor<8x16x32x32xf8E4M3FN>) -> tensor<8x16x32x32xf8E4M3FN>
// NT-HF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// TT-HF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d5, d3)>
// TT-HF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
// TT-HF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
// TT-HF8: func.func @entry(%arg0: tensor<8x8x64x32xf8E4M3FN>, %arg1: tensor<16x8x32x64xf8E4M3FN>, %arg2: tensor<8x16x32x32xf8E4M3FN>) -> tensor<8x16x32x32xf8E4M3FN>
// TT-HF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]

// ===== VNNI transpose combinations =====

// bf16, VNNI, A transposed + VNNI packed (5D pre-packed arg). A stays in the
// transposed VNNI-A layout and is consumed directly through the transposed
// VNNI-A input map (no relayout for the single external argument).
// TA-VNNI-BF16-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d0, d6, d4, d3)>
// TA-VNNI-BF16-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// TA-VNNI-BF16-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// TA-VNNI-BF16: func.func @entry(%arg0: tensor<16x8x16x32x2xbf16>, %arg1: tensor<16x16x16x32x2xbf16>, %arg2: tensor<8x16x32x32xbf16>) -> tensor<8x16x32x32xbf16>
// TA-VNNI-BF16-NOT: linalg.transpose
// TA-VNNI-BF16: linalg.generic {indexing_maps = [#map, #map1, #map2]{{.*}}ins(%arg0, %arg1 : tensor<16x8x16x32x2xbf16>, tensor<16x16x16x32x2xbf16>)

// i8, VNNI, A flat + B VNNI: A stays 4D and is expanded internally.
// VNNI-I8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// VNNI-I8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// VNNI-I8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// VNNI-I8: func.func @entry(%arg0: tensor<8x8x32x64xi8>, %arg1: tensor<16x8x16x32x4xi8>, %arg2: tensor<8x16x32x32xi32>) -> tensor<8x16x32x32xi32>
// VNNI-I8: tensor.expand_shape %arg0 {{\[\[}}0], [1], [2], [3, 4]] output_shape [8, 8, 32, 16, 4] : tensor<8x8x32x64xi8> into tensor<8x8x32x16x4xi8>
// VNNI-I8: linalg.contract indexing_maps = [#map, #map1, #map2]

// bf8, VNNI, A transposed + VNNI packed. A stays in the transposed VNNI-A
// layout and is consumed directly through the transposed VNNI-A input map.
// TA-VNNI-BF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d2, d0, d6, d4, d3)>
// TA-VNNI-BF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// TA-VNNI-BF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// TA-VNNI-BF8: func.func @entry(%arg0: tensor<8x8x16x32x4xf8E5M2>, %arg1: tensor<16x8x16x32x4xf8E5M2>, %arg2: tensor<8x16x32x32xf8E5M2>) -> tensor<8x16x32x32xf8E5M2>
// TA-VNNI-BF8-NOT: linalg.transpose
// TA-VNNI-BF8: linalg.generic {indexing_maps = [#map, #map1, #map2]{{.*}}ins(%arg0, %arg1 : tensor<8x8x16x32x4xf8E5M2>, tensor<16x8x16x32x4xf8E5M2>)

// hf8, VNNI, A flat + B VNNI.
// VNNI-HF8-DAG: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// VNNI-HF8-DAG: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// VNNI-HF8-DAG: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// VNNI-HF8: func.func @entry(%arg0: tensor<8x8x32x64xf8E4M3FN>, %arg1: tensor<16x8x16x32x4xf8E4M3FN>, %arg2: tensor<8x16x32x32xf8E4M3FN>) -> tensor<8x16x32x32xf8E4M3FN>
// VNNI-HF8: tensor.expand_shape %arg0 {{\[\[}}0], [1], [2], [3, 4]] output_shape [8, 8, 32, 16, 4] : tensor<8x8x32x64xf8E4M3FN> into tensor<8x8x32x16x4xf8E4M3FN>
// VNNI-HF8: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "reduction", "reduction", "parallel", "parallel", "reduction"]

// ===== bf16 un-tiled (pytorch-style) flat GEMM =====
// The pytorch path only emits the plain NN contraction.

// FLAT-NN-BF16-DAG: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// FLAT-NN-BF16-DAG: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// FLAT-NN-BF16-DAG: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// FLAT-NN-BF16: func.func @entry(%arg0: tensor<256x512xbf16>, %arg1: tensor<512x512xbf16>, %arg2: tensor<256x512xbf16>) -> tensor<256x512xbf16>
// FLAT-NN-BF16-NOT: linalg.transpose
// FLAT-NN-BF16: linalg.generic {{.*}}ins(%arg0, %arg1 : tensor<256x512xbf16>, tensor<512x512xbf16>)

// Packed (tiled + VNNI) requantize with narrow (f8E8M0FNU) scales.

// I8-REQUANT-I8SCALE-PACKED: #map = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// I8-REQUANT-I8SCALE-PACKED: #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// I8-REQUANT-I8SCALE-PACKED: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// I8-REQUANT-I8SCALE-PACKED: #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// I8-REQUANT-I8SCALE-PACKED: #map4 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
// I8-REQUANT-I8SCALE-PACKED: #map5 = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3, 0)>
// I8-REQUANT-I8SCALE-PACKED-LABEL:   func.func @entry(
// I8-REQUANT-I8SCALE-PACKED-SAME:                     %[[ARG0:[a-z0-9_]+]]: tensor<4x36x32x64xi8>,
// I8-REQUANT-I8SCALE-PACKED-SAME:                     %[[ARG1:[a-z0-9_]+]]: tensor<128xf8E8M0FNU>,
// I8-REQUANT-I8SCALE-PACKED-SAME:                     %[[ARG2:[a-z0-9_]+]]: tensor<24x36x16x32x4xi8>,
// I8-REQUANT-I8SCALE-PACKED-SAME:                     %[[ARG3:[a-z0-9_]+]]: tensor<768xf8E8M0FNU>,
// I8-REQUANT-I8SCALE-PACKED-SAME:                     %[[ARG4:[a-z0-9_]+]]: tensor<768xf8E8M0FNU>,
// I8-REQUANT-I8SCALE-PACKED-SAME:                     %[[ARG5:[a-z0-9_]+]]: tensor<4x24x32x32xi8>) -> tensor<4x24x32x32xi8> {
// I8-REQUANT-I8SCALE-PACKED:           linalg.fill{{.*}} -> tensor<4x24x32x32xi32>
// I8-REQUANT-I8SCALE-PACKED:           tensor.expand_shape %[[ARG0]]
// I8-REQUANT-I8SCALE-PACKED:           linalg.contract indexing_maps = [#map, #map1, #map2] ins({{.*}}, %[[ARG2]] : tensor<4x36x32x16x4xi8>, tensor<24x36x16x32x4xi8>) outs({{.*}} : tensor<4x24x32x32xi32>) -> tensor<4x24x32x32xi32>
// I8-REQUANT-I8SCALE-PACKED:           tensor.expand_shape %[[ARG1]] {{.*}} output_shape [4, 1, 32, 1] : tensor<128xf8E8M0FNU> into tensor<4x1x32x1xf8E8M0FNU>
// I8-REQUANT-I8SCALE-PACKED:           tensor.expand_shape %[[ARG3]] {{.*}} output_shape [24, 1, 32, 1] : tensor<768xf8E8M0FNU> into tensor<24x1x32x1xf8E8M0FNU>
// I8-REQUANT-I8SCALE-PACKED:           tensor.expand_shape %[[ARG4]] {{.*}} output_shape [24, 1, 32, 1] : tensor<768xf8E8M0FNU> into tensor<24x1x32x1xf8E8M0FNU>
// I8-REQUANT-I8SCALE-PACKED:           linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins({{.*}} : tensor<4x24x32x32xi32>, tensor<4x1x32x1xf8E8M0FNU>, tensor<24x1x32x1xf8E8M0FNU>, tensor<24x1x32x1xf8E8M0FNU>) outs(%[[ARG5]] : tensor<4x24x32x32xi8>) {
// I8-REQUANT-I8SCALE-PACKED:             arith.sitofp
// I8-REQUANT-I8SCALE-PACKED:             arith.extf {{.*}} f8E8M0FNU to f32
// I8-REQUANT-I8SCALE-PACKED:             arith.extf {{.*}} f8E8M0FNU to f32
// I8-REQUANT-I8SCALE-PACKED:             arith.extf {{.*}} f8E8M0FNU to f32
// I8-REQUANT-I8SCALE-PACKED:             arith.mulf
// I8-REQUANT-I8SCALE-PACKED:             arith.mulf
// I8-REQUANT-I8SCALE-PACKED:             arith.mulf
// I8-REQUANT-I8SCALE-PACKED:             arith.maximumf
// I8-REQUANT-I8SCALE-PACKED:             arith.minimumf
// I8-REQUANT-I8SCALE-PACKED:             arith.fptosi
// I8-REQUANT-I8SCALE-PACKED:             linalg.yield


// I8F32-PACKED-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// I8F32-PACKED-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// I8F32-PACKED-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// I8F32-PACKED-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// I8F32-PACKED-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
// I8F32-PACKED-DEQUANT-LABEL:   func.func @entry(
// I8F32-PACKED-DEQUANT-SAME:                     {{.*}}: tensor<128x128x32x64xi8>,
// I8F32-PACKED-DEQUANT-SAME:                     {{.*}}: tensor<4096xf32>, {{.*}}: tensor<128x128x16x32x4xi8>,
// I8F32-PACKED-DEQUANT-SAME:                     {{.*}}: tensor<4096xf32>,
// I8F32-PACKED-DEQUANT-SAME:                     {{.*}}: tensor<128x128x32x32xf32>) -> tensor<128x128x32x32xf32> {

// I8F32-PACKED-DEQUANT:           linalg.fill
// I8F32-PACKED-DEQUANT:           tensor.expand_shape
// I8F32-PACKED-DEQUANT:           linalg.contract  indexing_maps = [#[[$ATTR_0:.+]], #[[$ATTR_1:.+]], #[[$ATTR_2:.+]]]
// I8F32-PACKED-DEQUANT-2:         tensor.expand_shape
// I8F32-PACKED-DEQUANT:           linalg.generic {indexing_maps = [#[[$ATTR_3:.+]], #[[$ATTR_4:.+]], #[[$ATTR_4:.+]], #[[$ATTR_3:.+]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// I8F32-PACKED-DEQUANT:           ^bb0
// I8F32-PACKED-DEQUANT:             arith.mulf
// I8F32-PACKED-DEQUANT:             arith.sitofp
// I8F32-PACKED-DEQUANT:             arith.mulf
// I8F32-PACKED-DEQUANT:             linalg.yield


// I8F32-I8SCALE-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// I8F32-I8SCALE-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// I8F32-I8SCALE-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// I8F32-I8SCALE-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// I8F32-I8SCALE-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0)>
// I8F32-I8SCALE-DEQUANT-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d1)>
// I8F32-I8SCALE-DEQUANT-LABEL:   func.func @entry(
// I8F32-I8SCALE-DEQUANT-SAME:                     %[[ARG0:.*]]: tensor<128x2304xi8>,
// I8F32-I8SCALE-DEQUANT-SAME:                     %[[ARG1:.*]]: tensor<128xf8E8M0FNU>,
// I8F32-I8SCALE-DEQUANT-SAME:                     %[[ARG2:.*]]: tensor<2304x768xi8>,
// I8F32-I8SCALE-DEQUANT-SAME:                     %[[ARG3:.*]]: tensor<768xf8E8M0FNU>,
// I8F32-I8SCALE-DEQUANT-SAME:                     %[[ARG4:.*]]: tensor<128x768xf32>) -> tensor<128x768xf32> {
// I8F32-I8SCALE-DEQUANT:           arith.constant 0 : i32
// I8F32-I8SCALE-DEQUANT:           tensor.empty() : tensor<128x768xi32>
// I8F32-I8SCALE-DEQUANT:           linalg.fill
// I8F32-I8SCALE-DEQUANT:           linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]] ins(%[[ARG0]], %[[ARG2]]
// I8F32-I8SCALE-DEQUANT:           linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_4]], #[[$ATTR_5]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel"]}
// I8F32-I8SCALE-DEQUANT:           ^bb0
// I8F32-I8SCALE-DEQUANT-2:             arith.extf {{.*}} fastmath<nnan>
// I8F32-I8SCALE-DEQUANT:               arith.mulf
// I8F32-I8SCALE-DEQUANT:               arith.sitofp
// I8F32-I8SCALE-DEQUANT:               arith.mulf
// I8F32-I8SCALE-DEQUANT:               linalg.yield
// I8F32-I8SCALE-DEQUANT:           } -> tensor<128x768xf32>


// I8F32-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6, d3)>
// I8F32-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d2, d6, d5, d3)>
// I8F32-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d5)>
// I8F32-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// I8F32-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, 0)>
// I8F32-I8SCALE-PACKED-DEQUANT-DAG: #[[$ATTR_5:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3, 0)>
// I8F32-I8SCALE-PACKED-DEQUANT-LABEL:   func.func @entry(
// I8F32-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG0:.*]]: tensor<4x36x32x64xi8>,
// I8F32-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG1:.*]]: tensor<128xf8E8M0FNU>,
// I8F32-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG2:.*]]: tensor<24x36x16x32x4xi8>,
// I8F32-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG3:.*]]: tensor<768xf8E8M0FNU>,
// I8F32-I8SCALE-PACKED-DEQUANT-SAME:                     %[[ARG4:.*]]: tensor<4x24x32x32xf32>) -> tensor<4x24x32x32xf32> {
// I8F32-I8SCALE-PACKED-DEQUANT:           arith.constant 0 : i32
// I8F32-I8SCALE-PACKED-DEQUANT:           tensor.empty() : tensor<4x24x32x32xi32>
// I8F32-I8SCALE-PACKED-DEQUANT:           linalg.fill
// I8F32-I8SCALE-PACKED-DEQUANT:           tensor.expand_shape %[[ARG0]] {{\[\[}}0], [1], [2], [3, 4]] output_shape [4, 36, 32, 16, 4] : tensor<4x36x32x64xi8> into tensor<4x36x32x16x4xi8>
// I8F32-I8SCALE-PACKED-DEQUANT:           linalg.contract indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]], #[[$ATTR_2]]]
// I8F32-I8SCALE-PACKED-DEQUANT:           tensor.expand_shape %[[ARG1]] {{\[\[}}0, 1, 2, 3]] output_shape [4, 1, 32, 1] : tensor<128xf8E8M0FNU> into tensor<4x1x32x1xf8E8M0FNU>
// I8F32-I8SCALE-PACKED-DEQUANT:           tensor.expand_shape %[[ARG3]] {{\[\[}}0, 1, 2, 3]] output_shape [24, 1, 32, 1] : tensor<768xf8E8M0FNU> into tensor<24x1x32x1xf8E8M0FNU>
// I8F32-I8SCALE-PACKED-DEQUANT:           linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_4]], #[[$ATTR_5]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
// I8F32-I8SCALE-PACKED-DEQUANT:           ^bb0
// I8F32-I8SCALE-PACKED-DEQUANT-2:             arith.extf {{.*}} fastmath<nnan>
// I8F32-I8SCALE-PACKED-DEQUANT:               arith.mulf
// I8F32-I8SCALE-PACKED-DEQUANT:               arith.sitofp
// I8F32-I8SCALE-PACKED-DEQUANT:               arith.mulf
// I8F32-I8SCALE-PACKED-DEQUANT:             linalg.yield
// I8F32-I8SCALE-PACKED-DEQUANT:           } -> tensor<4x24x32x32xf32>
