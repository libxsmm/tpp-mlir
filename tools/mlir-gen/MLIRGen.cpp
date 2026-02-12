//===- MLIRGen.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinDialect.h"

#include "MLIRGen.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// #include "mlir/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

namespace {

void parseStringList(StringRef str, SmallVector<int64_t> &list) {
  if (str.empty())
    return;
  SmallVector<StringRef> sizeStrs;
  str.split(sizeStrs, ",");
  for (auto str : sizeStrs) {
    APInt i;
    str.getAsInteger(10, i);
    auto val = i.getZExtValue();
    assert(val != 0 && "Size cannot be zero");
    list.push_back(val);
  }
}

/// Returns the vector of boolean for the required broadcast dimensions
static SmallVector<bool> getBroadcastDims(ArrayRef<int64_t> sourceShape,
                                          ArrayRef<int64_t> targetShape) {
  SmallVector<bool> broadcastDims;
  int sourceIdx = sourceShape.size() - 1;
  int targetIdx = targetShape.size() - 1;

  while (targetIdx >= 0) {
    if (sourceIdx >= 0 && sourceShape[sourceIdx] == targetShape[targetIdx]) {
      broadcastDims.push_back(false);
      sourceIdx--;
    } else {
      broadcastDims.push_back(true);
    }
    targetIdx--;
  }

  std::reverse(broadcastDims.begin(), broadcastDims.end());
  return broadcastDims;
}

// Helper function to create the expand_tensor operation.
static Value createExpandedScaleTensor(OpBuilder &builder, Location loc,
                                       Value scale, SmallVector<int64_t> tiles,
                                       bool isInputScale = false) {
  auto outputScaleTy = cast<ShapedType>(scale.getType());
  assert(outputScaleTy.getRank() == 1 && "Scale must be 1-D");
  auto shape = outputScaleTy.getShape();
  SmallVector<int64_t, 4> scaleShapes = {1, 1, 1, 1};
  auto tiledDim = isInputScale ? 0 : 1;
  auto tileFactor = tiles[tiledDim];
  scaleShapes[0] = shape[0] / tileFactor;
  scaleShapes[2] = tileFactor;
  auto packedScaleTy =
      RankedTensorType::get(scaleShapes, outputScaleTy.getElementType());
  SmallVector<ReassociationIndices> reassociationIndices;
  reassociationIndices.push_back({0, 1, 2, 3});
  scale = tensor::ExpandShapeOp::create(builder, loc, packedScaleTy, scale,
                                                reassociationIndices);
  return scale;
}

} // anonymous namespace

MLIRGenerator::MLIRGenerator(StringRef outputOpKindStr, StringRef kernelStr,
                             unsigned batch, StringRef layersStr,
                             StringRef tilesStr, StringRef targetType,
                             StringRef scaleType, StringRef quantizationTypeStr,
                             int seed, bool identity, bool enableBias,
                             bool enableRelu, bool enableSoftmax,
                             bool keepGenericMatmul, int vnniBlockingFactor)
    : builder(&context), loc(builder.getUnknownLoc()), batch(batch), seed(seed),
      identity(identity), flops(0), enableBias(enableBias),
      enableRelu(enableRelu), enableSoftmax(enableSoftmax),
      keepGenericMatmul(keepGenericMatmul), vnniFactor(vnniBlockingFactor) {

  // Register all necessary dialects
  context
      .loadDialect<mlir::BuiltinDialect, func::FuncDialect,
                   bufferization::BufferizationDialect, tensor::TensorDialect,
                   linalg::LinalgDialect, math::MathDialect,
                   arith::ArithDialect, scf::SCFDialect>();

  // Parse output Op kind
  auto optOutputOpKind =
      llvm::StringSwitch<std::optional<OutputOpKind>>(outputOpKindStr)
          .CaseLower("generic", OutputOpKind::Generic)
          .CaseLower("contract", OutputOpKind::Contract)
          .CaseLower("named", OutputOpKind::NamedOp)
          .Default(std::nullopt);
  assert(optOutputOpKind && "Invalid output Op kind");
  assert(!(optOutputOpKind == OutputOpKind::Contract && keepGenericMatmul) &&
         "Can't keep generic matmul with contract");
  outputOpKind = *optOutputOpKind;

  // Parse kernel type
  auto optKernel = llvm::StringSwitch<std::optional<KernelType>>(kernelStr)
                       .CaseLower("const", KernelType::Const)
                       .CaseLower("args", KernelType::Args)
                       .Default(std::nullopt);
  assert(optKernel && "Invalid kernel type");
  kernelType = *optKernel;

  // Argument validation
  assert(batch != 0 && "Batch cannot be zero");

  // Parse hidden layer sizes
  parseStringList(layersStr, layers);
  assert(layers.size() >= 2 && "Must have at least input/output layers");

  // Parse tile sizes
  parseStringList(tilesStr, tiles);
  assert((tiles.size() == 0 || tiles.size() == 3) &&
         "Must have 3 tile sizes (or none)");

  // Pick data type
  auto elementType =
      llvm::StringSwitch<std::optional<SmallVector<mlir::Type>>>(targetType)
          .CaseLower("f32", SmallVector<Type>{builder.getF32Type(),
                                              builder.getF32Type()})
          .CaseLower("f16", SmallVector<Type>{builder.getF16Type(),
                                              builder.getF16Type()})
          .CaseLower("bf16", SmallVector<Type>{builder.getBF16Type(),
                                               builder.getBF16Type()})
          .CaseLower("mx-bf16", SmallVector<Type>{builder.getBF16Type(),
                                                  builder.getF32Type()})
          .CaseLower("mx-f16", SmallVector<Type>{builder.getF16Type(),
                                                 builder.getF32Type()})
          .CaseLower("mx-i8", SmallVector<Type>{builder.getIntegerType(8),
                                                builder.getI32Type()})
          .CaseLower("mx-i8-i32", SmallVector<Type>{builder.getIntegerType(8),
                                                    builder.getI32Type()})
          .CaseLower("mx-i8-f32", SmallVector<Type>{builder.getIntegerType(8),
                                                    builder.getF32Type()})
          .CaseLower("mx-f32-i8", SmallVector<Type>{builder.getF32Type(),
                                                    builder.getIntegerType(8)})
          .Default(std::nullopt);
  assert(elementType && "Unsupported data type");
  dataTypes.push_back((*elementType)[0]);
  dataTypes.push_back((*elementType)[1]);

  auto scaleTypeOpt = llvm::StringSwitch<std::optional<Type>>(scaleType)
                          .CaseLower("f32", builder.getF32Type())
                          .CaseLower("i32", builder.getI32Type())
                          .CaseLower("", builder.getF32Type())
                          .Default(std::nullopt);
  assert(scaleTypeOpt && "Unsupported scale type");
  dataTypes.push_back(*scaleTypeOpt);

  // Parse quantization type
  auto optQuantType =
      llvm::StringSwitch<std::optional<QuantizationType>>(quantizationTypeStr)
          .CaseLower("mixed", QuantizationType::Mixed)
          .CaseLower("quantize", QuantizationType::Quant)
          .CaseLower("dequantize", QuantizationType::Dequant)
          .CaseLower("testquant", QuantizationType::QuantDequant)
          .Default(QuantizationType::None);
  quantType = *optQuantType;

  // If the target type contains "mx", it is a mixed precision type. If
  // quantization type is not explicitly specified, we will default to Mixed
  // quantization type for mixed precision target types.
  bool hasMixedType = !targetType.empty() && targetType.contains("mx");
  if (hasMixedType && quantType == QuantizationType::None)
    quantType = QuantizationType::Mixed;

  // const kernelType is only supported for non quantization kernel.
  assert(!(kernelType == KernelType::Const &&
           quantType == QuantizationType::Quant) &&
         "Const kernel type is only supported for non quantization kernel");

  // Update output kind to 'contract' if quantization is enabled.
  if (quantType != QuantizationType::None)
    outputOpKind = OutputOpKind::Contract;

  // Disable VNNI packing if it is not a F16/BF16/I8 data type
  if (!dataTypes[0].isBF16() && !dataTypes[0].isF16() &&
      !dataTypes[0].isInteger(8))
    vnniFactor = 0;
  assert(((vnniFactor >= 0) && (vnniFactor % 2 == 0)) &&
         "Invalid VNNI packing factor");

  // Use VNNI packed format if both tiles and VNNI factor are specified.
  vnniPacked = tiles.size() > 0 && vnniFactor != 0;

  // Initialize random seed, if needed
  if (seed) {
    initType = TensorInitType::Normal;
    srand(seed);
  } else {
    initType = TensorInitType::Constant;
  }

  /// Initialize affine map expressions
  int numDims = (vnniFactor != 0) ? 7 : 6;
  for (int i = 0; i < numDims; i++)
    affineExprs.push_back(getAffineDimExpr(i, &context));

  // Create module
  module = ModuleOp::create(builder, loc);
  builder.setInsertionPoint(module);
}

void MLIRGenerator::getKernelTypes(KernelArgs &args) {
  // Input type, also first layer's input
  TensorType currentType = getShape({batch, layers.front()}, PACK_INPUT);

  // Weights and biases types (which is also relu and input to the next)
  for (unsigned i = 1, max = layers.size(); i < max; i++) {
    // Input to the layer is previous size
    unsigned inputSize = layers[i - 1];
    // Output to the layer is current size
    unsigned outputSize = layers[i];

    // Types: {MB, input} X {input, output} + Bcast(MB, {output}) -> ReLU
    LayerArgs arg;
    arg.index = i;
    arg.input.type = currentType;
    // Scale inputs are only needed for dequantization.
    if (quantType == QuantizationType::Dequant)
      arg.inputScale.type = getShape({batch}, INPUT_SCALE);
    arg.weight.type = getShape({inputSize, outputSize}, PACK_WEIGHT);
    if (quantType == QuantizationType::Dequant)
      arg.weightScale.type = getShape({outputSize}, WEIGHT_SCALE);
    arg.bias.type = getShape({outputSize}, PACK_OUTPUT);

    // For QuantDequant, such as F32->i8->F32, we need an intermediate type to
    // hold the quantized value.
    if (quantType == QuantizationType::QuantDequant) {
      arg.intermediate.type = getShape({batch, outputSize}, PACK_INTERMEDIATE);
      arg.output.type = getShape({batch, outputSize}, PACK_INPUT);
    } else {
      arg.output.type = getShape({batch, outputSize}, PACK_OUTPUT);
    }
    args.push_back(arg);

    // Update next input type with the output type of this layer
    currentType = arg.output.type;
  }
}

// Creates a quantize op around the gemm output and subsequently dequantize it.
// This is mainly to validate the quantization scheme.
Value MLIRGenerator::testQuantDequant(LayerArgs &args, Value input) {
  SmallVector<Value> scalingFactors = computeScalingFactor(input);
  Value chain = quantizeGemm(args, input, scalingFactors[0]);
  Value reScaleFactor = scalingFactors[1];
  Type rescaleType = reScaleFactor.getType();
  auto castedOutput =
      tensor::EmptyOp::create(builder, loc, rescaleType, ValueRange{});
  Value castedVal =
      linalg::GenericOp::create(builder, 
              loc, rescaleType, ValueRange{chain}, ValueRange{castedOutput},
              ArrayRef<AffineMap>{getMap(chain, MAP_PARALLEL),
                                  getMap(castedOutput, MAP_PARALLEL)},
              getIterators(MAP_PARALLEL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto casted = arith::SIToFPOp::create(nestedBuilder, 
                    loc, dataTypes[2], arg0);
                linalg::YieldOp::create(nestedBuilder, loc, ValueRange{casted});
              })
          .getResult(0);
  castedVal = linalg::MulOp::create(builder, loc, TypeRange{castedOutput.getType()},
                                         ValueRange{castedVal, reScaleFactor},
                                         ValueRange{castedOutput})
                  .getResult(0);
  return castedVal;
}

Value MLIRGenerator::createLayer(LayerArgs &args, bool hasMixedType) {
  OpBuilder::InsertionGuard guard(builder);

  Value chain;
  chain = lowerMatmul(args, hasMixedType);

  if (quantType == QuantizationType::QuantDequant)
    return testQuantDequant(args, chain);

  if (quantType == QuantizationType::Quant) {
    SmallVector<Value> scalingFactors = computeScalingFactor(chain);
    chain = quantizeGemm(args, chain, scalingFactors[0]);
  }

  if (quantType == QuantizationType::Dequant)
    chain = dequantizeGemm(args, chain);

  // These are optional and only emitted if enabled
  if (outputOpKind == OutputOpKind::Generic) {
    chain = lowerBiasAdd(chain, args.bias.value, args.output.value);
    chain = lowerRelu(chain, args.output.value);
  } else {
    chain = lowerNamedBiasAdd(chain, args.bias.value, args.output.value);
    chain = lowerNamedRelu(chain, args.output.value);
  }

  // Last layer may output softmax
  if (args.index == layers.size() - 1) {
    if (outputOpKind == OutputOpKind::Generic) {
      chain = lowerSoftmax(chain, args.output.value);
    } else {
      chain = lowerNamedSoftmax(chain, args.output.value);
    }
  }

  // Return output tensor to the next layer
  return chain;
}

void MLIRGenerator::createKernel(bool hasMixedType) {
  assert(((kernelType == KernelType::Const) ||
          (kernelType == KernelType::Args)) &&
         "Invalid kernel type");
  OpBuilder::InsertionGuard guard(builder);

  // Get all kernel types first
  KernelArgs args;
  getKernelTypes(args);
  assert(args.size() > 0 && "Invalid model size");
  unsigned lastLayer = args.size() - 1;
  auto &firstArg = args[0];
  auto &lastArg = args[lastLayer];

  // Model type only has `input`, while Layer type has everything
  // We need to create the function type list first, to set the values from
  // the function's arguments on the kernel type `layer`.
  SmallVector<Type, 1> inputTypes{firstArg.input.type};
  if (kernelType == KernelType::Args) {
    for (auto &layer : args) {
      if (quantType == QuantizationType::Dequant)
        inputTypes.push_back(layer.inputScale.type);

      inputTypes.push_back(layer.weight.type);
      if (quantType == QuantizationType::Dequant)
        inputTypes.push_back(layer.weightScale.type);

      if (enableBias)
        inputTypes.push_back(layer.bias.type);
      inputTypes.push_back(layer.output.type);
    }
  }

  // Create function with all necessary arguments
  auto func = createFunction(builder, module, "entry", inputTypes,
                             {lastArg.output.type});

  // Initialize the values depending on the KernelType
  //   * Model: input = arg, weights/bias = const, output = zero
  //   * Layer: input/weights/bias/output = args
  firstArg.input.value = func.getArgument(0);
  // Scales are only needed for dequantization
  if (quantType == QuantizationType::Dequant)
    firstArg.inputScale.value = func.getArgument(1);

  // Argument position is input + N * { weight/bias } + output
  // First weight is at position 1, every two
  unsigned argPos = !(quantType == QuantizationType::Dequant) ? 1 : 2;
  // Caches the output to chain into the next layer's input
  Value lastOutput;
  for (auto &arg : args) {
    // Chain the last output into this layer
    if (!arg.input.value)
      arg.input.value = lastOutput;

    // Initialize weights and biases
    if (kernelType == KernelType::Args) {
      arg.weight.value = func.getArgument(argPos++);
      if (quantType == QuantizationType::Dequant)
        arg.weightScale.value = func.getArgument(argPos++);
      if (enableBias)
        arg.bias.value = func.getArgument(argPos++);
      arg.output.value = func.getArgument(argPos++);
    } else { // Model
      if (identity) {
        // Identity weights / constant bias to test operations keeping the input
        // (A) predictable for testing.
        arg.weight.value = createDenseTensor(builder, TensorInitType::Identity,
                                             arg.weight.type, /* seed = */ 0);
        if (enableBias)
          arg.bias.value = createDenseTensor(builder, TensorInitType::Constant,
                                             arg.bias.type, /* seed = */ 0);
      } else {
        arg.weight.value =
            createDenseTensor(builder, initType, arg.weight.type, getRand());
        if (enableBias)
          arg.bias.value =
              createDenseTensor(builder, initType, arg.bias.type, getRand());
      }
      arg.output.value = getZeroInitTensor(arg.output.type);
    }

    lastOutput = createLayer(arg, hasMixedType);
    arg.output.value = lastOutput;
  }
  // Data is now output
  func::ReturnOp::create(builder, loc, lastArg.output.value);
}

int MLIRGenerator::generate(StringRef filename, bool hasMixedType) {
  // First, populate the module with all functions
  createKernel(hasMixedType);

  // Verify
  if (failed(module.verify())) {
    module->print(llvm::errs());
    module.emitError("Module verification failed");
    return 1;
  }

  // Now dump the module to the file of choice
  std::error_code error;
  if (filename.empty())
    filename = "-";
  auto outfile = llvm::raw_fd_ostream(filename, error);
  if (error) {
    module.emitError(filename + ": " + error.message());
    return 1;
  }

  outfile << createMetadata();
  module->print(outfile);

  return 0;
}

// ============================================= Helpers

std::string MLIRGenerator::createMetadata() {
  assert(flops && "FLOPS not computed?");
  std::string data = "";
  data += "// RUN: tpp-run %s -n 10 \\\n";
  data += "// RUN:  -e entry -entry-point-result=void\n";
  data += "\n";
  data += "// BENCH_TOTAL_FLOPS: " + std::to_string(flops);
  data += "\n";
  data += "\n";

  return data;
}

void MLIRGenerator::computeMatmulFlops(ShapedType inputShape,
                                       ShapedType outputShape) {
  // Matmul flops = 2 * M * N * K = 2 * prod(inputDims) * N (outShape[1])
  int64_t mkFlops = 1;
  for (int i = 0, max = inputShape.getRank(); i < max; i++)
    mkFlops *= inputShape.getDimSize(i);
  int outRank = outputShape.getRank();
  assert((outRank == 2 || outRank == 4) && "Invalid outRank");
  // Tiled: N = NB * n = outShape[0] + outShape[3]
  int64_t nFlops = outputShape.getDimSize(outRank - 1);
  if (outRank > 2)
    nFlops *= outputShape.getDimSize(1);
  flops += 2 * mkFlops * nFlops;
}

void MLIRGenerator::computeBiasOrReluFlops(ShapedType outputShape) {
  // Add flops = M * N = prod(outputDims)
  int64_t addReluFlops = 1;
  for (int i = 0, max = outputShape.getRank(); i < max; i++)
    addReluFlops *= outputShape.getDimSize(i);
  flops += addReluFlops;
}

Value MLIRGenerator::lowerNamedMatmul(Value input, Value weight, Value output) {
  auto inputShape = cast<ShapedType>(input.getType());
  auto weightShape = cast<ShapedType>(weight.getType());

  // TODO: VNNI produces mixed shape args, say 4D input and 5D weight. All
  // linalg named ops for matrix multiplication expects arguments of same
  // number of dimensions. Hence, such matmul patterns are not compatible to be
  // matched using named ops. Having a tuple or vector type as the element of
  // tensor had been discussed and can be revisited as potential solution.
  if (vnniFactor != 0) {
    llvm_unreachable(
        "Unsupported Lowering for VNNI, Try '--keep-generic-matmul'");
  }

  Value namedMatmul;
  if (inputShape.getRank() == 2) {
    namedMatmul = linalg::MatmulOp::create(builder, 
                          loc, TypeRange{output.getType()},
                          ValueRange{input, weight}, ValueRange{output})
                      .getResult(0);
  } else if (inputShape.getRank() == 4) {
    SmallVector<OpFoldResult, 4> dims =
        tensor::getMixedSizes(builder, loc, weight);
    applyPermutationToVector(dims, {0, 1, 3, 2});
    Value emptyTensor = tensor::EmptyOp::create(builder, 
        loc, dims, weightShape.getElementType());

    Value transpose = linalg::TransposeOp::create(builder, loc, weight, emptyTensor,
                                         ArrayRef<int64_t>{0, 1, 3, 2})
            .getResults()[0];
    namedMatmul = linalg::Mmt4DOp::create(builder, loc, TypeRange{output.getType()},
                                               ValueRange{input, transpose},
                                               ValueRange{output})
                      .getResult(0);
  }

  return namedMatmul;
}

Value MLIRGenerator::lowerMatmul(LayerArgs &args, bool hasMixedType = false) {
  Value chain;
  Value input = args.input.value;
  Value weight = args.weight.value;
  Value output = args.output.value;
  auto inputType = cast<ShapedType>(input.getType());
  auto outputType = cast<ShapedType>(output.getType());
  auto shape = outputType.getShape();
  auto zeroType = RankedTensorType::get(shape, inputType.getElementType());

  // For quant, derive the output type from input type.
  // TODO: Revisit to check we really need zero initalizer for mixed precision
  // float type?
  if (quantType == QuantizationType::Quant) {
    output = getZeroInitTensor(zeroType);
  } else if (quantType == QuantizationType::Dequant ||
             quantType == QuantizationType::Mixed) {
    Type elementType = inputType.getElementType();
    if (elementType.isInteger(8)) {
      // Get integer tensor accumulator type for dequantization.
      TensorType intAccumulatorType =
          RankedTensorType::get(shape, builder.getIntegerType(32));
      output = getZeroInitTensor(intAccumulatorType);
    } else if (elementType.isBF16() || elementType.isF16()) {
      output = getZeroInitTensor(cast<TensorType>(outputType));
    } else {
      llvm_unreachable("Unsupported dequantization data type");
    }
  }

  if (vnniPacked) {
    SmallVector<int64_t> vnniShape{inputType.getShape()};
    vnniShape.back() = vnniShape.back() / vnniFactor;
    vnniShape.push_back(vnniFactor);

    auto weightShape = cast<ShapedType>(weight.getType()).getShape();
    assert(weightShape.size() >= 3 && "Expected VNNI weights");
    assert(vnniShape.back() == weightShape.back() &&
           vnniShape.end()[-2] == weightShape.end()[-3] &&
           "Input and weights VNNI layout mismatch");

    auto vnniType =
        RankedTensorType::get(vnniShape, inputType.getElementType());

    auto inputRank = inputType.getRank();
    SmallVector<ReassociationIndices> reassociationIndices;
    for (int64_t index = 0; index < inputRank - 1; index++)
      reassociationIndices.push_back({index});
    reassociationIndices.push_back({inputRank - 1, inputRank});

    input = tensor::ExpandShapeOp::create(builder, loc, vnniType, input,
                                                  reassociationIndices);
  }

  if (outputOpKind == OutputOpKind::Generic || keepGenericMatmul) {
    chain = lowerGenericMatmul(input, weight, output);
  } else if (outputOpKind == OutputOpKind::Contract) {
    chain = lowerContract(input, weight, output);
  } else if (outputOpKind == OutputOpKind::NamedOp) {
    chain = lowerNamedMatmul(input, weight, output);
  }

  computeMatmulFlops(inputType, outputType);
  return chain;
}

Value MLIRGenerator::lowerGenericMatmul(Value input, Value weight,
                                        Value output) {
  // Matmul as a linalg.generic
  auto map1 = getMap(input, MAP_MATMUL_INPUT);   // { 0, 2 }
  auto map2 = getMap(weight, MAP_MATMUL_WEIGHT); // { 2, 1 }
  auto map3 = getMap(output, MAP_MATMUL_OUTPUT); // { 0, 1 }
  auto matmul =
      linalg::GenericOp::create(builder, 
              loc, output.getType(), ValueRange{input, weight},
              ValueRange{output}, ArrayRef<AffineMap>{map1, map2, map3},
              getIterators(MAP_MATMUL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto arg1 = blockArgs[1];
                auto arg2 = blockArgs[2];
                // If input and output type differs, up cast input to output
                // type using arith.extf/arith.extsi.
                Type inputElementType =
                    cast<ShapedType>(input.getType()).getElementType();
                Type weightElementType =
                    cast<ShapedType>(weight.getType()).getElementType();
                Type outputElementType =
                    cast<ShapedType>(output.getType()).getElementType();
                if (inputElementType != outputElementType) {
                  if (inputElementType.isFloat()) {
                    arg0 = arith::ExtFOp::create(builder, 
                        loc, outputElementType, arg0);
                  } else {
                    arg0 = arith::ExtSIOp::create(nestedBuilder, 
                        loc, outputElementType, arg0);
                  }
                }

                if (weightElementType != outputElementType) {
                  if (weightElementType.isFloat()) {
                    arg1 = arith::ExtFOp::create(builder, 
                        loc, outputElementType, arg1);
                  } else {
                    arg1 = arith::ExtSIOp::create(nestedBuilder, 
                        loc, outputElementType, arg1);
                  }
                }

                auto *mul =
                    outputElementType.isFloat()
                        ? arith::MulFOp::create(nestedBuilder, loc, arg0, arg1)
                        : arith::MulIOp::create(nestedBuilder, loc, arg0, arg1);
                auto *add = outputElementType.isFloat()
                                ? arith::AddFOp::create(nestedBuilder, 
                                      loc, arg2, mul->getResult(0))
                                : arith::AddIOp::create(nestedBuilder, 
                                      loc, arg2, mul->getResult(0));
                linalg::YieldOp::create(nestedBuilder, 
                    loc, ValueRange{add->getResults()});
              })
          .getResult(0);

  return matmul;
}

Value MLIRGenerator::lowerContract(Value input, Value weight, Value output) {
  // Matmul as a linalg.contract
  SmallVector<Attribute> maps;
  maps.push_back(AffineMapAttr::get(getMap(input, MAP_MATMUL_INPUT)));   // { 0, 2 }
  maps.push_back(AffineMapAttr::get(getMap(weight, MAP_MATMUL_WEIGHT))); // { 2, 1 }
  maps.push_back(AffineMapAttr::get(getMap(output, MAP_MATMUL_OUTPUT))); // { 0, 1 }
  auto contract = linalg::ContractOp::create(builder, 
                          loc, output.getType(), ValueRange{input, weight}, ValueRange{output},
                          builder.getArrayAttr(maps))
                      .getResult(0);

  return contract;
}

SmallVector<Value> MLIRGenerator::computeScalingFactor(Value input) {
  auto inputType = cast<ShapedType>(input.getType());
  assert(inputType.getRank() == 2 && "Input must be a 2D tensor");

  auto loc = input.getLoc();
  auto elementType = inputType.getElementType();

  // Initialize the reduction tensor with the minimum possible value
  Value initValue = arith::ConstantOp::create(builder, 
      loc, builder.getFloatAttr(elementType,
                                -std::numeric_limits<float>::infinity()));
  auto reductionType =
      RankedTensorType::get({inputType.getShape()[1]}, elementType);

  // Per channel scale factor output tensor
  Value scaleTensor =
      tensor::EmptyOp::create(builder, loc, reductionType, ValueRange{});
  Value scaleTensorInit =
      linalg::FillOp::create(builder, loc, initValue, scaleTensor).getResult(0);

  // Reduce along dimension 0 (rows) to find max of each column for per channel
  // quantization.
  Value absMax = linalg::ReduceOp::create(builder, 
              loc, input, scaleTensorInit, ArrayRef<int64_t>{0},
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange args) {
                Value absVal =
                    math::AbsFOp::create(nestedBuilder, nestedLoc, args[0]);
                Value maxVal = arith::MaximumFOp::create(nestedBuilder, 
                    nestedLoc, absVal, args[1]);
                linalg::YieldOp::create(nestedBuilder, nestedLoc, maxVal);
              })
          .getResult(0);

  // Compute the scaling factors (2^(-exponent)) from the absolute maximum
  // values.
  Value zeroVal = arith::ConstantIntOp::create(builder, loc, 0, 32);

  // Create two output tensors for the two results
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  Value channelScale =
      tensor::EmptyOp::create(builder, loc, reductionType, ValueRange{});
  Value channelReScale =
      tensor::EmptyOp::create(builder, loc, reductionType, ValueRange{});

  auto frExp = linalg::GenericOp::create(builder, 
      loc,
      TypeRange{reductionType, reductionType}, // Specify multiple result types
      ValueRange{absMax}, ValueRange{channelScale, channelReScale},
      ArrayRef<AffineMap>{getMap(absMax, MAP_PARALLEL),
                          getMap(channelScale, MAP_PARALLEL),
                          getMap(channelReScale, MAP_PARALLEL)},
      ArrayRef<utils::IteratorType>{utils::IteratorType::parallel},
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value frexpResult = LLVM::FractionExpOp::create(
            nestedBuilder, nestedLoc,
            LLVM::LLVMStructType::getLiteral(
                &context, ArrayRef<Type>{elementType, builder.getI32Type()}),
            ValueRange{args[0]});
        Value exponent =
            LLVM::ExtractValueOp::create(nestedBuilder, nestedLoc,
                                         builder.getI32Type(), frexpResult, 1)
                .getResult();
        Value unbiased = arith::SubIOp::create(nestedBuilder, 
            nestedLoc, exponent,
            arith::ConstantOp::create(builder, nestedLoc,
                                              builder.getI32IntegerAttr(7)));
        Value negExponent =
            arith::SubIOp::create(nestedBuilder, nestedLoc, zeroVal, unbiased);
        auto tchannleReScale =
            math::Exp2Op::create(nestedBuilder, nestedLoc,
                                      arith::SIToFPOp::create(nestedBuilder, 
                                          nestedLoc, elementType, unbiased))
                ->getResult(0);
        auto tchannleScale =
            math::Exp2Op::create(nestedBuilder, nestedLoc,
                                      arith::SIToFPOp::create(nestedBuilder, 
                                          nestedLoc, elementType, negExponent))
                ->getResult(0);
        linalg::YieldOp::create(nestedBuilder, 
            nestedLoc, ValueRange{tchannleScale, tchannleReScale});
      });

  SmallVector<Value> frExpVec;
  frExpVec.push_back(frExp.getResults()[0]);
  frExpVec.push_back(frExp.getResults()[1]);

  SmallVector<Value> scalingFactors;
  Value scalingFactor =
      tensor::EmptyOp::create(builder, loc, inputType, ValueRange{});
  Value filledTensor =
      linalg::FillOp::create(builder, loc, initValue, scalingFactor)
          .getResult(0);
  // Broadcast to match output shape
  auto broadcastScaleRes =
      linalg::BroadcastOp::create(builder, loc, frExpVec[0], filledTensor,
                                       ArrayRef<int64_t>{0})
          ->getResult(0);
  scalingFactors.push_back(broadcastScaleRes);

  broadcastScaleRes =
      linalg::BroadcastOp::create(builder, loc, frExpVec[1], filledTensor,
                                       ArrayRef<int64_t>{0})
          ->getResult(0);
  scalingFactors.push_back(broadcastScaleRes);

  return scalingFactors;
}

Value MLIRGenerator::quantizeGemm(LayerArgs &args, Value chain,
                                  Value scaleFactor) {
  Value input = args.input.value;
  Value weight = args.weight.value;
  Type outputType = quantType == QuantizationType::QuantDequant
                        ? args.intermediate.type
                        : args.output.type;

  auto inputShapedTy = cast<ShapedType>(input.getType());
  auto outputShapedTy = cast<ShapedType>(outputType);
  auto shape = outputShapedTy.getShape();
  // Create a output type for the quantized output using shape and input element
  // type.
  auto contractOutputTy =
      RankedTensorType::get(shape, inputShapedTy.getElementType());

  auto castedOutput =
      tensor::EmptyOp::create(builder, loc, outputShapedTy, ValueRange{});
  SmallVector<Attribute> maps;
  maps.push_back(AffineMapAttr::get(getMap(input, MAP_MATMUL_INPUT)));
  maps.push_back(AffineMapAttr::get(getMap(weight, MAP_MATMUL_WEIGHT)));
  maps.push_back(AffineMapAttr::get(getMap(castedOutput, MAP_MATMUL_OUTPUT)));
  auto dquantVal = getZeroInitTensor(contractOutputTy);

  auto dquantRes = linalg::MulOp::create(builder, loc, chain.getType(),
                                              ValueRange{chain, scaleFactor},
                                              ValueRange{dquantVal})
                       .getResult(0);

  dquantRes =
      linalg::GenericOp::create(builder, 
              loc, outputShapedTy, ValueRange{dquantRes},
              ValueRange{castedOutput},
              ArrayRef<AffineMap>{getMap(dquantRes, MAP_PARALLEL),
                                  getMap(castedOutput, MAP_PARALLEL)},
              getIterators(MAP_PARALLEL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto casted = arith::FPToSIOp::create(nestedBuilder, 
                    loc, outputShapedTy.getElementType(), arg0);
                linalg::YieldOp::create(nestedBuilder, loc, ValueRange{casted});
              })
          .getResult(0);

  // TODO: A place holder for flops computation for quantization.
  computeMatmulFlops(inputShapedTy, outputShapedTy);
  return dquantRes;
}

Value MLIRGenerator::dequantizeGemm(LayerArgs &args, Value chain) {
  // Chain is the contract/gemm output
  assert(chain && "Expected valid chain output from contract/gemm operation");

  Value input = args.input.value;
  Value inputScale = args.inputScale.value;
  Value weightScale = args.weightScale.value;
  Value output = args.output.value;

  // For mixed type, we need to handle input and weight scales to compute the
  // resultant scaleand then multiply the result with the contract output.
  auto inputScaleTy = cast<ShapedType>(inputScale.getType());
  assert(inputScaleTy.getRank() == 1 && "Input scale must be a vector");
  assert(inputScaleTy.getElementType() == dataTypes[2] &&
         "Input scale must be of scale type");

  auto weightScaleTy = cast<ShapedType>(weightScale.getType());
  assert(weightScaleTy.getRank() == 1 && "Weight scale must be a vector");
  assert(weightScaleTy.getElementType() == dataTypes[2] &&
         "Weight scale must be of scale type");

  // Create a 2-D ouput scale shape using input and weight scales
  auto outputScaleShape = SmallVector<int64_t>{inputScaleTy.getShape()[0],
                                               weightScaleTy.getShape()[0]};
  auto inputShapedTy = cast<ShapedType>(input.getType());
  auto outputShapedTy = cast<ShapedType>(output.getType());

  // Create map for outerproduct of input and weight scales
  MLIRContext *ctx = &context;
  auto dim0 = getAffineDimExpr(0, ctx);
  auto dim1 = getAffineDimExpr(1, ctx);
  auto inputScaleMap = AffineMap::get(2, 0, {dim0}, ctx);
  auto weightScaleMap = AffineMap::get(2, 0, {dim1}, ctx);
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel, utils::IteratorType::parallel};
  // Initialize the map for linalg.generic to perform dequantization of result
  // of gemm with scales.
  SmallVector<AffineMap> reshapeMap = {getMap(chain, MAP_PARALLEL),
                                       inputScaleMap, weightScaleMap,
                                       getMap(output, MAP_PARALLEL)};
  // If tiling is applied, we need to expand the scale tensors to match the
  // tiled dimensions and update the reshape map and iterator types accordingly.
  if (tiles.size() > 0) {
    // The expansion is essentially a reshape with some dimensions being marked
    // as unit size dim for broadcasting.
    inputScale =
        createExpandedScaleTensor(builder, loc, inputScale, tiles, true);
    weightScale =
        createExpandedScaleTensor(builder, loc, weightScale, tiles, false);
    // Update the reshape map to broadcast the unit dims for the expanded scale
    // tensors.
    SmallVector<AffineExpr> scaleAffineExprs;
    auto dim = 0;
    for (auto i : cast<ShapedType>(inputScale.getType()).getShape()) {
      scaleAffineExprs.push_back(i == 1 ? getAffineConstantExpr(0, &context)
                                        : getAffineDimExpr(dim, &context));
      dim++;
    }
    AffineMap packedScaleMap =
        AffineMap::get(outputShapedTy.getRank(), 0, scaleAffineExprs, &context);
    reshapeMap[1] = packedScaleMap;
    reshapeMap[2] = packedScaleMap;
    iteratorTypes = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::parallel};
  }

  auto result =
      linalg::GenericOp::create(
          builder, loc, TypeRange{outputShapedTy},
          ValueRange{chain, inputScale, weightScale}, ValueRange{output},
          reshapeMap, iteratorTypes,
          [&](OpBuilder &nestedBuilder, Location nestedLoc,
              ValueRange blockArgs) {
            auto arg0 = blockArgs[0];
            auto arg1 = blockArgs[1];
            auto arg2 = blockArgs[2];
            auto alu =
                inputScaleTy.getElementType().isInteger()
                    ? arith::MulIOp::create(nestedBuilder, loc, arg1, arg2)
                          .getResult()
                    : arith::MulFOp::create(nestedBuilder, loc, arg1, arg2)
                          .getResult();
            Value castToFloat = arg0;
            if (arg0.getType() != dataTypes[2]) {
              if (arg0.getType().isF16() || arg0.getType().isBF16()) {
                castToFloat = arith::ExtFOp::create(nestedBuilder, loc,
                                                    dataTypes[2], arg0);
              } else {
                castToFloat = arith::SIToFPOp::create(nestedBuilder, loc,
                                                      dataTypes[2], arg0);
              }
            }
            alu = arith::MulFOp::create(nestedBuilder, loc, castToFloat, alu)
                      .getResult();
            linalg::YieldOp::create(nestedBuilder, loc, ValueRange{alu});
          })
          .getResult(0);

  // TODO: A place holder for flops computation for dequantization.
  computeMatmulFlops(inputShapedTy, outputShapedTy);
  return result;
}

Value MLIRGenerator::lowerBiasAdd(Value input, Value bias, Value output) {
  if (!enableBias)
    return input;

  auto outTy = cast<ShapedType>(input.getType());
  auto mapA = getMap(input, MAP_BROADCAST);
  auto mapB = getMap(input, MAP_PARALLEL);
  auto sum =
      linalg::GenericOp::create(builder, 
              loc, outTy, ValueRange{bias}, ValueRange{input},
              ArrayRef<AffineMap>{mapA, mapB}, getIterators(MAP_PARALLEL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto arg1 = blockArgs[1];
                auto add = arith::AddFOp::create(nestedBuilder, loc, arg0, arg1);
                linalg::YieldOp::create(nestedBuilder, loc, ValueRange{add});
              })
          .getResult(0);

  computeBiasOrReluFlops(outTy);
  return sum;
}

Value MLIRGenerator::lowerNamedBiasAdd(Value input, Value bias, Value output) {
  if (!enableBias)
    return input;

  auto outTy = cast<ShapedType>(input.getType());
  auto biasTy = cast<ShapedType>(bias.getType());
  Value emptyTensor = tensor::EmptyOp::create(builder, loc, outTy, ValueRange{});
  SmallVector<int64_t> addedDimensions;
  SmallVector<bool> dimsNeeded =
      getBroadcastDims(biasTy.getShape(), outTy.getShape());
  for (int64_t dim : llvm::seq<int64_t>(0, outTy.getRank() - 1)) {
    if (dimsNeeded[dim])
      addedDimensions.push_back(dim);
  }

  Value broadcast =
      linalg::BroadcastOp::create(builder, loc, bias, emptyTensor, addedDimensions)
          .getResult()[0];
  Value biasAdd = linalg::AddOp::create(builder, loc, TypeRange{output.getType()},
                                             ValueRange{broadcast, input},
                                             ValueRange{emptyTensor})
                      .getResult(0);

  computeBiasOrReluFlops(outTy);
  return biasAdd;
}

Value MLIRGenerator::lowerNamedRelu(Value input, Value output) {
  if (!enableRelu)
    return input;

  auto outTy = cast<ShapedType>(input.getType());
  auto zero =
      getConstFloat(builder, 0.0, cast<FloatType>(outTy.getElementType()));
  Value emptyTensor = tensor::EmptyOp::create(builder, loc, outTy, ValueRange{});
  auto fill =
      linalg::FillOp::create(builder, loc, zero, emptyTensor)->getResult(0);
  Value relu = linalg::MaxOp::create(builder, loc, TypeRange{output.getType()},
                                 ValueRange{input, fill}, ValueRange{emptyTensor})
          .getResult(0);

  computeBiasOrReluFlops(outTy);
  return relu;
}

Value MLIRGenerator::lowerRelu(Value input, Value output) {
  if (!enableRelu)
    return input;

  auto zero = getConstFloat(builder, 0.0, cast<FloatType>(dataTypes[0]));
  auto outTy = cast<ShapedType>(input.getType());
  auto map = getMap(input, MAP_PARALLEL);
  auto relu =
      linalg::GenericOp::create(builder, 
              loc, outTy, ValueRange{}, ValueRange{input},
              ArrayRef<AffineMap>{map}, getIterators(MAP_PARALLEL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto max =
                    arith::MaximumFOp::create(nestedBuilder, loc, arg0, zero);
                linalg::YieldOp::create(nestedBuilder, loc, ValueRange{max});
              })
          .getResult(0);

  computeBiasOrReluFlops(outTy);
  return relu;
}

Value MLIRGenerator::lowerNamedSoftmax(Value input, Value output) {
  if (!enableSoftmax)
    return input;

  // TODO: Add lowering of softmax to sequence of named Ops
  llvm_unreachable("Linalg named ops for softmax not implemented yet");
  
  auto outTy = cast<ShapedType>(input.getType());
  // Softmax flops = 4 * M * N = 4 * prod(outputDims)
  int64_t softmaxFlops = 1;
  for (int i = 0, max = outTy.getRank(); i < max; i++)
    softmaxFlops *= outTy.getDimSize(i);
  flops += 4 * softmaxFlops;

  return input;
}

Value MLIRGenerator::lowerSoftmax(Value input, Value output) {
  if (!enableSoftmax)
    return input;

  assert(cast<ShapedType>(input.getType()).getRank() == 2 &&
         "Packed softmax not implemented yet");
  auto map1 = getMap(input, MAP_PARALLEL);
  auto map2 = getMap(input, MAP_REDUCTION);
  auto outTy = cast<ShapedType>(input.getType());

  // First, we calculate the element-wise exp
  Value expTensor = tensor::EmptyOp::create(builder, loc, outTy, ValueRange{});
  auto exp = linalg::GenericOp::create(builder, 
      loc, outTy, ValueRange{input}, ValueRange{expTensor},
      ArrayRef<AffineMap>{map1, map1}, getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto exp = math::ExpOp::create(nestedBuilder, loc, arg0);
        linalg::YieldOp::create(nestedBuilder, loc, ValueRange{exp});
      });

  // Second, we sum-reduce and splat
  SmallVector<int64_t> dims{batch, 1};
  auto redTy = getShape(dims, PACK_OUTPUT);
  Value redTensor =
      tensor::EmptyOp::create(builder, loc, dims, outTy.getElementType());
  auto zero = getConstFloat(builder, 0.0, cast<FloatType>(dataTypes[0]));
  auto fill = linalg::FillOp::create(builder, loc, zero, redTensor);
  auto redux = linalg::GenericOp::create(builder, 
      loc, redTy, ValueRange{exp.getResult(0)}, ValueRange{fill.getResult(0)},
      ArrayRef<AffineMap>{map1, map2}, getIterators(MAP_REDUCTION),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto arg1 = blockArgs[1];
        auto add = arith::AddFOp::create(nestedBuilder, loc, arg0, arg1);
        linalg::YieldOp::create(nestedBuilder, loc, ValueRange{add});
      });
  // Splat back to the same dims
  Value meanTensor = tensor::EmptyOp::create(builder, loc, outTy, ValueRange{});
  auto mean = linalg::GenericOp::create(builder, 
      loc, outTy, ValueRange{redux.getResult(0)}, ValueRange{meanTensor},
      ArrayRef<AffineMap>{map2, map1}, getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        linalg::YieldOp::create(nestedBuilder, loc, ValueRange{arg0});
      });

  // Third, we update the exp/sum(exp) onto the output tensor
  auto softmax =
      linalg::GenericOp::create(builder, 
              loc, outTy, ValueRange{exp.getResult(0), mean.getResult(0)},
              ValueRange{output}, ArrayRef<AffineMap>{map1, map1, map1},
              getIterators(MAP_PARALLEL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto arg1 = blockArgs[1];
                auto div = arith::DivFOp::create(nestedBuilder, loc, arg0, arg1);
                linalg::YieldOp::create(nestedBuilder, loc, ValueRange{div});
              })
          .getResult(0);

  // Softmax flops = 4 * M * N = 4 * prod(outputDims)
  int64_t softmaxFlops = 1;
  for (int i = 0, max = outTy.getRank(); i < max; i++)
    softmaxFlops *= outTy.getDimSize(i);
  flops += 4 * softmaxFlops;

  return softmax;
}

TensorType MLIRGenerator::getShape(ArrayRef<int64_t> dims, PackingType type) {
  // Already packed type, just return ND tensor
  if (dims.size() > 2)
    return RankedTensorType::get(dims, type == PACK_OUTPUT ? dataTypes[1]
                                                           : dataTypes[0]);

  if (!tiles.size()) {
    if (quantType != QuantizationType::None) {
      if (type == INPUT_SCALE || type == WEIGHT_SCALE) {
        return RankedTensorType::get(dims, dataTypes[2]);
      } else if (type == PACK_OUTPUT) {
        return RankedTensorType::get(dims, dataTypes[1]);
      } else if (type == PACK_INPUT) {
        return RankedTensorType::get(dims, dataTypes[0]);
      } else if (type == PACK_INTERMEDIATE) {
        return RankedTensorType::get(dims, dataTypes[1]);
      }
    }
    // Unpacked type, just return 2D tensor
    return RankedTensorType::get(dims, dataTypes[0]);
  }

  // Packed types block by tile size
  assert(tiles.size() == 3 && "Invalid tile size format");
  auto n = tiles[0];
  auto k = tiles[1];
  auto c = tiles[2];
  auto x = dims[0];
  // Broadcast is 1D
  auto y = dims.size() == 2 ? dims[1] : 0;

  switch (type) {
  case PACK_INPUT:
    assert(x % n == 0 && "Invalid tile size for N dim");
    assert(y % c == 0 && "Invalid tile size for C dim");
    // N x C -> BN x BC x bn x bc
    return RankedTensorType::get({x / n, y / c, n, c}, dataTypes[0]);
  case PACK_WEIGHT:
    // VNNI packing can be done via tpp-opt --vnni-pack
    assert(x % k == 0 && "Invalid tile size for K dim");
    assert(y % c == 0 && "Invalid tile size for C dim");

    // VNNI: C x K -> BK x BC x bc/vnni x bk x vnni
    if (vnniFactor != 0)
      return RankedTensorType::get(
          {y / k, x / c, c / vnniFactor, k, vnniFactor}, dataTypes[0]);

    // C x K -> BK x BC x bc x bk
    return RankedTensorType::get({y / k, x / c, c, k}, dataTypes[0]);
  case PACK_OUTPUT:
    assert(x % n == 0 && "Invalid tile size for N dim");

    // Broadcast 1D -> 2D is Bk x bk only
    if (!y)
      return RankedTensorType::get({x / k, k}, dataTypes[1]);

    // N x K -> BN x BK x bn x bk
    assert(y % k == 0 && "Invalid tile size for K dim");
    return RankedTensorType::get({x / n, y / k, n, k}, dataTypes[1]);
  case INPUT_SCALE:
    return RankedTensorType::get({dims[0]}, dataTypes[2]);
  case WEIGHT_SCALE:
    return RankedTensorType::get({dims[0]}, dataTypes[2]);
  case PACK_INTERMEDIATE:
    llvm_unreachable("Unknown intermediate packing type");
  }

  llvm_unreachable("Unknown packing type");
}

AffineMap MLIRGenerator::getMap(Value tensor, MapType type) {
  auto n = cast<ShapedType>(tensor.getType()).getRank();
  // Packed tensors are either 4 or 5 dim, map needs to be 6 or 7
  bool packed = (n > 2);
  SmallVector<AffineExpr> list;
  auto zero = getAffineConstantExpr(0, builder.getContext());
  auto pushDim = [&](size_t index, ArrayRef<int64_t> order) {
    if (order.size() > index) {
      list.push_back(affineExprs[order[index]]);
    } else if (order.size()) {
      // Means we use less dims than the total number (ex. matmul)
      return;
    } else {
      list.push_back(affineExprs[index]);
    }
  };

  auto getDims = [&](ArrayRef<int64_t> dims) {
    for (auto &dim : dims)
      list.push_back(affineExprs[dim]);
  };

  // For each map type, check if it's packed or not, build the order and
  // return the map.
  SmallVector<int64_t, 5> iter;
  switch (type) {
  case MAP_MATMUL:
    assert(false && "Invalid map type");
  case MAP_PARALLEL:
    // Parallel only depends on the tensor rank
    for (unsigned i = 0; i < n; i++)
      pushDim(i, iter);
    break;
  case MAP_REDUCTION:
    // TODO: Work out how reduction works on packed tensors
    for (unsigned i = 0; i < n - 1; i++)
      pushDim(i, iter);
    list.push_back(zero);
    break;
  case MAP_BROADCAST:
    // Broadcast from ND to (N+1)D is (0, 1) -> (1)
    // Packed broadcast (BN, bn) is (0, 1, 2, 3) -> (1, 3)
    for (unsigned i = 1; i < n; i+=2)
      pushDim(i, iter);
    break;
  case MAP_MATMUL_INPUT:
    // Packed tensors have 4/5 dims and 6 loops (ppr-ppr)
    n = packed ? 6 : 3;
    if (vnniPacked) {
      // Extra VNNI packing reduction dim
      n += 1;
      getDims({0, 2, 4, 6, 3});
    } else if (packed)
      getDims({0, 2, 3, 5});
    else
      getDims({0, 2});
    break;
  case MAP_MATMUL_WEIGHT:
    // Packed tensors have 4/5 dims and 6 loops (ppr-ppr)
    n = packed ? 6 : 3;
    if (vnniPacked) {
      // Extra VNNI packing reduction dim
      n += 1;
      getDims({1, 2, 6, 5, 3});
    } else if (packed)
      getDims({1, 2, 5, 4});
    else
      getDims({2, 1});
    break;
  case MAP_MATMUL_OUTPUT:
    // Packed tensors have 4/5 dims and 6 loops (ppr-ppr)
    n = packed ? 6 : 3;
    if (vnniPacked) {
      // Extra VNNI packing reduction dim
      n += 1;
      getDims({0, 1, 4, 5});
    } else if (packed)
      getDims({0, 1, 3, 4});
    else
      getDims({0, 1});
    break;
  }

  auto map = AffineMap::get(n, 0, list, &context);
  return map;
}

SmallVector<utils::IteratorType> MLIRGenerator::getIterators(MapType type) {
  bool packed = tiles.size();
  switch (type) {
  case MAP_PARALLEL:
  case MAP_BROADCAST:
    if (packed)
      return {utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::parallel, utils::IteratorType::parallel};
    else
      return {utils::IteratorType::parallel, utils::IteratorType::parallel};
    break;
  case MAP_REDUCTION:
    // TODO: Work out how reduction works on packed tensors
    if (packed)
      return {utils::IteratorType::parallel, utils::IteratorType::reduction,
              utils::IteratorType::parallel, utils::IteratorType::reduction};
    else
      return {utils::IteratorType::parallel, utils::IteratorType::reduction};
    break;
  case MAP_MATMUL_INPUT:
  case MAP_MATMUL_WEIGHT:
  case MAP_MATMUL_OUTPUT:
  case MAP_MATMUL:
    if (vnniPacked)
      // Extra VNNI packing reduction dim
      return {utils::IteratorType::parallel,  utils::IteratorType::parallel,
              utils::IteratorType::reduction, utils::IteratorType::reduction,
              utils::IteratorType::parallel,  utils::IteratorType::parallel,
              utils::IteratorType::reduction};
    else if (packed)
      return {utils::IteratorType::parallel,  utils::IteratorType::parallel,
              utils::IteratorType::reduction, utils::IteratorType::parallel,
              utils::IteratorType::parallel,  utils::IteratorType::reduction};
    else
      return {utils::IteratorType::parallel, utils::IteratorType::parallel,
              utils::IteratorType::reduction};
  }
  return {};
}

int MLIRGenerator::getRand() {
  // Not random
  if (!seed) {
    return 0;
  }
  // Update and return previous
  int temp = seed;
  seed = rand();
  return temp;
}

Value MLIRGenerator::getZeroInitTensor(TensorType type) {
  // Initialize tensor with zeros of all appropriate types such as f32, i32,
  // bf16, i8
  Value zero = nullptr;
  auto elTy = type.getElementType();
  if (elTy.isFloat()) {
    zero = getConstFloat(builder, 0.0, cast<FloatType>(elTy));
  } else if (elTy.isInteger()) {
    zero = getConstInt(builder, 0, elTy.getIntOrFloatBitWidth());
  } else {
    llvm_unreachable("Unsupported element type for zero initialization");
  }

  Value tensor =
      tensor::EmptyOp::create(builder, loc, type, ValueRange{}).getResult();
  tensor = linalg::FillOp::create(builder, loc, zero, tensor).getResult(0);
  return tensor;
}