//===- MLIRGen.cpp -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/DLTI/DLTI.h"
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

static Value createCastToFloat(OpBuilder &builder, Location loc, Value value,
                              mlir::Type dstType,
                              arith::FastMathFlagsAttr fmf = nullptr) {
  assert(dstType.isFloat() && "Unsupported target type for cast");

  auto srcType = value.getType();
  if (srcType == dstType)
    return value;

  auto srctypeSize = srcType.getIntOrFloatBitWidth();
  auto dstTypeSize = dstType.getIntOrFloatBitWidth();

  Value castToFloat = value;
  // Cast value to float if element types differ
  if (srcType.isInteger()) {
    castToFloat = arith::SIToFPOp::create(builder, loc, dstType, value);
  } else if (srctypeSize < dstTypeSize) {
    castToFloat = arith::ExtFOp::create(builder, loc, dstType, value, fmf);
  } else {
    castToFloat = arith::TruncFOp::create(builder, loc, dstType, value);
  }

  return castToFloat;
}

// Downcasts the matmul accumulator to the output element type when they differ
// and share the same domain (float or integer), via an element-wise
// linalg.generic using arith.truncf/arith.trunci. Cross-domain conversions
// (e.g. quantization/dequantization) are handled by their dedicated lowerings.
static Value downcastToOutput(OpBuilder &builder, Location loc,
                              Value accumulator, ShapedType outputType) {
  auto accTy = cast<ShapedType>(accumulator.getType());
  Type accElem = accTy.getElementType();
  Type outElem = outputType.getElementType();
  bool sameFloat = outElem.isFloat() && accElem.isFloat();
  bool sameInt = outElem.isInteger() && accElem.isInteger();
  if (accElem == outElem || (!sameFloat && !sameInt))
    return accumulator;

  // So far, only the accumulator was created by the matmul. If they're of the same type
  // we have returned above. If not, then we need to allocate a new buffer, since they
  // have different types. This will be the `chain` to the next operation.
  auto resultTy = RankedTensorType::get(accTy.getShape(), outElem);
  Value dest = tensor::EmptyOp::create(builder, loc, resultTy, ValueRange{});
  int64_t rank = accTy.getRank();
  SmallVector<AffineMap> maps(2, builder.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  return linalg::GenericOp::create(
             builder, loc, resultTy, ValueRange{accumulator}, ValueRange{dest},
             maps, iterators,
             [&](OpBuilder &nestedBuilder, Location nestedLoc,
                 ValueRange blockArgs) {
               Value trunc;
               if (sameFloat)
                 trunc = arith::TruncFOp::create(nestedBuilder, nestedLoc,
                                                 outElem, blockArgs[0]);
               else
                 trunc = arith::TruncIOp::create(nestedBuilder, nestedLoc,
                                                 outElem, blockArgs[0]);
               linalg::YieldOp::create(nestedBuilder, nestedLoc,
                                       ValueRange{trunc});
             })
      .getResult(0);
}

} // anonymous namespace

MLIRGenerator::MLIRGenerator(StringRef outputOpKindStr, StringRef kernelStr,
                             unsigned batch, StringRef layersStr,
                             StringRef tilesStr, StringRef registerUnrollStr, StringRef dataType,
                             StringRef scaleType, bool quant,
                             int seed, bool identity, bool enableBias,
                             bool enableRelu, bool enableSoftmax,
                             int vnniBlockingFactor)
    : builder(&context), loc(builder.getUnknownLoc()), batch(batch), seed(seed),
      identity(identity), flops(0), enableBias(enableBias),
      enableRelu(enableRelu), enableSoftmax(enableSoftmax), quant(quant),
      vnniFactor(vnniBlockingFactor) {

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

  // Parse matmul tile / unroll sizes
  parseStringList(tilesStr, tiles);
  assert((tiles.size() == 0 || tiles.size() == 3) &&
         "Must have 3 tile sizes (or none)");
  parseStringList(registerUnrollStr, registerUnroll);
  assert((registerUnroll.size() == 0 || registerUnroll.size() == 3) &&
         "Must have 3 register unrolling or none");

  // Pick data types. Each case sets {input, output, accumulator}; the scale
  // types are filled in below.
  dataTypes =
      llvm::StringSwitch<DataTypes>(dataType)
          .CaseLower("f32", DataTypes{builder.getF32Type(),
                                      builder.getF32Type(),
                                      builder.getF32Type()})
          .CaseLower("f16", DataTypes{builder.getF16Type(),
                                      builder.getF16Type(),
                                      builder.getF32Type()})
          .CaseLower("bf16", DataTypes{builder.getBF16Type(),
                                       builder.getBF16Type(),
                                       builder.getF32Type()})
          // FP8 pure types. E5M2 is named bf8 and E4M3 is named hf8 by libxsmm.
          .CaseLower("bf8", DataTypes{builder.getF8E5M2Type(),
                                      builder.getF8E5M2Type(),
                                      builder.getF8E5M2Type()})
          .CaseLower("hf8", DataTypes{builder.getF8E4M3FNType(),
                                      builder.getF8E4M3FNType(),
                                      builder.getF8E4M3FNType()})
          // Low-precision integer GEMMs, used on their own or with --quant.
          .CaseLower("i8", DataTypes{builder.getIntegerType(8),
                                     builder.getI32Type(),
                                     builder.getI32Type()})
          .CaseLower("i8-f32", DataTypes{builder.getIntegerType(8),
                                         builder.getF32Type(),
                                         builder.getF32Type()})
          .Default(DataTypes{});
  assert(dataTypes.input && "Unsupported data type");

  auto scaleTypeOpt = llvm::StringSwitch<std::optional<Type>>(scaleType)
                          .CaseLower("f32", builder.getF32Type())
                          .CaseLower("f8E8M0FNU", builder.getF8E8M0Type())
                          .CaseLower("", builder.getF32Type())
                          .Default(std::nullopt);
  assert(scaleTypeOpt && "Unsupported scale type");
  dataTypes.inputScale = *scaleTypeOpt;
  dataTypes.weightScale = *scaleTypeOpt;
  dataTypes.outputScale = *scaleTypeOpt;

  // const kernelType is only supported for non quantization kernel.
  assert(!(kernelType == KernelType::Const && quant) &&
         "Const kernel type is only supported for non quantization kernel");

  // Quantized kernels are lowered as linalg.contract.
  if (quant)
    outputOpKind = OutputOpKind::Contract;

  // Disable VNNI packing if it is not a F16/BF16/I8/FP8 data type
  if (!dataTypes.input.isBF16() && !dataTypes.input.isF16() &&
      !dataTypes.input.isInteger(8) &&
      !llvm::isa<Float8E5M2Type, Float8E4M3FNType>(dataTypes.input))
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
    // Quantized kernels carry a per-row input scale and a per-output-channel
    // weight scale to dequantize the wide accumulator.
    if (quant)
      arg.inputScale.type = getShape({batch}, INPUT_SCALE);
    arg.weight.type = getShape({inputSize, outputSize}, PACK_WEIGHT);
    if (quant)
      arg.weightScale.type = getShape({outputSize}, WEIGHT_SCALE);

    // TODO: Bias should be of accumulator type when it differs from the output
    // type AND we want to propagate the truncation through the element-wise ops.
    arg.bias.type = getShape({outputSize}, PACK_OUTPUT);

    // Integer output additionally carries a per-output-channel output scale to
    // requantize the value back down.
    bool hasOutputScale = quant && dataTypes.output.isInteger();
    if (hasOutputScale)
      arg.outputScale.type = getShape({outputSize}, OUTPUT_SCALE);

    if (hasOutputScale) {
      // Requantized GEMM keeps the N x K output layout but stores i8 values.
      auto packedOutTy = getShape({batch, outputSize}, PACK_OUTPUT);
      arg.output.type =
          RankedTensorType::get(packedOutTy.getShape(), dataTypes.input);
    } else {
      arg.output.type = getShape({batch, outputSize}, PACK_OUTPUT);
      arg.accumulator.type = getShape({batch, outputSize}, PACK_ACCUMULATOR);
    }
    args.push_back(arg);

    // Update next input type with the output type of this layer
    currentType = arg.output.type;
  }
}

Value MLIRGenerator::createLayer(LayerArgs &args) {
  OpBuilder::InsertionGuard guard(builder);

  Value chain;
  chain = lowerMatmul(args);

  // These are optional and only emitted if enabled
  if (outputOpKind == OutputOpKind::Generic) {
    chain = lowerBiasAdd(args, chain);
    chain = lowerRelu(args, chain);
  } else {
    chain = lowerNamedBiasAdd(args, chain);
    chain = lowerNamedRelu(args, chain);
  }

  // Last layer may output softmax
  if (args.index == layers.size() - 1)
    chain = lowerSoftmax(args, chain);

  // Return output tensor to the next layer
  return chain;
}

void MLIRGenerator::createKernel() {
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
      if (layer.inputScale.type)
        inputTypes.push_back(layer.inputScale.type);

      inputTypes.push_back(layer.weight.type);
      if (layer.weightScale.type)
        inputTypes.push_back(layer.weightScale.type);
      if (layer.outputScale.type)
        inputTypes.push_back(layer.outputScale.type);

      if (enableBias)
        inputTypes.push_back(layer.bias.type);
      inputTypes.push_back(layer.output.type);
    }
  }

  // Create function with all necessary arguments
  auto func = createFunction(builder, module, "entry", inputTypes,
                             {lastArg.output.type});

  // Add the register unroll user input as a DLTI attribute.
  if (registerUnroll.size() == 3) {
    builder.getContext()->getOrLoadDialect<mlir::DLTIDialect>();
    auto i64 = IntegerType::get(builder.getContext(), 64);

    SmallVector<Attribute> unrollVals = {
        IntegerAttr::get(i64, registerUnroll[0]),
        IntegerAttr::get(i64, registerUnroll[1]),
        IntegerAttr::get(i64, registerUnroll[2])
    };

    auto unrollArray = ArrayAttr::get(builder.getContext(), unrollVals);
    auto keyAttr = StringAttr::get(builder.getContext(), "reg_gemm_unroll");
    auto entry = DataLayoutEntryAttr::get(keyAttr, unrollArray);
    auto deviceSpec = TargetDeviceSpecAttr::get(builder.getContext(), {entry});
    auto systemKey = StringAttr::get(builder.getContext(), "CPU");
    TargetSystemSpecAttr systemSpec = TargetSystemSpecAttr::get(
        builder.getContext(),
        {DataLayoutEntryAttr::get(systemKey, deviceSpec)}
    );

    func->setAttr("dlti.target_system_spec", systemSpec);
  }


  // Initialize the values depending on the KernelType
  //   * Model: input = arg, weights/bias = const, output = zero
  //   * Layer: input/weights/bias/output = args
  firstArg.input.value = func.getArgument(0);
  // Integer inputs carry an input scale right after the model input.
  if (firstArg.inputScale.type)
    firstArg.inputScale.value = func.getArgument(1);

  // Argument position is input + N * { weight/bias } + output
  unsigned argPos = firstArg.inputScale.type ? 2 : 1;
  // Caches the output to chain into the next layer's input
  Value lastOutput;
  for (auto &arg : args) {
    // Chain the last output into this layer
    if (!arg.input.value)
      arg.input.value = lastOutput;

    // Initialize weights and biases
    if (kernelType == KernelType::Args) {
      arg.weight.value = func.getArgument(argPos++);
      if (arg.weightScale.type)
        arg.weightScale.value = func.getArgument(argPos++);
      if (arg.outputScale.type)
        arg.outputScale.value = func.getArgument(argPos++);
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

    lastOutput = createLayer(arg);
    arg.output.value = lastOutput;
  }
  // Data is now output
  func::ReturnOp::create(builder, loc, lastArg.output.value);
}

int MLIRGenerator::generate(StringRef filename) {
  // First, populate the module with all functions
  createKernel();

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

// For dequantization, we have an elementwise scaling after gemm, so the flops
// would be the double of number of elements in the output as it involves two
// multiplications.
void MLIRGenerator::computeElementwiseScalingFlops(ShapedType outputShape) {
  int64_t scalingFlops = 1;
  for (int i = 0, max = outputShape.getRank(); i < max; i++)
    scalingFlops *= outputShape.getDimSize(i);

  // For combining dequantization scales, we have an additional elementwise
  // multiplication, so we count that as well.
  flops += 2 * scalingFlops;
}

Value MLIRGenerator::lowerMatmul(LayerArgs &args) {
  auto inputType = cast<ShapedType>(args.input.value.getType());
  auto outputType = cast<ShapedType>(args.output.value.getType());
  auto shape = outputType.getShape();

  // Quantized kernels always accumulate into i32; regular kernels either
  // accumulate directly into the output tensor or into an explicit accumulator.
  if (quant) {
    args.accumulator.value = getZeroInitTensor(
        RankedTensorType::get(shape, builder.getIntegerType(32)));
  } else if (!args.accumulator.type ||
             args.accumulator.type.getElementType() ==
                 outputType.getElementType()) {
    // Accumulator matches the output type; accumulate directly into it.
    args.accumulator.value = args.output.value;
  } else {
      args.accumulator.value = getZeroInitTensor(args.accumulator.type);
  }

  if (vnniPacked) {
    SmallVector<int64_t> vnniShape{inputType.getShape()};
    vnniShape.back() = vnniShape.back() / vnniFactor;
    vnniShape.push_back(vnniFactor);

    auto weightShape = cast<ShapedType>(args.weight.value.getType()).getShape();
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

    args.input.value = tensor::ExpandShapeOp::create(
        builder, loc, vnniType, args.input.value, reassociationIndices);
  }

  computeMatmulFlops(inputType, outputType);
  Value accumulator;
  switch(outputOpKind) {
    case OutputOpKind::Generic:
      accumulator = lowerGenericMatmul(args, args.input.value);
      break;
    case OutputOpKind::Contract:
      accumulator = lowerContract(args, args.input.value);
      break;
    case OutputOpKind::NamedOp:
      accumulator = lowerNamedMatmul(args, args.input.value);
      break;
  }

  // The quantization epilogue consumes the raw wide accumulator and performs
  // the final cast, so skip the same-domain downcast here.
  if (quant)
    return quantizeEpilogue(args, accumulator);

  return downcastToOutput(builder, loc, accumulator, args.output.type);
}

Value MLIRGenerator::lowerGenericMatmul(LayerArgs &args, Value chain) {
  // Matmul as a linalg.generic
  auto map1 = getMap(chain, MAP_MATMUL_INPUT);   // { 0, 2 }
  auto map2 = getMap(args.weight.value, MAP_MATMUL_WEIGHT); // { 2, 1 }
  auto map3 = getMap(args.accumulator.value, MAP_MATMUL_OUTPUT); // { 0, 1 }
  return linalg::GenericOp::create(
             builder, loc, args.accumulator.value.getType(),
             ValueRange{chain, args.weight.value},
             ValueRange{args.accumulator.value},
             ArrayRef<AffineMap>{map1, map2, map3}, getIterators(MAP_MATMUL),
             [&](OpBuilder &nestedBuilder, Location nestedLoc,
                 ValueRange blockArgs) {
               auto arg0 = blockArgs[0];
               auto arg1 = blockArgs[1];
               auto arg2 = blockArgs[2];
               // If input and output type differs, up cast input to output
               // type using arith.extf/arith.extsi.
               Type inputElementType =
                   cast<ShapedType>(chain.getType()).getElementType();
               Type weightElementType =
                   cast<ShapedType>(args.weight.value.getType())
                       .getElementType();
               Type outputElementType =
                   cast<ShapedType>(args.accumulator.value.getType())
                       .getElementType();
               if (inputElementType != outputElementType) {
                 if (inputElementType.isFloat()) {
                   arg0 = arith::ExtFOp::create(nestedBuilder, nestedLoc,
                                                outputElementType, arg0);
                 } else {
                   arg0 = arith::ExtSIOp::create(nestedBuilder, nestedLoc,
                                                 outputElementType, arg0);
                 }
               }

               if (weightElementType != outputElementType) {
                 if (weightElementType.isFloat()) {
                   arg1 = arith::ExtFOp::create(nestedBuilder, nestedLoc,
                                                outputElementType, arg1);
                 } else {
                   arg1 = arith::ExtSIOp::create(nestedBuilder, nestedLoc,
                                                 outputElementType, arg1);
                 }
               }

               auto *mul = outputElementType.isFloat()
                               ? arith::MulFOp::create(nestedBuilder, nestedLoc,
                                                       arg0, arg1)
                               : arith::MulIOp::create(nestedBuilder, nestedLoc,
                                                       arg0, arg1);
               auto *add = outputElementType.isFloat()
                               ? arith::AddFOp::create(nestedBuilder, nestedLoc,
                                                       arg2, mul->getResult(0))
                               : arith::AddIOp::create(nestedBuilder, nestedLoc,
                                                       arg2, mul->getResult(0));
               linalg::YieldOp::create(nestedBuilder, nestedLoc,
                                       ValueRange{add->getResults()});
             })
      .getResult(0);
}

Value MLIRGenerator::lowerContract(LayerArgs &args, Value chain) {
  // Matmul as a linalg.contract
  SmallVector<Attribute> maps;
  maps.push_back(AffineMapAttr::get(getMap(chain, MAP_MATMUL_INPUT)));   // { 0, 2 }
  maps.push_back(AffineMapAttr::get(getMap(args.weight.value, MAP_MATMUL_WEIGHT))); // { 2, 1 }
  maps.push_back(AffineMapAttr::get(getMap(args.accumulator.value, MAP_MATMUL_OUTPUT))); // { 0, 1 }
  return linalg::ContractOp::create(
             builder, loc, args.accumulator.value.getType(),
             ValueRange{chain, args.weight.value},
             ValueRange{args.accumulator.value}, builder.getArrayAttr(maps))
      .getResult(0);
}

Value MLIRGenerator::lowerNamedMatmul(LayerArgs &args, Value chain) {
  // VNNI produces mixed shape args, say 4D input and 5D weight. All
  // linalg named ops for matrix multiplication expects arguments of same
  // number of dimensions. Hence, such matmul patterns are not compatible to be
  // matched using named ops.
  auto inputShape = cast<ShapedType>(chain.getType());
  assert((vnniFactor != 0 || inputShape.getRank() == 2) &&
         "Unsupported Lowering for VNNI/input rank > 2. "
         "Try 'generic' or 'contract' lowering");

  return linalg::MatmulOp::create(builder, loc,
                                  TypeRange{args.accumulator.value.getType()},
                                  ValueRange{chain, args.weight.value},
                                  ValueRange{args.accumulator.value})
      .getResult(0);
}

Value MLIRGenerator::quantizeEpilogue(LayerArgs &args, Value chain) {
  // Choose the epilogue from the available types:
  //  * integer input, integer output -> requantize (uses the output scale);
  //  * integer input, float output -> dequantize.
  if (dataTypes.output.isInteger())
    return requantizeGemm(args, chain);
  return dequantizeGemm(args, chain);
}

Value MLIRGenerator::requantizeGemm(LayerArgs &args, Value chain) {
  // Chain is the wide integer GEMM accumulator to be requantized down to the
  // output integer type. Requantization first dequantizes the accumulator
  // using the per-row input scale and the per-output-channel weight scale,
  // then rescales it with the per-output-channel output scale and saturates to
  // the signed range of the output integer type:
  //   out = clamp(round(acc * S_input * S_weight * S_output), INT_MIN, INT_MAX)
  assert(chain && "Expected valid chain output from contract/gemm operation");

  Value inputScale = args.inputScale.value;
  Value weightScale = args.weightScale.value;
  Value outputScale = args.outputScale.value;
  Value output = args.output.value;

  auto outputShapedTy = cast<ShapedType>(output.getType());
  Type outElemTy = outputShapedTy.getElementType();
  assert(outElemTy.isInteger() &&
         "Requantization output must be an integer type");

  auto inputScaleTy = cast<ShapedType>(inputScale.getType());
  assert(inputScaleTy.getRank() == 1 && "Input scale must be a vector");
  assert(inputScaleTy.getElementType() == dataTypes.inputScale &&
         "Input scale must be of scale type");
  auto weightScaleTy = cast<ShapedType>(weightScale.getType());
  assert(weightScaleTy.getRank() == 1 && "Weight scale must be a vector");
  assert(weightScaleTy.getElementType() == dataTypes.weightScale &&
         "Weight scale must be of scale type");
  auto outputScaleTy = cast<ShapedType>(outputScale.getType());
  assert(outputScaleTy.getRank() == 1 && "Output scale must be a vector");
  assert(outputScaleTy.getElementType() == dataTypes.outputScale &&
         "Output scale must be of scale type");

  MLIRContext *ctx = &context;
  // Input scale is per-row (batch); weight and output scales are per-output
  // channel (K dimension).
  auto dim0 = getAffineDimExpr(0, ctx);
  auto dim1 = getAffineDimExpr(1, ctx);
  AffineMap inputScaleMap = AffineMap::get(2, 0, {dim0}, ctx);
  AffineMap weightScaleMap = AffineMap::get(2, 0, {dim1}, ctx);
  AffineMap outputScaleMap = AffineMap::get(2, 0, {dim1}, ctx);
  SmallVector<utils::IteratorType> iteratorTypes(2,
                                                 utils::IteratorType::parallel);
  SmallVector<AffineMap> maps = {getMap(chain, MAP_PARALLEL), inputScaleMap,
                                 weightScaleMap, outputScaleMap,
                                 getMap(output, MAP_PARALLEL)};

  // If tiling is applied, expand each scale tensor to match the tiled output
  // dimensions and broadcast over the remaining unit dimensions.
  if (tiles.size() > 0) {
    inputScale =
        createExpandedScaleTensor(builder, loc, inputScale, tiles, true);
    weightScale =
        createExpandedScaleTensor(builder, loc, weightScale, tiles, false);
    outputScale =
        createExpandedScaleTensor(builder, loc, outputScale, tiles, false);

    auto outputShape = outputShapedTy.getShape();
    // Map an expanded scale tensor's dimensions onto the output dimensions,
    // broadcasting unit-sized dimensions. Input scales start matching from the
    // outermost (row) dimension, weight/output scales from the channel one.
    auto createScaleAffineExprs = [&](ArrayRef<int64_t> scaleShape,
                                      bool isInputScale) {
      SmallVector<AffineExpr> affineExprs;
      unsigned outputDim = isInputScale ? 0 : 1;
      unsigned inputDim = isInputScale ? 0 : 1;
      for (auto size : scaleShape) {
        if (size == 1) {
          affineExprs.push_back(getAffineConstantExpr(0, ctx));
        } else {
          while (outputDim < outputShape.size() &&
                 outputShape[outputDim] != size)
            outputDim++;
          affineExprs.push_back(getAffineDimExpr(inputDim, ctx));
          outputDim++;
        }
        inputDim++;
      }
      return affineExprs;
    };

    unsigned rank = outputShapedTy.getRank();
    auto inScaleShape = cast<ShapedType>(inputScale.getType()).getShape();
    auto wScaleShape = cast<ShapedType>(weightScale.getType()).getShape();
    auto oScaleShape = cast<ShapedType>(outputScale.getType()).getShape();
    maps[1] = AffineMap::get(rank, 0,
                             createScaleAffineExprs(inScaleShape, true), ctx);
    maps[2] = AffineMap::get(rank, 0,
                             createScaleAffineExprs(wScaleShape, false), ctx);
    maps[3] = AffineMap::get(rank, 0,
                             createScaleAffineExprs(oScaleShape, false), ctx);
    iteratorTypes.assign(rank, utils::IteratorType::parallel);
  }

  Type floatTy = builder.getF32Type();
  // Saturation bounds derived from the signed output integer width.
  unsigned outBitWidth = outElemTy.getIntOrFloatBitWidth();
  double lowBound = llvm::APInt::getSignedMinValue(outBitWidth).getSExtValue();
  double highBound = llvm::APInt::getSignedMaxValue(outBitWidth).getSExtValue();
  Value lowClamp = arith::ConstantOp::create(
      builder, loc, builder.getFloatAttr(floatTy, lowBound));
  Value highClamp = arith::ConstantOp::create(
      builder, loc, builder.getFloatAttr(floatTy, highBound));

  auto result =
      linalg::GenericOp::create(
          builder, loc, TypeRange{outputShapedTy},
          ValueRange{chain, inputScale, weightScale, outputScale},
          ValueRange{output}, maps, iteratorTypes,
          [&](OpBuilder &nestedBuilder, Location nestedLoc,
              ValueRange blockArgs) {
            // Cast the wide accumulator to float and combine all scales.
            Value accF = createCastToFloat(nestedBuilder, nestedLoc,
                                           blockArgs[0], floatTy);
            Value inS = createCastToFloat(nestedBuilder, nestedLoc,
                                          blockArgs[1], floatTy);
            Value wS = createCastToFloat(nestedBuilder, nestedLoc,
                                         blockArgs[2], floatTy);
            Value oS = createCastToFloat(nestedBuilder, nestedLoc,
                                         blockArgs[3], floatTy);
            // Dequantize with input/weight scales, then apply output scale.
            Value scale =
                arith::MulFOp::create(nestedBuilder, nestedLoc, inS, wS)
                    ->getResult(0);
            scale = arith::MulFOp::create(nestedBuilder, nestedLoc, scale, oS)
                        ->getResult(0);
            Value scaled =
                arith::MulFOp::create(nestedBuilder, nestedLoc, accF, scale)
                    ->getResult(0);
            // Saturate to the output integer range before truncating.
            scaled = arith::MaximumFOp::create(nestedBuilder, nestedLoc, scaled,
                                               lowClamp);
            scaled = arith::MinimumFOp::create(nestedBuilder, nestedLoc, scaled,
                                               highClamp);
            Value quantized = arith::FPToSIOp::create(nestedBuilder, nestedLoc,
                                                      outElemTy, scaled);
            linalg::YieldOp::create(nestedBuilder, nestedLoc,
                                    ValueRange{quantized});
          })
          .getResult(0);

  computeElementwiseScalingFlops(outputShapedTy);
  return result;
}

Value MLIRGenerator::dequantizeGemm(LayerArgs &args, Value chain) {
  // Chain is the contract/gemm output
  assert(chain && "Expected valid chain output from contract/gemm operation");

  Value inputScale = args.inputScale.value;
  Value weightScale = args.weightScale.value;
  Value output = args.output.value;

  // For mixed type, we need to handle input and weight scales to compute the
  // resultant scaleand then multiply the result with the contract output.
  auto inputScaleTy = cast<ShapedType>(inputScale.getType());
  assert(inputScaleTy.getRank() == 1 && "Input scale must be a vector");
  assert(inputScaleTy.getElementType() == dataTypes.inputScale &&
         "Input scale must be of scale type");

  auto weightScaleTy = cast<ShapedType>(weightScale.getType());
  assert(weightScaleTy.getRank() == 1 && "Weight scale must be a vector");
  assert(weightScaleTy.getElementType() == dataTypes.weightScale &&
         "Weight scale must be of scale type");

  // Create a 2-D ouput scale shape using input and weight scales
  auto outputScaleShape = SmallVector<int64_t>{inputScaleTy.getShape()[0],
                                               weightScaleTy.getShape()[0]};
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
    SmallVector<AffineExpr> inputScaleAffineExprs;
    SmallVector<AffineExpr> weightScaleAffineExprs;

    // Infer the affine expressions for input and weight scales based on the
    // output shape and the scale shapes.
    auto inputScaleShape = cast<ShapedType>(inputScale.getType()).getShape();
    auto weightScaleShape = cast<ShapedType>(weightScale.getType()).getShape();
    auto outputShape = cast<ShapedType>(outputShapedTy).getShape();

    // Map scale dimensions to output dimensions
    auto createScaleAffineExprs = [&](ArrayRef<int64_t> scaleShape,
                                      bool isInputScale) {
      SmallVector<AffineExpr> affineExprs;
      // Input scale maps to output dim 0, weight scale maps to output dim 1
      unsigned outputDim = isInputScale ? 0 : 1;
      unsigned inputDim = isInputScale ? 0 : 1;
      for (auto size : scaleShape) {
        if (size == 1) {
          affineExprs.push_back(getAffineConstantExpr(0, &context));
        } else {
          // Find matching dimension in output shape
          while (outputDim < outputShape.size() &&
                 outputShape[outputDim] != size)
            outputDim++;
          affineExprs.push_back(getAffineDimExpr(inputDim, &context));
          outputDim++;
        }
        inputDim++;
      }
      return affineExprs;
    };

    inputScaleAffineExprs = createScaleAffineExprs(inputScaleShape, true);
    weightScaleAffineExprs = createScaleAffineExprs(weightScaleShape, false);
    AffineMap packedInputScaleMap = AffineMap::get(
        outputShapedTy.getRank(), 0, inputScaleAffineExprs, &context);
    AffineMap packedWeightScaleMap = AffineMap::get(
        outputShapedTy.getRank(), 0, weightScaleAffineExprs, &context);
    reshapeMap[1] = packedInputScaleMap;
    reshapeMap[2] = packedWeightScaleMap;
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

            // For int8(f8E8M0FNU) scales, we need to convert the int8 scales to
            // float scales before computing the resultant scale by
            // multiplying the two scales.
            auto floatTy = builder.getF32Type();
            bool isNarrowFloatType =
                dataTypes.inputScale.isFloat() &&
                dataTypes.inputScale.getIntOrFloatBitWidth() < 32;
            arith::FastMathFlags fmf = isNarrowFloatType
                                           ? arith::FastMathFlags::nnan
                                           : arith::FastMathFlags::none;
            arg1 =
                createCastToFloat(nestedBuilder, nestedLoc, arg1, floatTy,
                                 arith::FastMathFlagsAttr::get(&context, fmf));
            arg2 =
                createCastToFloat(nestedBuilder, nestedLoc, arg2, floatTy,
                                 arith::FastMathFlagsAttr::get(&context, fmf));
            Value alu = arith::MulFOp::create(nestedBuilder, loc, arg1, arg2)
                            ->getResult(0);
            Value castToFloat =
                createCastToFloat(nestedBuilder, nestedLoc, arg0,
                                 outputShapedTy.getElementType());
            alu = arith::MulFOp::create(nestedBuilder, loc, castToFloat, alu)
                      ->getResult(0);
            linalg::YieldOp::create(nestedBuilder, loc, ValueRange{alu});
          })
          .getResult(0);

  // Compute flop for dequantization by combining scales and then applying the
  // combined scale on output of gemm.
  computeElementwiseScalingFlops(outputShapedTy);
  return result;
}

Value MLIRGenerator::lowerBiasAdd(LayerArgs &args, Value chain) {
  if (!enableBias)
    return chain;

  auto outTy = cast<ShapedType>(chain.getType());
  auto mapA = getMap(chain, MAP_BROADCAST);
  auto mapB = getMap(chain, MAP_PARALLEL);
  auto sum =
      linalg::GenericOp::create(builder, 
              loc, outTy, ValueRange{args.bias.value}, ValueRange{chain},
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

Value MLIRGenerator::lowerNamedBiasAdd(LayerArgs &args, Value chain) {
  if (!enableBias)
    return chain;

  auto outTy = cast<ShapedType>(chain.getType());
  auto biasTy = cast<ShapedType>(args.bias.value.getType());
  Value emptyTensor = tensor::EmptyOp::create(builder, loc, outTy, ValueRange{});
  SmallVector<int64_t> addedDimensions;
  SmallVector<bool> dimsNeeded =
      getBroadcastDims(biasTy.getShape(), outTy.getShape());
  for (int64_t dim : llvm::seq<int64_t>(0, outTy.getRank() - 1)) {
    if (dimsNeeded[dim])
      addedDimensions.push_back(dim);
  }

  Value broadcast =
      linalg::BroadcastOp::create(builder, loc, args.bias.value, emptyTensor, addedDimensions)
          .getResult()[0];
  Value biasAdd = linalg::AddOp::create(builder, loc, TypeRange{args.output.value.getType()},
                                             ValueRange{broadcast, chain},
                                             ValueRange{emptyTensor})
                      .getResult(0);

  computeBiasOrReluFlops(outTy);
  return biasAdd;
}

Value MLIRGenerator::lowerNamedRelu(LayerArgs &args, Value chain) {
  if (!enableRelu)
    return chain;

  auto outTy = cast<ShapedType>(chain.getType());
  auto zero =
      getConstFloat(builder, 0.0, cast<FloatType>(outTy.getElementType()));
  Value emptyTensor = tensor::EmptyOp::create(builder, loc, outTy, ValueRange{});
  auto fill =
      linalg::FillOp::create(builder, loc, zero, emptyTensor)->getResult(0);
  Value relu = linalg::MaxOp::create(builder, loc, TypeRange{args.output.value.getType()},
                                 ValueRange{chain, fill}, ValueRange{emptyTensor})
          .getResult(0);

  computeBiasOrReluFlops(outTy);
  return relu;
}

Value MLIRGenerator::lowerRelu(LayerArgs &args, Value chain) {
  if (!enableRelu)
    return chain;

  auto zero = getConstFloat(
      builder, 0.0,
      cast<FloatType>(cast<ShapedType>(chain.getType()).getElementType()));
  auto outTy = cast<ShapedType>(chain.getType());
  auto map = getMap(chain, MAP_PARALLEL);
  auto relu =
      linalg::GenericOp::create(builder, 
              loc, outTy, ValueRange{}, ValueRange{chain},
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

Value MLIRGenerator::lowerSoftmax(LayerArgs &args, Value chain) {
  if (!enableSoftmax)
    return chain;

  assert(cast<ShapedType>(chain.getType()).getRank() == 2 &&
         "Packed softmax not implemented yet");
  auto map1 = getMap(chain, MAP_PARALLEL);
  auto map2 = getMap(chain, MAP_REDUCTION);
  auto outTy = cast<ShapedType>(chain.getType());

  // First, we calculate the element-wise exp
  Value expTensor = tensor::EmptyOp::create(builder, loc, outTy, ValueRange{});
  auto exp = linalg::GenericOp::create(builder, 
      loc, outTy, ValueRange{chain}, ValueRange{expTensor},
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
  auto zero = getConstFloat(builder, 0.0, cast<FloatType>(dataTypes.input));
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
              ValueRange{args.output.value}, ArrayRef<AffineMap>{map1, map1, map1},
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
    return RankedTensorType::get(dims, type == PACK_OUTPUT ? dataTypes.output
                                                           : dataTypes.input);

  if (!tiles.size()) {
    if (quant) {
      if (type == INPUT_SCALE || type == WEIGHT_SCALE) {
        return RankedTensorType::get(dims, type == INPUT_SCALE
                                               ? dataTypes.inputScale
                                               : dataTypes.weightScale);
      } else if (type == OUTPUT_SCALE) {
        return RankedTensorType::get(dims, dataTypes.outputScale);
      } else if (type == PACK_OUTPUT) {
        return RankedTensorType::get(dims, dataTypes.output);
      } else if (type == PACK_ACCUMULATOR) {
        return RankedTensorType::get(dims, dataTypes.accumulator);
      } else if (type == PACK_INPUT) {
        return RankedTensorType::get(dims, dataTypes.input);
      }
    }
    // Unpacked type, just return 2D tensor
    return RankedTensorType::get(dims, dataTypes.input);
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
    return RankedTensorType::get({x / n, y / c, n, c}, dataTypes.input);
  case PACK_WEIGHT:
    // VNNI packing can be done via tpp-opt --vnni-pack
    assert(x % k == 0 && "Invalid tile size for K dim");
    assert(y % c == 0 && "Invalid tile size for C dim");

    // VNNI: C x K -> BK x BC x bc/vnni x bk x vnni
    if (vnniFactor != 0)
      return RankedTensorType::get(
          {y / k, x / c, c / vnniFactor, k, vnniFactor}, dataTypes.input);

    // C x K -> BK x BC x bc x bk
    return RankedTensorType::get({y / k, x / c, c, k}, dataTypes.input);
  case PACK_OUTPUT:
    assert(x % n == 0 && "Invalid tile size for N dim");

    // Broadcast 1D -> 2D is Bk x bk only
    if (!y)
      return RankedTensorType::get({x / k, k}, dataTypes.output);

    // N x K -> BN x BK x bn x bk
    assert(y % k == 0 && "Invalid tile size for K dim");
    return RankedTensorType::get({x / n, y / k, n, k}, dataTypes.output);
  case PACK_ACCUMULATOR:
    assert(x % n == 0 && "Invalid tile size for N dim");

    // Broadcast 1D -> 2D is Bk x bk only
    if (!y)
      return RankedTensorType::get({x / k, k}, dataTypes.accumulator);

    // N x K -> BN x BK x bn x bk
    assert(y % k == 0 && "Invalid tile size for K dim");
    return RankedTensorType::get({x / n, y / k, n, k}, dataTypes.accumulator);
  case INPUT_SCALE:
    return RankedTensorType::get({dims[0]}, dataTypes.inputScale);
  case WEIGHT_SCALE:
    return RankedTensorType::get({dims[0]}, dataTypes.weightScale);
  case OUTPUT_SCALE:
    return RankedTensorType::get({dims[0]}, dataTypes.outputScale);
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
