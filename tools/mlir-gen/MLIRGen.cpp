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
#include "llvm/Support/ErrorHandling.h"

#include <optional>

using namespace mlir;

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

} // anonymous namespace

MLIRGenerator::MLIRGenerator(StringRef outputOpKindStr, StringRef kernelStr,
                             unsigned batch, StringRef layersStr,
                             StringRef tilesStr, StringRef targetType, int seed,
                             bool enableBias, bool enableRelu,
                             bool enableSoftmax, bool keepGenericMatmul,
                             int vnniBlockingFactor)
    : builder(&context), loc(builder.getUnknownLoc()), batch(batch), seed(seed),
      flops(0), enableBias(enableBias), enableRelu(enableRelu),
      enableSoftmax(enableSoftmax), keepGenericMatmul(keepGenericMatmul),
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
  auto elementType = llvm::StringSwitch<std::optional<Type>>(targetType)
                         .CaseLower("f32", builder.getF32Type())
                         .CaseLower("f16", builder.getF16Type())
                         .CaseLower("bf16", builder.getBF16Type())
                         .Default(std::nullopt);
  assert(elementType && "Unsupported data type");
  dataType = *elementType;

  // Disable VNNI packing if it is not a F16/BF16 data type
   if (!dataType.isBF16() && !dataType.isF16())
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
  module = builder.create<ModuleOp>(loc);
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
    arg.weight.type = getShape({inputSize, outputSize}, PACK_WEIGHT);
    arg.bias.type = getShape({outputSize}, PACK_OUTPUT);
    arg.output.type = getShape({batch, outputSize}, PACK_OUTPUT);
    args.push_back(arg);

    // Update next input type with the output type of this layer
    currentType = arg.output.type;
  }
}

Value MLIRGenerator::createLayer(LayerArgs &args) {
  OpBuilder::InsertionGuard guard(builder);

  Value chain;
  chain = lowerMatmul(args.input.value, args.weight.value, args.output.value);

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
      inputTypes.push_back(layer.weight.type);
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

  // Argument position is input + N * { weight/bias } + output
  // First weight is at position 1, every two
  unsigned argPos = 1;
  // Caches the output to chain into the next layer's input
  Value lastOutput;
  for (auto &arg : args) {
    // Chain the last output into this layer
    if (!arg.input.value)
      arg.input.value = lastOutput;

    // Initialize weights and biases
    if (kernelType == KernelType::Args) {
      arg.weight.value = func.getArgument(argPos++);
      if (enableBias)
        arg.bias.value = func.getArgument(argPos++);
      arg.output.value = func.getArgument(argPos++);
    } else { // Model
      arg.weight.value =
          createDenseTensor(builder, initType, arg.weight.type, getRand());
      if (enableBias)
        arg.bias.value =
            createDenseTensor(builder, initType, arg.bias.type, getRand());
      arg.output.value = getZeroInitTensor(arg.output.type);
    }

    // Now pass the input through all layers
    lastOutput = createLayer(arg);
    arg.output.value = lastOutput;
  }
  // Data is now output
  builder.create<func::ReturnOp>(loc, lastArg.output.value);
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
    namedMatmul = builder
                      .create<linalg::MatmulOp>(
                          loc, TypeRange{output.getType()},
                          ValueRange{input, weight}, ValueRange{output})
                      .getResult(0);
  } else if (inputShape.getRank() == 4) {
    SmallVector<OpFoldResult, 4> dims =
        tensor::getMixedSizes(builder, loc, weight);
    applyPermutationToVector(dims, {0, 1, 3, 2});
    Value emptyTensor = builder.create<tensor::EmptyOp>(
        loc, dims, weightShape.getElementType());

    Value transpose =
        builder
            .create<linalg::TransposeOp>(loc, weight, emptyTensor,
                                         ArrayRef<int64_t>{0, 1, 3, 2})
            .getResults()[0];
    namedMatmul = builder
                      .create<linalg::Mmt4DOp>(loc, TypeRange{output.getType()},
                                               ValueRange{input, transpose},
                                               ValueRange{output})
                      .getResult(0);
  }

  return namedMatmul;
}

Value MLIRGenerator::lowerMatmul(Value input, Value weight, Value output) {
  Value chain;
  auto inputType = cast<ShapedType>(input.getType());
  auto outputType = cast<ShapedType>(output.getType());

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

    input = builder.create<tensor::ExpandShapeOp>(loc, vnniType, input,
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
      builder
          .create<linalg::GenericOp>(
              loc, output.getType(), ValueRange{input, weight},
              ValueRange{output}, ArrayRef<AffineMap>{map1, map2, map3},
              getIterators(MAP_MATMUL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto arg1 = blockArgs[1];
                auto arg2 = blockArgs[2];
                auto mul = nestedBuilder.create<arith::MulFOp>(loc, arg0, arg1);
                auto add = nestedBuilder.create<arith::AddFOp>(loc, arg2, mul);
                nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{add});
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
  auto contract = builder
                      .create<linalg::ContractOp>(
                          loc, output.getType(), ValueRange{input, weight}, ValueRange{output},
                          builder.getArrayAttr(maps))
                      .getResult(0);

  return contract;
}

Value MLIRGenerator::lowerBiasAdd(Value input, Value bias, Value output) {
  if (!enableBias)
    return input;

  auto outTy = cast<ShapedType>(input.getType());
  auto mapA = getMap(input, MAP_BROADCAST);
  auto mapB = getMap(input, MAP_PARALLEL);
  auto sum =
      builder
          .create<linalg::GenericOp>(
              loc, outTy, ValueRange{bias}, ValueRange{input},
              ArrayRef<AffineMap>{mapA, mapB}, getIterators(MAP_PARALLEL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto arg1 = blockArgs[1];
                auto add = nestedBuilder.create<arith::AddFOp>(loc, arg0, arg1);
                nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{add});
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
  Value emptyTensor = builder.create<tensor::EmptyOp>(loc, outTy, ValueRange{});
  SmallVector<int64_t> addedDimensions;
  SmallVector<bool> dimsNeeded =
      getBroadcastDims(biasTy.getShape(), outTy.getShape());
  for (int64_t dim : llvm::seq<int64_t>(0, outTy.getRank() - 1)) {
    if (dimsNeeded[dim])
      addedDimensions.push_back(dim);
  }

  Value broadcast =
      builder
          .create<linalg::BroadcastOp>(loc, bias, emptyTensor, addedDimensions)
          .getResult()[0];
  Value biasAdd = builder
                      .create<linalg::AddOp>(loc, TypeRange{output.getType()},
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
  auto zero = getConstFloat(builder, 0.0, cast<FloatType>(dataType));
  Value emptyTensor = builder.create<tensor::EmptyOp>(loc, outTy, ValueRange{});
  auto fill =
      builder.create<linalg::FillOp>(loc, zero, emptyTensor)->getResult(0);
  Value relu =
      builder
          .create<linalg::MaxOp>(loc, TypeRange{output.getType()},
                                 ValueRange{input, fill}, ValueRange{emptyTensor})
          .getResult(0);

  computeBiasOrReluFlops(outTy);
  return relu;
}

Value MLIRGenerator::lowerRelu(Value input, Value output) {
  if (!enableRelu)
    return input;

  auto zero = getConstFloat(builder, 0.0, cast<FloatType>(dataType));
  auto outTy = cast<ShapedType>(input.getType());
  auto map = getMap(input, MAP_PARALLEL);
  auto relu =
      builder
          .create<linalg::GenericOp>(
              loc, outTy, ValueRange{}, ValueRange{input},
              ArrayRef<AffineMap>{map}, getIterators(MAP_PARALLEL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto max =
                    nestedBuilder.create<arith::MaximumFOp>(loc, arg0, zero);
                nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{max});
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
  Value expTensor = builder.create<tensor::EmptyOp>(loc, outTy, ValueRange{});
  auto exp = builder.create<linalg::GenericOp>(
      loc, outTy, ValueRange{input}, ValueRange{expTensor},
      ArrayRef<AffineMap>{map1, map1}, getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto exp = nestedBuilder.create<math::ExpOp>(loc, arg0);
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{exp});
      });

  // Second, we sum-reduce and splat
  SmallVector<int64_t> dims{batch, 1};
  auto redTy = getShape(dims, PACK_OUTPUT);
  Value redTensor =
      builder.create<tensor::EmptyOp>(loc, dims, outTy.getElementType());
  auto zero = getConstFloat(builder, 0.0, cast<FloatType>(dataType));
  auto fill = builder.create<linalg::FillOp>(loc, zero, redTensor);
  auto redux = builder.create<linalg::GenericOp>(
      loc, redTy, ValueRange{exp.getResult(0)}, ValueRange{fill.getResult(0)},
      ArrayRef<AffineMap>{map1, map2}, getIterators(MAP_REDUCTION),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        auto arg1 = blockArgs[1];
        auto add = nestedBuilder.create<arith::AddFOp>(loc, arg0, arg1);
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{add});
      });
  // Splat back to the same dims
  Value meanTensor = builder.create<tensor::EmptyOp>(loc, outTy, ValueRange{});
  auto mean = builder.create<linalg::GenericOp>(
      loc, outTy, ValueRange{redux.getResult(0)}, ValueRange{meanTensor},
      ArrayRef<AffineMap>{map2, map1}, getIterators(MAP_PARALLEL),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto arg0 = blockArgs[0];
        nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{arg0});
      });

  // Third, we update the exp/sum(exp) onto the output tensor
  auto softmax =
      builder
          .create<linalg::GenericOp>(
              loc, outTy, ValueRange{exp.getResult(0), mean.getResult(0)},
              ValueRange{output}, ArrayRef<AffineMap>{map1, map1, map1},
              getIterators(MAP_PARALLEL),
              [&](OpBuilder &nestedBuilder, Location nestedLoc,
                  ValueRange blockArgs) {
                auto arg0 = blockArgs[0];
                auto arg1 = blockArgs[1];
                auto div = nestedBuilder.create<arith::DivFOp>(loc, arg0, arg1);
                nestedBuilder.create<linalg::YieldOp>(loc, ValueRange{div});
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
    return RankedTensorType::get(dims, dataType);

  // Unpacked type, just return 2D tensor
  if (!tiles.size())
    return RankedTensorType::get(dims, dataType);

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
    return RankedTensorType::get({x / n, y / c, n, c}, dataType);
  case PACK_WEIGHT:
    // VNNI packing can be done via tpp-opt --vnni-pack
    assert(x % k == 0 && "Invalid tile size for K dim");
    assert(y % c == 0 && "Invalid tile size for C dim");

    // VNNI: C x K -> BK x BC x bc/vnni x bk x vnni
    if (vnniFactor != 0)
      return RankedTensorType::get(
          {y / k, x / c, c / vnniFactor, k, vnniFactor}, dataType);

    // C x K -> BK x BC x bc x bk
    return RankedTensorType::get({y / k, x / c, c, k}, dataType);
  case PACK_OUTPUT:
    assert(x % n == 0 && "Invalid tile size for N dim");

    // Broadcast 1D -> 2D is Bk x bk only
    if (!y)
      return RankedTensorType::get({x / k, k}, dataType);

    // N x K -> BN x BK x bn x bk
    assert(y % k == 0 && "Invalid tile size for K dim");
    return RankedTensorType::get({x / n, y / k, n, k}, dataType);
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
  auto zero = getConstFloat(builder, 0.0, cast<FloatType>(dataType));
  Value tensor =
      builder.create<tensor::EmptyOp>(loc, type, ValueRange{}).getResult();
  tensor = builder.create<linalg::FillOp>(loc, zero, tensor).getResult(0);
  return tensor;
}
