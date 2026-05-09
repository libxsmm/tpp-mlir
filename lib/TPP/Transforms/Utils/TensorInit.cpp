//===- TensorInit.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/TensorInit.h"
#include "TPP/Transforms/Utils/TensorInitFloat.h"
#include "TPP/Transforms/Utils/TensorInitInt.h"

#include <functional>
#include <unordered_map>

using namespace mlir;

namespace {

struct InitKey {
  InitKey() = default;
  explicit InitKey(TensorInitType type, mlir::Type elmType, int seed,
                   bool isScaleArgument = false)
      : type(type), isScaleArgument(isScaleArgument) {
    // Seed only matters for randomized types.
    switch (type) {
    case TensorInitType::Random:
    case TensorInitType::Normal:
    case TensorInitType::Quant:
    case TensorInitType::Mixed:
      this->seed = seed;
      break;
    default:
      this->seed = 0;
      break;
    }

    floatType = TensorInitFloat::getTensorInitDataType(elmType);
    intType = TensorInitInt::getTensorInitDataType(elmType);
  }

  bool operator==(const InitKey &ik) const {
    return type == ik.type && floatType == ik.floatType &&
           intType == ik.intType && seed == ik.seed &&
           isScaleArgument == ik.isScaleArgument;
  }

  TensorInitType type;
  TensorInitFloat::DataType floatType;
  TensorInitInt::DataType intType;
  int seed;
  bool isScaleArgument;
};

struct InitKeyHash_fn {
  std::size_t operator()(const InitKey &ik) const {
    std::size_t h1 = std::hash<TensorInitType>{}(ik.type);
    std::size_t h2 = std::hash<TensorInitFloat::DataType>{}(ik.floatType);
    std::size_t h3 = std::hash<TensorInitInt::DataType>{}(ik.intType);
    std::size_t h4 = std::hash<int>{}(ik.seed);
    std::size_t h5 = std::hash<bool>{}(ik.isScaleArgument);
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
  }
};

std::unordered_map<InitKey, TensorInitPtr, InitKeyHash_fn> tensorInitializers;
} // namespace

TensorInitType parseTensorInitType(StringRef name) {
  auto type = StringSwitch<TensorInitType>(name)
                  .Case("", TensorInitType::Auto)
                  .Case("const", TensorInitType::Constant)
                  .Case("random", TensorInitType::Random)
                  .Case("normal", TensorInitType::Normal)
                  .Case("identity", TensorInitType::Identity)
                  .Case("mixed", TensorInitType::Mixed)
                  .Case("quant", TensorInitType::Quant)
                  .Default(TensorInitType::Invalid);
  return type;
}

TensorInitPtr getTensorInit(TensorInitType type, mlir::Type elmType,
                            mlir::Type nextElmType, int seed,
                            bool isScaleArgument) {
  // Defaults for seed or not
  if (type == TensorInitType::Auto) {
    if (seed)
      type = TensorInitType::Normal;
    else
      type = TensorInitType::Constant;
  }

  InitKey key(type, elmType, seed, isScaleArgument);
  if (tensorInitializers.find(key) != tensorInitializers.end())
    return tensorInitializers[key];

  TensorInitPtr initPtr = nullptr;

  if (TensorInitFloat::isTypeSupported(elmType)) {
    auto dataType = TensorInitFloat::getTensorInitDataType(elmType);
    switch (type) {
    case TensorInitType::Constant:
      initPtr = std::make_shared<ConstantTensorInitFloat>(dataType);
      break;
    case TensorInitType::Random:
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_shared<RandomTensorInitFloat>(dataType, seed);
      break;
    case TensorInitType::Normal:
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_shared<NormalTensorInitFloat>(dataType, seed);
      break;
    case TensorInitType::Identity:
      initPtr = std::make_shared<IdentityTensorInitFloat>(dataType);
      break;
    default:
      assert(false && "Invalid tensor initializer type");
    }
  }

  if (TensorInitInt::isTypeSupported(elmType)) {
    auto dataType = TensorInitInt::getTensorInitDataType(elmType);
    switch (type) {
    case TensorInitType::Constant:
      initPtr = std::make_shared<ConstantTensorInitInt>(dataType);
      break;
    case TensorInitType::Random:
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_shared<RandomTensorInitInt>(dataType, seed);
      break;
    case TensorInitType::Normal:
      assert(seed && "Can't call random initializers without seed");
      initPtr = std::make_shared<NormalTensorInitInt>(dataType, seed);
      break;
    case TensorInitType::Identity:
      initPtr = std::make_shared<IdentityTensorInitInt>(dataType);
      break;
    case TensorInitType::Mixed:
      initPtr = std::make_shared<QuantTensorInitInt>(dataType, nullptr, seed,
                                                     nullptr);
      break;
    case TensorInitType::Quant: {
      // Create a float initializer for the dequant scale factors to do
      // the initialization for the quantized argument and corresponding
      // dequant scale factors.
      assert(seed && "Can't call random initializers without seed");
      auto floatScaleDataType =
          TensorInitFloat::getTensorInitDataType(nextElmType);
      TensorInitPtr floatScaleInit =
          std::make_shared<QuantScaleTensorInitFloat>(floatScaleDataType, seed);
      if (!isScaleArgument) {
        initPtr = std::make_shared<QuantTensorInitInt>(dataType, nextElmType,
                                                       seed, floatScaleInit);
        // Store the float/int initializer for dequant scale factors into hash.
        InitKey keyFloatScale(type, nextElmType, seed, true);
        tensorInitializers[keyFloatScale] = floatScaleInit;
      }
      break;
    }
    default:
      assert(false && "Invalid tensor initializer type");
    }
  }

  assert(initPtr && "Unsupported tensor element type");
  tensorInitializers[key] = initPtr;

  return initPtr;
}

TensorInitPtr getTensorInit(StringRef type, mlir::Type elmType,
                            mlir::Type nextElmType, int seed,
                            bool isScaleArgument) {
  auto initType = parseTensorInitType(type);
  return getTensorInit(initType, elmType, nextElmType, seed, isScaleArgument);
}
