//===- TensorInitInt.h - MLIR Tensor Initialization -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Initializes tensors for kernel input/output handling with some reasonable
// distribution to allow for layout testing (reorder, pad) without vanishing
// or exploding values at the end of a large model - uses quantization range
// within <0 - 255> integer values.
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_TENSORINITINT_H
#define TPP_TRANSFORMS_UTILS_TENSORINITINT_H

#include "TPP/Transforms/Utils/TensorInit.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

#include <algorithm>
#include <random>

// Base class for integer values.
struct TensorInitInt : public TensorInit<llvm::APInt> {
  // Supported data types.
  // TODO: Support signed (si32) and unsinged (ui32) integers
  enum class DataType { AUTO, I8, I16, I32, I64 };

  static bool isTypeSupported(const mlir::Type &type) {
    return type.isSignlessInteger(8) || type.isSignlessInteger(16) ||
           type.isSignlessInteger(32) || type.isSignlessInteger(64);
  }

  // Get data type from element type.
  static DataType getTensorInitDataType(mlir::Type type);

  // Get bit width from data type.
  static unsigned getDataTypeBitWidth(DataType type);

  // True if the data type is signed.
  static bool isDataTypeSigned(DataType type);

  TensorInitInt(DataType type)
      : type(type), bitWidth(getDataTypeBitWidth(type)),
        isSigned(isDataTypeSigned(type)) {}
  virtual ~TensorInitInt() = default;

protected:
  // Tensor element data type.
  DataType type;

  // Bit width of the data type.
  unsigned bitWidth;

  // True if the data type is signed.
  bool isSigned;

  // Insert element indexed on the buffer.
  using TensorInit::insert;
  virtual void insert(size_t index, uint64_t value);

  // Insert element at the end of the buffer.
  using TensorInit::push;
  virtual void push(uint64_t value);

  // Convert value to the tensor's data type (by reference).
  void convertType(llvm::APInt &value) override final;

  // Actual implementation that fills the buffer
  // To be implemented by derived classes.
  virtual void fillData() override = 0;
};

// Constant init (all-ones, do not use!).
struct ConstantTensorInitInt : TensorInitInt {
  ConstantTensorInitInt(DataType type) : TensorInitInt(type) {}

  // Return a dense<1> repeated throughout the shape.
  mlir::DenseElementsAttr get(mlir::ShapedType shape) override;

  void fillData() override;
};

// Simple init (basic example, not useful).
struct SimpleTensorInitInt : TensorInitInt {
  SimpleTensorInitInt(DataType type) : TensorInitInt(type) {}

  // Return a dense<0, 1, 2> repeated throughout the shape.
  void fillData() override;
};

// Continuous init (quantized normalized affine range).
struct ContinuousTensorInitInt : TensorInitInt {
  ContinuousTensorInitInt(DataType type)
      : TensorInitInt(type), upperBound(255) {
    if (type == DataType::I8)
      upperBound = 127;
  }

  // Return a dense<0 ... upperBound> throughout the shape.
  void fillData() override;

  // Upper bound for quantization.
  int upperBound;
};

// Random init (uniform).
struct RandomTensorInitInt : TensorInitInt {
  RandomTensorInitInt(DataType type, int seed)
      : TensorInitInt(type), generator(seed), distribution(0, 255) {
    if (type == DataType::I8)
      distribution = std::uniform_int_distribution<uint64_t>(0, 127);
  }

  // Next random uniform number.
  float next() { return distribution(generator); }

  // Return a dense<uniform(0, distribution)> throughout the shape.
  void fillData() override;

private:
  // Random generator.
  std::default_random_engine generator;
  // Random distribution.
  std::uniform_int_distribution<uint64_t> distribution;
};

// Random init (normal).
struct NormalTensorInitInt : TensorInitInt {
  NormalTensorInitInt(DataType type, int seed)
      : TensorInitInt(type), generator(seed), distribution(255) {
    if (type == DataType::I8)
      distribution = std::binomial_distribution<uint64_t>(127);
  }

  // Next random number.
  float next() {
    auto value = distribution(generator);
    return value;
  }

  // Return a dense<normal(0, distribution)> throughout the shape.
  void fillData() override;

private:
  // Random generator.
  std::default_random_engine generator;
  // Random distribution.
  std::binomial_distribution<uint64_t> distribution;
};

#endif // TPP_TRANSFORMS_UTILS_TENSORINITINT_H
