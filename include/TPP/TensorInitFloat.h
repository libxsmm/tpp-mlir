//===- TensorInitFloat.h - MLIR Tensor Initialization ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Initializes tensors for kernel input/output handling with some reasonable
// distribution to allow for layout testing (reorder, pad) without vanishing
// or exploding values at the end of a large model (0.0 ~ 1.0).
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TENSORINITFLOAT_H
#define TPP_TENSORINITFLOAT_H

#include "TPP/TensorInit.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"

#include <algorithm>
#include <random>

// Base class for float values.
struct TensorInitFloat : public TensorInit<llvm::APFloat> {
  // Supported data types. (TODO: Support 8-bit data types)
  enum class DataType { AUTO, FP32, FP64, BF16 };

  static bool isTypeSupported(const mlir::Type &type) {
    return type.isF32() || type.isF64() || type.isBF16();
  }

  // Get data type from element type.
  static DataType getTensorInitDataType(mlir::Type type);

  TensorInitFloat(DataType type) : type(type) {}
  virtual ~TensorInitFloat() = default;

protected:
  // FP32 conversion (by reference).
  static void toFP32(llvm::APFloat &value) {
    bool ignored;
    value.convert(llvm::APFloat::IEEEsingle(),
                  llvm::APFloat::rmNearestTiesToEven, &ignored);
  }

  // FP64 conversion (by reference).
  static void toFP64(llvm::APFloat &value) {
    bool ignored;
    value.convert(llvm::APFloat::IEEEdouble(),
                  llvm::APFloat::rmNearestTiesToEven, &ignored);
  }

  // BF16 conversion (by reference).
  static void toBF16(llvm::APFloat &value) {
    bool ignored;
    value.convert(llvm::APFloat::BFloat(), llvm::APFloat::rmNearestTiesToEven,
                  &ignored);
  }

  // Tensor element data type.
  DataType type;

  // Insert element indexed on the buffer.
  using TensorInit::insert;
  virtual void insert(size_t index, float value);

  // Insert element at the end of the buffer.
  using TensorInit::push;
  virtual void push(float value);

  // Convert value to the tensor's data type (by reference).
  void convertType(llvm::APFloat &value) override final;

  // Actual implementation that fills the buffer
  // To be implemented by derived classes.
  virtual void fillData() override = 0;
};

// Constant init (all-ones, do not use!).
struct ConstantTensorInitFloat : TensorInitFloat {
  ConstantTensorInitFloat(DataType type) : TensorInitFloat(type) {}

  // Return a dense<1.0> repeated throughout the shape.
  mlir::DenseElementsAttr get(mlir::ShapedType shape) override;

  void fillData() override;
};

// Simple init (basic example, not useful).
struct SimpleTensorInitFloat : TensorInitFloat {
  SimpleTensorInitFloat(DataType type) : TensorInitFloat(type) {}

  // Return a dense<0.3, 0.6, 0.9> repeated throughout the shape.
  void fillData() override;
};

// Continuous init (normalized affine range).
struct ContinuousTensorInitFloat : TensorInitFloat {
  ContinuousTensorInitFloat(DataType type) : TensorInitFloat(type) {}

  // Return a dense<0.0 ... 1.0> throughout the shape.
  void fillData() override;
};

// Random init (uniform).
struct RandomTensorInitFloat : TensorInitFloat {
  RandomTensorInitFloat(DataType type, int seed)
      : TensorInitFloat(type), generator(seed), distribution(0.0, 1.0) {}

  // Next random uniform number.
  float next() { return distribution(generator); }

  // Return a dense<uniform(0.0, 1.0)> throughout the shape.
  void fillData() override;

private:
  // Random generator.
  std::default_random_engine generator;
  // Random distribution.
  std::uniform_real_distribution<float> distribution;
};

// Random init (normal).
struct NormalTensorInitFloat : TensorInitFloat {
  NormalTensorInitFloat(DataType type, int seed)
      : TensorInitFloat(type), generator(seed), distribution(0.0, 0.2) {}

  // Next random number.
  float next() {
    auto value = distribution(generator);
    value = std::clamp(value, 0.0f, 1.0f);
    return value;
  }

  // Return a dense<normal(0.0, 1.0)> throughout the shape.
  void fillData() override;

private:
  // Random generator.
  std::default_random_engine generator;
  // Random distribution.
  std::normal_distribution<float> distribution;
};

#endif // TPP_TENSORINITFLOAT_H
