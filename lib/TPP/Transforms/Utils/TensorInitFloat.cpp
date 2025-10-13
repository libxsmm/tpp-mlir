//===- TensorInitFloat.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/TensorInitFloat.h"

using namespace mlir;

TensorInitFloat::DataType
TensorInitFloat::getTensorInitDataType(mlir::Type type) {
  if (type.isBF16())
    return DataType::BF16;
  if (type.isF16())
    return DataType::FP16;
  if (type.isF32())
    return DataType::FP32;
  if (type.isF64())
    return DataType::FP64;
  return DataType::AUTO;
}

void TensorInitFloat::insert(size_t index, float value) {
  this->TensorInit::insert(index, APFloat(value));
}

void TensorInitFloat::push(float value) {
  this->TensorInit::push(APFloat(value));
}

void TensorInitFloat::convertType(llvm::APFloat &value) {
  switch (type) {
  case DataType::FP16:
    toFP16(value);
    break;
  case DataType::FP32:
    toFP32(value);
    break;
  case DataType::FP64:
    toFP64(value);
    break;
  case DataType::BF16:
    toBF16(value);
    break;
  case DataType::AUTO:
    toFP32(value);
    break;
  }
}

FailureOr<DenseElementsAttr> ConstantTensorInitFloat::get(ShapedType shape) {
  auto floatValue = APFloat(1.0F);
  if (!isTypeSupported(shape.getElementType()))
    assert(false && "Element type not supported");
  convertType(floatValue);

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(shape.getShape(), shape.getElementType());
  return mlir::DenseElementsAttr::get(tensorType, floatValue);
}

void RandomTensorInitFloat::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

void NormalTensorInitFloat::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

void IdentityTensorInitFloat::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  APFloat zero = APFloat(0.0);
  convertType(zero);
  buffer.resize(size, zero);
  size_t ld = dims[0];

  // Shape is guaranteed to be "2D square" by `checkShape()`
  for (size_t i=0; i < ld; i++) {
    size_t offset = i*ld + i;
    insert(offset, APFloat(1.0));
  }
}
