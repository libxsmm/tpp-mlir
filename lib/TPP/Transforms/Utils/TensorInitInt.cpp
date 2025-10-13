//===- TensorInitInt.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/TensorInitInt.h"

using namespace mlir;

TensorInitInt::DataType TensorInitInt::getTensorInitDataType(mlir::Type type) {
  if (type.isSignlessInteger(8))
    return DataType::I8;
  if (type.isSignlessInteger(16))
    return DataType::I16;
  if (type.isSignlessInteger(32))
    return DataType::I32;
  if (type.isSignlessInteger(64))
    return DataType::I64;
  return DataType::AUTO;
}

unsigned TensorInitInt::getDataTypeBitWidth(TensorInitInt::DataType type) {
  switch (type) {
  case DataType::I8:
    return 8;
  case DataType::I16:
    return 16;
  case DataType::I32:
    return 32;
  case DataType::I64:
    return 64;
  case DataType::AUTO:
    return 32;
  }
  llvm_unreachable("unknown type");
}

bool TensorInitInt::isDataTypeSigned(TensorInitInt::DataType type) {
  switch (type) {
  case DataType::I8:
  case DataType::I16:
  case DataType::I32:
  case DataType::I64:
  case DataType::AUTO:
    return true;
  }
  llvm_unreachable("unknown type");
}

void TensorInitInt::insert(size_t index, uint64_t value) {
  this->TensorInit::insert(index, APInt(bitWidth, value, isSigned));
}

void TensorInitInt::push(uint64_t value) {
  this->TensorInit::push(APInt(bitWidth, value, isSigned));
}

void TensorInitInt::convertType(llvm::APInt &value) {
  assert(value.getBitWidth() == bitWidth && "Invalid element size");
}

FailureOr<DenseElementsAttr> ConstantTensorInitInt::get(ShapedType shape) {
  auto value = APInt(bitWidth, 1, isSigned);
  if (!isTypeSupported(shape.getElementType()))
    assert(false && "Element type not supported");
  convertType(value);

  // For some reason, memref global op needs dense tensor type
  // See: lib/Dialect/MemRef/IR/MemRefOps.cpp :: GlobalOp::verify
  auto tensorType =
      RankedTensorType::get(shape.getShape(), shape.getElementType());
  return mlir::DenseElementsAttr::get(tensorType, value);
}

void RandomTensorInitInt::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

void NormalTensorInitInt::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  for (size_t i = 0; i < size; i++)
    push(next());
}

void IdentityTensorInitInt::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");
  APInt zero = APInt(bitWidth, 0, isSigned);
  convertType(zero);
  buffer.resize(size, zero);
  size_t ld = dims[0];

  // Shape is guaranteed to be "2D square" by `checkShape()`
  for (size_t i=0; i < ld; i++) {
    size_t offset = i*ld + i;
    insert(offset, APInt(bitWidth, 1, isSigned));
  }
}
