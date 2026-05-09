//===- TensorInitInt.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/TensorInitInt.h"
#include "TPP/Transforms/Utils/TensorInitFloat.h"

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

namespace {
// Compute rows and columns for the collapsed 2-D matrix from the original
// tensor dimensions.
static void getUnpackedShapeFromDims(const std::vector<float> &samples,
                                     const std::vector<size_t> &dims,
                                     bool isRowWiseReduce, size_t &rows,
                                     size_t &columns) {
  // Compute full size of the matrix by collapsing all dimensions
  size_t fullSize = 1;
  for (size_t i = 0; i < dims.size(); i++)
    fullSize *= dims[i];
  assert(fullSize == samples.size() && "Mismatch in samples size");

  // Initialize rows computing the product of collapsed outer dimension sizes
  // chosen based on reduction dimension.
  if (dims.size() > 2)
    rows = isRowWiseReduce ? dims[0] * dims[2] : dims[1] * dims[2] * dims[4];
  else if (dims.size() == 2)
    rows = dims[0];

  // Initialize column size by factoring out the row size.
  columns = fullSize / rows;
}
} // namespace

// Compute a simple dynamic fixed point per channel(row or column) quantization
// scale for the 2-D flattened tensor data or packed data.Find the max absolute
// value in the distribution channel wise and use it to determine the unbiased
// exponent using frexp().
std::vector<int>
QuantTensorInitInt::computeScales(const std::vector<float> &samples,
                                  bool isRowWiseReduce) {
  std::vector<int> channelwiseScales;
  assert(dims.size() >= 2 && dims.size() <= 5 &&
         "Only 2D for unpacked and <=5D for packed tensor dims are supported");

  size_t rows = 1, columns = 1;
  getUnpackedShapeFromDims(samples, dims, isRowWiseReduce, rows, columns);
  size_t reductionDimSize = isRowWiseReduce ? rows : columns;
  size_t nonReductionDimSize = isRowWiseReduce ? columns : rows;
  channelwiseScales.resize(isRowWiseReduce ? rows : columns, 0);

  for (size_t c = 0; c < reductionDimSize; c++) {
    // Initialize with minimum absolute value and compute max abs value.
    float maxAbsValue = 0.0f;
    for (size_t r = 0; r < nonReductionDimSize; r++) {
      size_t index = isRowWiseReduce ? (c * nonReductionDimSize + r)
                                     : (r * reductionDimSize + c);
      float value = samples[index];
      float absValue = std::abs(value);
      if (absValue > maxAbsValue)
        maxAbsValue = absValue;
    }
    int exponent;
    if (maxAbsValue == 0.0f) {
      exponent = 0; // Handle zero case
    } else {
      std::frexp(maxAbsValue, &exponent); // Get unbiased exponent
    }
    exponent = exponent - 7;

    // Compute the scale as 2^(-exponent)
    channelwiseScales[c] = std::exp2(-(exponent));
  }
  return channelwiseScales;
}

// Quantize the float samples into dynamic fixed point representation
// using the provided channel-wise scales.
std::vector<APInt>
QuantTensorInitInt::quantizeDFP(const std::vector<float> &samples,
                                const std::vector<int> &channelwiseScales,
                                bool isRowWiseReduce) {
  std::vector<APInt> quantizedValues;
  assert(dims.size() >= 2 && dims.size() <= 5 &&
         "Only 2D for unpacked and <=5D for packed tensor dims are supported");

  size_t rows = 1, columns = 1;
  getUnpackedShapeFromDims(samples, dims, isRowWiseReduce, rows, columns);
  size_t reductionDimSize = isRowWiseReduce ? rows : columns;
  size_t nonReductionDimSize = isRowWiseReduce ? columns : rows;
  quantizedValues.resize(samples.size(), APInt(bitWidth, 0, isSigned));

  for (size_t c = 0; c < reductionDimSize; c++) {
    float scale = static_cast<float>(channelwiseScales[c]);
    for (size_t r = 0; r < nonReductionDimSize; r++) {
      size_t index = isRowWiseReduce ? (c * nonReductionDimSize + r)
                                     : (r * reductionDimSize + c);
      float value = samples[index];
      int quantized = static_cast<int>(std::round(value * scale));

      // Clamp to the representable range
      int64_t minVal = isSigned ? -(1LL << (bitWidth - 1)) : 0;
      int64_t maxVal =
          isSigned ? ((1LL << (bitWidth - 1)) - 1) : ((1LL << bitWidth) - 1);
      if (quantized < minVal)
        quantized = minVal;
      if (quantized > maxVal)
        quantized = maxVal;
      quantizedValues[index] = APInt(bitWidth, quantized, isSigned);
    }
  }
  return quantizedValues;
}

void QuantTensorInitInt::fillData() {
  assert(buffer.size() == 0 && "Buffer not empty");

  // Random float samples to be quantized.
  std::vector<float> samples;
  for (size_t i = 0; i < size; i++) {
    auto p = distribution(generator);
    samples.push_back(p);
  }

  std::vector<int> channelwiseScales = computeScales(samples, isInputMatrix);

  // Quantize each sample data using the channel-wise scale.
  std::vector<APInt> quantizedValues =
      quantizeDFP(samples, channelwiseScales, isInputMatrix);

  // Update the internal buffer with quantized values.
  buffer = quantizedValues;

  // Update the corresponding float/f8 dequant scale factors into member
  // variable of class object.
  if (floatScaleInit) {
    std::vector<llvm::APFloat> floatDequantScales;
    for (size_t i = 0; i < channelwiseScales.size(); i++) {
      float scale = 1.0f / static_cast<float>(channelwiseScales[i]);
      if (scaleDataType.isFloat(32))
        floatDequantScales.emplace_back(scale);
      else if (scaleDataType.isFloat(8)) {
        uint32_t shiftedValue;
        std::memcpy(&shiftedValue, &scale, sizeof(uint32_t));
        shiftedValue = shiftedValue >> 23;
        llvm::APInt intScale(bitWidth, shiftedValue, false);
        floatDequantScales.emplace_back(
            llvm::APFloat(llvm::APFloat::Float8E8M0FNU(), intScale));
      } else {
        llvm_unreachable("Unsupported scale data type");
      }
    }
    static_cast<QuantScaleTensorInitFloat *>(floatScaleInit.get())
        ->setScaleBuffer(floatDequantScales);
  }

  // Update the matrix type to indicate next argument would be weight matrix.
  isInputMatrix = false;
}
