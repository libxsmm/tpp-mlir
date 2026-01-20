//===- XsmmOps.cpp - Xsmm dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "TPP/Dialect/Xsmm/XsmmOps.cpp.inc"

using namespace mlir::xsmm;

namespace {
constexpr std::string_view INPUTS = "inputs";
constexpr std::string_view DATA_TYPE = "data_type";
constexpr std::string_view FLAGS_NAME = "flags";
constexpr std::string_view KIND = "kind";
constexpr std::string_view UNARY_FLAGS_NAME = "unary_flags";
constexpr std::string_view BINARY_FLAGS_NAME = "binary_flags";
constexpr std::string_view BINARY_KIND = "binary_kind";
constexpr std::string_view UNARY_KIND = "unary_kind";
} // namespace

template <typename EnumClass>
static ParseResult parseEnum(EnumClass &value, OpAsmParser &parser) {
  StringRef flag;
  auto loc = parser.getCurrentLocation();
  if (parser.parseKeyword(&flag))
    return failure();
  auto flagAttr = symbolizeEnum<EnumClass>(flag);
  if (!flagAttr)
    return parser.emitError(loc, "invalid enum ") << flag;
  value = *flagAttr;
  return success();
}

static ParseResult parseInputImpl(OpAsmParser &parser, OperationState &result) {
  DenseI64ArrayAttr kindAttr;
  if (parser.parseCustomAttributeWithFallback(kindAttr, Type{}, INPUTS,
                                              result.attributes)) {
    return failure();
  }
  return success();
}

static ParseResult parseDataTypeImpl(OpAsmParser &parser,
                                     OperationState &result) {
  auto &builder = parser.getBuilder();
  if (parser.parseKeyword(DATA_TYPE) || parser.parseEqual())
    return failure();
  DataType dataType;
  if (parseEnum(dataType, parser))
    return failure();
  result.addAttribute(DATA_TYPE,
                      DataTypeAttr::get(builder.getContext(), dataType));
  result.addTypes(builder.getIntegerType(64));

  // Parse the optional attribute list
  return parser.parseOptionalAttrDict(result.attributes);
}

template <typename FLAGS>
static ParseResult parserFlagsImpl(OpAsmParser &parser, OperationState &result,
                                   const std::string_view &flagsName) {
  auto &builder = parser.getBuilder();
  if (parser.parseKeyword(flagsName) || parser.parseEqual() ||
      parser.parseLParen())
    return failure();

  SmallVector<Attribute, 4> flags;
  auto parseFlags = [&]() -> ParseResult {
    FLAGS flag;
    if (parseEnum<FLAGS>(flag, parser))
      return failure();
    flags.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(flag)));
    return success();
  };
  if (parser.parseCommaSeparatedList(parseFlags) || parser.parseRParen())
    return failure();
  result.addAttribute(flagsName, builder.getArrayAttr(flags));
  return success();
}

ParseResult GemmDispatchOp::parse(OpAsmParser &parser, OperationState &result) {
  if (failed(parseInputImpl(parser, result)))
    return failure();
  if (failed(parserFlagsImpl<GemmFlags>(parser, result, FLAGS_NAME)))
    return failure();
  return parseDataTypeImpl(parser, result);
}

ParseResult BrgemmDispatchOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  if (failed(parseInputImpl(parser, result)) ||
      failed(parserFlagsImpl<GemmFlags>(parser, result, FLAGS_NAME)))
    return failure();
  return parseDataTypeImpl(parser, result);
}

ParseResult FusedBrgemmDispatchOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  // Parse inputs.
  if (failed(parseInputImpl(parser, result)))
    return failure();
  // Parse the unary and binary kind.
  BinaryKind binaryKind;
  UnaryKind unaryKind;
  if (parser.parseLSquare() || parseEnum(binaryKind, parser) ||
      parser.parseComma() || parseEnum(unaryKind, parser) ||
      parser.parseRSquare()) {
    return failure();
  }
  auto *ctx = parser.getBuilder().getContext();
  result.addAttribute(BINARY_KIND, BinaryKindAttr::get(ctx, binaryKind));
  result.addAttribute(UNARY_KIND, UnaryKindAttr::get(ctx, unaryKind));
  // Parse different flags (gemm, binary and unary).
  if (failed(parserFlagsImpl<GemmFlags>(parser, result, FLAGS_NAME)) ||
      failed(parserFlagsImpl<BinaryFlags>(parser, result, BINARY_FLAGS_NAME)) ||
      failed(parserFlagsImpl<UnaryFlags>(parser, result, UNARY_FLAGS_NAME))) {
    return failure();
  }
  // Parse data type.
  return parseDataTypeImpl(parser, result);
}

ParseResult UnaryDispatchOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  // Parse the type of unary
  UnaryKind kind;
  if (parseEnum(kind, parser))
    return failure();
  result.addAttribute(
      KIND, UnaryKindAttr::get(parser.getBuilder().getContext(), kind));
  if (failed(parseInputImpl(parser, result)) ||
      failed(parserFlagsImpl<UnaryFlags>(parser, result, FLAGS_NAME)))
    return failure();
  return parseDataTypeImpl(parser, result);
}

ParseResult BinaryDispatchOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  // Parse the type of binary
  BinaryKind kind;
  if (parseEnum(kind, parser))
    return failure();
  result.addAttribute(
      KIND, BinaryKindAttr::get(parser.getBuilder().getContext(), kind));
  if (failed(parseInputImpl(parser, result)) ||
      failed(parserFlagsImpl<BinaryFlags>(parser, result, FLAGS_NAME)))
    return failure();
  return parseDataTypeImpl(parser, result);
}

template <typename OpTy>
static void printerInputImpl(OpAsmPrinter &printer, OpTy op) {
  printer << " [" << op.getInputs() << ']';
};

template <typename OpTy>
static void printerDataTypeImpl(OpAsmPrinter &printer, OpTy op) {
  printer << DATA_TYPE << " = ";
  auto dataType = op.getDataType();
  printer << xsmm::stringifyDataType(dataType);
  printer.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{DATA_TYPE, FLAGS_NAME, INPUTS, KIND, FLAGS_NAME,
                       UNARY_FLAGS_NAME, BINARY_FLAGS_NAME, BINARY_KIND,
                       UNARY_KIND});
}

template <typename AttrTy>
static void printerFlagsImpl(OpAsmPrinter &printer,
                             const std::function<ArrayAttr()> &fn,
                             const std::string_view &flagsName) {
  printer << " " << flagsName << " = (";
  llvm::interleaveComma(fn(), printer, [&](auto &flag) {
    printer << stringifyEnum(cast<AttrTy>(flag).getValue());
  });
  printer << ") ";
}

void GemmDispatchOp::print(OpAsmPrinter &printer) {
  printerInputImpl<GemmDispatchOp>(printer, *this);
  auto getOpFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printerFlagsImpl<GemmFlagsAttr>(printer, getOpFlags, FLAGS_NAME);
  printerDataTypeImpl<GemmDispatchOp>(printer, *this);
}

void BrgemmDispatchOp::print(OpAsmPrinter &printer) {
  printerInputImpl<BrgemmDispatchOp>(printer, *this);
  auto getOpFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printerFlagsImpl<GemmFlagsAttr>(printer, getOpFlags, FLAGS_NAME);
  printerDataTypeImpl<BrgemmDispatchOp>(printer, *this);
}

void FusedBrgemmDispatchOp::print(OpAsmPrinter &printer) {
  printerInputImpl<FusedBrgemmDispatchOp>(printer, *this);
  printer << "[" << getBinaryKind() << "," << getUnaryKind() << "] ";
  auto getOpGemmFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printerFlagsImpl<GemmFlagsAttr>(printer, getOpGemmFlags, FLAGS_NAME);
  auto getOpBinaryFlags = [this]() -> ArrayAttr {
    return this->getBinaryFlags();
  };
  printerFlagsImpl<BinaryFlagsAttr>(printer, getOpBinaryFlags,
                                    BINARY_FLAGS_NAME);
  auto getOpUnaryFlags = [this]() -> ArrayAttr {
    return this->getUnaryFlags();
  };
  printerFlagsImpl<UnaryFlagsAttr>(printer, getOpUnaryFlags, UNARY_FLAGS_NAME);
  printerDataTypeImpl<FusedBrgemmDispatchOp>(printer, *this);
}

void UnaryDispatchOp::print(OpAsmPrinter &printer) {
  printer << " " << getKind();
  printerInputImpl<UnaryDispatchOp>(printer, *this);
  auto getOpFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printerFlagsImpl<UnaryFlagsAttr>(printer, getOpFlags, FLAGS_NAME);
  printerDataTypeImpl<UnaryDispatchOp>(printer, *this);
}

void BinaryDispatchOp::print(OpAsmPrinter &printer) {
  printer << " " << getKind();
  printerInputImpl<BinaryDispatchOp>(printer, *this);
  auto getOpFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printerFlagsImpl<BinaryFlagsAttr>(printer, getOpFlags, FLAGS_NAME);
  printerDataTypeImpl<BinaryDispatchOp>(printer, *this);
}

void IntelAMXTileConfigDispatchOp::print(OpAsmPrinter &printer) {
  printerInputImpl<IntelAMXTileConfigDispatchOp>(printer, *this);
  auto getOpFlags = [this]() -> ArrayAttr { return this->getFlags(); };
  printerFlagsImpl<GemmFlagsAttr>(printer, getOpFlags, FLAGS_NAME);
  printerDataTypeImpl<IntelAMXTileConfigDispatchOp>(printer, *this);
}

ParseResult IntelAMXTileConfigDispatchOp::parse(OpAsmParser &parser,
                                                OperationState &result) {
  if (failed(parseInputImpl(parser, result)) ||
      failed(parserFlagsImpl<GemmFlags>(parser, result, FLAGS_NAME)))
    return failure();
  return parseDataTypeImpl(parser, result);
}

template <typename FLAGS>
static LogicalResult
verifyUniquenessAndConsistency(ArrayAttr flags, Operation *op,
                               const std::string_view &flagsName) {
  SmallVector<int64_t> flagsAsInt;
  for (auto flag : flags)
    flagsAsInt.push_back(cast<IntegerAttr>(flag).getInt());

  // check uniqueness
  std::sort(flagsAsInt.begin(), flagsAsInt.end());
  auto *it = std::unique(flagsAsInt.begin(), flagsAsInt.end());
  if (it != flagsAsInt.end())
    return op->emitOpError() << "expected " << flagsName << " to be unique";
  // none flag conflicts with all the others
  if (llvm::is_contained(flagsAsInt, static_cast<int64_t>(FLAGS::NONE)) &&
      flagsAsInt.size() != 1) {
    return op->emitOpError()
           << "'none' " << flagsName << " conflicts with others";
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyGemmFlags(ArrayAttr flags, DataType dataType,
                                     OpTy op,
                                     const std::string_view &flagsName) {
  static_assert(llvm::is_one_of<OpTy, xsmm::BrgemmDispatchOp, GemmDispatchOp,
                                xsmm::FusedBrgemmDispatchOp>::value,
                "applies to xsmm gemms dispatch operations only");

  // Verify flags.
  if (failed(verifyUniquenessAndConsistency<GemmFlags>(flags, op, flagsName)))
    return failure();

  SmallVector<int64_t> flagsAsInt;
  for (auto flag : flags) {
    flagsAsInt.push_back(cast<IntegerAttr>(flag).getInt());
  }
  // VNNI flags must be specified only for bf16 type
  if (dataType != DataType::BF16 && llvm::any_of(flagsAsInt, [](int64_t flag) {
        return (flag == static_cast<int64_t>(GemmFlags::VNNI_B) ||
                flag == static_cast<int64_t>(GemmFlags::VNNI_A) ||
                flag == static_cast<int64_t>(GemmFlags::VNNI_C));
      })) {
    return op->emitOpError() << "VNNI flags but type is not bf16";
  }

  return success();
}

template <typename OpTy>
static LogicalResult verifyDispatchInputs(OpTy op, size_t expected) {
  static_assert(llvm::is_one_of<OpTy, xsmm::UnaryDispatchOp,
                                xsmm::BinaryDispatchOp, GemmDispatchOp,
                                BrgemmDispatchOp, FusedBrgemmDispatchOp>::value,
                "applies to xsmm dispatch operations only");

  // `inputs` are leading dimensions and sizes
  size_t numInputs = op.getInputs().size();
  if (numInputs != expected) {
    return op.emitOpError()
           << "expect " << expected << " args but got: " << numInputs;
  }
  return success();
}

template <typename OpTy> static LogicalResult verifyGemmLikeOp(OpTy op) {
  // 'inputs' = [m, n, k, lda, ldb, ldc] for GEMM.
  // 'inputs' = [m, n, k, lda, ldb, ldc, stride_a, stride_b] for BRGEMM.
  bool isBrgemm = isa<BrgemmDispatchOp>(op.getOperation()) ||
                  isa<FusedBrgemmDispatchOp>(op.getOperation());
  size_t expected = (isBrgemm) ? 8 : 6;
  if (failed(verifyDispatchInputs(op, expected)))
    return failure();

  // Verify leading dims.
  ArrayRef<int64_t> inputs = op.getInputs();
  int64_t n = inputs[1];
  int64_t k = inputs[2];
  int64_t lda = inputs[3];
  int64_t ldb = inputs[4];
  int64_t ldc = inputs[5];
  if (lda < k)
    return op.emitOpError() << "expect lda to be >= of dimension k\n";
  if (ldb < n)
    return op.emitOpError() << "expect ldb to be >= of dimension n\n";
  if (ldc < n)
    return op.emitOpError() << "expect ldc to be >= of dimension n\n";

  // Verify dispatch flags.
  return verifyGemmFlags(op.getFlags(), op.getDataType(), op, FLAGS_NAME);
}

LogicalResult GemmDispatchOp::verify() {
  return verifyGemmLikeOp<GemmDispatchOp>(*this);
}

LogicalResult BrgemmDispatchOp::verify() {
  return verifyGemmLikeOp<BrgemmDispatchOp>(*this);
}

LogicalResult UnaryDispatchOp::verify() {
  if (failed(verifyUniquenessAndConsistency<UnaryFlags>(
          getFlags(), getOperation(), FLAGS_NAME))) {
    return failure();
  }
  // 'inputs' = [m, n, lda, ldo]
  return verifyDispatchInputs(*this, /*expected=*/4);
}

LogicalResult BinaryDispatchOp::verify() {
  if (failed(verifyUniquenessAndConsistency<UnaryFlags>(
          getFlags(), getOperation(), FLAGS_NAME))) {
    return failure();
  }
  // 'inputs' = [m, n, lda, ldb, ldo]
  return verifyDispatchInputs(*this, /*expected=*/5);
}

LogicalResult FusedBrgemmDispatchOp::verify() {
  if (failed(verifyUniquenessAndConsistency<BinaryFlags>(
          getBinaryFlags(), getOperation(), BINARY_FLAGS_NAME)) ||
      failed(verifyUniquenessAndConsistency<UnaryFlags>(
          getUnaryFlags(), getOperation(), UNARY_FLAGS_NAME))) {
    return failure();
  }

  if (failed(verifyGemmLikeOp<FusedBrgemmDispatchOp>(*this)))
    return failure();

  // Verify the flags are consistent with the type of unary or binary specified.
  auto unaryKind = getUnaryKind();
  if (unaryKind == xsmm::UnaryKind::NONE) {
    auto unaryFlags = getUnaryFlags();
    if (unaryFlags.size() != 1 ||
        cast<xsmm::UnaryFlagsAttr>(unaryFlags[0]).getValue() !=
            xsmm::UnaryFlags::NONE) {
      return emitOpError() << "invalid unary flags for kind none";
    }
  }
  auto binaryKind = getBinaryKind();
  if (binaryKind == xsmm::BinaryKind::NONE) {
    auto binaryFlags = getBinaryFlags();
    if (binaryFlags.size() != 1 ||
        cast<xsmm::BinaryFlagsAttr>(binaryFlags[0]).getValue() !=
            xsmm::BinaryFlags::NONE) {
      return emitOpError() << "invalid binary flags for kind none";
    }
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyXsmmCommon(OpTy invokeOp,
                                      const size_t expectedInputs) {
  SmallVector<Value> inputs = invokeOp.getInputs();

  if (inputs.size() != expectedInputs) {
    return invokeOp.emitOpError() << "expect " << expectedInputs
                                  << " inputs but got " << inputs.size();
  }

  Value dispatch = invokeOp.getDispatch();
  if (!dispatch.getType().isInteger(64)) {
    return invokeOp.emitOpError()
           << "expect an i64 but got " << dispatch.getType()
           << " for operand 0 (dispatch)";
  }

  auto isCompatible = [](xsmm::DataType dataType, Type type) {
    if (dataType == xsmm::DataType::F32)
      return type.isF32();
    return type.isBF16();
  };

  // Skip dispatch at index 0. In case of a brgemm operation
  // skip the last operand (batch).
  size_t upTo = inputs.size();
  if (llvm::is_one_of<OpTy, xsmm::BrgemmOp, xsmm::FusedBrgemmOp>::value)
    upTo--;

  for (size_t idx = 1; idx < upTo; idx++) {
    Type elementType = getElementTypeOrSelf(inputs[idx].getType());
    if (!isCompatible(invokeOp.getDataType(), elementType)) {
      return invokeOp.emitOpError()
             << "expect " << xsmm::stringifyDataType(invokeOp.getDataType())
             << " but got: " << elementType << " for operand at index: " << idx;
    }
  }
  return success();
}

LogicalResult GemmOp::verify() {
  if (failed(verifyXsmmCommon(*this, /*expectedInputs=*/4)))
    return failure();

  // Verify the rank of the shaped operands.
  SmallVector<Value> memrefOperands = {getOperandA(), getOperandB(),
                                       getOutput()};

  for (size_t idx = 0; idx < memrefOperands.size(); idx++) {
    size_t actualIdx = idx + 1 /*skip dispatch*/;
    auto memref = dyn_cast<MemRefType>(memrefOperands[idx].getType());
    assert(memref && (memref.getRank() == 2 || memref.getRank() == 3));

    if (memref.getRank() == 3) {
      if (memref.getShape().back() % 2 != 0 ||
          !vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::GEMM,
                                       memref)) {
        return emitOpError() << "expect VNNI layout for operand: " << actualIdx;
      }
    }
  }
  return success();
}

template <typename OpTy>
static LogicalResult verifyBrgemmLikeOpCommon(OpTy brgemmOp,
                                              const size_t expectedInputs) {
  static_assert(
      llvm::is_one_of<OpTy, xsmm::BrgemmOp, xsmm::FusedBrgemmOp>::value);

  if (failed(verifyXsmmCommon(brgemmOp, expectedInputs)))
    return failure();

  // Verify the rank of the shaped operands.
  SmallVector<Value> memrefOperands = {
      brgemmOp.getOperandA(), brgemmOp.getOperandB(), brgemmOp.getOutput()};

  for (size_t idx = 0; idx < memrefOperands.size(); idx++) {
    size_t actualIdx = idx + 1 /*skip dispatch*/;
    auto memref = dyn_cast<MemRefType>(memrefOperands[idx].getType());
    // Output memref. Must be of rank 2 or in VNNI layout with rank 3.
    if (idx == 2 && (memref.getRank() != 2 &&
                     (memref.getRank() == 3 &&
                      !vnni::utils::isInVnniLayout(
                          vnni::utils::VnniOperandRank::BRGEMM_INS, memref)))) {
      return brgemmOp.emitOpError()
             << "expect a 2d or 3d VNNI layout for operand: " << actualIdx;
    }
    // Input memref. Must be of rank 3 or in VNNI layout with rank 4.
    if (idx != 2 &&
        (memref.getRank() != 3 &&
         (memref.getRank() != 4 &&
          !vnni::utils::isInVnniLayout(
              vnni::utils::VnniOperandRank::BRGEMM_OUTS, memref)))) {
      return brgemmOp.emitOpError()
             << "expect a 3d or 4d VNNI memref for operand: " << actualIdx;
    }
  }
  // Verify the batch to be an i64.
  Value batch = brgemmOp.getBatch();
  if (!batch.getType().isInteger(64)) {
    return brgemmOp.emitOpError() << "expect an i64 but got " << batch.getType()
                                  << " for last operand (batch)";
  }
  return success();
}

LogicalResult BrgemmOp::verify() {
  return verifyBrgemmLikeOpCommon(*this, /*expectedInputs=*/5);
}

LogicalResult FusedBrgemmOp::verify() {
  return verifyBrgemmLikeOpCommon(*this, /*expectedInputs=*/6);
}

LogicalResult UnaryOp::verify() {
  return verifyXsmmCommon(*this, /*expectedInputs=*/3);
}

LogicalResult BinaryOp::verify() {
  return verifyXsmmCommon(*this, /*expectedInputs=*/4);
}
