//===- XsmmUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"

#include "TPP/Transforms/Utils/BuilderUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "xsmm-utils"

namespace mlir {
namespace xsmm {
namespace utils {

// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
static void
computeBcastShapeInput(ArrayRef<int64_t> higherRankShape,
                       ArrayRef<int64_t> lowerRankShape,
                       SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim)) {
      reshapeOutputShape[i] = lowerRankDim;
    } else if (higherRankDim != lowerRankDim) {
      llvm_unreachable("bCast semantics for identity op broken");
    }
  }
}

DataTypeAttr getDataType(RewriterBase &rewriter, Type type) {
  auto elemType = getElementTypeOrSelf(type);
  if (elemType.isBF16())
    return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16);
  return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
}

void replaceOpWithUnary(RewriterBase &rewriter, Operation *operation,
                        ArrayRef<Value> operands, UnaryInfo unaryInfo,
                        ArrayAttr flags, xsmm::UnaryKindAttr kind) {
  Location loc = operation->getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(), ArrayRef<int64_t>{unaryInfo.m, unaryInfo.n,
                                               unaryInfo.ldi, unaryInfo.ldo});
  auto dtype = xsmm::utils::getDataType(rewriter, operands.back().getType());
  Value dispatched = xsmm::UnaryDispatchOp::create(rewriter, 
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(operands.begin(), operands.end());
  rewriter.replaceOpWithNewOp<xsmm::UnaryOp>(operation, dtype, kind,
                                             invokeOperands);
}

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  UnaryFlags inputFlag) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  UnaryInfo unaryInfo;
  unaryInfo.m = outputShapedType.getShape()[0];
  unaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldi = 1;
  if (ShapedType inputShapedType = dyn_cast<ShapedType>(input.getType())) {
    auto stridesOnInput = mlir::utils::getStaticStrides(input);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1 ||
        !inputShapedType.hasStaticShape()) {
      return failure();
    }

    // If we are broascasting a row into cols, the leading
    // dimension is 1, same for scalar broadcast.
    if (inputFlag == UnaryFlags::BCAST_ROW ||
        inputFlag == UnaryFlags::BCAST_SCALAR) {
      ldi = 1;
    }
    // If we are broascasting a col into rows, the leading
    // dimension is the size of the tensor.
    else if (inputFlag == UnaryFlags::BCAST_COL) {
      ldi = inputShapedType.getShape().back();
    } else {
      ldi = stridesOnInput->front();
    }
  }
  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();

  unaryInfo.ldi = ldi;
  unaryInfo.ldo = stridesOnOutput->front();
  return unaryInfo;
}

FailureOr<UnaryInfo>
getVectorUnaryInfo(MemRefType shapedType, MemRefType inputType,
                   MemRefType outputType, VectorType inputVectorType,
                   VectorType outputVectorType, UnaryFlags inputFlag) {

  UnaryInfo unaryInfo;

  unaryInfo.m = 1;
  for (unsigned i = 0; i < shapedType.getShape().size() - 1; i++) {
    unaryInfo.m *= shapedType.getShape()[i];
  }
  unaryInfo.n = shapedType.getShape()[shapedType.getShape().size() - 1];

  auto getStrideLoc = [&](MemRefType memrefType) -> FailureOr<int64_t> {
    int64_t strideAtLoc;
    SmallVector<int64_t> strides;
    int64_t offset;

    if (failed(memrefType.getStridesAndOffset(strides, offset))) {
      return failure();
    }
    if (strides.empty()) {
      return failure();
    }
    strideAtLoc = strides[0];
    return strideAtLoc;
  };

  auto strideLdi = getStrideLoc(inputType);
  if (failed(strideLdi)) {
    return failure();
  }
  unaryInfo.ldi = *strideLdi;
  int ldo = 1;
  // If we are broascasting a row into cols, the leading
  // dimension is 1, same for scalar broadcast.
  if (inputFlag == UnaryFlags::BCAST_ROW ||
      inputFlag == UnaryFlags::BCAST_SCALAR)
    ldo = 1;
  // If we are broascasting a col into rows, the leading
  // dimension is the size of the tensor.
  else if (inputFlag == UnaryFlags::BCAST_COL) {
    if (inputVectorType.getShape().size() == 0)
      ldo = 1;
    else
      ldo = inputVectorType.getShape()[0];
  } else {
    auto strideLdo = getStrideLoc(outputType);
    if (failed(strideLdo)) {
      return failure();
    }
    ldo = *strideLdo;
  }
  unaryInfo.ldo = ldo;
  return unaryInfo;
}

FailureOr<BinaryInfo> getBinaryInfo(Value lhs, BinaryFlags lhsFlag, Value rhs,
                                    BinaryFlags rhsFlag, Value output) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  BinaryInfo binaryInfo;
  binaryInfo.m = outputShapedType.getShape()[0];
  binaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldiLhs = 1;
  if (ShapedType lhsShapedType = dyn_cast<ShapedType>(lhs.getType())) {
    auto stridesOnLhs = mlir::utils::getStaticStrides(lhs);
    if (failed(stridesOnLhs) || stridesOnLhs->back() != 1 ||
        !lhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (lhsFlag == BinaryFlags::BCAST_SCALAR_IN_0 ||
        lhsFlag == BinaryFlags::BCAST_ROW_IN_0) {
      ldiLhs = 1;
    } else if (lhsFlag == BinaryFlags::BCAST_COL_IN_0) {
      ldiLhs = lhsShapedType.getShape().back();
    } else {
      ldiLhs = stridesOnLhs->front();
    }
  }

  int64_t ldiRhs = 1;
  if (ShapedType rhsShapedType = dyn_cast<ShapedType>(rhs.getType())) {
    auto stridesOnRhs = mlir::utils::getStaticStrides(rhs);
    if (failed(stridesOnRhs) || stridesOnRhs->back() != 1 ||
        !rhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (rhsFlag == BinaryFlags::BCAST_SCALAR_IN_1 ||
        rhsFlag == BinaryFlags::BCAST_ROW_IN_1) {
      ldiRhs = 1;
    } else if (rhsFlag == BinaryFlags::BCAST_COL_IN_1) {
      ldiRhs = rhsShapedType.getShape().back();
    } else {
      ldiRhs = stridesOnRhs->front();
    }
  }

  binaryInfo.ldiLhs = ldiLhs;
  binaryInfo.ldiRhs = ldiRhs;

  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();
  binaryInfo.ldo = stridesOnOutput->front();
  return binaryInfo;
}

FailureOr<UnaryFlags> getUnaryFlags(Type inputType, Type outputType) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");

  if (!isa<ShapedType>(inputType) ||
      cast<ShapedType>(inputType).getRank() == 0) {
    return xsmm::UnaryFlags::BCAST_SCALAR;
  }

  ArrayRef<int64_t> shapeOutput = cast<ShapedType>(outputType).getShape();
  ArrayRef<int64_t> shapeInput = cast<ShapedType>(inputType).getShape();
  assert(shapeOutput.size() >= shapeInput.size() &&
         "output rank must be >= input rank");
  SmallVector<int64_t> bShapeInput;
  computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
  assert(shapeOutput.size() == bShapeInput.size());
  shapeInput = bShapeInput;

  // Same shape for input and output, no bcast.
  if (shapeInput == shapeOutput)
    return xsmm::UnaryFlags::NONE;

  // Input is a memref but it is all ones, bcast = scalar.
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(shapeInput, isOne))
    return xsmm::UnaryFlags::BCAST_SCALAR;

  if (shapeInput[shapeInput.size() - 1] == 1 &&
      shapeOutput[shapeInput.size() - 1] > 1)
    return xsmm::UnaryFlags::BCAST_ROW;

  if (shapeInput[0] == 1 && shapeOutput[0] > 1)
    return xsmm::UnaryFlags::BCAST_COL;

  return failure();
}

FailureOr<BinaryFlags> getBinaryFlags(Type operandType, Type outputType,
                                      OperandPos operandNumber) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(operandType) ||
      cast<ShapedType>(operandType).getRank() == 0) {
    if (operandNumber == OperandPos::LHS)
      return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
    return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  }

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto shapeOutput = cast<MemRefType>(outputType).getShape();
  auto shapeOperand = cast<MemRefType>(operandType).getShape();
  assert(shapeOutput.size() >= shapeOperand.size() &&
         "Output rank must be >= operand rank");
  SmallVector<int64_t> bOperandShape;
  computeBcastShapeInput(shapeOutput, shapeOperand, bOperandShape);
  assert(shapeOutput.size() == bOperandShape.size());
  assert(shapeOutput.size() == 2);

  auto getBCastEnum = [](BCastType bCastType,
                         OperandPos operandPos) -> xsmm::BinaryFlags {
    switch (bCastType) {
    case BCastType::NONE:
      return xsmm::BinaryFlags::NONE;
    case BCastType::SCALAR:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
    case BCastType::ROW:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_ROW_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_ROW_IN_1;
    case BCastType::COL:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_COL_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_COL_IN_1;
    }
    assert(false && "unrechable");
    abort();
  };

  if (bOperandShape == shapeOutput)
    return getBCastEnum(BCastType::NONE, operandNumber);

  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(bOperandShape, isOne))
    return getBCastEnum(BCastType::SCALAR, operandNumber);

  if (bOperandShape[1] == 1 && shapeOutput[1] > 1)
    return getBCastEnum(BCastType::ROW, operandNumber);

  if (bOperandShape[0] == 1 && shapeOutput[0] > 1)
    return getBCastEnum(BCastType::COL, operandNumber);

  return failure();
}

SmallVector<Type> extractOperandTypes(OpBuilder &builder,
                                      SmallVector<XsmmOperand> operands) {
  SmallVector<Type> results;
  for (XsmmOperand operand : operands) {
    if (std::holds_alternative<int64_t>(operand))
      results.push_back(IntegerType::get(builder.getContext(), 64));
    else if (std::holds_alternative<Value>(operand)) {
      Value valueOperand = std::get<Value>(operand);
      if (isa<MemRefType>(valueOperand.getType())) {
        Type basePtrType = LLVM::LLVMPointerType::get(builder.getContext());
        results.push_back(basePtrType);
        results.push_back(builder.getIndexType()); // offset
      }
    } else if (std::holds_alternative<XsmmCall>(operand)) {
      XsmmCall callOperand = std::get<XsmmCall>(operand);
      if (callOperand.CallType == XsmmCallType::DISPATCH) {
        results.push_back(callOperand.CallResult.getType());
      } else {
        llvm_unreachable("Unknown XSMM function type");
      }
    }
  }
  return results;
}

SmallVector<Value> getXsmmOperands(OpBuilder &builder, Location loc,
                                   SmallVector<XsmmOperand> operands,
                                   IntegerAttr dataTypeAttr,
                                   Operation *parentOp) {
  SmallVector<Value> res;
  for (XsmmOperand operand : operands) {
    if (std::holds_alternative<int64_t>(operand))
      res.push_back(getConstInt(builder, std::get<int64_t>(operand), 64));
    else if (std::holds_alternative<Value>(operand)) {
      Value valueOperand = std::get<Value>(operand);
      if (isa<MemRefType>(valueOperand.getType())) {
        auto [ptr, offset] = ::mlir::utils::getPtrAndOffset(
            builder, std::get<Value>(operand), loc);
        res.push_back(ptr);
        res.push_back(offset);
      }
    } else if (std::holds_alternative<XsmmCall>(operand)) {
      XsmmCall callOperand = std::get<XsmmCall>(operand);
      if (callOperand.CallType == XsmmCallType::DISPATCH) {
        res.push_back(callOperand.CallResult);
      } else {
        llvm_unreachable("Unknown XSMM function type");
      }
    }
  }
  return res;
}

func::CallOp buildXsmmCall(RewriterBase &rewriter, XsmmCallType callType,
                           Location loc, DataTypeAttr dtype,
                           SmallVector<XsmmOperand> operands, TypeRange results,
                           FlatSymbolRefAttr fnName, Operation *parentOp,
                           Operation *insertBefore) {
  auto module = parentOp->getParentOfType<ModuleOp>();
  OpBuilder::InsertionGuard guard(rewriter);

  auto operandTypes = xsmm::utils::extractOperandTypes(rewriter, operands);
  createFunction(rewriter, module, fnName.getValue(), operandTypes, results,
                 false);
  if (callType == XsmmCallType::INVOKE) {
    assert(insertBefore != NULL);
    rewriter.setInsertionPoint(insertBefore);
  } else if (callType == XsmmCallType::DISPATCH) {
    auto functionOp = parentOp->getParentOfType<func::FuncOp>();
    rewriter.setInsertionPointAfter(&*functionOp.getBlocks().begin()->begin());
  } else {
    llvm_unreachable("Unknown XSMM function type");
  }

  SmallVector<Value> finalOperands =
      xsmm::utils::getXsmmOperands(rewriter, loc, operands, dtype, parentOp);
  return func::CallOp::create(rewriter, loc, fnName.getValue(), results,
                                       finalOperands);
}

FailureOr<FusedMatch> getFusedBrgemmSequenceFromProducer(Operation *op) {
  // The loop is in reverse order, so we deduplicate the list making sure we
  // only have one type of each
  SmallVector<Operation *, 4> chain;
  Operation *prev = nullptr;
  for (auto *user : op->getUsers()) {
    // Deduplicate, only take each operation once
    if (dyn_cast<func::ReturnOp>(user) || user == prev)
      continue;
    chain.push_back(user);
    prev = user;

    // BRGEMM is the last one, we can stop looking
    if (auto brgemmOp = (dyn_cast<xsmm::BrgemmOp>(user))) {
      // Make sure the BRGEMM outputs to the chain value
      // (it could be one of BRGEMM's inputs in the chain)
      if (brgemmOp.getOperand(3).getDefiningOp() != op)
        return failure();
      continue;
    }

    // Make sure this is a chain, ie. at least once in inputs and outputs
    int numUses = std::count(user->getOperands().begin(),
                             user->getOperands().end(), op->getResult(0));
    // At least one input and the last operand (output) is the same buffer
    if (((dyn_cast<xsmm::UnaryOp>(user) &&
          dyn_cast<xsmm::UnaryOp>(user).getCallee() != UnaryKind::ZERO) &&
         numUses < 2) ||
        user->getOperands()[user->getOperands().size() - 1] != op->getResult(0))
      return failure();
  }
  // We don't know how to fuse more than two tail ops after and a zero op before
  // BRGEMM
  if (chain.size() > 4)
    return failure();
  if (!(isa<xsmm::BrgemmOp>(chain[0]) ||
        (dyn_cast<xsmm::UnaryOp>(chain[0]) &&
         dyn_cast<xsmm::UnaryOp>(chain[0]).getCallee() == UnaryKind::ZERO)))
    // List is in reverse order, put the brgemm or zero at the top
    std::reverse(chain.begin(), chain.end());

  // If we haven't found a BRGEMM or zero, this are not the droids we're looking
  // for
  if (!(isa<xsmm::BrgemmOp>(chain[0]) ||
        (dyn_cast<xsmm::UnaryOp>(chain[0]) &&
         dyn_cast<xsmm::UnaryOp>(chain[0]).getCallee() == UnaryKind::ZERO &&
         isa<xsmm::BrgemmOp>(chain[1]))))
    return failure();

  // Now, we're sure we have a chain, but not yet if it has the right types
  // and in the right order: (ZER0) -> BRGEMM -> BINARY -> UNARY
  // Allowed patterns are:
  //  - (ZERO) + GEMM + BINARY
  //  - (ZERO)+ GEMM + UNARY
  //  - (ZERO) + GEMM + BINARY + UNARY
  xsmm::FusedMatch fusedMatch;
  for (auto *user : chain) {
    if (auto unaryOp = dyn_cast<xsmm::UnaryOp>(user)) {
      if (dyn_cast<xsmm::UnaryOp>(user).getCallee() == UnaryKind::ZERO) {
        fusedMatch.zeroOp = unaryOp;
        continue;
      }
    }
    if (auto brgemmOp = (dyn_cast<xsmm::BrgemmOp>(user))) {
      // We only accept one of each
      if (fusedMatch.brgemmOp)
        return failure();

      fusedMatch.brgemmOp = brgemmOp;
      continue;
    }

    if (auto binOp = (dyn_cast<xsmm::BinaryOp>(user))) {
      // We only accept one of each
      if (fusedMatch.binaryOp)
        return failure();

      // We cannot accept binary *after* unary
      if (fusedMatch.unaryOp)
        return failure();

      // For now we only support ADD as binary
      if (binOp.getCallee() != BinaryKind::ADD)
        return failure();

      // Make sure the op is new or the same as before
      fusedMatch.binaryOp = binOp;
      fusedMatch.binaryKind = binOp.getCallee();
      continue;
    }

    if (auto unOp = dyn_cast<xsmm::UnaryOp>(user)) {
      // We only accept one of each
      if (fusedMatch.unaryOp)
        return failure();

      // Binary op may have come earlier, we don't know
      // We have already made sure it didn't come before this
      // unary in the binary check above

      // For now we only support RELU as unary
      if (unOp.getCallee() != UnaryKind::RELU)
        return failure();

      // Make sure the op is new or the same as before
      fusedMatch.unaryOp = unOp;
      fusedMatch.unaryKind = unOp.getCallee();
      continue;
    }

    // If found anything else in the users, bail
    return failure();
  }

  return fusedMatch;
}

template <typename DispatchOpTy>
FailureOr<SmallVector<Attribute>> getBrgemmFlags(PatternRewriter &rewriter,
                                                 DispatchOpTy dispatchOpTy,
                                                 bool returnNone) {
  SmallVector<Attribute> attributes;
  auto flags = dispatchOpTy.getFlags();
  for (auto flagItr : flags) {
    if (flagItr == xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                            xsmm::GemmFlags::NONE)) {
      if (returnNone) {
        attributes.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                                      xsmm::GemmFlags::NONE));
        return attributes;
      } else {
        return failure();
      }
    }
    attributes.push_back(flagItr);
  }

  if (attributes.empty())
    attributes.push_back(
        xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE));
  return attributes;
}

template FailureOr<SmallVector<Attribute>>
getBrgemmFlags<xsmm::BrgemmDispatchOp>(PatternRewriter &rewriter,
                                       xsmm::BrgemmDispatchOp dispatchOpTy,
                                       bool returnNone);
template FailureOr<SmallVector<Attribute>>
getBrgemmFlags<xsmm::FusedBrgemmDispatchOp>(
    PatternRewriter &rewriter, xsmm::FusedBrgemmDispatchOp dispatchOpTy,
    bool returnNone);
} // namespace utils
} // namespace xsmm
} // namespace mlir
