//===- XsmmUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Transforms/Utils/BuilderUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Support/Compiler.h"
#include <map>
#define DEBUG_TYPE "xsmm-utils"

using namespace mlir;
using namespace mlir::linalg;
using namespace structured_match;

namespace mlir {
namespace xsmm {
namespace utils {

// Callable object to verify if `operand` has static shape.
struct HasStaticShape {
  SmallVectorImpl<int64_t> *shape = nullptr;
  HasStaticShape() = default;
  HasStaticShape(SmallVectorImpl<int64_t> *shape) : shape(shape){};

  bool operator()(Value operand, Operation *op) const {
    auto operandType = operand.getType();
    if (auto shapedType = dyn_cast_or_null<ShapedType>(operandType)) {
      if (!shapedType.hasStaticShape())
        return false;
      if (shape) {
        for (int64_t shapeOnDim : shapedType.getShape())
          shape->push_back(shapeOnDim);
      }
    }
    return true;
  }
};

// Callable object to verify if `operand` has static strides.
// If `operand` is a tensor type or a scalar, return true.
struct HasStaticStrides {
  SmallVectorImpl<int64_t> *strides = nullptr;
  HasStaticStrides() = default;
  HasStaticStrides(SmallVector<int64_t> *strides) : strides(strides){};

  bool operator()(Value operand, Operation *op) const {
    auto operandType = operand.getType();
    SmallVector<int64_t> strides;
    if (auto memRefType = dyn_cast_or_null<MemRefType>(operandType)) {
      int64_t offset;
      if (failed(getStridesAndOffset(memRefType, strides, offset)))
        return false;
      if (llvm::any_of(strides, [](int64_t stride) {
            return stride == ShapedType::kDynamic;
          })) {
        return false;
      }
      if (this->strides)
        this->strides->append(strides.begin(), strides.end());
    }
    return true;
  }
};

// Structural matcher.
static FailureOr<ContractionDimensions>
checkStructure(vector::ContractionOp contractOp, SmallVector<Value> &inputs,
               SmallVector<Value> &outputs, ArrayRef<AffineMap> indexingMap) {
  if (!HasStaticShape()(inputs[0], inputs[0].getDefiningOp()) ||
      !HasStaticShape()(inputs[1], inputs[1].getDefiningOp()) ||
      !HasStaticShape()(inputs[2], inputs[2].getDefiningOp()) ||
      (outputs[0] != nullptr &&
       !HasStaticShape()(outputs[0], outputs[0].getDefiningOp())) ||
      !HasStaticStrides()(inputs[0], inputs[0].getDefiningOp()) ||
      !HasStaticStrides()(inputs[1], inputs[1].getDefiningOp()) ||
      !HasStaticStrides()(inputs[2], inputs[2].getDefiningOp()) ||
      (outputs[0] != nullptr &&
       !HasStaticStrides()(outputs[0], outputs[0].getDefiningOp()))) {
    return failure();
  }

  return inferContractionDims(indexingMap);
}

// Return the position of `dim` in the codomain of `operand`.
std::optional<unsigned> getPosInCodomain(unsigned dim, Value operand,
                                         vector::ContractionOp contractOp,
                                         AffineMap map) {
  return map.getResultPosition(getAffineDimExpr(dim, contractOp.getContext()));
}

static SmallVector<int64_t, 4>
createFlatListOfOperandStaticDims(vector::ContractionOp contractOp) {
  SmallVector<int64_t, 4> res;
  for (OpOperand &opOperand : contractOp.getOperation()->getOpOperands())
    llvm::append_range(
        res, dyn_cast<VectorType>(opOperand.get().getType()).getShape());
  return res;
}

static SmallVector<int64_t, 4>
computeStaticLoopSizes(vector::ContractionOp contractOp,
                       ArrayRef<AffineMap> maps) {
  AffineMap map = concatAffineMaps(maps);
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  SmallVector<int64_t, 4> allShapeSizes =
      createFlatListOfOperandStaticDims(contractOp);
  SmallVector<int64_t, 4> res(numDims, 0);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = dyn_cast<AffineDimExpr>(result))
      res[d.getPosition()] = allShapeSizes[idx];
  }
  return res;
}

static FailureOr<SmallVector<int64_t>>
getVNNIStaticStrides(MemRefType valueType) {
  SmallVector<int64_t> strides;
  int64_t offset;
  SmallVector<int64_t> shape;
  for (size_t i = 0; i < valueType.getShape().size(); i++) {
    shape.push_back(valueType.getShape()[i]);
  }
  auto temp = shape[shape.size() - 1];
  shape[shape.size() - 1] = shape[shape.size() - 2];
  shape[shape.size() - 2] = temp;
  auto memrefType = MemRefType::get(shape, valueType.getElementType());
  if (failed(getStridesAndOffset(memrefType, strides, offset))) {
    return failure();
  }
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      })) {
    return failure();
  }
  return strides;
}

// Access matcher.
FailureOr<xsmm::BrgemmInfo>
checkAccess(PatternRewriter &rewriter, vector::ContractionOp contractOp,
            unsigned m, unsigned n, SmallVector<unsigned, 2> kVector,
            std::optional<unsigned> batchPos, SmallVector<Value> inputs,
            ArrayRef<AffineMap> indexingMap, bool checkForTransposes) {
  Value operandA = inputs[0];
  Value operandB = inputs[1];
  Value operandC = inputs[2];

  unsigned k = kVector[0];
  if (*xsmm::utils::getPosInCodomain(
          kVector[0], contractOp->getOpOperand(1).get(), contractOp,
          contractOp.getIndexingMapsArray()[1]) <
          *xsmm::utils::getPosInCodomain(
              n, contractOp->getOpOperand(1).get(), contractOp,
              contractOp.getIndexingMapsArray()[1]) ||
      kVector.size() == 1) {
    k = kVector[0];
  } else if (kVector.size() > 1) {
    k = kVector[1];
  }
  auto checkStridesAndGetLdaAndBatch =
      [&](Value operand, AffineMap map, AffineMap transformMap,
          int operandIndex, std::map<bool, std::pair<int, int>> dimMap,
          std::optional<int> batchPos) -> FailureOr<std::pair<int64_t, int>> {
    AffineMap indexingMap = map;
    bool permutation = false;
    if (map != transformMap) {
      Value source = operand.getDefiningOp()->getOperand(0);
      auto partialIndexingMap = inversePermutation(compressUnusedDims(map));
      if (isa<mlir::vector::TransferReadOp>(operand.getDefiningOp())) {
        if (checkForTransposes) {
          auto invokeOpSource =
              mlir::utils::getPtrFromOp(rewriter, source, source.getLoc());
          if (succeeded(invokeOpSource)) {
            auto invokeOp = dyn_cast<mlir::func::CallOp>(
                *(*invokeOpSource->first.user_begin()));
            auto dispatchOp = invokeOp->getOperand(1).getDefiningOp();
            int constVal =
                dyn_cast<IntegerAttr>(dyn_cast<arith::ConstantOp>(
                                          dyn_cast<func::CallOp>(dispatchOp)
                                              .getOperand(0)
                                              .getDefiningOp())
                                          .getValue())
                    .getInt();
            if (constVal == xsmm::UnaryKindAttr::get(rewriter.getContext(),
                                                     xsmm::UnaryKind::VNNI2)
                                .getInt() ||
                constVal == xsmm::UnaryKindAttr::get(rewriter.getContext(),
                                                     xsmm::UnaryKind::TRANSPOSE)
                                .getInt()) {
              partialIndexingMap = inversePermutation(partialIndexingMap);
            }

            indexingMap =
                transformMap.compose(partialIndexingMap.getMinorIdentityMap(
                    map.getNumDims(), map.getNumDims(), rewriter.getContext()));
            permutation = true;
          }
        }
      }
    }
    int minorDim = dimMap[permutation].first;
    int majorDim = dimMap[permutation].second;
    auto minorDimPosInCodomain = xsmm::utils::getPosInCodomain(
        minorDim, operand, contractOp, indexingMap);
    auto majorDimPosInCodomain = xsmm::utils::getPosInCodomain(
        majorDim, operand, contractOp, indexingMap);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain) {
      return failure();
    }

    auto dataType = xsmm::utils::getDataType(rewriter, operand.getType());
    MemRefType type;
    if (operand.getDefiningOp() != NULL) {

      if (isa<memref::ExpandShapeOp>(operand.getDefiningOp()) ||
          isa<memref::SubViewOp>(operand.getDefiningOp())) {
        type = dyn_cast<MemRefType>(
            operand.getDefiningOp()->getResult(0).getType());
      } else if (isa<mlir::vector::TransferReadOp>(operand.getDefiningOp())) {
        type = dyn_cast<MemRefType>(
            operand.getDefiningOp()->getOperand(0).getType());
      } else if (isa<MemRefType>(operand.getType())) {
        type = dyn_cast<MemRefType>(operand.getType());
      } else {
        type = dyn_cast<MemRefType>(
            operand.getDefiningOp()->getOperand(0).getType());
      }
    }
    auto shape = type.getShape();
    if (permutation) {
      shape = applyPermutationMap<int64_t>(compressUnusedDims(indexingMap),
                                           type.getShape());
      type = MemRefType::get(
          {shape}, dyn_cast<ShapedType>(operand.getType()).getElementType());
    }
    FailureOr<SmallVector<int64_t>> stridesOnOperand;
    if (dataType == xsmm::DataTypeAttr::get(contractOp.getContext(),
                                            xsmm::DataType::BF16) &&
        operandIndex == 1) {
      stridesOnOperand = getVNNIStaticStrides(type);
    } else {
      stridesOnOperand = ::mlir::utils::getStaticStrides(type);
    }
    if (failed(stridesOnOperand) ||
        ((dataType != xsmm::DataTypeAttr::get(contractOp.getContext(),
                                              xsmm::DataType::BF16) &&
          (*stridesOnOperand)[*minorDimPosInCodomain] != 1))) {
      return failure();
    }

    auto stride = 1;
    if (batchPos >= 0) {
      auto batchPosCodomainA =
          getPosInCodomain(batchPos.value(), operand, contractOp, indexingMap);
      auto stridesOnA = ::mlir::utils::getStaticStrides(type);
      if (succeeded(stridesOnA)) {
        stride = (*stridesOnA)[*batchPosCodomainA];
      }
    }
    if (dataType == xsmm::DataTypeAttr::get(contractOp.getContext(),
                                            xsmm::DataType::BF16) &&
        operandIndex == 1) {
      if (*majorDimPosInCodomain == (*stridesOnOperand).size() - 2) {
        return std::make_pair((*stridesOnOperand)[*majorDimPosInCodomain + 1],
                              stride);
      } else if (*majorDimPosInCodomain == (*stridesOnOperand).size() - 1) {
        return std::make_pair((*stridesOnOperand)[*majorDimPosInCodomain - 1],
                              stride);
      } else {
        return std::make_pair((*stridesOnOperand)[*majorDimPosInCodomain],
                              stride);
      }
    } else {
      return std::make_pair((*stridesOnOperand)[*majorDimPosInCodomain],
                            stride);
    }
  };

  std::map<bool, std::pair<int, int>> ldaIndexMap;
  ldaIndexMap[false] = std::make_pair(k, m);
  ldaIndexMap[true] = std::make_pair(n, k);
  auto ldaVal = checkStridesAndGetLdaAndBatch(
      operandA, indexingMap[0], indexingMap[1], 0, ldaIndexMap, batchPos);

  if (failed(ldaVal)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute lda\n");
    return failure();
  }
  auto lda = (*ldaVal).first;
  auto strideA = (*ldaVal).second;
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "A: OK "
                          << lda << "\n");

  std::map<bool, std::pair<int, int>> ldbIndexMap;
  ldbIndexMap[false] = std::make_pair(n, k);
  ldbIndexMap[true] = std::make_pair(k, m);
  auto ldbVal = checkStridesAndGetLdaAndBatch(
      operandB, indexingMap[1], indexingMap[0], 1, ldbIndexMap, batchPos);

  if (failed(ldbVal)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute ldb\n");
    return failure();
  }
  auto ldb = (*ldbVal).first;
  auto strideB = (*ldbVal).second;
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "B: OK "
                          << ldb << "\n");

  // C(m, n)
  std::map<bool, std::pair<int, int>> ldcIndexMap;
  ldcIndexMap[true] = std::make_pair(n, m);
  ldcIndexMap[false] = std::make_pair(n, m);
  int batch = -1;
  auto ldcVal = checkStridesAndGetLdaAndBatch(
      operandC, indexingMap[2], indexingMap[2], 2, ldcIndexMap, batch);
  if (failed(ldcVal)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute ldc\n");
    return failure();
  }
  auto ldc = (*ldcVal).first;
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "C: OK "
                          << ldc << "\n");

  auto loops = computeStaticLoopSizes(contractOp, indexingMap);
  int64_t batchVal = (batchPos) ? loops[batchPos.value()] : 0;
  auto loopsK = 1;
  for (auto kItr : kVector)
    loopsK *= loops[kItr];

  xsmm::BrgemmInfo info{loops[m], loops[n], loopsK,  batchVal, lda,
                        ldb,      ldc,      strideA, strideB};
  return info;
}

// Check if the given
// generic is mappable to a
// brgemm xsmm op.
// - It is a contraction,
// with:
// -- 1 m and 1 n and 2 k
// dimensions.
// -- m appears on the LHS
// and OUT but not in RHS.
// -- n appears on the RHS
// and OUT but not in LHS.
// -- k and k' appear on the
// RHS and LHS but not OUT.
// -- the stride of the
// minor dimension for A, k
// is 1.
// -- the stride of the
// minor dimension for B, n
// is 1.
// -- the stride of the
// minor dimension for C, n
// is 1.
FailureOr<BrgemmInfo>
isMappableToBrgemm(PatternRewriter &rewriter, vector::ContractionOp contractOp,
                   SmallVector<Value> &inputs, SmallVector<Value> &output,
                   ArrayRef<AffineMap> indexingMap, bool checkForTransposes) {
  auto contractionDims =
      checkStructure(contractOp, inputs, output, indexingMap);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] Failed "
                               "on "
                               "checkStructure"
                               "\n");
    return failure();
  }
  unsigned m = contractionDims->m.back();
  unsigned n = contractionDims->n.back();
  SmallVector<unsigned, 2> kVector;
  std::optional<unsigned> batch;
  auto pos = xsmm::utils::getPosInCodomain(
      contractionDims->k[0], inputs[0], contractOp,
      contractOp.getIndexingMapsArray()[0]);
  int index = 0;
  if (contractionDims->k.size() >= 2) {
    for (int i = 1; i < contractionDims->k.size(); i++) {
      auto posTwo = xsmm::utils::getPosInCodomain(
          contractionDims->k[i], inputs[0], contractOp,
          contractOp.getIndexingMapsArray()[0]);
      if (*posTwo < *pos) {
        index = i;
        pos = posTwo;
      }
    }
  }
  if (contractionDims->k.size() >= 2) {
    batch = contractionDims->k[index];
    for (int i = 0; i < contractionDims->k.size(); i++) {
      if (i == index)
        continue;
      kVector.push_back(contractionDims->k[i]);
    }
  } else {
    for (size_t i = 0; i < contractionDims->k.size(); i++)
      kVector.push_back(contractionDims->k[i]);
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Candidate "
                             "dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] m: "
                          << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] n: "
                          << n << "\n");
  if (batch)
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] batch: "
                            << batch << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] no batch "
                               "dim\n");
  auto retval = checkAccess(rewriter, contractOp, m, n, kVector, batch, inputs,
                            indexingMap, checkForTransposes);
  if (failed(retval)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to check access\n");
    return failure();
  }
  return retval;
}

DataTypeAttr getDataType(RewriterBase &rewriter, Type type) {
  auto elemType = getElementTypeOrSelf(type);
  if (elemType.isBF16())
    return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16);
  return xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
}

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  Value inputVectorType, Value outputVectorType,
                                  UnaryFlags inputFlag) {
  Type outputType = outputVectorType.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  UnaryInfo unaryInfo;
  unaryInfo.m = outputShapedType.getShape()[1];
  unaryInfo.n = outputShapedType.getShape()[0];
  int ldi = 1;
  if (ShapedType inputShapedType = dyn_cast<ShapedType>(input.getType())) {
    SmallVector<int64_t> strides;
    int64_t offset;
    getStridesAndOffset(dyn_cast<MemRefType>(inputShapedType), strides, offset);
    ldi = strides.front();
  }

  unaryInfo.ldi = ldi;
  int ldo = 1;
  if (ShapedType outputShapedType = dyn_cast<ShapedType>(output.getType())) {
    SmallVector<int64_t> strides;
    int64_t offset;
    auto memrefType = MemRefType::get(outputShapedType.getShape(),
                                      outputShapedType.getElementType());
    getStridesAndOffset(memrefType, strides, offset);
    ldo = strides.front();
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
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      assert(false && "bCast semantics for identity op broken");
  }
}

FailureOr<UnaryFlags> getUnaryFlags(Type inputType, Type outputType) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

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

  if (shapeInput[1] == 1 && shapeOutput[1] > 1)
    return xsmm::UnaryFlags::BCAST_ROW;

  if (shapeInput[0] == 1 && shapeOutput[0] > 1)
    return xsmm::UnaryFlags::BCAST_COL;

  return failure();
}

FailureOr<BinaryFlags> getBinFlags(ArrayRef<int64_t> shapeOutput,
                                   ArrayRef<int64_t> shapeOperand,
                                   OperandPos operandNumber) {
  assert(shapeOutput.size() >= shapeOperand.size() &&
         "Output rank must be >= operand rank");
  SmallVector<int64_t> bOperandShape;
  computeBcastShapeInput(shapeOutput, shapeOperand, bOperandShape);
  assert(shapeOutput.size() == bOperandShape.size());
  assert(shapeOutput.size() == 2);
  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
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
  return getBinFlags(shapeOutput, shapeOperand, operandNumber);
}

FailureOr<BinaryFlags> getBinaryFlagsVectorType(Type operandType,
                                                Type outputType,
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
  auto shapeOutput = cast<VectorType>(outputType).getShape();
  auto shapeOperand = cast<MemRefType>(operandType).getShape();
  return getBinFlags(shapeOutput, shapeOperand, operandNumber);
}

FailureOr<int64_t> getLeadingDim(Type type, size_t pos) {
  // Not shaped type, the leading dimension is the single scalar.
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return 1;
  // For 1d memref we cannot use the stride as leading dimension, but the
  // leading dimension is the dimension itself.
  if (memref.getRank() == 1)
    return memref.getShape()[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset)))
    return failure();
  // fail if the strides are non-constant
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      }))
    return failure();
  return strides[pos];
}

static bool isInnerMostDim(OpOperand *operand, unsigned minorDim,
                           vector::ContractionOp contractOp,
                           xsmm::DataTypeAttr dtype, int operandNumber) {
  auto shapedType = cast<VectorType>(operand->get().getType());
  int64_t rank = shapedType.getRank();
  if (dtype == xsmm::DataTypeAttr::get(contractOp.getContext(),
                                       xsmm::DataType::BF16) &&
      (operandNumber == 1 || operandNumber == 0)) {
    return minorDim == rank - 2;
  }
  return minorDim == rank - 1;
}
// Emit a transpose operation for `operand` by swapping `dim` with `newDim`.
// Emit a transpose operation for `operand` by swapping the dimensions at index
// `dim` with `newDim`.
static void emitTransposeOnOperand(RewriterBase &rewriter,
                                   vector::ContractionOp contractOp,
                                   Value operand, unsigned dim, unsigned newDim,
                                   int operandNumber) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(contractOp);
  Location loc = contractOp.getLoc();
  auto operandType = cast<ShapedType>(operand.getType());
  auto rank = operandType.getRank();
  SmallVector<int64_t> shape = llvm::to_vector(operandType.getShape());
  auto permutation = llvm::to_vector(llvm::seq<int64_t>(0, rank));
  std::swap(permutation[dim], permutation[newDim]);
  assert(isPermutationVector(permutation));
  LLVM_DEBUG(llvm::interleaveComma(
      permutation, llvm::dbgs() << "[emitTransposeOnOperand] Perm: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");
  applyPermutationToVector<int64_t>(shape, permutation);
  auto vectorType = VectorType::get(
      shape, cast<ShapedType>(operand.getType()).getElementType());

  vector::TransposeOp transposeResult = rewriter.create<vector::TransposeOp>(
      loc, vectorType, operand, permutation);
  /*  auto writeMemrefSubviewDefinition =
        operand.getDefiningOp()->getOperand(0).getDefiningOp();
    assert(isa<memref::SubViewOp>(writeMemrefSubviewDefinition));
    auto writeMemrefSubview =
    dyn_cast<memref::SubViewOp>(writeMemrefSubviewDefinition); auto
    writeMemrefDefinition =
        operand.getDefiningOp()->getOperand(0).getDefiningOp()->getOperand(0);*/
  /*  auto memref = rewriter.create<memref::AllocaOp>(
        operand.getLoc(),
        MemRefType::get(
            dyn_cast<ShapedType>(writeMemrefDefinition.getType()).getShape(),
            operandType.getElementType()));
    auto writeSubviewType = MemRefType::get(
        dyn_cast<ShapedType>(writeMemrefSubview.getResult().getType()).getShape(),
        dyn_cast<MemRefType>(writeMemrefSubview.getType()).getElementType());

    auto [originalStride, originalOffset] = getStridesAndOffset(
        dyn_cast<MemRefType>(writeMemrefSubview.getResult().getType()));
    writeSubviewType = MemRefType::get(
        dyn_cast<ShapedType>(writeMemrefSubview.getResult().getType()).getShape(),
        dyn_cast<MemRefType>(writeSubviewType).getElementType(),
        StridedLayoutAttr::get(rewriter.getContext(), ShapedType::kDynamic,
                               originalStride));

    auto writeMemrefSubviewOp = rewriter.create<memref::SubViewOp>(
        operand.getLoc(), dyn_cast<MemRefType>(writeSubviewType), memref,
        writeMemrefSubview.getOffsets(), writeMemrefSubview.getSizes(),
        writeMemrefSubview.getStrides(), writeMemrefSubview.getStaticOffsets(),
        writeMemrefSubview.getStaticSizes(),
    writeMemrefSubview.getStaticStrides());
   */
  SmallVector<Value> indices(vectorType.getRank(),
                             rewriter.create<arith::ConstantIndexOp>(loc, 0));

  auto memref = rewriter.create<memref::AllocaOp>(
      operand.getLoc(),
      MemRefType::get(
          dyn_cast<ShapedType>(vectorType).getShape(),
          dyn_cast<ShapedType>(operand.getType()).getElementType()));

  auto transferWrite = rewriter.create<vector::TransferWriteOp>(
      operand.getLoc(), transposeResult->getResult(0), memref, indices);

  SmallVector<AffineMap> transposeIndexingMaps =
      contractOp.getIndexingMapsArray();

  rewriter.setInsertionPoint(contractOp);

  Value transferRead = rewriter.create<vector::TransferReadOp>(
      operand.getLoc(), vectorType, memref, indices);

  contractOp->setOperand(operandNumber, transferRead);

  auto map = AffineMap::getPermutationMap(permutation, rewriter.getContext());
  map = map.compose(contractOp.getIndexingMapsArray()[operandNumber]);

  transposeIndexingMaps[operandNumber] = map;
  rewriter.modifyOpInPlace(contractOp, [&]() {
    contractOp.setIndexingMapsAttr(ArrayAttr::get(
        contractOp.getContext(),
        llvm::to_vector(llvm::map_range(transposeIndexingMaps,
                                        [](AffineMap map) -> Attribute {
                                          return AffineMapAttr::get(map);
                                        }))));
  });
}

FailureOr<vector::ContractionOp>
makeMinorDimensionsInnerMost(RewriterBase &rewriter,
                             vector::ContractionOp contractOp, unsigned m,
                             unsigned n, unsigned k, xsmm::DataTypeAttr type) {
  OpOperand *operandA = &contractOp->getOpOperand(0);
  OpOperand *operandB = &contractOp->getOpOperand(1);
  OpOperand &operandC = contractOp->getOpOperand(2);

  // C(m,n) += A(m,k) * B(k,n)
  // n is expected to be the innermost for C
  // k is expected to be the innermost for A
  // n is expected to be the innermost for B
  auto minorKInCodomainOpA = xsmm::utils::getPosInCodomain(
      k, operandA->get(), contractOp, contractOp.getIndexingMapsArray()[0]);
  auto minorMInCodomainOpA = xsmm::utils::getPosInCodomain(
      m, operandA->get(), contractOp, contractOp.getIndexingMapsArray()[0]);
  if (!minorKInCodomainOpA || !minorMInCodomainOpA) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for A\n");
    return failure();
  }
  auto minorNInCodomainOpB = xsmm::utils::getPosInCodomain(
      n, operandB->get(), contractOp, contractOp.getIndexingMapsArray()[1]);
  auto minorKInCodomainOpB = xsmm::utils::getPosInCodomain(
      k, operandB->get(), contractOp, contractOp.getIndexingMapsArray()[1]);
  if (!minorNInCodomainOpB || !minorKInCodomainOpB) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for B\n");
    return failure();
  }
  auto minorNInCodomainOpC = xsmm::utils::getPosInCodomain(
      n, operandC.get(), contractOp, contractOp.getIndexingMapsArray()[2]);
  auto minorMInCodomainOpC = xsmm::utils::getPosInCodomain(
      m, operandC.get(), contractOp, contractOp.getIndexingMapsArray()[2]);
  if (!minorNInCodomainOpC || !minorMInCodomainOpC) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for C\n");
    return failure();
  }
  if (!isInnerMostDim(&operandC, *minorNInCodomainOpC, contractOp, type, 2)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for C\n");
    if (isInnerMostDim(&operandC, *minorMInCodomainOpC, contractOp, type, 2)) {
      if (isInnerMostDim(operandA, *minorKInCodomainOpA, contractOp, type, 0)) {
        emitTransposeOnOperand(rewriter, contractOp, operandA->get(),
                               *minorKInCodomainOpA, *minorMInCodomainOpA, 0);
      }
      if (isInnerMostDim(operandB, *minorNInCodomainOpB, contractOp, type, 1)) {
        emitTransposeOnOperand(rewriter, contractOp, operandB->get(),
                               *minorNInCodomainOpB, *minorKInCodomainOpB, 1);
      }
    }
    // Avoid transpose on the output by swapping A and B.
    OpOperand *operandA = &contractOp->getOpOperand(0);
    OpOperand *operandB = &contractOp->getOpOperand(1);
    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
    std::swap(indexingMaps[0], indexingMaps[1]);
    rewriter.modifyOpInPlace(contractOp, [&]() {
      Value operandATmp = operandA->get();
      contractOp->setOperand(0, operandB->get());
      contractOp->setOperand(1, operandATmp);
      contractOp.setIndexingMapsAttr(
          ArrayAttr::get(contractOp.getContext(),
                         llvm::to_vector(llvm::map_range(
                             indexingMaps, [](AffineMap map) -> Attribute {
                               return AffineMapAttr::get(map);
                             }))));
    });
    return contractOp;
  }

  if (!isInnerMostDim(operandA, *minorKInCodomainOpA, contractOp, type, 0)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for A\n");
    if (isInnerMostDim(operandA, *minorMInCodomainOpA, contractOp, type, 0)) {
      emitTransposeOnOperand(rewriter, contractOp, operandA->get(),
                             *minorKInCodomainOpA, *minorMInCodomainOpA, 0);
    }
  }
  if (!isInnerMostDim(operandB, *minorNInCodomainOpB, contractOp, type, 1)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for B\n");
    if (isInnerMostDim(operandB, *minorKInCodomainOpB, contractOp, type, 1)) {
      emitTransposeOnOperand(rewriter, contractOp, operandB->get(),
                             *minorKInCodomainOpB, *minorNInCodomainOpB, 1);
    }
  }
  return contractOp;
}

bool isTwoDTransposeOp(vector::TransposeOp transposeOp) {
  if (!(dyn_cast<VectorType>(transposeOp.getOperand().getType()).getRank() ==
            2 &&
        dyn_cast<VectorType>(transposeOp.getResult().getType()).getRank() ==
            2) ||
      !(isa<scf::ForallOp>(transposeOp->getParentOp()) &&
        dyn_cast<scf::ForallOp>(transposeOp->getParentOp()).getRank() == 2))
    return false;
  return true;
}

// Extract the operands to be used in the function call. For each memref operand
// extract the aligned pointer and the offset.
SmallVector<Value> getOperands(OpBuilder &builder, Location loc,
                               ValueRange operands, IntegerAttr dataTypeAttr,
                               Operation *parentOp, bool getResults) {
  SmallVector<Value> res;
  builder.setInsertionPoint(parentOp);
  for (Value operand : operands) {
    auto memrefType = dyn_cast<MemRefType>(operand.getType());
    if (!memrefType) {
      res.push_back(operand);
      continue;
    }
    auto [ptr, offset] = ::mlir::utils::getPtrAndOffset(builder, operand, loc);
    res.push_back(ptr);
    res.push_back(offset);
  }
  return res;
}

SmallVector<Type> extractInvokeOperandTypes(OpBuilder &builder,
                                            ValueRange operands) {
  SmallVector<Type> results;
  // One extra operand for datatype
  for (Value operand : operands) {
    Type operandType = operand.getType();
    if (auto memrefType = dyn_cast<MemRefType>(operandType)) {
      // TODO: non-POD will require an LLVMTypeConverter.
      Type basePtrType = LLVM::LLVMPointerType::get(builder.getContext());
      results.push_back(basePtrType);
      results.push_back(builder.getIndexType()); // offset
    } else {
      results.push_back(operand.getType());
    }
  }
  return results;
}

int64_t getOredFlags(ArrayAttr flags) {
  int64_t oredFlag = 0;
  for (auto flag : flags) {
    int64_t intAttr = dyn_cast<IntegerAttr>(flag).getInt();
    // LIBXSMM is col-major, swap A and B flags.
    if (auto gemmFlag = dyn_cast_or_null<xsmm::GemmFlagsAttr>(flag)) {
      if (gemmFlag.getValue() == GemmFlags::VNNI_A)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_B);
      if (gemmFlag.getValue() == GemmFlags::VNNI_B)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_A);
    }
    oredFlag |= intAttr;
  }
  return oredFlag;
}

func::CallOp buildDispatchCall(RewriterBase &rewriter, Location loc,
                               ArrayRef<Value> dispatchOperands,
                               ArrayRef<Type> dispatchOperandTypes,
                               ModuleOp module, FlatSymbolRefAttr fnName) {
  auto libFnType = rewriter.getFunctionType(
      dispatchOperandTypes, IntegerType::get(rewriter.getContext(), 64));

  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    funcOp.setPrivate();
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(dispatchOperands.back().getDefiningOp());

  func::CallOp call = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), IntegerType::get(rewriter.getContext(), 64),
      dispatchOperands);
  return call;
}

func::CallOp buildInvokeCall(RewriterBase &rewriter, Operation *parentOp,
                             ModuleOp module, SmallVector<Value> inputRange,
                             SmallVector<Value> prependOperands,
                             int prependIndex, SmallVector<Value> operandRange,
                             StringRef invokeName, DataTypeAttr dtype,
                             bool getResult) {
  SmallVector<Value> finalOperands;
  finalOperands.append(operandRange.begin(), operandRange.end());
  SmallVector<Value> extraOperands;
  int i = 0;
  while (i < inputRange.size()) {
    if (i == prependIndex) {
      extraOperands.append(prependOperands.begin(), prependOperands.end());
    }
    extraOperands.push_back(inputRange[i]);
    i++;
  }
  if (prependIndex >= 0 && inputRange.size() == 0) {
    extraOperands.append(prependOperands.begin(), prependOperands.end());
  }
  finalOperands.append(extraOperands.begin(), extraOperands.end());
  SmallVector<Type> invokeTypes =
      xsmm::utils::extractInvokeOperandTypes(rewriter, finalOperands);
  auto loc = parentOp->getLoc();
  auto libFnType = rewriter.getFunctionType(invokeTypes, {});
  FlatSymbolRefAttr fnName =
      SymbolRefAttr::get(rewriter.getContext(), invokeName);

  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, invokeName, libFnType);
    funcOp.setPrivate();
  }

  SmallVector<Value> operands = xsmm::utils::getOperands(
      rewriter, loc, finalOperands, dtype, parentOp, getResult);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(parentOp);
  func::CallOp call =
      rewriter.create<func::CallOp>(loc, fnName, TypeRange(), operands);

  return call;
}

} // namespace utils
} // namespace xsmm
} // namespace mlir
