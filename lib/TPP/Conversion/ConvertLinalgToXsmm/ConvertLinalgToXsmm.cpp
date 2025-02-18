//===- ConvertLinalgToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/IR/MatcherUtils.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTLINALGTOXSMM
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_FOLDXSMMFLAGS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "convert-linalg-to-xsmm"

namespace {

struct ConvertLinalgToXsmm
    : public tpp::impl::ConvertLinalgToXsmmBase<ConvertLinalgToXsmm> {
  using ConvertLinalgToXsmmBase::ConvertLinalgToXsmmBase;
  void runOnOperation() override;
};

struct FoldXsmmFlags : public tpp::impl::FoldXsmmFlagsBase<FoldXsmmFlags> {
  void runOnOperation() override;
};

namespace {
struct BrgemmInfo {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t batch;

  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t strideA;
  int64_t strideB;

  bool isVnni = false;
};

} // namespace

// Return the position of `dim` in the codomain of `operand`.
std::optional<unsigned> getPosInCodomain(unsigned dim, OpOperand *operand,
                                         linalg::LinalgOp linalgOp) {
  assert(operand->getOwner() == linalgOp);
  return linalgOp.getMatchingIndexingMap(operand).getResultPosition(
      getAffineDimExpr(dim, linalgOp.getContext()));
}

// Replace `linalgOp` with a binary dispatch plus invoke.
static void replaceOpWithBinary(RewriterBase &rewriter,
                                linalg::LinalgOp linalgOp,
                                ArrayRef<Value> operands,
                                xsmm::BinaryInfo binaryInfo, ArrayAttr flags,
                                xsmm::BinaryKindAttr kind) {
  Location loc = linalgOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
      rewriter.getContext(),
      ArrayRef<int64_t>{binaryInfo.m, binaryInfo.n, binaryInfo.ldiLhs,
                        binaryInfo.ldiRhs, binaryInfo.ldo});
  auto dtype =
      xsmm::utils::getDataType(rewriter, linalgOp.getDpsInits()[0].getType());
  Value dispatched = rewriter.create<xsmm::BinaryDispatchOp>(
      loc, integer64, kind, dims, flags, dtype);
  SmallVector<Value> invokeOperands;
  invokeOperands.push_back(dispatched);
  invokeOperands.append(operands.begin(), operands.end());
  rewriter.replaceOpWithNewOp<xsmm::BinaryOp>(linalgOp, dtype, kind,
                                              invokeOperands);
}

// Convert a linalg.fill to XSMM zero, if the fill fills with zeros.
struct ConvertFillOpToUnaryZero : public OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern<linalg::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    if (!structured_match::utils::isTwoDFillOpWithZeros(fillOp, &operands) ||
        operands.size() != 2) {
      return failure();
    }

    auto unaryInfo = xsmm::utils::getUnaryInfo(operands[0], operands[1],
                                               xsmm::UnaryFlags::BCAST_SCALAR);
    if (failed(unaryInfo))
      return failure();

    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::BCAST_SCALAR));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::ZERO);
    xsmm::utils::replaceOpWithUnary(rewriter, fillOp, operands, *unaryInfo,
                                    flags, kind);
    return success();
  }
};

// Convert a linalg.transpose to a XSMM unary transpose.
struct ConvertTransposeOpToUnaryTranspose
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value> operands;
    if (!structured_match::utils::isTwoDTransposeOp(transposeOp, &operands) ||
        operands.size() != 2) {
      return failure();
    }

    auto unaryInfo = xsmm::utils::getUnaryInfo(operands[0], operands[1],
                                               xsmm::UnaryFlags::NONE);
    if (failed(unaryInfo))
      return failure();

    // LIBXSMM for transpose wants the input dims and not the output.
    std::swap((*unaryInfo).m, (*unaryInfo).n);
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
        rewriter.getContext(), xsmm::UnaryKind::TRANSPOSE);
    xsmm::utils::replaceOpWithUnary(rewriter, transposeOp, operands, *unaryInfo,
                                    flags, kind);
    return success();
  }
};

// Get the OpOperand matching 'input', assert if 'input' is not found.
static OpOperand *getOperandFromValue(linalg::GenericOp genericOp, Value val) {
  SmallVector<OpOperand *> allOperands = genericOp.getDpsInputOperands();
  SmallVector<OpOperand *> initOperands = llvm::to_vector(llvm::map_range(
      genericOp.getDpsInitsMutable(), [](OpOperand &o) { return &o; }));
  allOperands.append(initOperands.begin(), initOperands.end());

  OpOperand *valAsOperand = nullptr;
  for (OpOperand *operand : allOperands) {
    if (operand->get() == val) {
      valAsOperand = operand;
      break;
    }
  }
  assert(valAsOperand && "expect to find input");
  return valAsOperand;
}

namespace {
enum class BroadCastType { NONE = 0, SCALAR, ROW, COL };
} // namespace

static FailureOr<BroadCastType> getBroadCastFromMap(AffineMap map) {
  if (map.getNumResults() > map.getNumInputs() || map.getNumInputs() != 2 ||
      map.getNumSymbols() != 0) {
    return failure();
  }

  if (map.getNumResults() == 0)
    return BroadCastType::SCALAR;

  if (!map.isProjectedPermutation(/*allowZeroInResults=*/true))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "[getBroadCastFromMap] map: " << map << "\n");

  SmallVector<bool> isPresent(map.getNumInputs(), false);
  for (auto expr : map.getResults()) {
    if (auto cstExpr = dyn_cast<AffineConstantExpr>(expr)) {
      if (cstExpr.getValue() != 0)
        return failure();
    } else if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      isPresent[dimExpr.getPosition()] = true;
    } else {
      return failure();
    }
  }

  // None of the dimensions are available, scalar broadcast.
  if (llvm::all_of(isPresent, [](bool dim) { return !dim; })) {
    return BroadCastType::SCALAR;
  }

  // All the dimensions are available, no broadcast.
  if (llvm::all_of(isPresent, [](bool dim) { return dim; })) {
    return BroadCastType::NONE;
  }

  size_t rowPos = 0;
  if (isPresent[rowPos] == false) // Broadcast the cols into the rows.
    return BroadCastType::COL;
  return BroadCastType::ROW;
}

// Get the xsmm unary broadcast flags by looking at the map. Example,
// (d0, d1) -> (d0, d1) = NONE
// (d0, d1) -> (0, d1) = COL
// (d0, d1) -> (d0, 0) = ROW
// (d0, d1) -> () = SCALAR
static FailureOr<xsmm::UnaryFlags> getBroadCastUnaryFlagFromMap(AffineMap map) {
  auto broadCastType = getBroadCastFromMap(map);
  if (failed(broadCastType))
    return failure();

  switch (*broadCastType) {
  case BroadCastType::SCALAR:
    return xsmm::UnaryFlags::BCAST_SCALAR;
  case BroadCastType::ROW:
    return xsmm::UnaryFlags::BCAST_ROW;
  case BroadCastType::COL:
    return xsmm::UnaryFlags::BCAST_COL;
  default:
    return xsmm::UnaryFlags::NONE;
  }
}

static Value makeOperandShapeRowBroadCastable(RewriterBase &rewriter,
                                              Location loc, Value output,
                                              Value operand) {
  assert(isa<ShapedType>(output.getType()));
  assert(isa<ShapedType>(operand.getType()));

  ShapedType shapedOutput = cast<ShapedType>(output.getType());
  if (shapedOutput.getRank() != 2)
    return operand;

  ShapedType shapedOperand = cast<ShapedType>(operand.getType());
  if (shapedOperand.getRank() != 1)
    return operand;

  SmallVector<int64_t> shapeOperand = llvm::to_vector(shapedOperand.getShape());
  shapeOperand.push_back(1);
  auto newShapedOperand =
      MemRefType::get(shapeOperand, shapedOperand.getElementType());
  auto reassoc =
      getReassociationIndicesForReshape(shapedOperand, newShapedOperand);
  assert(reassoc.has_value());
  return linalgx::utils::expand(rewriter, loc, operand, newShapedOperand,
                                *reassoc);
}

// Convert linalg.generic to xsmm unary relu or identity op.
struct ConvertGenericToUnary : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    if (!genericOp.hasPureBufferSemantics())
      return failure();

    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr();
    if (structured_match::utils::isTwoDReluOp(genericOp, &operands)) {
      kind = xsmm::UnaryKindAttr::get(rewriter.getContext(),
                                      xsmm::UnaryKind::RELU);
    } else if (structured_match::utils::isTwoDIdentityOp(genericOp,
                                                         &operands)) {
      kind = xsmm::UnaryKindAttr::get(rewriter.getContext(),
                                      xsmm::UnaryKind::IDENTITY);
    }

    if (!kind || operands.size() != 2)
      return failure();

    OpOperand *inputOperand = getOperandFromValue(genericOp, operands[0]);
    auto broadCastFlag = getBroadCastUnaryFlagFromMap(
        genericOp.getMatchingIndexingMap(inputOperand));
    if (failed(broadCastFlag))
      return failure();

    // Make shape broadcast compatible.
    // For later XSMM verification we need to introduce back
    // unit dimension if we are dealing with a row broadcast.
    // Example: memref<10xf32> -> memref<10x1xf32>
    if (*broadCastFlag == xsmm::UnaryFlags::BCAST_ROW) {
      operands[0] = makeOperandShapeRowBroadCastable(
          rewriter, genericOp.getLoc(), operands[1], operands[0]);
    }

    auto unaryInfo =
        xsmm::utils::getUnaryInfo(operands[0], operands[1], *broadCastFlag);
    if (failed(unaryInfo))
      return failure();
    auto flags = rewriter.getArrayAttr(
        xsmm::UnaryFlagsAttr::get(rewriter.getContext(), *broadCastFlag));
    xsmm::utils::replaceOpWithUnary(rewriter, genericOp, operands, *unaryInfo,
                                    flags, kind);
    return success();
  }
};

static FailureOr<xsmm::BinaryFlags>
getBroadCastBinaryFlagFromMap(AffineMap map, unsigned operandIdx) {
  auto broadCastType = getBroadCastFromMap(map);
  if (failed(broadCastType))
    return failure();

  assert(operandIdx == 0 || operandIdx == 1);
  switch (*broadCastType) {
  case BroadCastType::SCALAR:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_SCALAR_IN_0
                             : xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  case BroadCastType::ROW:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_ROW_IN_0
                             : xsmm::BinaryFlags::BCAST_ROW_IN_1;
  case BroadCastType::COL:
    return (operandIdx == 0) ? xsmm::BinaryFlags::BCAST_COL_IN_0
                             : xsmm::BinaryFlags::BCAST_COL_IN_1;
  default:
    return xsmm::BinaryFlags::NONE;
  }
}

static LogicalResult rewriteBinaryOp(RewriterBase &rewriter,
                                     linalg::GenericOp genericOp,
                                     MutableArrayRef<Value> operands,
                                     xsmm::BinaryKind xsmmTy) {
  assert(operands.size() == 3);
  Location loc = genericOp.getLoc();
  auto &lhs = operands[0];
  auto &rhs = operands[1];
  auto &output = operands[2];

  OpOperand *lhsOperand = getOperandFromValue(genericOp, lhs);
  auto broadCastFlagLhs = getBroadCastBinaryFlagFromMap(
      genericOp.getMatchingIndexingMap(lhsOperand), /*operandIdx=*/0);
  if (failed(broadCastFlagLhs))
    return failure();
  if (*broadCastFlagLhs == xsmm::BinaryFlags::BCAST_ROW_IN_0) {
    lhs = makeOperandShapeRowBroadCastable(rewriter, loc, output, lhs);
  }

  OpOperand *rhsOperand = getOperandFromValue(genericOp, rhs);
  auto broadCastFlagRhs = getBroadCastBinaryFlagFromMap(
      genericOp.getMatchingIndexingMap(rhsOperand), /*operandIdx=*/1);
  if (failed(broadCastFlagRhs))
    return failure();
  if (*broadCastFlagRhs == xsmm::BinaryFlags::BCAST_ROW_IN_1) {
    operands[1] = makeOperandShapeRowBroadCastable(rewriter, loc, output, rhs);
  }

  auto binaryInfo = xsmm::utils::getBinaryInfo(lhs, *broadCastFlagLhs, rhs,
                                               *broadCastFlagRhs, output);
  if (failed(binaryInfo))
    return failure();

  auto flagLhs =
      xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *broadCastFlagLhs);
  auto flagRhs =
      xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *broadCastFlagRhs);

  // Spaghetti code to handle 'NONE' as it conflicts with other flags; we
  // cannot add it if at least the RHS or the LHS is not 'NONE'. Maybe the
  // best solution is to get rid of it.
  SmallVector<Attribute> flagsVec;
  if (flagLhs.getValue() != xsmm::BinaryFlags::NONE)
    flagsVec.push_back(flagLhs);
  if (flagRhs.getValue() != xsmm::BinaryFlags::NONE)
    flagsVec.push_back(flagRhs);
  if (flagsVec.empty()) {
    flagsVec.push_back(xsmm::BinaryFlagsAttr::get(rewriter.getContext(),
                                                  xsmm::BinaryFlags::NONE));
  }
  ArrayAttr flags = rewriter.getArrayAttr(flagsVec);

  xsmm::BinaryKindAttr kind =
      xsmm::BinaryKindAttr::get(rewriter.getContext(), xsmmTy);
  replaceOpWithBinary(rewriter, genericOp, operands, *binaryInfo, flags, kind);
  return success();
}

// Convert linalg.generic to xsmm binary:
// 1. Add
// 2. Mul
// 3. Sub
struct ConvertGenericToBinary : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    if (!genericOp.hasPureBufferSemantics())
      return failure();
    xsmm::BinaryKind kind = xsmm::BinaryKind::NONE;

    if (structured_match::utils::isTwoDAddOp(genericOp, &operands))
      kind = xsmm::BinaryKind::ADD;
    else if (structured_match::utils::isTwoDMulOp(genericOp, &operands))
      kind = xsmm::BinaryKind::MUL;
    else if (structured_match::utils::isTwoDSubOp(genericOp, &operands))
      kind = xsmm::BinaryKind::SUB;

    if (kind == xsmm::BinaryKind::NONE || operands.size() != 3)
      return failure();
    return rewriteBinaryOp(rewriter, genericOp, operands, kind);
  }
};

// Replace linalgOp with a matmul or a batch reduce matmul.
static void replaceOpWithGemmLikeOp(RewriterBase &rewriter,
                                    linalg::LinalgOp linalgOp,
                                    BrgemmInfo brgemmInfo) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto m = brgemmInfo.m;
  auto n = brgemmInfo.n;
  auto k = brgemmInfo.k;
  auto batch = brgemmInfo.batch;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;

  auto dtype =
      xsmm::utils::getDataType(rewriter, linalgOp.getDpsInits()[0].getType());
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  Location loc = linalgOp.getLoc();
  xsmm::GemmFlagsAttr gemmFlags;
  if (brgemmInfo.isVnni) {
    gemmFlags = xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                         xsmm::GemmFlags::VNNI_B);
  } else {
    gemmFlags =
        xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE);
  }
  auto flags = rewriter.getArrayAttr(gemmFlags);
  SmallVector<Value> invokeOperands;
  SmallVector<Value> inputs = {linalgOp->getOperands()};

  // Collapse VNNI factor dimension for matrix A:
  // A <32x8x2> -> A <32x16>
  if (brgemmInfo.isVnni) {
    auto rankA = cast<ShapedType>(inputs[0].getType()).getRank();
    assert(rankA >= 3 && "Invalid A mat rank for VNNI");
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t index = 0; index < rankA - 2; index++)
      reassoc.push_back({index});
    reassoc.push_back(ReassociationIndices{rankA - 2, rankA - 1});

    inputs[0] =
        rewriter.create<memref::CollapseShapeOp>(loc, inputs[0], reassoc);
  }

  if (batch != 0) {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{m, n, k, lda, ldb, ldc, strideA, strideB});
    Value dispatched = rewriter.create<xsmm::BrgemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batch));
    invokeOperands.push_back(dispatched);
    invokeOperands.append(inputs);
    invokeOperands.push_back(batchDim);
    rewriter.replaceOpWithNewOp<xsmm::BrgemmOp>(linalgOp, dtype,
                                                invokeOperands);
  } else {
    DenseI64ArrayAttr dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), ArrayRef<int64_t>{m, n, k, lda, ldb, ldc});
    Value dispatched = rewriter.create<xsmm::GemmDispatchOp>(
        loc, integer64, dims, flags, dtype);
    invokeOperands.push_back(dispatched);
    invokeOperands.append(inputs);
    rewriter.replaceOpWithNewOp<xsmm::GemmOp>(linalgOp, dtype, invokeOperands);
  }
}

// Structural matcher.
static FailureOr<linalg::ContractionDimensions>
checkStructure(linalg::LinalgOp linalgOp) {
  // clang-format off
  using namespace structured_match;
  auto maybeBrgemmMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>()
      .output(MatchAll(), HasStaticShape())
      .input(MatchAll(), HasStaticShape())
      .output(MatchAll(), HasStaticStrides())
      .input(MatchAll(), HasStaticStrides())
      .operation(NumOfLoops(GreaterThanOrEqualTo(3)));
  // clang-format on
  if (!maybeBrgemmMatcher.match(linalgOp))
    return failure();

  auto contractionDims = linalgx::utils::isContraction(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Not a contraction\n");
    return failure();
  }
  if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1 ||
      contractionDims->k.size() > 3 || contractionDims->k.size() < 1 ||
      contractionDims->batch.size() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "[checkStructure] Wrong dimensions\n");
    return failure();
  }
  unsigned classifiedLoops =
      contractionDims->m.size() + contractionDims->n.size() +
      contractionDims->k.size() + contractionDims->batch.size();
  if (linalgOp.getNumLoops() != classifiedLoops) {
    LLVM_DEBUG(llvm::dbgs()
               << "[checkStructure] Not all loops are classified\n");
    return failure();
  }
  return contractionDims;
}

// Access matcher.
static FailureOr<BrgemmInfo> checkAccess(linalg::LinalgOp linalgOp, unsigned m,
                                         unsigned n, unsigned k,
                                         std::optional<unsigned> batchPos) {
  assert(linalgOp.getNumDpsInputs() == 2 && linalgOp.getNumDpsInits() == 1);
  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
  OpOperand *operandC = &linalgOp.getDpsInitsMutable()[0];

  auto checkStridesAndGetLda = [&](unsigned minorDim, unsigned majorDim,
                                   OpOperand *operand) -> FailureOr<int64_t> {
    auto minorDimPosInCodomain = getPosInCodomain(minorDim, operand, linalgOp);
    auto majorDimPosInCodomain = getPosInCodomain(majorDim, operand, linalgOp);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain)
      return failure();
    auto stridesOnOperand = utils::getStaticStrides(operand->get());
    if (failed(stridesOnOperand) ||
        (*stridesOnOperand)[*minorDimPosInCodomain] != 1)
      return failure();
    return (*stridesOnOperand)[*majorDimPosInCodomain];
  };

  // A(m, k)
  auto lda = checkStridesAndGetLda(k, m, operandA);
  if (failed(lda))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on A: OK\n");

  // B(k, n)
  auto ldb = checkStridesAndGetLda(n, k, operandB);
  if (failed(ldb))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on B: OK\n");

  // C(m, n)
  auto ldc = checkStridesAndGetLda(n, m, operandC);
  if (failed(ldc))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Strides on C: OK\n");

  int64_t strideA = 1;
  int64_t strideB = 1;
  if (batchPos) {
    auto batchPosCodomainA =
        getPosInCodomain(batchPos.value(), operandA, linalgOp);
    auto stridesOnA = utils::getStaticStrides(operandA->get());
    strideA = (*stridesOnA)[*batchPosCodomainA];

    auto batchPosCodomainB =
        getPosInCodomain(batchPos.value(), operandB, linalgOp);
    auto stridesOnB = utils::getStaticStrides(operandB->get());
    strideB = (*stridesOnB)[*batchPosCodomainB];
  }

  auto loops = linalgOp.getStaticLoopRanges();
  int64_t batchVal = (batchPos) ? loops[batchPos.value()] : 0;

  bool isVnni = vnni::utils::isInVnniLayout(linalgOp);

  BrgemmInfo info{loops[m], loops[n], loops[k], batchVal, *lda,
                  *ldb,     *ldc,     strideA,  strideB,  isVnni};
  return info;
}

// Check if the given generic is mappable to a brgemm xsmm op.
// - It is a contraction, with:
// -- 1 m, 1 n, and 2 or 3 (VNNI) k dimensions.
// -- m appears on the LHS and OUT but not in RHS.
// -- n appears on the RHS and OUT but not in LHS.
// -- k and k' appear on the RHS and LHS but not OUT.
// -- the stride of the minor dimension for A, k is 1.
// -- the stride of the minor dimension for B, n is 1.
// -- the stride of the minor dimension for C, n is 1.
static FailureOr<BrgemmInfo> isMappableToBrgemm(linalg::LinalgOp linalgOp) {
  auto contractionDims = checkStructure(linalgOp);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[isMappableToBrgemm] Failed on checkStructure\n");
    return failure();
  }

  unsigned m = contractionDims->m[0];
  unsigned n = contractionDims->n[0];
  unsigned k = contractionDims->k.back();

  // Check if there is a batch reduce dimension.
  // At least one K-dim is the GEMM reduction.
  // In case of VNNI layout, there is additional reduction dimension
  // representing VNNI blocking factor.
  std::optional<unsigned> batch;
  unsigned numBrgemmReductionDims =
      vnni::utils::isInVnniLayout(linalgOp) ? 3 : 2;
  if (contractionDims->k.size() == numBrgemmReductionDims)
    batch = contractionDims->k.front();

  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] Candidate dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] m: " << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] n: " << n << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] k: " << k << "\n");
  if (batch)
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] batch: " << batch << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrgemm] no batch dim\n");

  return checkAccess(linalgOp, m, n, k, batch);
}

// Check if we can map `genericOp` to a BRGEMM and rewrite it to XSMM brgemm op.
struct ConvertGenericToBrgemm : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto brgemmInfo = isMappableToBrgemm(genericOp);
    if (failed(brgemmInfo))
      return failure();
    replaceOpWithGemmLikeOp(rewriter, genericOp, *brgemmInfo);
    return success();
  }
};

// Emit a transpose operation for `operand` by swapping `dim` with `newDim`.
// Emit a transpose operation for `operand` by swapping the dimensions at index
// `dim` with `newDim`.
static void emitTransposeOnOperand(RewriterBase &rewriter,
                                   linalg::GenericOp linalgOp,
                                   OpOperand *operand, unsigned dim,
                                   unsigned newDim) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(linalgOp);

  Location loc = linalgOp.getLoc();
  auto operandType = cast<ShapedType>(operand->get().getType());
  auto rank = operandType.getRank();
  SmallVector<int64_t> shape = llvm::to_vector(operandType.getShape());
  auto permutation = llvm::to_vector(llvm::seq<int64_t>(0, rank));
  std::swap(permutation[dim], permutation[newDim]);
  assert(isPermutationVector(permutation));
  LLVM_DEBUG(llvm::interleaveComma(
      permutation, llvm::dbgs() << "[emitTransposeOnOperand] Perm: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  applyPermutationToVector<int64_t>(shape, permutation);
  Value buffer;
  if (linalgOp.hasPureTensorSemantics()) {
    buffer = rewriter.create<tensor::EmptyOp>(loc, shape,
                                              operandType.getElementType());
    buffer = rewriter
                 .create<linalg::TransposeOp>(loc, operand->get(), buffer,
                                              permutation)
                 .getResults()[0];
  } else {
    buffer = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(shape, operandType.getElementType()));
    rewriter.create<linalg::TransposeOp>(loc, operand->get(), buffer,
                                         permutation);
  }

  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  AffineMap operandMap = indexingMaps[operand->getOperandNumber()];
  LLVM_DEBUG(llvm::dbgs() << "[emitTransposeOnOperand] Old map: " << operandMap
                          << "\n");
  SmallVector<AffineExpr> mapResults = llvm::to_vector(operandMap.getResults());
  applyPermutationToVector<AffineExpr>(mapResults, permutation);
  AffineMap newMap =
      AffineMap::get(operandMap.getNumDims(), operandMap.getNumSymbols(),
                     mapResults, linalgOp.getContext());
  LLVM_DEBUG(llvm::dbgs() << "[emitTransposeOnOperand] New map: " << newMap
                          << "\n");
  indexingMaps[operand->getOperandNumber()] = newMap;
  // TODO: We probably cannot update the result in place.
  rewriter.modifyOpInPlace(linalgOp, [&]() {
    linalgOp->setOperand(operand->getOperandNumber(), buffer);
    linalgOp.setIndexingMapsAttr(
        ArrayAttr::get(linalgOp.getContext(),
                       llvm::to_vector(llvm::map_range(
                           indexingMaps, [](AffineMap map) -> Attribute {
                             return AffineMapAttr::get(map);
                           }))));
  });
  if (linalgOp.hasPureBufferSemantics()) {
    rewriter.setInsertionPointAfter(linalgOp);
    rewriter.create<memref::DeallocOp>(linalgOp.getLoc(), buffer);
  }
}

static bool isInnerMostDim(OpOperand *operand, unsigned minorDim) {
  auto shapedType = cast<ShapedType>(operand->get().getType());
  unsigned rank = shapedType.getRank();
  return minorDim == rank - 1;
}

static FailureOr<linalg::GenericOp>
makeMinorDimensionsInnerMost(RewriterBase &rewriter, linalg::GenericOp linalgOp,
                             unsigned m, unsigned n, unsigned k) {
  assert(linalgOp.getNumDpsInputs() == 2 && linalgOp.getNumDpsInits() == 1);
  OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
  OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
  OpOperand &operandC = linalgOp.getDpsInitsMutable()[0];

  // C(m,n) += A(m,k) * B(k,n)
  // n is expected to be the innermost for C
  // k is expected to be the innermost for A
  // n is expected to be the innermost for B
  auto minorKInCodomainOpA = getPosInCodomain(k, operandA, linalgOp);
  auto minorMInCodomainOpA = getPosInCodomain(m, operandA, linalgOp);
  if (!minorKInCodomainOpA || !minorMInCodomainOpA) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for A\n");
    return failure();
  }

  auto minorNInCodomainOpB = getPosInCodomain(n, operandB, linalgOp);
  auto minorKInCodomainOpB = getPosInCodomain(k, operandB, linalgOp);
  if (!minorNInCodomainOpB || !minorKInCodomainOpB) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for B\n");
    return failure();
  }

  auto minorNInCodomainOpC = getPosInCodomain(n, &operandC, linalgOp);
  auto minorMInCodomainOpC = getPosInCodomain(m, &operandC, linalgOp);
  if (!minorNInCodomainOpC || !minorMInCodomainOpC) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for C\n");
    return failure();
  }

  if (!isInnerMostDim(&operandC, *minorNInCodomainOpC)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for C\n");
    assert(isInnerMostDim(&operandC, *minorMInCodomainOpC));
    if (isInnerMostDim(operandA, *minorKInCodomainOpA)) {
      emitTransposeOnOperand(rewriter, linalgOp, operandA, *minorKInCodomainOpA,
                             *minorMInCodomainOpA);
    }
    if (isInnerMostDim(operandB, *minorNInCodomainOpB)) {
      emitTransposeOnOperand(rewriter, linalgOp, operandB, *minorNInCodomainOpB,
                             *minorKInCodomainOpB);
    }
    // Avoid transpose on the output by swapping A and B.
    OpOperand *operandA = linalgOp.getDpsInputOperands()[0];
    OpOperand *operandB = linalgOp.getDpsInputOperands()[1];
    SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
    std::swap(indexingMaps[0], indexingMaps[1]);
    rewriter.modifyOpInPlace(linalgOp, [&]() {
      Value operandATmp = operandA->get();
      linalgOp->setOperand(operandA->getOperandNumber(), operandB->get());
      linalgOp->setOperand(operandB->getOperandNumber(), operandATmp);
      linalgOp.setIndexingMapsAttr(
          ArrayAttr::get(linalgOp.getContext(),
                         llvm::to_vector(llvm::map_range(
                             indexingMaps, [](AffineMap map) -> Attribute {
                               return AffineMapAttr::get(map);
                             }))));
    });
    return linalgOp;
  }

  bool isVnni = vnni::utils::isInVnniLayout(linalgOp);

  if (!isVnni && !isInnerMostDim(operandA, *minorKInCodomainOpA)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for A\n");
    if (!isInnerMostDim(operandA, *minorMInCodomainOpA))
      return failure();
    emitTransposeOnOperand(rewriter, linalgOp, operandA, *minorKInCodomainOpA,
                           *minorMInCodomainOpA);
  }
  // Do not inject transposes in case of VNNI format.
  // Otherwise, it breaks later VNNI layout validation.
  if (!isVnni && !isInnerMostDim(operandB, *minorNInCodomainOpB)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for B\n");
    if (!isInnerMostDim(operandB, *minorKInCodomainOpB))
      return failure();
    emitTransposeOnOperand(rewriter, linalgOp, operandB, *minorKInCodomainOpB,
                           *minorNInCodomainOpB);
  }
  return linalgOp;
}

void ConvertLinalgToXsmm::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  IRRewriter rewriter(&getContext());

  // Enable conversion for linalg.generic to XSMM Brgemm if possible.
  getOperation()->walk([&](linalg::GenericOp genericOp) {
    auto contractionDims = checkStructure(genericOp);
    // If the generic does not match the structure of a Brgemm op, skip it.
    if (failed(contractionDims))
      return WalkResult::skip();
    unsigned m = contractionDims->m[0];
    unsigned n = contractionDims->n[0];
    unsigned k = contractionDims->k.back();
    std::optional<unsigned> batch;
    if (contractionDims->k.size() == 3)
      contractionDims->k.front();

    if (failed(checkAccess(genericOp, m, n, k, batch))) {
      // The generic is a Brgemm but the strides of the selected dims (m, n, k)
      // are not unit strides. Inject transposes to bring them innermost.
      if (failed(makeMinorDimensionsInnerMost(rewriter, genericOp, m, n, k))) {
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });
  SmallVector<StringRef> skipPatterns(skipOperations.begin(),
                                      skipOperations.end());
  tpp::populateLinalgToXsmmPatterns(patterns, skipPatterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

// Set the beta flags of a gemm dispatch to zero by cloning and updating the
// clone.
template <typename XsmmDisTy, typename XsmmTy>
static void updateGemmOpFlags(RewriterBase &rewriter, XsmmDisTy gemmDispatchOp,
                              XsmmTy gemmOp) {
  static_assert(
      llvm::is_one_of<XsmmDisTy, xsmm::GemmDispatchOp, xsmm::BrgemmDispatchOp,
                      xsmm::FusedBrgemmDispatchOp>::value);
  static_assert(llvm::is_one_of<XsmmTy, xsmm::GemmOp, xsmm::BrgemmOp,
                                xsmm::FusedBrgemmOp>::value);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(gemmDispatchOp);

  auto clonedOp =
      cast<XsmmDisTy>(rewriter.clone(*gemmDispatchOp.getOperation()));
  rewriter.modifyOpInPlace(clonedOp, [&]() {
    ArrayAttr flags = gemmDispatchOp.getFlags();
    SmallVector<Attribute> newFlags;
    for (auto flag : flags) {
      if (auto gemmFlag = dyn_cast<xsmm::GemmFlagsAttr>(flag)) {
        if ((gemmFlag.getValue() == xsmm::GemmFlags::NONE) ||
            (gemmFlag.getValue() == xsmm::GemmFlags::BETA_0)) {
          continue;
        }
      }
      newFlags.push_back(flag);
    }
    newFlags.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                                xsmm::GemmFlags::BETA_0));
    clonedOp.setFlagsAttr(rewriter.getArrayAttr(newFlags));
  });
  rewriter.replaceUsesWithIf(
      gemmDispatchOp->getResults(), clonedOp->getResults(),
      [&](OpOperand &operand) { return operand.getOwner() == gemmOp; });
}

// Given `rootOp` return the first gemm-like operation that is zero initialized
// by `rootOp`.
static std::optional<Operation *> getZeroInitGemmLikeOp(xsmm::UnaryOp rootOp) {
  // Walk the bb and make sure there are only side-effect free operations
  // between the zero op and the gemm. Bail out if any operations take a subview
  // from `dest`.
  Value dest = rootOp.getInputs().back();
  DenseSet<Operation *> destUsers(dest.getUsers().begin(),
                                  dest.getUsers().end());

  Block *blck = nullptr;
  if (auto bbArg = dyn_cast<BlockArgument>(dest)) {
    blck = bbArg.getOwner();
  } else {
    Operation *defOp = dest.getDefiningOp();
    if (!defOp)
      return std::nullopt;
    blck = defOp->getBlock();
  }
  assert(blck && "must be a valid ptr");
  auto it = blck->begin();
  auto itEnd = blck->end();
  while (it != itEnd && &*it != rootOp.getOperation()) {
    // View may introduce aliasing.
    if (auto view = dyn_cast<ViewLikeOpInterface>(&*it)) {
      if (view.getViewSource() == dest)
        return std::nullopt;
    }
    it++;
  }

  if (it == itEnd)
    return std::nullopt;

  while (++it != itEnd) {
    // Skip operations that do not touch `dest`.
    if (!destUsers.count(&*it))
      continue;
    // No memory effects other than read.
    if (mlir::hasSingleEffect<MemoryEffects::Read>(&*it, dest))
      continue;
    // View may introduce aliasing.
    if (auto view = dyn_cast<ViewLikeOpInterface>(&*it)) {
      if (view.getViewSource() == dest)
        return std::nullopt;
    }
    // A gemm or brgemm operation touching `dest`, fold if the
    // output (i.e. C matrix) is `dest`.
    if (auto gemmOp = dyn_cast<xsmm::GemmOp>(*it)) {
      Value outVal = gemmOp.getOutput();
      if (outVal == dest)
        break;
    }
    if (auto brgemmOp = dyn_cast<xsmm::BrgemmOp>(*it)) {
      Value outVal = brgemmOp.getOutput();
      if (outVal == dest)
        break;
    }
    if (auto fusedBrgemmOp = dyn_cast<xsmm::FusedBrgemmOp>(*it)) {
      Value outVal = fusedBrgemmOp.getOutput();
      if (outVal == dest)
        break;
    }
    // Fail.
    return std::nullopt;
  }
  if (it == itEnd)
    return std::nullopt;
  return &*it;
}

static void fuseZeroWithGemmOrBrgemm(RewriterBase &rewriter,
                                     xsmm::UnaryOp rootOp) {
  LLVM_DEBUG(llvm::dbgs() << "[fuseZeroWithGemmOrBrgemm] Candidate op: "
                          << rootOp << "\n");
  // 1. Check if we have a gemm zero initialized by rootOp.
  auto gemmLikeOp = getZeroInitGemmLikeOp(rootOp);
  if (!gemmLikeOp)
    return;

  LLVM_DEBUG(llvm::dbgs() << "[fuseZeroWithGemmOrBrgemm] Candidate op OK: "
                          << rootOp << "\n");

  // 2. Update flags.
  assert(isa<xsmm::GemmOp>(*gemmLikeOp) || isa<xsmm::BrgemmOp>(*gemmLikeOp) ||
         isa<xsmm::FusedBrgemmOp>(*gemmLikeOp));
  if (auto gemmOp = dyn_cast<xsmm::GemmOp>(*gemmLikeOp)) {
    xsmm::GemmDispatchOp gemmDispatchOp =
        cast<xsmm::GemmDispatchOp>(gemmOp.getInputs()[0].getDefiningOp());
    updateGemmOpFlags(rewriter, gemmDispatchOp, gemmOp);
  } else if (auto brgemmOp = dyn_cast<xsmm::BrgemmOp>(*gemmLikeOp)) {
    xsmm::BrgemmDispatchOp brgemmDispatchOp =
        cast<xsmm::BrgemmDispatchOp>(brgemmOp.getInputs()[0].getDefiningOp());
    updateGemmOpFlags(rewriter, brgemmDispatchOp, brgemmOp);
  } else {
    auto fusedBrgemm = cast<xsmm::FusedBrgemmOp>(*gemmLikeOp);
    xsmm::FusedBrgemmDispatchOp fusedBrgemmDispatchOp =
        cast<xsmm::FusedBrgemmDispatchOp>(
            fusedBrgemm.getInputs()[0].getDefiningOp());
    updateGemmOpFlags(rewriter, fusedBrgemmDispatchOp, fusedBrgemm);
  }
  rewriter.eraseOp(rootOp);
}

void FoldXsmmFlags::runOnOperation() {
  SmallVector<xsmm::UnaryOp> producers;
  IRRewriter rewriter(&getContext());
  getOperation()->walk([&](xsmm::UnaryOp unaryOp) {
    auto kind = unaryOp.getCallee();
    if (kind == xsmm::UnaryKind::ZERO)
      fuseZeroWithGemmOrBrgemm(rewriter, unaryOp);
  });
}

// Convert a linalg.matmul to a XSMM matmul op.
struct ConvertMatmulToMatmul : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto gemmInfo = isMappableToBrgemm(matmulOp);
    if (failed(gemmInfo))
      return failure();
    replaceOpWithGemmLikeOp(rewriter, matmulOp, *gemmInfo);
    return success();
  }
};

// Convert a linalg.batch_reduce_matmul to a XSMM brgemm op.
struct ConvertBatchReduceMatmulToBatchReduceMatmul
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp batchReduceOp,
                                PatternRewriter &rewriter) const override {
    auto brgemmInfo = isMappableToBrgemm(batchReduceOp);
    if (failed(brgemmInfo))
      return failure();
    replaceOpWithGemmLikeOp(rewriter, batchReduceOp, *brgemmInfo);
    return success();
  }
};

// Convert a vnni pack to xsmm norm to vnni op. It assumes the pack to be
// decomposed as an expand.shape + linalg.transpose.
struct ConvertVnniPacking : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    if (!transposeOp.hasPureBufferSemantics())
      return failure();

    Value out = transposeOp.getInit();
    Value source = transposeOp.getInput();
    MemRefType outType = cast<MemRefType>(out.getType());
    MemRefType sourceType = cast<MemRefType>(source.getType());
    if (!outType.hasStaticShape() || !sourceType.hasStaticShape() ||
        !vnni::utils::isInVnniLayout(vnni::utils::VnniOperandRank::TRANSPOSE,
                                     outType)) {
      return failure();
    }

    memref::ExpandShapeOp expandShapeOp =
        dyn_cast<memref::ExpandShapeOp>(source.getDefiningOp());
    if (!expandShapeOp || expandShapeOp.getSrcType().getRank() != 2)
      return failure();

    source = expandShapeOp.getSrc();
    xsmm::UnaryInfo unaryInfo;
    unaryInfo.m = expandShapeOp.getSrcType().getShape()[0];
    unaryInfo.n = expandShapeOp.getSrcType().getShape()[1];
    auto stridesOnInput = mlir::utils::getStaticStrides(source);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1)
      return failure();
    unaryInfo.ldi = stridesOnInput->front();
    auto stridesOnOutput = mlir::utils::getStaticStrides(out);
    if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
      return failure();
    // Ajust ldo based on the VNNI factor.
    auto vnniFactor =
        vnni::utils::getVnniBlockingFactor(out.getType(), transposeOp);
    assert(vnniFactor && "Failed to get VNNI blocking factor");
    unaryInfo.ldo = stridesOnOutput->front() / vnniFactor;
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind =
        xsmm::UnaryKindAttr::get(rewriter.getContext(), xsmm::UnaryKind::VNNI2);
    xsmm::utils::replaceOpWithUnary(rewriter, transposeOp, {source, out},
                                    unaryInfo, flags, kind);
    return success();
  }
};

// Converts linalg.generic with the following layout:
// [i][j] = [i][k] [k/VNNI][j][VNNI] -> xsmm.matmul
// [i][j] = [b][i][k] [b][k/VNNI][j][VNNI] -> xsmm.brgemm
struct ConvertGenericToVnniMatmulLikeOp
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(genericOp, "expects buffer semantics");
    }

    auto [isMatmulVnni, hasBatch] = structured_match::utils::isMatmulVnniOp(
        genericOp, /*operands=*/nullptr);
    if (!isMatmulVnni) {
      return rewriter.notifyMatchFailure(
          genericOp, "expects an operation mappable to VNNI contraction");
    }

    Value bufferA = genericOp.getDpsInputs()[0];
    Value bufferB = genericOp.getDpsInputs()[1];
    Value bufferC = genericOp.getDpsInits()[0];

    int64_t m = cast<ShapedType>(bufferC.getType()).getShape()[0];
    int64_t n = cast<ShapedType>(bufferC.getType()).getShape()[1];
    int64_t kPos = 1;
    if (hasBatch)
      kPos++;
    // Take the whole reduction dim size. Account for the VNNI factor (ensured
    // by the earlier check) that splits the K dim in the shape.
    std::optional<int64_t> vnniFactor =
        vnni::utils::getVnniBlockingFactor(bufferB.getType(), genericOp);
    if (!vnniFactor)
      return rewriter.notifyMatchFailure(genericOp,
                                         "failed to determine VNNI factor");
    int64_t k =
        cast<ShapedType>(bufferA.getType()).getShape()[kPos] * *vnniFactor;
    int64_t batch = 0;
    if (hasBatch)
      batch = cast<ShapedType>(bufferA.getType()).getShape()[0];

    auto stridesOnLhs = utils::getStaticStrides(bufferA);
    auto stridesOnRhs = utils::getStaticStrides(bufferB);
    auto stridesOnOutput = utils::getStaticStrides(bufferC);
    if (failed(stridesOnLhs) || failed(stridesOnRhs) ||
        failed(stridesOnOutput)) {
      return rewriter.notifyMatchFailure(genericOp, "expects static strides");
    }
    if (stridesOnLhs->back() != 1 || stridesOnRhs->back() != 1 ||
        stridesOnOutput->back() != 1) {
      return rewriter.notifyMatchFailure(
          genericOp, "expect stride 1 in the fastest-varying dimension");
    }

    int64_t leadingDimPosOnAandB = 0;
    if (hasBatch)
      leadingDimPosOnAandB++;
    int64_t lda = (*stridesOnLhs)[leadingDimPosOnAandB];
    int64_t ldb = (*stridesOnRhs)[leadingDimPosOnAandB] / *vnniFactor;
    int64_t ldc = (*stridesOnOutput)[0];

    BrgemmInfo brgemmInfo{m,   n,   k,       batch,   lda,
                          ldb, ldc, lda * m, ldb * k, /*isVnni=*/true};
    replaceOpWithGemmLikeOp(rewriter, genericOp, brgemmInfo);
    return success();
  }
};

struct ConvertCopyOp : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    if (!copyOp.hasPureBufferSemantics())
      return failure();
    Value source = copyOp.getInputs()[0];
    Value dest = copyOp.getOutputs()[0];
    auto unaryInfo =
        xsmm::utils::getUnaryInfo(source, dest, xsmm::UnaryFlags::NONE);
    if (failed(unaryInfo))
      return failure();
    auto flags = rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
        rewriter.getContext(), xsmm::UnaryFlags::NONE));
    xsmm::UnaryKindAttr kind = xsmm::UnaryKindAttr::get(
        rewriter.getContext(), xsmm::UnaryKind::IDENTITY);
    SmallVector<Value> operands{source, dest};
    xsmm::utils::replaceOpWithUnary(rewriter, copyOp, operands, *unaryInfo,
                                    flags, kind);
    return success();
  }
};

} // namespace

void mlir::tpp::populateLinalgToXsmmPatterns(
    RewritePatternSet &patterns, ArrayRef<StringRef> skipPatterns) {
  std::vector<StringRef> patternsToAdd = {"fill",   "transpose", "unary",
                                          "binary", "brgemm",    "matmul",
                                          "copy",   "vnni"};
  // If skipping all patterns, just don't do anything.
  if (skipPatterns.size() == 1 && skipPatterns[0] == "all") {
    LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] ignoring all patterns\n");
    return;
  }

  // If skip list is not empty, remove all that were listed.
  // This is O(n^2), but the lists are small
  if (!skipPatterns.empty()) {
    assert(skipPatterns.size() <= 8);
    auto newEnd = std::remove_if(patternsToAdd.begin(), patternsToAdd.end(),
                   [&skipPatterns](StringRef elm) -> bool {
                     return std::find(skipPatterns.begin(),
                                      skipPatterns.end(),
                                      elm) != skipPatterns.end();
                   });
    if (newEnd != patternsToAdd.end())
      patternsToAdd.erase(newEnd, patternsToAdd.end());
  }

  // Now, add all the remaining patterns
  auto ctx = patterns.getContext();
  for (auto pattern : patternsToAdd) {
    if (pattern == "fill") {
      patterns.add<ConvertFillOpToUnaryZero>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] adding fill\n");
    } else if (pattern == "transpose") {
      patterns.add<ConvertTransposeOpToUnaryTranspose>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] adding transpose\n");
    } else if (pattern == "copy") {
      patterns.add<ConvertCopyOp>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] adding copy\n");
    } else if (pattern == "unary") {
      patterns.add<ConvertGenericToUnary>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] adding unary\n");
    } else if (pattern == "binary") {
      patterns.add<ConvertGenericToBinary>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] adding binary\n");
    } else if (pattern == "brgemm") {
      patterns.add<ConvertGenericToBrgemm,
                   ConvertBatchReduceMatmulToBatchReduceMatmul>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] adding brgem\n");
    } else if (pattern == "matmul") {
      patterns.add<ConvertMatmulToMatmul>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] adding matmul\n");
    } else if (pattern == "vnni") {
      patterns.add<ConvertVnniPacking, ConvertGenericToVnniMatmulLikeOp>(ctx);
      LLVM_DEBUG(llvm::dbgs() << "[LinalgToXsmm] adding vnni\n");
    }
  }
}
