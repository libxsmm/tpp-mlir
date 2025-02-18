//===CombineXsmmPass.cpp --------------------------------------*----C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. / See https://llvm.org/LICENSE.txt for license information. /
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_COMBINEXSMMOPPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;

namespace {

struct CombineXsmmOp : public OpRewritePattern<xsmm::BrgemmOp> {

  using OpRewritePattern<xsmm::BrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmm::BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    auto *output = brgemmOp.getOperand(3).getDefiningOp();
    if (!output)
      return failure();

    // First, match the required fused ops
    auto result = xsmm::utils::getFusedBrgemmSequenceFromProducer(output);
    if (failed(result))
      return failure();
    auto fusedMatch = *result;
    // TODO: Support BRGEMM + BINARY && BRGEMM + UNARY patterns
    if (!fusedMatch.binaryOp || !fusedMatch.unaryOp)
      return failure();
    // Validate broadcast flags
    auto unaryFlags =
        xsmm::utils::getUnaryFlags(fusedMatch.unaryOp.getOperand(0).getType(),
                                   fusedMatch.unaryOp.getOperand(2).getType());
    if (unaryFlags != mlir::xsmm::UnaryFlags::BCAST_SCALAR &&
        unaryFlags != mlir::xsmm::UnaryFlags::NONE)
      return failure();

    // TODO: Support more than just COL_0 BCAST
    auto binaryFlags =
        xsmm::utils::getBinaryFlags(fusedMatch.binaryOp.getOperand(1).getType(),
                                    fusedMatch.binaryOp.getOperand(3).getType(),
                                    mlir::xsmm::utils::OperandPos::LHS);
    int binaryArg = 0;
    switch (*binaryFlags) {
    case mlir::xsmm::BinaryFlags::BCAST_COL_IN_0:
      binaryArg = 1;
      break;
    case mlir::xsmm::BinaryFlags::BCAST_COL_IN_1:
      binaryArg = 2;
      binaryFlags = mlir::xsmm::BinaryFlags::BCAST_COL_IN_0;
      break;
    default:
      return failure();
    }
    // Now, replace the ops with a fused BRGEMM
    auto dtype =
        xsmm::utils::getDataType(rewriter, brgemmOp.getOperand(1).getType());
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    Location loc = brgemmOp.getLoc();
    auto dims = DenseI64ArrayAttr::get(
        rewriter.getContext(), dyn_cast<mlir::xsmm::BrgemmDispatchOp>(
                                   brgemmOp.getOperand(0).getDefiningOp())
                                   .getInputs());
    auto memrefB = brgemmOp.getOperand(2);
    int64_t batchSize = cast<ShapedType>(memrefB.getType()).getShape()[0];
    auto brgemmFlags = xsmm::utils::getBrgemmFlags<xsmm::BrgemmDispatchOp>(
        rewriter,
        dyn_cast<xsmm::BrgemmDispatchOp>(
            brgemmOp.getOperand(0).getDefiningOp()),
        true);
    if (failed(brgemmFlags))
      return failure();
    auto attributes = *brgemmFlags;
    if (fusedMatch.zeroOp) {
      if (attributes[0] == xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                                    xsmm::GemmFlags::NONE)) {
        attributes.clear();
      }
      attributes.push_back(xsmm::GemmFlagsAttr::get(rewriter.getContext(),
                                                    xsmm::GemmFlags::BETA_0));
    }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(fusedMatch.binaryOp);
    Value dispatched = rewriter.create<xsmm::FusedBrgemmDispatchOp>(
        loc, integer64, dims,
        xsmm::BinaryKindAttr::get(rewriter.getContext(), fusedMatch.binaryKind),
        xsmm::UnaryKindAttr::get(rewriter.getContext(), fusedMatch.unaryKind),
        rewriter.getArrayAttr(attributes),
        rewriter.getArrayAttr(xsmm::UnaryFlagsAttr::get(
            rewriter.getContext(), xsmm::UnaryFlags::NONE)),
        rewriter.getArrayAttr(
            xsmm::BinaryFlagsAttr::get(rewriter.getContext(), *binaryFlags)),
        dtype);

    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batchSize));
    SmallVector<Value, 6> invokeOperands;
    invokeOperands.push_back(dispatched);
    auto opItr = brgemmOp->getOperands().begin();
    // Skipping dispatch operand
    std::advance(opItr, 1);
    invokeOperands.append(opItr, brgemmOp->getOperands().end());
    invokeOperands.pop_back();
    invokeOperands.push_back(fusedMatch.binaryOp->getOperand(binaryArg));
    invokeOperands.push_back(batchDim);

    // Replace and delete the old invokes and their dispatches
    rewriter.create<xsmm::FusedBrgemmOp>(loc, dtype, invokeOperands);
    rewriter.eraseOp(brgemmOp);
    rewriter.eraseOp(brgemmOp.getOperand(0).getDefiningOp());
    if (fusedMatch.binaryOp) {
      rewriter.eraseOp(fusedMatch.binaryOp);
      rewriter.eraseOp(fusedMatch.binaryOp->getOperand(0).getDefiningOp());
    }
    if (fusedMatch.unaryOp) {
      rewriter.eraseOp(fusedMatch.unaryOp);
      rewriter.eraseOp(fusedMatch.unaryOp->getOperand(0).getDefiningOp());
    }
    if (fusedMatch.zeroOp) {
      rewriter.eraseOp(fusedMatch.zeroOp);
      rewriter.eraseOp(fusedMatch.zeroOp->getOperand(0).getDefiningOp());
    }
    return success();
  }
};

void populateCombinePatterns(RewritePatternSet &patterns) {
  patterns.add<CombineXsmmOp>(patterns.getContext());
}

struct CombineXsmmOpPass
    : public tpp::impl::CombineXsmmOpPassBase<CombineXsmmOpPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace
