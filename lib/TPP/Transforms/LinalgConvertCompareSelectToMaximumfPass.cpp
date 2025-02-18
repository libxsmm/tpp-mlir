//===-LinalgConvertCompareSelectToMaximumfPass.cpp ---------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file lowers Compare select generic to maximumf generic
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGCONVERTCOMPARESELECTTOMAXIMUMFPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace tpp {

struct LinalgConvertCompareSelectToMaximumf
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getBody()->getOperations().size() != 3)
      return failure();
    auto cmpf = dyn_cast<arith::CmpFOp>(&op.getBody()->getOperations().front());
    if (!cmpf || cmpf.getPredicate() != arith::CmpFPredicate::UGT)
      return failure();

    if (!matchPattern(cmpf.getOperand(1), m_AnyZeroFloat()))
      return failure();
    auto select = dyn_cast<arith::SelectOp>(
        std::next(op.getBody()->getOperations().begin(), 1));
    if (!select)
      return failure();
    if (select.getOperand(0) != cmpf.getResult() ||
        select.getOperand(1) != cmpf.getOperand(0))
      return failure();
    rewriter.setInsertionPointAfter(&op.getBody()->front());
    auto maxf = rewriter.create<arith::MaximumFOp>(
        op.getLoc(),
        dyn_cast<arith::CmpFOp>(op.getBody()->getOperations().begin())
            ->getOperands());
    dyn_cast<YieldOp>(op.getBody()->getTerminator()).setOperand(0, maxf);

    // Deletion in reverse order due to dependences
    rewriter.eraseOp(select);
    rewriter.eraseOp(cmpf);
    return success();
  }
};

struct LinalgConvertCompareSelectToMaximumfPass
    : public impl::LinalgConvertCompareSelectToMaximumfPassBase<
          LinalgConvertCompareSelectToMaximumfPass> {
  void populateCombinePatterns(RewritePatternSet &patterns) {
    patterns.add<LinalgConvertCompareSelectToMaximumf>(patterns.getContext());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace tpp
} // namespace mlir
