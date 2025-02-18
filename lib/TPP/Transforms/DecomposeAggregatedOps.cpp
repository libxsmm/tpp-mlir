//===- DecomposeAggregatedOps.cpp --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_DECOMPOSEAGGREGATEDOPS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct DecomposeAggregateOpsImpl : public OpRewritePattern<linalg::SoftmaxOp> {
  using OpRewritePattern<linalg::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::SoftmaxOp softmaxOp,
                                PatternRewriter &rewriter) const override {
    auto decomposableOp =
        cast<linalg::AggregatedOpInterface>(softmaxOp.getOperation());
    FailureOr<SmallVector<Value>> maybeNewResult =
        decomposableOp.decomposeOperation(rewriter);
    if (failed(maybeNewResult))
      return failure();
    rewriter.replaceOp(softmaxOp, *maybeNewResult);
    return success();
  }
};

struct DecomposeAggregatedOps
    : public tpp::impl::DecomposeAggregatedOpsBase<DecomposeAggregatedOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<DecomposeAggregateOpsImpl>(patterns.getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
