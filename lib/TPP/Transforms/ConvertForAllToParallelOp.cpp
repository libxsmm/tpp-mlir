//===- ConvertForAllToParallelOp.cpp -----------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTFORALLTOPARALLELOP
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Wrapper for upstream transform API.
struct ConvertForAllToParallelOpImpl : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    return scf::forallToParallelLoop(rewriter, forallOp);
  }
};

struct ConvertForAllToParallelOp
    : public tpp::impl::ConvertForAllToParallelOpBase<
          ConvertForAllToParallelOp> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<ConvertForAllToParallelOpImpl>(patterns.getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace
