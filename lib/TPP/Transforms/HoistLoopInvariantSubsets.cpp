//===-HoistLoopInvariantSubsets.cpp -------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements tile configuration hoisting on parallel loops.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_HOISTLOOPINVARIANTSUBSETS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

struct HoistLISubsetOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    Operation *current = contractOp;
    hoistLoopInvariantSubsets(rewriter, current->getParentOfType<scf::ForOp>());
    hoistLoopInvariantSubsets(rewriter, current->getParentOfType<scf::ForOp>()->getParentOfType<scf::ForOp>());

    return success();
  }
};

void populateHoistLoopInvariantSubsetPatterns(RewritePatternSet &patterns) {
  patterns.add<HoistLISubsetOp>(patterns.getContext());
}

struct HoistLoopInvariantSubsets
    : public impl::HoistLoopInvariantSubsetsBase<HoistLoopInvariantSubsets> {
  using HoistLoopInvariantSubsetsBase::HoistLoopInvariantSubsetsBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateHoistLoopInvariantSubsetPatterns(patterns);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
