//===- IntelAMXTileConfigHoisting.cpp ----------------------------*- C++-*-===//
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
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_INTELAMXTILECONFIGHOISTINGPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::xsmm;

namespace mlir {
namespace tpp {

struct IntelAMXTileConfigHoisting : OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp alloca,
                                PatternRewriter &rewriter) const override {

    xsmm::IntelAMXTileConfigOp firstTileConfig, secondTileConfig;
    for (auto *user : alloca->getUsers()) {
      if (!dyn_cast<xsmm::IntelAMXTileConfigOp>(user)) {
        return failure();
      }
      auto flags = dyn_cast<xsmm::IntelAMXTileConfigDispatchOp>(
                       dyn_cast<xsmm::IntelAMXTileConfigOp>(user)
                           .getOperand(0)
                           .getDefiningOp())
                       .getFlags();
      for (auto flagItr : flags) {
        if (flagItr == xsmm::GemmFlagsAttr::get(
                           rewriter.getContext(),
                           mlir::xsmm::GemmFlags::NO_RESET_TILECONFIG)) {
          firstTileConfig = dyn_cast<xsmm::IntelAMXTileConfigOp>(user);

        } else if (flagItr == xsmm::GemmFlagsAttr::get(
                                  rewriter.getContext(),
                                  mlir::xsmm::GemmFlags::NO_SETUP_TILECONFIG)) {
          secondTileConfig = dyn_cast<xsmm::IntelAMXTileConfigOp>(user);
        }
      }
    }

    scf::ParallelOp parallelOpParent = NULL;
    auto op = alloca.getOperation();
    while (op) {
      if (op->getParentOfType<scf::ParallelOp>()) {
        if (&op->getParentOfType<scf::ParallelOp>().getRegion() ==
            alloca->getParentRegion()) {
          return failure();
        }
        parallelOpParent = op->getParentOfType<scf::ParallelOp>();
        break;
      }
      op = op->getParentOp();
    }

    if (parallelOpParent == NULL)
      return failure();

    rewriter.moveOpBefore(alloca, parallelOpParent.getBody(),
                          parallelOpParent.getBody()->begin());
    rewriter.moveOpAfter(firstTileConfig, alloca);
    rewriter.moveOpBefore(secondTileConfig, parallelOpParent.getBody(),
                          std::prev(parallelOpParent.getBody()->end(), 1));
    return success();
  }
};

struct IntelAMXTileConfigHoistingPass
    : public impl::IntelAMXTileConfigHoistingPassBase<
          IntelAMXTileConfigHoistingPass> {
  void populateCombinePatterns(RewritePatternSet &patterns) {
    patterns.add<IntelAMXTileConfigHoisting>(patterns.getContext());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace tpp
} // namespace mlir
