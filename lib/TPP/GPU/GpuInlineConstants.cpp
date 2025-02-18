//===- GpuInlineConstants.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUINLINECONSTANTS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Inlines constants into GPU launch body.
struct InlineConstantsIntoGPULaunch : public OpRewritePattern<gpu::LaunchOp> {
  using OpRewritePattern<gpu::LaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchOp launchOp,
                                PatternRewriter &rewriter) const override {
    Region &launchOpBody = launchOp.getBody();

    // Identify values defined outside of the launch operation.
    SetVector<Value> aboveVals;
    getUsedValuesDefinedAbove(launchOpBody, aboveVals);

    // Gather operations representing constants.
    SetVector<Operation *> constantOps;
    for (auto val : aboveVals) {
      auto *op = val.getDefiningOp();
      // TODO: Add more constant representations.
      if (op && isa<arith::ConstantOp>(op))
        constantOps.insert(op);
    }

    // Clone the constants into the gpu.launch body.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&launchOpBody.front());

    for (auto *op : constantOps) {
      auto *clonedOp = rewriter.clone(*op);

      // Replace uses within the body with the inlined values.
      for (auto [oldVal, newVal] :
           llvm::zip_equal(op->getResults(), clonedOp->getResults())) {
        replaceAllUsesInRegionWith(oldVal, newVal, launchOpBody);
      }
    }

    return success();
  }
};

void populateGpuInlineConstantsPatterns(RewritePatternSet &patterns) {
  patterns.add<InlineConstantsIntoGPULaunch>(patterns.getContext());
}

struct GpuInlineConstants
    : public tpp::impl::GpuInlineConstantsBase<GpuInlineConstants> {
  using GpuInlineConstantsBase::GpuInlineConstantsBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGpuInlineConstantsPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
