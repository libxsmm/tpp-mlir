//===- ConvertPerfToLoops.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::perf;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTPERFTOLOOPS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct ConvertBenchToLoops : public OpRewritePattern<perf::BenchOp> {
  using OpRewritePattern<perf::BenchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::BenchOp benchOp,
                                PatternRewriter &rewriter) const override {
    auto loc = benchOp.getLoc();
    auto *benchYield = benchOp.getRegion().front().getTerminator();
    assert(dyn_cast_or_null<perf::YieldOp>(benchYield) &&
           "expect perf.yield in perf.bench");

    auto zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto numIters = arith::IndexCastOp::create(rewriter, 
        loc, rewriter.getIndexType(), benchOp.getNumIters());

    // Create benchmark loop up to perf.bench numIters.
    // Wrap the benchmark kernel in timer calls.
    auto timer = perf::StartTimerOp::create(rewriter, 
        loc, TimerType::get(rewriter.getContext()));
    auto loop = scf::ForOp::create(rewriter, loc, zero, numIters, one,
                                            benchOp.getIterArgs());
    auto delta = perf::StopTimerOp::create(rewriter, loc, rewriter.getF64Type(),
                                                    timer.getTimer());

    if (benchOp.getIterArgs().empty()) {
      // Erase the default loop yield, it will be inserted later.
      auto *yield = loop.getRegion().front().getTerminator();
      assert(isa<scf::YieldOp>(yield) && "Last op must be yield");
      rewriter.eraseOp(yield);
    }

    // Move perf.bench region inside the loop.
    rewriter.mergeBlocks(&benchOp.getRegion().front(), loop.getBody(),
                         benchOp.getIterArgs());

    // Replace uses of bench args within the benchmark body with their
    // equivalent loop-carried variables.
    assert((benchOp.getIterArgs().size() == loop.getRegionIterArgs().size()) &&
           "expect equal number of iter_args variables");
    for (auto [benchArg, loopArg] :
         llvm::zip_equal(benchOp.getIterArgs(), loop.getRegionIterArgs()))
      replaceAllUsesInRegionWith(benchArg, loopArg, loop.getRegion());

    // Pass perf.yield values through the scf.yield.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(loop.getBody());
    scf::YieldOp::create(rewriter, loc, benchYield->getOperands());
    rewriter.eraseOp(benchYield);

    // Swap bench results with loop results.
    assert((benchOp.getBodyResults().size() == loop.getResults().size() + 1) &&
           "expect equal number of return variables");

    // First, we add the timer delta as a loop result
    SmallVector<Value> loopResults;
    loopResults.push_back(delta);
    // Then we add the rest of the iter args
    loopResults.append(loop.getResults().begin(), loop.getResults().end());
    // And replace everything
    for (auto [benchRes, loopRes] :
         llvm::zip_equal(benchOp.getBodyResults(), loopResults))
      benchRes.replaceAllUsesWith(loopRes);

    // Erase bench op & return
    rewriter.eraseOp(benchOp);
    return success();
  }
};

void populatePerfToLoopsPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertBenchToLoops>(patterns.getContext());
}

struct ConvertPerfToLoops
    : public tpp::impl::ConvertPerfToLoopsBase<ConvertPerfToLoops> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePerfToLoopsPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
