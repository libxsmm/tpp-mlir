//===- ConvertPerfToLoops.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
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
    auto numIters = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), benchOp.getNumIters());

    // Allocate a scalar f64 accumulator on the stack and zero-initialize it.
    // Per-iteration timings accumulate here so that any extra ops inserted
    // between iterations are excluded from measurement.
    auto acc = memref::AllocaOp::create(
        rewriter, loc, MemRefType::get({}, rewriter.getF64Type()));
    auto zeroF64 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getFloatAttr(rewriter.getF64Type(), 0.0));
    memref::StoreOp::create(rewriter, loc, zeroF64, acc, ValueRange{});

    // Allocate cache-nuke buffers on the heap (too large for the stack).
    // A<16384x512xf32>, B<512x16384xf32>, C<16384x16384xf32>.
    // These will be used to run a large GEMM before each timed iteration to
    // flush all cache levels, ensuring the benchmark measures cold-cache
    // performance.
    constexpr int64_t nukeM = 16384, nukeN = 16384, nukeK = 512;
    auto f32Type = rewriter.getF32Type();
    auto nukeA = memref::AllocOp::create(
        rewriter, loc, MemRefType::get({nukeM, nukeK}, f32Type));
    auto nukeB = memref::AllocOp::create(
        rewriter, loc, MemRefType::get({nukeK, nukeN}, f32Type));
    auto nukeC = memref::AllocOp::create(
        rewriter, loc, MemRefType::get({nukeM, nukeN}, f32Type));

    // Dispatch the cache-nuke GEMM once.
    auto integer64 = IntegerType::get(rewriter.getContext(), 64);
    auto dtype =
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
    auto nukeFlags = rewriter.getArrayAttr(
        xsmm::GemmFlagsAttr::get(rewriter.getContext(), xsmm::GemmFlags::NONE));
    auto nukeDims = DenseI64ArrayAttr::get(
        rewriter.getContext(),
        ArrayRef<int64_t>{nukeM, nukeN, nukeK, /*lda=*/nukeK, /*ldb=*/nukeN,
                          /*ldc=*/nukeN});
    Value nukeDispatch = xsmm::GemmDispatchOp::create(
        rewriter, loc, integer64, nukeDims, nukeFlags, dtype);

    // Create benchmark loop up to perf.bench numIters.
    auto loop = scf::ForOp::create(rewriter, loc, zero, numIters, one,
                                   benchOp.getIterArgs());

    // Load the total accumulated time after the loop – this becomes the delta.
    auto finalVal = memref::LoadOp::create(rewriter, loc, acc, ValueRange{});

    // Free the cache-nuke buffers after the loop.
    memref::DeallocOp::create(rewriter, loc, nukeA);
    memref::DeallocOp::create(rewriter, loc, nukeB);
    memref::DeallocOp::create(rewriter, loc, nukeC);

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

    // Insert cache-nuke GEMM at the start of the loop body to flush all cache
    // levels, then start the timer immediately after so GEMM time is excluded.
    Value timerVal;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      auto nukeGemm = xsmm::GemmOp::create(
          rewriter, loc, dtype,
          ValueRange{nukeDispatch, nukeA.getResult(), nukeB.getResult(),
                     nukeC.getResult()});
      // Advance past the GEMM so the timer is inserted after it.
      rewriter.setInsertionPointAfter(nukeGemm);
      timerVal = perf::StartTimerOp::create(
                     rewriter, loc, TimerType::get(rewriter.getContext()))
                     .getTimer();
    }

    // Insert timer stop and time accumulation just before the perf.yield.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(benchYield);
      auto iterDelta = perf::StopTimerOp::create(
          rewriter, loc, rewriter.getF64Type(), timerVal);
      auto currVal = memref::LoadOp::create(rewriter, loc, acc, ValueRange{});
      auto newVal = arith::AddFOp::create(rewriter, loc, currVal, iterDelta);
      memref::StoreOp::create(rewriter, loc, newVal, acc, ValueRange{});
    }

    // Pass perf.yield values through the scf.yield.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(loop.getBody());
      scf::YieldOp::create(rewriter, loc, benchYield->getOperands());
      rewriter.eraseOp(benchYield);
    }

    // Swap bench results with loop results.
    assert((benchOp.getBodyResults().size() == loop.getResults().size() + 1) &&
           "expect equal number of return variables");

    // Use finalVal (total accumulated time) as the delta result.
    SmallVector<Value> loopResults;
    loopResults.push_back(finalVal);
    // Then add the iter_args results.
    loopResults.append(loop.getResults().begin(), loop.getResults().end());
    // Replace everything.
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
