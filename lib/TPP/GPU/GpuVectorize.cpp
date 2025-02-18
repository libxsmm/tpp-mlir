//===- GpuVectorize.cpp ------------------------------------------*- C++-*-===//
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
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUVECTORIZE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Vectorize ops within GPU kernel.
struct VectorizeGpuLaunch : public OpRewritePattern<gpu::LaunchOp> {
  using OpRewritePattern<gpu::LaunchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchOp launchOp,
                                PatternRewriter &rewriter) const override {
    // Vectorize all linalg ops within GPU kernel.
    // It is expected that the ops operate on statically sized tiles.
    auto walkResult = launchOp->walk([&](linalg::LinalgOp linalgOp) {
      if (linalgOp.hasDynamicShape())
        return WalkResult::interrupt();

      if (failed(vectorize(rewriter, linalgOp, /*inputVectorSizes=*/{},
                           /*scalableVecDims=*/{})))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return rewriter.notifyMatchFailure(
          launchOp, "Failed to vectorize ops within GPU launch");

    return success();
  }
};

// Vectorize linalg ops targeting GPU.
struct GpuVectorizeLinalg : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Vectorize all Linalg ops within parallelized loops.
    if (!linalgOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(linalgOp, "Expects tensor semantics");

    if (linalgOp.hasDynamicShape())
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Expects static shapes only");

    // Only process operations within parallelized loops.
    // TODO: Use some different mechanism like annotations to determine which
    //       ops target GPU.
    if (!linalgOp->getParentOfType<scf::ForallOp>())
      return rewriter.notifyMatchFailure(linalgOp,
                                         "Expects parallel loop parent");

    return vectorize(rewriter, linalgOp, /*inputVectorSizes=*/{},
                     /*scalableVecDims=*/{});
  }
};

// Vectorize operations targeting GPU.
struct GpuVectorize : public tpp::impl::GpuVectorizeBase<GpuVectorize> {
  using GpuVectorizeBase::GpuVectorizeBase;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);

    // Vectorize core computation ops within kernel launch.
    patterns.add<VectorizeGpuLaunch, GpuVectorizeLinalg>(ctx);

    // Vector postprocessing patterns.
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);
    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
