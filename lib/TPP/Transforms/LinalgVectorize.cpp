//===- LinalgVectorize.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGVECTORIZE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!linalg::hasVectorizationImpl(op))
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Op, cannot vectorize");
    // Direct naive pack/unpack vectorization has terrible performance.
    // Disable these for now and rely on other lowering patterns.
    if (isa<linalg::PackOp, linalg::UnPackOp>(op))
      return rewriter.notifyMatchFailure(op, "Packing vectorization disabled");
    // Insert slice is vectorized into a vector read-write pair which by default
    // introduces unnecessary extra operations.
    // Disable insert vectorization for now and allow bufferization to fold it
    // into subview in many cases.
    if (isa<tensor::InsertSliceOp>(op))
      return rewriter.notifyMatchFailure(op,
                                         "Insert slice vectorization disabled");

    auto vectorizeResult = linalg::vectorize(rewriter, op);
    if (failed(vectorizeResult))
            return failure();

    rewriter.replaceOp(op, vectorizeResult->replacements);

    return success();
  }
};

// Based on 'vectorize_children_and_apply_patterns' transform op.
struct LinalgVectorize
    : public tpp::impl::LinalgVectorizeBase<LinalgVectorize> {
  using LinalgVectorizeBase::LinalgVectorizeBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<VectorizationPattern>(ctx);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorReductionToContractPatterns(patterns);
    vector::populateSinkVectorOpsPatterns(patterns);
    patterns.add<linalg::LinalgCopyVTRForwardingPattern,
                 linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                         /*benefit=*/2);
    vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
    vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);
    patterns.add<linalg::CopyVectorizationPattern>(ctx);
    vector::populateVectorStepLoweringPatterns(patterns);
    vector::populateFoldArithExtensionPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
