//===- ConstantFoldPack.cpp --------------------------------------*- C++-*-===//
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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/Threading.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONSTANTFOLDPACK
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Helper pattern - lower linalg.pack operations that pack constants.
struct LowerConstantPacking : public OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern<linalg::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto constOp = packOp.getSource().getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return failure();
    // Must be a dense constant.
    auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
    if (!denseAttr)
      return failure();

    // Bail out if the pack is used as a writing operation i.e., the destination
    // is not a tensor.empty.
    if (!packOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return rewriter.notifyMatchFailure(packOp,
                                         "expects empty tensor destination");
    // Pack destination must have static shape.
    if (!packOp.getDestType().hasStaticShape())
      return rewriter.notifyMatchFailure(
          packOp, "expects destination with static shape");

    // If it is a splat constant, skip and let linalg.pack folder to handle this
    // case.
    if (denseAttr.isSplat())
      return rewriter.notifyMatchFailure(
          packOp, "skip pack - existing folder covers constant splats");

    return linalg::lowerPack(rewriter, packOp);
  }
};

// Rewrite constant packing operation as a compile-time packed constant.
struct ConstantFoldPack
    : public tpp::impl::ConstantFoldPackBase<ConstantFoldPack> {

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // Temporarily lower constant packing operation to allow other existing
    // patterns to fold the operation completely.
    patterns.add<LowerConstantPacking>(ctx);
    // Apply canonicalization to fold trivial cases and linalg constant folders
    // to cleanup lowered packs.
    linalg::FillOp::getCanonicalizationPatterns(patterns, ctx);
    linalg::PackOp::getCanonicalizationPatterns(patterns, ctx);
    tensor::populateRewriteAsConstantPatterns(
        patterns, [](OpOperand *) -> bool { return true; });
    linalg::populateConstantFoldLinalgOperations(
        patterns, [](OpOperand *) -> bool { return true; });

    (void)applyPatternsAndFoldGreedily(module, std::move(patterns));
  }
};

} // namespace
