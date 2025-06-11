//===- SplitReductionDim.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements serial reduction dimension splitting.
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/TransformUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_SPLITREDUCTIONDIM
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Split contraction's innermost reduction dimension.
struct SplitContractionReduction
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  SplitContractionReduction(MLIRContext *ctx, SplitReductionDimOptions options)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx), options(options) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (options.tileSize <= 0)
      return rewriter.notifyMatchFailure(linalgOp,
                                         "invalid reduction tile size");

    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(linalgOp);
    if (failed(dims))
      return rewriter.notifyMatchFailure(linalgOp, "not a contraction");

    scf::SCFTilingOptions tilingOpts;
    // Tile using a serial loop.
    tilingOpts.setLoopType(scf::SCFTilingOptions::LoopType::ForOp);
    // Tile only the innermost reduction dimension - disable tiling for all
    // other dims.
    SmallVector<OpFoldResult> tiles(
        linalgOp.getNumLoops(),
        getAsIndexOpFoldResult(rewriter.getContext(), 0));
    tiles[dims->k.back()] =
        getAsIndexOpFoldResult(rewriter.getContext(), options.tileSize);
    tilingOpts.setTileSizes(tiles);

    FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(linalgOp.getOperation()), tilingOpts);
    if (failed(tilingResult))
      return rewriter.notifyMatchFailure(linalgOp,
                                         "failed to tile contraction");

    rewriter.replaceOp(linalgOp, tilingResult->replacements);

    return success();
  }

private:
  SplitReductionDimOptions options;
};

// Split reduction dimension.
struct SplitReductionDim
    : public tpp::impl::SplitReductionDimBase<SplitReductionDim> {
  using SplitReductionDimBase::SplitReductionDimBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    SplitReductionDimOptions options{tileSize};

    RewritePatternSet patterns(ctx);
    patterns.add<SplitContractionReduction>(ctx, options);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};

} // namespace
