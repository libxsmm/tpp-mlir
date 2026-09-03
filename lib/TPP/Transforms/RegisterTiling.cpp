//===- RegisterTiling.cpp------------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements parallel loop insertion for tiling.
//
//===----------------------------------------------------------------------===//
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "register-tiling"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_REGISTERTILING
#define GEN_PASS_DEF_REGISTERTILING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {

template <typename GemmOp> struct LinalgOpTiling : OpRewritePattern<GemmOp> {
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  LinalgOpTiling(MLIRContext *ctx, RegisterTilingOptions tilingoptions)
      : OpRewritePattern<GemmOp>(ctx), options(tilingoptions) {}

  LogicalResult matchAndRewrite(GemmOp gemmOp,
                                PatternRewriter &rewriter) const override {

    // Check whether the tile sizes are valid
    if (options.registerTileShape.size() != 3)
      return rewriter.notifyMatchFailure(
          gemmOp, "Invalid user input tile sizes. Should be <m,n,k>");

    // Only the three known contraction forms are supported: gemm, batch gemm,
    // and batch-reduce gemm (each in plain or vnni layout). Use the upstream
    // contraction matchers to robustly reject anything else. The body check in
    // isaContractionOpInterface transparently skips the extf casts used by the
    // mixed-precision (bf16) variants.
    auto linalgOp = cast<linalg::LinalgOp>(gemmOp.getOperation());
    if (!linalg::isaContractionOpInterface(linalgOp))
      return rewriter.notifyMatchFailure(gemmOp, "Expected a contraction");

    FailureOr<linalg::ContractionDimensions> contractionDims =
        linalg::inferContractionDims(linalgOp);
    if (failed(contractionDims))
      return rewriter.notifyMatchFailure(gemmOp,
                                         "Could not infer contraction dims");

    // gemm/batch-gemm/batch-reduce-gemm all have a single M and single N dim.
    if (contractionDims->m.size() != 1 || contractionDims->n.size() != 1)
      return rewriter.notifyMatchFailure(gemmOp, "Expected a single M and N");

    auto shapeTypeLhs = dyn_cast<ShapedType>(gemmOp.getOperand(0).getType());
    auto shapeTypeRhs = dyn_cast<ShapedType>(gemmOp.getOperand(1).getType());
    if (!shapeTypeLhs || !shapeTypeRhs)
      return rewriter.notifyMatchFailure(gemmOp, "Expected shaped operands");

    auto shapeLhs = shapeTypeLhs.getShape();
    auto vnniOpt = vnni::utils::isInVnniLayout(gemmOp);

    // Tiling with the help of upstream APIs
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setLoopType(linalg::LinalgTilingLoopType::Loops);

    unsigned rankA = shapeTypeLhs.getRank();
    AffineMap mapA = gemmOp.getMatchingIndexingMap(&gemmOp->getOpOperand(0));

    // baseRank is 2 for the plain layout ([...][M][K]) and 3 for the vnni
    // layout ([...][M][K/vnni][vnni]).
    unsigned baseRank = vnniOpt ? 3 : 2;
    if (rankA < baseRank)
      return rewriter.notifyMatchFailure(gemmOp, "Unexpected operand rank");

    // M and N come from the inferred contraction dims. The innermost K (and the
    // vnni dim) are read positionally from the LHS trailing layout, which is
    // fixed for all supported forms: A = [...][M][K] or [...][M][K/vnni][vnni].
    unsigned dimM = contractionDims->m[0];
    unsigned dimN = contractionDims->n[0];
    unsigned dimK, vnniDim = 0;
    if (vnniOpt) {
      vnniDim =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1))).getPosition();
      dimK = (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 2))).getPosition();
    } else {
      dimK = (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1))).getPosition();
    }

    SmallVector<unsigned> batchDims(contractionDims->batch.begin(),
                                    contractionDims->batch.end());
    for (unsigned kDim : contractionDims->k)
      if (kDim != dimK && (!vnniOpt || kDim != vnniDim))
        batchDims.push_back(kDim);

    // Set the tile sizes.
    // M, N, and K tiles are inputted by user.
    // Batch/batch-reduction tile is set to 1.
    // Vnni tile is set to vnni factor (2 or 4).
    SmallVector<int64_t> tileSizes(linalgOp.getNumLoops(), 0);
    tileSizes[dimM] = options.registerTileShape[0];
    tileSizes[dimN] = options.registerTileShape[1];
    for (unsigned dim : batchDims)
      tileSizes[dim] = 1;

    // Order the tiled loops.
    // Any extra (outer) batch dimensions come first, followed by
    // the M, N, innermost-batch (if any), K (and vnni) loop.
    SmallVector<unsigned> interchange;
    for (unsigned i = 0; i + 1 < batchDims.size(); ++i)
      interchange.push_back(batchDims[i]);
    interchange.push_back(dimM);
    interchange.push_back(dimN);
    if (!batchDims.empty())
      interchange.push_back(batchDims.back());
    interchange.push_back(dimK);

    if (vnniOpt) {
      // k-tile size adjusted based on the vnni layout.
      int64_t vnniFactor = shapeLhs[rankA - 1];
      auto kTileVnni = options.registerTileShape[2] / vnniFactor;

      // Note: We make an assumption that the k tile size is divisible to
      // the powers of 2.
      if (kTileVnni < 1 || (options.registerTileShape[2] % vnniFactor != 0))
        return rewriter.notifyMatchFailure(
            gemmOp, "Failed matching K tile size for vnni layout. K tile "
                    "size should be >= vnni layout and divisible by vnni "
                    "layout");

      tileSizes[dimK] = kTileVnni;
      tileSizes[vnniDim] = 0;
      interchange.push_back(vnniDim);
    } else {
      tileSizes[dimK] = options.registerTileShape[2];
    }

    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setInterchange(interchange);

    // Upstream API to tile linalg op.
    FailureOr<linalg::TiledLinalgOp> tiledOp =
        linalg::tileLinalgOp(rewriter, gemmOp, tilingOptions);
    if (failed(tiledOp)) {
      return failure();
    }
    rewriter.replaceOp(gemmOp, tiledOp->tensorResults);

    return success();
  }

private:
  RegisterTilingOptions options;
};

void populateRegisterTilingPatterns(RewritePatternSet &patterns,
                                    RegisterTilingOptions options) {
  patterns
      .add<LinalgOpTiling<linalg::GenericOp>, LinalgOpTiling<linalg::MatmulOp>,
           LinalgOpTiling<linalg::BatchMatmulOp>,
           LinalgOpTiling<linalg::BatchReduceMatmulOp>>(patterns.getContext(),
                                                        options);
}

struct RegisterTiling : public tpp::impl::RegisterTilingBase<RegisterTiling> {

  using RegisterTilingBase::RegisterTilingBase;

  void runOnOperation() override {
    RegisterTilingOptions options;
    options.registerTileShape = SmallVector<unsigned>{*registerTileShape};
    RewritePatternSet patterns(&getContext());
    populateRegisterTilingPatterns(patterns, options);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
