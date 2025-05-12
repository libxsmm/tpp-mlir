//===- BrgemmLinalgTiling.cpp--------------------------------------*-C++-*-===//
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
#define DEBUG_TYPE "brgemm-linalg-tiling"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_BRGEMMLINALGTILING
#define GEN_PASS_DEF_BRGEMMLINALGTILING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {

template <typename BrgemmOp>
struct LinalgOpTiling : OpRewritePattern<BrgemmOp> {
  using OpRewritePattern<BrgemmOp>::OpRewritePattern;

  LinalgOpTiling(MLIRContext *ctx, BrgemmLinalgTilingOptions tilingoptions)
      : OpRewritePattern<BrgemmOp>(ctx), options(tilingoptions) {}

  LogicalResult matchAndRewrite(BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {

    // Check whether the tile sizes are valid
    if (options.registerTileShape.size() != 3)
      return rewriter.notifyMatchFailure(
          brgemmOp, "Invalid user input tile sizes. Should be <m,n,k>");

    // Check the whether the operation is brmatmul fp32 or bf16 type using
    // reduction count
    SmallVector<utils::IteratorType> brgemmIteratorTypes =
        brgemmOp.getIteratorTypesArray();
    int reductionCount =
        std::count(brgemmIteratorTypes.begin(), brgemmIteratorTypes.end(),
                   utils::IteratorType::reduction);

    int parallelCount =
        std::count(brgemmIteratorTypes.begin(), brgemmIteratorTypes.end(),
                   utils::IteratorType::parallel);

    if (reductionCount == 0 || reductionCount > 3 || parallelCount != 2)
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "Excepted GEMM like operation");

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          brgemmOp, "Batch matmul operation not supported yet");

    auto shapeTypeLhs =
        dyn_cast<ShapedType>(brgemmOp.getOperand(0).getType());
    auto shapeTypeRhs =
        dyn_cast<ShapedType>(brgemmOp.getOperand(1).getType());

    auto shapeLhs = shapeTypeLhs.getShape();
    auto shapeRhs = shapeTypeRhs.getShape();

    if (reductionCount == 2 &&
        (shapeLhs.size() != 3 || shapeRhs.size() != 3))
      return rewriter.notifyMatchFailure(
          brgemmOp, "Invalid rank for batch reduce operation");

    auto vnniOpt = vnni::utils::isInVnniLayout(brgemmOp);
    if (reductionCount == 3 && !vnniOpt)
      return rewriter.notifyMatchFailure(
          brgemmOp,
          "Failed matching for batch reduce operation with vnni layout");

    // Tiling with the help of upstream APIs
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setLoopType(linalg::LinalgTilingLoopType::Loops);

    // Get rank and map of linalg op
    unsigned rankA = shapeTypeLhs.getRank();
    unsigned rankB = shapeTypeRhs.getRank();
    AffineMap mapA =
        brgemmOp.getMatchingIndexingMap(&brgemmOp->getOpOperand(0));
    AffineMap mapB =
        brgemmOp.getMatchingIndexingMap(&brgemmOp->getOpOperand(1));

    if (vnniOpt) {
      // k-tile size adjusted based on the vnni layout for bf16 type
      auto kTileVnni = options.registerTileShape[2] / shapeLhs[3];

      // Note: We make an assumption that the k tile size is divisible to
      // the powers of 2.
      if (kTileVnni < 1 || (options.registerTileShape[2] % shapeLhs[3] != 0))
        return rewriter.notifyMatchFailure(
            brgemmOp, "Failed matching K tile size for batch reduce operation "
                      "with vnni layout. K tile size should be >= vnni layout "
                      "and divisible by vnni layout");

      // Calculating the tile sizes based on affine map for bf16 type with vnni
      auto vnniDim =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1))).getPosition();
      auto dimM =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 3))).getPosition();
      auto dimN =
          (dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 2))).getPosition();
      auto dimBR =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 4))).getPosition();
      auto dimK =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 2))).getPosition();

      // To set the loop interchange options
      SmallVector<int64_t> tileSizes(5);
      tileSizes[dimBR] = 1;
      tileSizes[dimM] = options.registerTileShape[0];
      tileSizes[dimN] = options.registerTileShape[1];
      tileSizes[dimK] = kTileVnni;
      tileSizes[vnniDim] = 0;

      tilingOptions.setTileSizes(tileSizes);
      tilingOptions.setInterchange({dimM, dimN, dimBR, dimK, vnniDim});

    } else {

      // Calculating the tile sizes based on affine map for fp32 type
      auto dimM =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 2))).getPosition();
      auto dimN =
          (dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 1))).getPosition();
      auto dimBR =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 3))).getPosition();
      auto dimK =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1))).getPosition();

      // Checks dimensions are aligned with the iterator types
      if (brgemmIteratorTypes[dimM] != mlir::utils::IteratorType::parallel ||
          brgemmIteratorTypes[dimN] != mlir::utils::IteratorType::parallel ||
          brgemmIteratorTypes[dimBR] != mlir::utils::IteratorType::reduction ||
          brgemmIteratorTypes[dimK] != mlir::utils::IteratorType::reduction)
        return rewriter.notifyMatchFailure(
            brgemmOp, "Failed matching with iterator types and dimension");

      // To set the loop interchange options
      SmallVector<int64_t> tileSizes(4);
      tileSizes[dimBR] = 1;
      tileSizes[dimM] = options.registerTileShape[0];
      tileSizes[dimN] = options.registerTileShape[1];
      tileSizes[dimK] = options.registerTileShape[2];

      tilingOptions.setTileSizes(tileSizes);
      tilingOptions.setInterchange({dimM, dimN, dimBR, dimK});
    }

    FailureOr<linalg::TiledLinalgOp> tiledOp = linalg::tileLinalgOp(rewriter, brgemmOp, tilingOptions);
    if (failed(tiledOp)) {
      return failure();
    }
    rewriter.replaceOp(brgemmOp, tiledOp->tensorResults);

    return success();
  }

private:
  BrgemmLinalgTilingOptions options;
};

void populateBrgemmLinalgTilingPatterns(RewritePatternSet &patterns,
                                        BrgemmLinalgTilingOptions options) {
  patterns.add<LinalgOpTiling<linalg::GenericOp>,
               LinalgOpTiling<linalg::BatchReduceMatmulOp>>(
      patterns.getContext(), options);
}

struct BrgemmLinalgTiling
    : public tpp::impl::BrgemmLinalgTilingBase<BrgemmLinalgTiling> {

  using BrgemmLinalgTilingBase::BrgemmLinalgTilingBase;

  void runOnOperation() override {
    BrgemmLinalgTilingOptions options;
    options.registerTileShape = SmallVector<unsigned>{*registerTileShape};
    RewritePatternSet patterns(&getContext());
    populateBrgemmLinalgTilingPatterns(patterns, options);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
