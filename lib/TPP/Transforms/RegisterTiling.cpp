//===- RegisterTiling.cpp--------------------------------------*-C++-*-===//
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

template <typename GemmOp>
struct LinalgOpTiling : OpRewritePattern<GemmOp> {
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  LinalgOpTiling(MLIRContext *ctx, RegisterTilingOptions tilingoptions)
      : OpRewritePattern<GemmOp>(ctx), options(tilingoptions) {}

  LogicalResult matchAndRewrite(GemmOp gemmOp,
                                PatternRewriter &rewriter) const override {

    // Check whether the tile sizes are valid
    if (options.registerTileShape.size() != 3)
      return rewriter.notifyMatchFailure(
          gemmOp, "Invalid user input tile sizes. Should be <m,n,k>");

    // Classify the operation using the iterator types. Supported GEMM-like
    // ops, each in plain or vnni layout, with any number of leading batch
    // (parallel) or batch-reduce (reduction) dimensions, all tiled by 1:
    //   - matmul:       2 parallel (M, N),           reduction over K (+ vnni)
    //   - batch matmul: 2 + b parallel (batch, M, N), reduction over K (+ vnni)
    //   - batch reduce: 2 parallel (M, N),           reduction over batch, K (+ vnni)
    SmallVector<utils::IteratorType> gemmIteratorTypes =
        gemmOp.getIteratorTypesArray();
    int reductionCount =
        std::count(gemmIteratorTypes.begin(), gemmIteratorTypes.end(),
                   utils::IteratorType::reduction);

    int parallelCount =
        std::count(gemmIteratorTypes.begin(), gemmIteratorTypes.end(),
                   utils::IteratorType::parallel);

    // Reject anything that is not a GEMM-like contraction with two inputs.
    if (reductionCount == 0 || parallelCount < 2 ||
        gemmOp.getNumDpsInputs() != 2)
      return rewriter.notifyMatchFailure(gemmOp,
                                         "Expected GEMM like operation");

    auto shapeTypeLhs =
        dyn_cast<ShapedType>(gemmOp.getOperand(0).getType());
    auto shapeTypeRhs =
        dyn_cast<ShapedType>(gemmOp.getOperand(1).getType());
    if (!shapeTypeLhs || !shapeTypeRhs)
      return rewriter.notifyMatchFailure(gemmOp, "Expected shaped operands");

    auto shapeLhs = shapeTypeLhs.getShape();

    auto vnniOpt = vnni::utils::isInVnniLayout(gemmOp);

    // Tiling with the help of upstream APIs
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setLoopType(linalg::LinalgTilingLoopType::Loops);

    // Get rank and map of linalg op
    unsigned rankA = shapeTypeLhs.getRank();
    unsigned rankB = shapeTypeRhs.getRank();
    AffineMap mapA =
        gemmOp.getMatchingIndexingMap(&gemmOp->getOpOperand(0));
    AffineMap mapB =
        gemmOp.getMatchingIndexingMap(&gemmOp->getOpOperand(1));

    // Every dimension before the base matmul operands is a leading batch
    // (parallel) or batch-reduce (reduction) dimension. baseRank is 2 for the
    // plain layout ([...][M][K]) and 3 for the vnni layout ([...][K/vnni][vnni]).
    unsigned baseRank = vnniOpt ? 3 : 2;
    if (rankA < baseRank)
      return rewriter.notifyMatchFailure(gemmOp, "Unexpected operand rank");
    unsigned numBatch = rankA - baseRank;

    unsigned dimM, dimN, dimK, vnniDim = 0;
    if (vnniOpt) {
      // Layout: A = [...][M][K/vnni][vnni], B = [...][K/vnni][N][vnni].
      vnniDim =
          (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1))).getPosition();
      dimM = (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 3))).getPosition();
      dimK = (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 2))).getPosition();
      dimN = (dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 2))).getPosition();
    } else {
      // Layout: A = [...][M][K], B = [...][K][N].
      dimM = (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 2))).getPosition();
      dimK = (dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1))).getPosition();
      dimN = (dyn_cast<AffineDimExpr>(mapB.getResult(rankB - 1))).getPosition();
    }

    // Collect the leading batch dimensions in operand order (outermost first).
    SmallVector<unsigned> batchDims;
    for (unsigned i = 0; i < numBatch; ++i)
      batchDims.push_back(
          (dyn_cast<AffineDimExpr>(mapA.getResult(i))).getPosition());

    // Check dimensions are aligned with the iterator types. The batch
    // dimensions may be parallel (batch matmul) or reduction (batch reduce).
    if (gemmIteratorTypes[dimM] != mlir::utils::IteratorType::parallel ||
        gemmIteratorTypes[dimN] != mlir::utils::IteratorType::parallel ||
        gemmIteratorTypes[dimK] != mlir::utils::IteratorType::reduction)
      return rewriter.notifyMatchFailure(
          gemmOp, "Failed matching with iterator types and dimension");

    // Every batch/batch-reduce dimension is tiled by 1.
    SmallVector<int64_t> tileSizes(gemmIteratorTypes.size(), 0);
    tileSizes[dimM] = options.registerTileShape[0];
    tileSizes[dimN] = options.registerTileShape[1];
    for (unsigned dim : batchDims)
      tileSizes[dim] = 1;

    // Interchange: any extra (outer) batch dimensions come first, followed by
    // the M, N, innermost-batch, K (and vnni) loop ordering.
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

    FailureOr<linalg::TiledLinalgOp> tiledOp = linalg::tileLinalgOp(rewriter, gemmOp, tilingOptions);
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
  patterns.add<LinalgOpTiling<linalg::GenericOp>,
               LinalgOpTiling<linalg::MatmulOp>,
               LinalgOpTiling<linalg::BatchMatmulOp>,
               LinalgOpTiling<linalg::BatchReduceMatmulOp>>(
      patterns.getContext(), options);
}

struct RegisterTiling
    : public tpp::impl::RegisterTilingBase<RegisterTiling> {

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
