//===- BrgemmLinalgTiling.cpp -----------------------------------------*-C++-*-===//
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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
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
    if (!brgemmOp.hasPureBufferSemantics())
      return failure();

    // Check whether the tile sizes are valid
    if (options.registerTileShape.size() != 3) 
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "Invalid user input tile sizes. Should be <m,n,k>");

    // Check the whether the operation is brmatmul fp32 or bf16 type using
    // reduction count
    SmallVector<utils::IteratorType> brgemmIteratorTypes =
        brgemmOp.getIteratorTypesArray();
    int reductionCount =
        std::count(brgemmIteratorTypes.begin(), brgemmIteratorTypes.end(),
                   utils::IteratorType::reduction);

    if (reductionCount == 0)
      return rewriter.notifyMatchFailure(
          brgemmOp, "Batch matmul operation not supported yet");

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          brgemmOp, "Matmul operation not supported yet");

    if (reductionCount > 3)
      return rewriter.notifyMatchFailure(
          brgemmOp, "The operation is not a gemm");

    auto tensorShapeLhs =
        dyn_cast<MemRefType>(brgemmOp.getOperand(0).getType()).getShape();
    auto tensorShapeRhs =
        dyn_cast<MemRefType>(brgemmOp.getOperand(1).getType()).getShape();

    if (reductionCount == 2 &&
        (tensorShapeLhs.size() != 3 || tensorShapeRhs.size() != 3))
      return rewriter.notifyMatchFailure(
          brgemmOp, "Invalid rank for batch reduce operation");

    if (reductionCount == 3 && !vnni::utils::isInVnniLayout(brgemmOp))
      return rewriter.notifyMatchFailure(
          brgemmOp, "Failed matching for batch reduce operation with vnni layout");

    //  Get the register blocking tile shape from the user input
    SmallVector<int64_t> mxnxkTile(options.registerTileShape.begin(),
                                    options.registerTileShape.end());

    // k-tile size adjusted based on the vnni layout for bf16 type
    if (vnni::utils::isInVnniLayout(brgemmOp)) {
      auto tensorShape =
        dyn_cast<MemRefType>(brgemmOp.getOperand(0).getType()).getShape();
      mxnxkTile[2] = mxnxkTile[2] / tensorShape[3];
    }

    size_t i = 0;
    SmallVector<int> swap_i = {0, 2, 1};
    std::map<int, std::map<int, Value>> inductionVars;

    // For M, N, and K loops
    scf::ForOp innermostForLoop;
    // Creating the tiled loops
    for (auto itrShapeMNK = mxnxkTile.begin(); itrShapeMNK != mxnxkTile.end();
         itrShapeMNK++, i++) {
      auto upperBound =
          dyn_cast<MemRefType>(brgemmOp.getOperand(swap_i[i]).getType())
              .getShape()[1];
      // Tile size should not be greater than the upperBound
      if ((*itrShapeMNK) > upperBound)
        return failure();

      Location loc = brgemmOp.getLoc();
      Value zeroCst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value ubCstTiledLoop =
          rewriter.create<arith::ConstantIndexOp>(loc, upperBound);
      Value stepCstTiledLoop =
          rewriter.create<arith::ConstantIndexOp>(loc, *itrShapeMNK);
      // Creates M, N, and K tile loops
      scf::ForOp loopOp = rewriter.create<scf::ForOp>(
          brgemmOp.getLoc(), zeroCst, ubCstTiledLoop, stepCstTiledLoop);
      rewriter.setInsertionPointToStart(loopOp.getBody());
      innermostForLoop = loopOp;

      // Stores the induction variable with respect to the operands mapping it's
      // subview.
      if (i == 0) { // Stores iv for M loop
        inductionVars[0][1] = loopOp.getInductionVar();
        inductionVars[2][0] = loopOp.getInductionVar();
      } else if (i == 1) { //stores iv for N loop, creates batch loop, and maps iv of batch loop
        inductionVars[1][2] = loopOp.getInductionVar();
        inductionVars[2][1] = loopOp.getInductionVar();
        // Creates reduction loop after the N loop
        Value ubCstReduction = rewriter.create<arith::ConstantIndexOp>(
            loc, dyn_cast<MemRefType>(brgemmOp.getOperand(0).getType())
                     .getShape()[0]);
        Value stepCstReduction =
            rewriter.create<arith::ConstantIndexOp>(loc, 1);
        scf::ForOp redloopOp = rewriter.create<scf::ForOp>(
            brgemmOp.getLoc(), zeroCst, ubCstReduction, stepCstReduction);
        rewriter.setInsertionPointToStart(redloopOp.getBody());
        inductionVars[0][0] = redloopOp.getInductionVar();
        inductionVars[1][0] = redloopOp.getInductionVar();

      } else if (i == 2) { // stores iv for k-loop
        inductionVars[0][2] = loopOp.getInductionVar();
        inductionVars[1][1] = loopOp.getInductionVar();
      }
    }

    // DS to assist while creating new subviews with correct indices and shapes
    SmallVector<int64_t> mxkTile{mxnxkTile[0], mxnxkTile[2]};
    SmallVector<int64_t> kxnTile{mxnxkTile[2], mxnxkTile[1]};
    SmallVector<int64_t> mxnTile{mxnxkTile[0], mxnxkTile[1]};

    SmallVector<SmallVector<int64_t>> tileshapes{mxkTile, kxnTile, mxnTile};
    // Creating subviews
    for (size_t i = 0; i < brgemmOp.getNumOperands(); i++) {
      SmallVector<OpFoldResult> offsets;
      SmallVector<int64_t> indices;
      SmallVector<OpFoldResult> shape;
      SmallVector<OpFoldResult> strides;

      auto input = brgemmOp.getOperand(i);
      auto tensorShape = dyn_cast<MemRefType>(input.getType()).getShape();
      auto tileItr = tileshapes[i].begin();

      // Iterates over the shape of each tensor and update its offsets, indices,
      // shapes, strides with respect to tile sizes
      for (size_t j = 0; j < tensorShape.size(); j++) {
        if (j == 0 && (i < 2)) { // Updates the batch dimension
          offsets.push_back(inductionVars[i][j]);
          indices.push_back(1);
          shape.push_back(rewriter.getIndexAttr(1));
          strides.push_back(rewriter.getIndexAttr(1));
        } else if (j < 3) { // Updates the M, N, and K dimensions
          offsets.push_back(inductionVars[i][j]);
          indices.push_back((*tileItr));
          shape.push_back(rewriter.getIndexAttr(*tileItr));
          strides.push_back(rewriter.getIndexAttr(1));
          tileItr++;
        } else { // Just copies the vnni layout dimensions
          offsets.push_back(rewriter.getIndexAttr(0));
          indices.push_back(tensorShape[j]);
          shape.push_back(rewriter.getIndexAttr(tensorShape[j]));
          strides.push_back(rewriter.getIndexAttr(1));
        }
      }

      auto subview = rewriter.create<memref::SubViewOp>(
          brgemmOp.getLoc(), MemRefType(), input, offsets, shape, strides);
      brgemmOp.setOperand(i, subview);
    }

    rewriter.setInsertionPoint(innermostForLoop.getBody(),
                               std::prev(innermostForLoop.getBody()->end(), 1));
    auto clone = rewriter.clone(*brgemmOp);
    brgemmOp.replaceAllUsesWith(clone);
    if (brgemmOp->use_empty())
      rewriter.eraseOp(brgemmOp);
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
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};
} // namespace tpp
} // namespace mlir
