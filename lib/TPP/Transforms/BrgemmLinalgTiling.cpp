//===- BrgemmLinalgTiling.cpp -----------------------------------------*- C++-*-===//
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
struct LinalgOpTiling : OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  LinalgOpTiling(MLIRContext *ctx, BrgemmLinalgTilingOptions tilingoptions)
      : OpRewritePattern(ctx), options(tilingoptions) {}

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {

    if (!brgemmOp.hasPureBufferSemantics())
      return failure();
    //  Get the register blocking tile shape from the user input
    SmallVector<int64_t> tileShapeM(options.registerTileShape.begin(),
                                    options.registerTileShape.end());

    if (tileShapeM.size() != 2)
           return failure();

    SmallVector<int64_t> tileShapeN(2);
    tileShapeN[0] = 1;
    tileShapeN[1] = tileShapeM[1];
    tileShapeM[1] = 1;

    // Stores the M, N, and K Tile Sizes
    SmallVector<int64_t> mxnxkTile(3);
     // Stores the M, and N Tile Sizes
    SmallVector<int64_t> mxnTile(2);

    mxnxkTile[0] = tileShapeM[0];
    mxnxkTile[1] = tileShapeN[1];
    mxnxkTile[2] = tileShapeM[1];
    mxnTile[0] = tileShapeM[0];
    mxnTile[1] = tileShapeN[1];

    // To assist in calculating the argument and step values for the tiled loop.
    SmallVector<int64_t> boundariesOne{1,
                                       static_cast<long>(tileShapeM.size() - 1),
                                       static_cast<long>(mxnxkTile.size() - 1)};

    SmallVector<int64_t> tileSizesIndex{static_cast<long>(tileShapeM.size()),
                                        static_cast<long>(tileShapeN.size()),
                                        static_cast<long>(mxnTile.size())};
    SmallVector<SmallVector<int64_t>> tileshapes{tileShapeM, tileShapeN, mxnTile};
    SmallVector<int> swap_i = {0, 2, 1};
    size_t i = 0;
    std::map<int, std::map<int, Value>> inductionVars;

    // For M, N, and K loops
    scf::ForOp innermostForLoop;
    // For brgemm reduction loop
    scf::ForOp reductionForLoop;

    // Creating the tiled loops
    for (auto itrShapeM = mxnxkTile.begin(); itrShapeM != mxnxkTile.end();
         itrShapeM++, i++) {
      int index = swap_i[i] / boundariesOne[swap_i[i]];
      int offset = swap_i[i] / (mxnxkTile.size() - 1);

      int operandSize =
          dyn_cast<MemRefType>(brgemmOp.getOperand(index).getType())
              .getShape()
              .size();
      int effectiveOffset = operandSize - tileSizesIndex[index] + offset;
      auto upperBound =
          dyn_cast<MemRefType>(brgemmOp.getOperand(index).getType())
              .getShape()[effectiveOffset];
      Location loc = brgemmOp.getLoc();
      Value zeroCst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value ubCstTiledLoop = rewriter.create<arith::ConstantIndexOp>(loc, upperBound);
      //Tile size should not be greater than the upperBound
      if ((*itrShapeM) > upperBound)
              return failure();
      Value stepCstTiledLoop = rewriter.create<arith::ConstantIndexOp>(loc, *itrShapeM);
      // Creates M, N, and K tile loops
      scf::ForOp loopOp = rewriter.create<scf::ForOp>(brgemmOp.getLoc(),
                                                      zeroCst, ubCstTiledLoop, stepCstTiledLoop);
      rewriter.setInsertionPointToStart(loopOp.getBody());
      int indexTwo = offset;
      int operandSizeTwo =
          dyn_cast<MemRefType>(brgemmOp.getOperand(indexTwo).getType())
              .getShape()
              .size();
      int effectiveOffsetTwo = operandSizeTwo - tileSizesIndex[index] + index;

      inductionVars[index][effectiveOffset] = loopOp.getInductionVar();

      inductionVars[indexTwo][effectiveOffsetTwo] = loopOp.getInductionVar();
      int indexThree = mxnTile.size();
      int effectiveOffsetThree =
          index +
          dyn_cast<MemRefType>(brgemmOp.getOperand(indexThree).getType())
              .getShape()
              .size() -
          tileSizesIndex[indexThree];
      if (inductionVars[indexThree][effectiveOffsetThree] == NULL) {
        inductionVars[indexThree][effectiveOffsetThree] =
            loopOp.getInductionVar();
      }

      innermostForLoop = loopOp;
      if ((mxnxkTile.size() - 1) == (i + 1)) {
        //Creates the brgemm reduction loop
        Value ubCstReduction = rewriter.create<arith::ConstantIndexOp>(
            loc, dyn_cast<MemRefType>(brgemmOp.getOperand(0).getType())
                     .getShape()[0]);
        Value stepCstReduction = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        scf::ForOp redloopOp = rewriter.create<scf::ForOp>(
            brgemmOp.getLoc(), zeroCst, ubCstReduction, stepCstReduction);
        rewriter.setInsertionPointToStart(redloopOp.getBody());
        reductionForLoop = redloopOp;
      }
    }

    // Creating subviews
    SmallVector<SmallVector<int64_t>> tiles = {tileShapeM, tileShapeN};
    for (size_t i = 0; i < brgemmOp.getNumOperands(); i++) {
      SmallVector<int64_t> indices;
      auto input = brgemmOp.getOperand(i);
      auto operandType = input.getType();
      SmallVector<OpFoldResult> offsets;
      size_t k = 0;
      auto tileItr = tileshapes[i].begin();
      auto tensorShape = dyn_cast<MemRefType>(operandType).getShape();
      SmallVector<OpFoldResult> shape;
      SmallVector<OpFoldResult> strides;
      for (size_t j = 0; j < tensorShape.size(); j++) {
        if (j < tensorShape.size() - tileSizesIndex[i]) {
          if (j == ((tensorShape.size() - tileSizesIndex[i]) - 1) &&
              i < (brgemmOp.getNumOperands() - 1)) {
            offsets.push_back(reductionForLoop.getInductionVar());
            indices.push_back(tensorShape[j] / tensorShape[j]);
            shape.push_back(rewriter.getIndexAttr(tensorShape[j] / tensorShape[j]));
            strides.push_back(rewriter.getIndexAttr(1));

          } else {
            offsets.push_back(rewriter.getIndexAttr(0));
            indices.push_back(tensorShape[j]);
            shape.push_back(rewriter.getIndexAttr(tensorShape[j]));
            strides.push_back(rewriter.getIndexAttr(1));
          }
        } else {
          shape.push_back(rewriter.getIndexAttr(*tileItr));
          indices.push_back((*tileItr));
          strides.push_back(rewriter.getIndexAttr(1));
          offsets.push_back(
              inductionVars[i][tensorShape.size() - tileSizesIndex[i] + k]);
          k++;
          tileItr++;
        }
      }

      auto subview = rewriter.create<memref::SubViewOp>(
          brgemmOp.getLoc(), MemRefType(),
          input, offsets, shape, strides);
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
  patterns.add<LinalgOpTiling>(patterns.getContext(), options);
}

struct BrgemmLinalgTiling : public tpp::impl::BrgemmLinalgTilingBase<BrgemmLinalgTiling> {

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
