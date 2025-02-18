//===- LinalgDeGeneralize.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGDEGENERALIZE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct LinalgDeGeneralize
    : tpp::impl::LinalgDeGeneralizeBase<LinalgDeGeneralize> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    (void)applyPatternsGreedily(func.getBody(), std::move(patterns));
  }
};

// From linalg.generic to linalg.matmul.
struct MatmulOpDeGeneralizationPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isMatmulOp(linalg::GenericOp linalgOp) const {
    using namespace mlir::structured_match;
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, linalgOp.getContext());
    };
    AffineExpr i, j, k;
    bindDims(linalgOp->getContext(), i, j, k);
    auto mapList = infer({{i, k}, {k, j}, {i, j}});
    auto matmulMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(2)))
            .operation(NumRegions(EqualsTo(1)))
            .dim(MatchAll(), {mlir::utils::IteratorType::reduction,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::parallel})
            .input(MatchOne(0), HasMap(EqualsTo(mapList[0])))
            .input(MatchOne(1), HasMap(EqualsTo(mapList[1])))
            .output(MatchOne(0), HasMap(EqualsTo(mapList[2])))
            .region(MatchOne(0), WithOpChain<arith::MulFOp, arith::AddFOp>(
                                     /*captures=*/nullptr));
    return matmulMatcher.match(linalgOp);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isMatmulOp(linalgOp))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
    SmallVector<Value> outputOperands = linalgOp.getDpsInits();
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

// From linalg.generic to linalg.batch_reduce_matmul.
struct BatchReduceOpDeGeneralizationPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isBrgemmOp(linalg::GenericOp linalgOp) const {
    using namespace mlir::structured_match;
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, linalgOp.getContext());
    };
    AffineExpr r, i, j, k;
    bindDims(linalgOp->getContext(), r, i, j, k);
    auto mapList = infer({{r, i, k}, {r, k, j}, {i, j}});
    auto brgemmMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(2)))
            .operation(NumRegions(EqualsTo(1)))
            .dim(MatchAll(), {mlir::utils::IteratorType::reduction,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::parallel,
                              mlir::utils::IteratorType::reduction})
            .input(MatchOne(0), HasMap(EqualsTo(mapList[0])))
            .input(MatchOne(1), HasMap(EqualsTo(mapList[1])))
            .output(MatchOne(0), HasMap(EqualsTo(mapList[2])))
            .region(MatchOne(0), WithOpChain<arith::MulFOp, arith::AddFOp>(
                                     /*captures=*/nullptr));
    return brgemmMatcher.match(linalgOp);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isBrgemmOp(linalgOp))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
    SmallVector<Value> outputOperands = linalgOp.getDpsInits();
    rewriter.replaceOpWithNewOp<linalg::BatchReduceMatmulOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

// From linalg.generic to linalg.transpose.
struct TransposeOpPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isIdentityPermutation(ArrayRef<int64_t> permutation) const {
    for (auto i : llvm::seq<int64_t>(0, permutation.size()))
      if (permutation[i] != i)
        return false;
    return true;
  }

  FailureOr<SmallVector<int64_t>>
  getPermutationFromMap(AffineMap map, int64_t numLoops) const {
    assert(map.isProjectedPermutation());
    if (numLoops != map.getNumResults())
      return failure();

    SmallVector<int64_t> perm;
    for (auto dim : llvm::seq<int64_t>(0, numLoops)) {
      auto dimExpr = getAffineDimExpr(dim, map.getContext());
      for (auto [idx, result] : llvm::enumerate(map.getResults())) {
        if (result == dimExpr)
          perm.push_back(idx);
      }
    }

    if (isIdentityPermutation(perm))
      return failure();
    return perm;
  }

  FailureOr<SmallVector<int64_t>>
  isTransposeOp(linalg::GenericOp linalgOp) const {
    using namespace mlir::structured_match;
    AffineMap inputMap;
    auto transposeMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(1)))
            .operation(NumRegions(EqualsTo(1)))
            .dim(MatchAll(), mlir::utils::IteratorType::parallel)
            .input(MatchOne(0), HasMap(ProjectedPermutation(), &inputMap))
            .output(MatchOne(0), HasMap(Identity()))
            .region(MatchOne(0),
                    WithSingleOp<linalg::YieldOp>(/*captures=*/nullptr));
    if (!transposeMatcher.match(linalgOp))
      return failure();
    return getPermutationFromMap(inputMap, linalgOp.getNumLoops());
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    auto maybePerm = isTransposeOp(linalgOp);
    if (failed(maybePerm))
      return failure();
    Value inputOperand = linalgOp.getDpsInputs()[0];
    Value outputOperand = linalgOp.getDpsInits()[0];
    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(linalgOp, inputOperand,
                                                     outputOperand, *maybePerm);
    return success();
  }
};

// From linalg.generic to linalg.fillOp.
struct FillOpDeGeneralizationPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  bool isFillOp(linalg::GenericOp linalgOp) const {
    using namespace mlir::structured_match;
    auto fillMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(1)))
            .operation(NumRegions(EqualsTo(1)))
            .dim(MatchAll(), mlir::utils::IteratorType::parallel)
            .output(MatchAll(), HasMap(Identity()))
            .input(MatchAll(), HasMap(BroadcastableProjectedPermutation()))
            .input(MatchAll(), HasRank({HasRank::SCALAR}))
            .region(MatchOne(0),
                    WithSingleOp<linalg::YieldOp>(/*captures=*/nullptr));
    return fillMatcher.match(linalgOp);
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isFillOp(linalgOp))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getDpsInputs();
    SmallVector<Value> outputOperands = linalgOp.getDpsInits();
    rewriter.replaceOpWithNewOp<linalg::FillOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

} // namespace

void mlir::linalg::populateLinalgDeGeneralizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FillOpDeGeneralizationPattern, MatmulOpDeGeneralizationPattern,
               BatchReduceOpDeGeneralizationPattern, TransposeOpPattern>(
      patterns.getContext());
}
