//===-X86Vectorizer.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_X86VECTORIZER
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

// template <typename IntType>
// static SmallVector<IntType> extractVector(ArrayAttr arrayAttr) {
//   return llvm::to_vector(llvm::map_range(
//       arrayAttr.getAsRange<IntegerAttr>(),
//       [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
// }

static SmallVector<OpFoldResult> getRegisterBlocks(Operation *op) {
  auto res = dlti::query(op, {"CPU", "reg_blocks"});
  if (failed(res))
    return {};

  auto vals = llvm::dyn_cast<ArrayAttr>(*res);
  if (!vals)
    return {};

  return getAsOpFoldResult(vals);
}

struct VectorizeMatmul : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(matmulOp, "expects tensor semantics");

    if (matmulOp.hasDynamicShape())
      return rewriter.notifyMatchFailure(matmulOp, "expects static shape");

    SmallVector<OpFoldResult> regBlocks = getRegisterBlocks(matmulOp);
    if (regBlocks.size() != 3)
      return rewriter.notifyMatchFailure(matmulOp, "invalid register blocking");

    scf::SCFTilingOptions options;
    options.setTileSizes(regBlocks);
    FailureOr<scf::SCFTilingResult> tilingRes = scf::tileUsingSCF(
        rewriter, cast<TilingInterface>(matmulOp.getOperation()), options);

    if (failed(tilingRes))
      return rewriter.notifyMatchFailure(matmulOp, "failed to block");

    rewriter.replaceOp(matmulOp, tilingRes->mergeResult.replacements);

    // Apply loop peeling to split tail partial iterations and allow for
    // canonicalization to ensure all blocked ops operate on static values. The
    // peeling is applied in reverse order from the inner-most loop to ensure
    // that only and all tiling loops are affected.
    //
    // Result is ignored as peeling can fail when tiling cleanly divides
    // a dimension which means there is no need for peeling anyway.
    for (LoopLikeOpInterface loop : llvm::reverse(tilingRes->loops)) {
      scf::ForOp partialIteration;
      (void)scf::peelForLoopAndSimplifyBounds(rewriter, cast<scf::ForOp>(loop),
                                              partialIteration);
    }

    return success();
  }
};

struct X86Vectorizer
    : public impl::X86VectorizerBase<X86Vectorizer> {
  using X86VectorizerBase::X86VectorizerBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<VectorizeMatmul>(ctx);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
