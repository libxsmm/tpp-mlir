//===-RegisterBlocking.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/VNNIUtils.h"

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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REGISTERBLOCKING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

template <typename IntType>
static SmallVector<IntType> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

// Returns register blocks for innermost dims: [M, N, K]
static SmallVector<int64_t> getRegisterBlocks(Operation *op) {
  auto res = dlti::query(op, {"CPU", "reg_blocks"});
  if (failed(res))
    return {};
  auto vals = llvm::dyn_cast<ArrayAttr>(*res);
  if (!vals)
    return {};
  return extractVector<int64_t>(vals);
}

static std::optional<unsigned>
mapIteratorToDim(PatternRewriter &rewriter, AffineMap map, unsigned iterPos) {
  return map.getResultPosition(rewriter.getAffineDimExpr(iterPos));
}

struct RegBlockContraction : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(matmulOp, "expects tensor semantics");

    if (matmulOp.hasDynamicShape())
      return rewriter.notifyMatchFailure(matmulOp, "expects static shape");

    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(matmulOp);
    if (failed(dims))
      return rewriter.notifyMatchFailure(matmulOp, "not a contraction");

    // Matching is constrained to support only one M and one N dimensions.
    // If multiple are present then it is unclear what they represent and
    // how the register blocking (currently assumed to control only 3
    // dimensions) maps to them.
    // This could be generalized or the constrain can remain in place if
    // the operation is expected to be preprocessed earlier.
    //
    // Multiple reduction dimensions must be supported to handle VNNI and
    // BRGEMM cases.
    if (dims->m.size() != 1 || dims->n.size() != 1)
      return rewriter.notifyMatchFailure(
          matmulOp, "expects at only 2 parallel non-batch dimensions");

    SmallVector<int64_t> regBlocks = getRegisterBlocks(matmulOp);
    if (regBlocks.size() != 3)
      return rewriter.notifyMatchFailure(matmulOp, "invalid register blocking");

    auto matA = matmulOp->getOperand(0);
    unsigned rankA = dyn_cast<ShapedType>(matA.getType()).getRank();
    AffineMap mapA =
        matmulOp.getMatchingIndexingMap(&matmulOp->getOpOperand(0));

    // Find the innermost reduction dimension for tiling.
    // In case of VNNI, take the second inner dimension as the VNNI
    // dimension is guaranteed to be the innermost.
    bool isVnni = vnni::utils::isInVnniLayout(matmulOp);
    std::optional<unsigned> dimVnni = std::nullopt;
    if (isVnni)
      dimVnni =
          dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1)).getPosition();
    unsigned dimK = 0;
    unsigned innermostDim = 0;
    for (auto pos : dims->k) {
      auto dimPos = mapIteratorToDim(rewriter, mapA, pos);
      assert(dimPos && "failed to map iterator to dim");
      if (*dimPos > innermostDim && (!isVnni || pos != *dimVnni)) {
        innermostDim = *dimPos;
        dimK = pos;
      }
    }

    // The register blocking is applied to the remaining innermost dimensions.
    // NOTE: It is assumed that all batch-reduce dimensions are outer w.r.t.
    //       K-dim reduce dimensions.
    //
    // Scalarize batch dimensions - it is a fallback option, ideally
    // user should've preprocessed batch dimension earlier.
    // Do not tile the VNNI dimension if present.
    //
    // TODO: Move all the dimension analysis and interchanges into separate
    // contraction canonicalization before vectorization.
    SmallVector<int64_t> tileSizes(matmulOp.getNumLoops(), 1);
    if (isVnni)
      tileSizes[*dimVnni] = 0;
    tileSizes[dims->m[0]] = regBlocks[0];
    tileSizes[dims->n[0]] = regBlocks[1];

    // Place parallel dimensions first as outer loops.
    // Move batch-reduce dimensions inside, then K-dim reductions.
    SmallVector<unsigned> interchange;
    interchange.append(dims->batch);
    interchange.append(dims->m);
    interchange.append(dims->n);
    for (auto redDim : dims->k) {
      if (redDim != dimK && ((!isVnni || redDim != *dimVnni)))
        interchange.push_back(redDim);
    }
    interchange.push_back(dimK);
    if (isVnni)
      interchange.push_back(*dimVnni);

    // Apply tiling and replace the original op.
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setLoopType(linalg::LinalgTilingLoopType::Loops);
    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setInterchange(interchange);

    FailureOr<linalg::TiledLinalgOp> tiledOp =
        linalg::tileLinalgOp(rewriter, matmulOp, tilingOptions);
    if (failed(tiledOp))
      return rewriter.notifyMatchFailure(matmulOp, "failed to block");

    rewriter.replaceOp(matmulOp, tiledOp->tensorResults);

    // Tiling cleanup.
    // It is easier to post-process loops now without need for complex matching.
    //
    // Apply loop peeling to split tail iterations and allow for
    // canonicalization to ensure all blocked ops operate on static values.
    // Peeling is applied in reverse order from the innermost loop to ensure
    // that only and all tiling loops are affected.
    //
    // Result is ignored as peeling can fail when tiling cleanly divides
    // a dimension which means there is no need for peeling anyway.
    for (Operation *loop : llvm::reverse(tiledOp->loops)) {
      scf::ForOp partialIteration;
      (void)scf::peelForLoopAndSimplifyBounds(
          rewriter, dyn_cast<scf::ForOp>(loop), partialIteration);
    }

    return success();
  }
};

struct RegisterBlocking : public impl::RegisterBlockingBase<RegisterBlocking> {
  using RegisterBlockingBase::RegisterBlockingBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<RegBlockContraction>(ctx);

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};
} // namespace tpp
} // namespace mlir
