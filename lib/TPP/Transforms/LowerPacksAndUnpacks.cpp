//===- LowerPacksAndUnpacks.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LOWERPACKSANDUNPACKS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Tile by one up to the first dimension involved in packing.
// 1) In case the pack as a consumer we can tile up to the first tiled dim.
// 2) If the pack is a standalone op just make it 2d
template <typename OpTy>
static SmallVector<int64_t> getTileSizes(OpTy packingOp,
                                         bool isConsumer = false) {
  static_assert(llvm::is_one_of<OpTy, linalg::PackOp, linalg::UnPackOp>::value,
                "applies to only pack or unpack operations");
  SmallVector<int64_t> tiledDims = llvm::to_vector(packingOp.getInnerDimsPos());
  assert(!tiledDims.empty());
  if (isConsumer) {
    int64_t upTo = *std::min_element(tiledDims.begin(), tiledDims.end());
    return SmallVector<int64_t>(upTo, 1);
  }
  if (std::is_same<OpTy, linalg::PackOp>::value) {
    int64_t upTo = packingOp.getDestType().getRank() - 2;
    return SmallVector<int64_t>(upTo, 1);
  }
  return llvm::to_vector(packingOp.getStaticTiles());
}

// Tile`operation` using `tileSizes`, the loops are annotated with
// `kLoopParallel` and later converted to scf.forall.
static FailureOr<scf::SCFTilingResult> tileOp(RewriterBase &rewriter,
                                              TilingInterface operation,
                                              ArrayRef<int64_t> tileSizes) {
  auto options = scf::SCFTilingOptions().setTileSizes(
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes));
  FailureOr<scf::SCFTilingResult> tilingResult =
      scf::tileUsingSCF(rewriter, operation, options);
  if (failed(tilingResult))
    return failure();
  if (!tilingResult->loops.empty()) {
    tilingResult->loops[0]->setAttr(
        linalgx::utils::kLoopParallel,
        rewriter.getStringAttr(linalgx::utils::kLoopRoot));
  }
  return tilingResult;
}

// Fuse producer and consumer pack. Standalone packs are tiled into 2d tiles.
static void fuseOrTilePacks(RewriterBase &rewriter, FunctionOpInterface func) {
  SmallVector<linalg::PackOp> chainedPackOps;
  SmallVector<linalg::PackOp> otherPacks;
  SmallVector<linalg::UnPackOp> unPacks;
  func->walk<WalkOrder::PostOrder>([&](linalg::PackOp consumerPackOp) {
    Value source = consumerPackOp.getSource();
    linalg::PackOp producerPackOp =
        dyn_cast_or_null<linalg::PackOp>(source.getDefiningOp());
    if (producerPackOp)
      chainedPackOps.push_back(consumerPackOp);
    else
      otherPacks.push_back(consumerPackOp);
  });
  func->walk<WalkOrder::PostOrder>(
      [&](linalg::UnPackOp unpackOp) { unPacks.push_back(unpackOp); });

  // Tile and fuse.
  for (auto consumerPackOp : chainedPackOps) {
    // Step 2. Tile the operation.
    SmallVector<int64_t> tileSizes =
        getTileSizes(consumerPackOp, /*isConsumer=*/true);
    if (tileSizes.empty())
      continue;
    auto tilingResult =
        tileOp(rewriter, cast<TilingInterface>(consumerPackOp.getOperation()),
               tileSizes);
    if (failed(tilingResult))
      continue;
    auto tiledPack = dyn_cast<linalg::PackOp>(tilingResult->tiledOps.back());
    assert(tiledPack);
    // Step 3. Fuse consumer and producer.
    auto forLoops =
        llvm::to_vector(llvm::map_range(tilingResult->loops, [](Operation *op) {
          return cast<LoopLikeOpInterface>(op);
        }));
    std::optional<scf::SCFFuseProducerOfSliceResult> fusedProducer =
        scf::tileAndFuseProducerOfSlice(
            rewriter,
            cast<tensor::ExtractSliceOp>(tiledPack.getSource().getDefiningOp()),
            forLoops);
    if (!fusedProducer)
      continue;
    rewriter.replaceOp(consumerPackOp, tilingResult->replacements);
  }

  // Tile packs.
  for (auto packOp : otherPacks) {
    SmallVector<int64_t> tileSizes = getTileSizes(packOp);
    if (tileSizes.empty())
      continue;
    auto tilingResult = tileOp(
        rewriter, cast<TilingInterface>(packOp.getOperation()), tileSizes);
    if (failed(tilingResult))
      continue;
    rewriter.replaceOp(packOp, tilingResult->replacements);
  }

  // Tile unpacks.
  for (auto unPackOp : unPacks) {
    SmallVector<int64_t> tileSizes = getTileSizes(unPackOp);
    if (tileSizes.empty())
      continue;
    auto tilingResult = tileOp(
        rewriter, cast<TilingInterface>(unPackOp.getOperation()), tileSizes);
    if (failed(tilingResult))
      continue;
    rewriter.replaceOp(unPackOp, tilingResult->replacements);
  }
}

// A wrapper pattern that calls linalg::lowerPack on linalg::PackOp. It lowers
// a linalg.pack op to tensor.pad + tensor.expand_shape + linalg.transpose ops.
struct LowerPackPattern : public OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern<linalg::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PackOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::LowerPackResult> res = linalg::lowerPack(rewriter, op);
    if (failed(res)) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to pad + expand + transpose");
    }
    return success();
  }
};

// A wrapper pattern that calls linalg::lowerUnPack on linalg::UnPackOp. It
// lowers a linalg.unpack op to tensor.empty + linalg.transpose +
// tensor.collapse_shape + tensor.extract_slice ops.
struct LowerUnPackPattern : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern<linalg::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::UnPackOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(linalg::lowerUnPack(rewriter, op))) {
      return rewriter.notifyMatchFailure(
          op, "cannot lower to empty + transpose + reshape + extract_slice");
    }
    return success();
  }
};

class LowerPacksAndUnPacks
    : public tpp::impl::LowerPacksAndUnPacksBase<LowerPacksAndUnPacks> {
  void runOnOperation() override {

    // Step1. Tile and fuse pack consumer and producer.
    auto *ctx = &getContext();
    IRRewriter rewriter(ctx);
    fuseOrTilePacks(rewriter, getOperation());

    // Step2. Raise scf.for to scf.forall.
    {
      RewritePatternSet patterns(ctx);
      linalgx::utils::populateScfForToForAllRewritePattern(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    // Step3. Simplify packs and unpacks.
    {
      RewritePatternSet patterns(ctx);
      mlir::tpp::populateSimplifyPacking(patterns);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    // Step4. Generalize to linalg.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<LowerPackPattern, LowerUnPackPattern>(ctx);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    // Step5. Fallback on tile by one + generalization patterns.
    {
      IRRewriter rewriter(ctx);
      getOperation()->walk([&](linalg::UnPackOp unPackOp) {
        scf::SCFTilingOptions unpackTilingOptions;
        SmallVector<int64_t> tiles(unPackOp.getDestType().getRank(), 1);
        unpackTilingOptions.setTileSizes(getAsIndexOpFoldResult(ctx, tiles));
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(unPackOp.getOperation()),
            unpackTilingOptions);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(unPackOp, tilingResult->replacements);
      });
      getOperation()->walk([&](linalg::PackOp packOp) {
        SmallVector<int64_t> tiles(packOp.getSourceType().getRank(), 1);
        scf::SCFTilingOptions packTilingOptions;
        packTilingOptions.setTileSizes(getAsIndexOpFoldResult(ctx, tiles));
        FailureOr<scf::SCFTilingResult> tilingResult = scf::tileUsingSCF(
            rewriter, cast<TilingInterface>(packOp.getOperation()),
            packTilingOptions);
        if (failed(tilingResult))
          return signalPassFailure();
        rewriter.replaceOp(packOp, tilingResult->replacements);
      });
      RewritePatternSet patterns(&getContext());
      patterns.add<linalg::DecomposeOuterUnitDimsUnPackOpPattern,
                   linalg::DecomposeOuterUnitDimsPackOpPattern>(&getContext());
      tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Step6. Canonicalize.
    {
      RewritePatternSet patterns(ctx);
      linalg::GenericOp::getCanonicalizationPatterns(patterns, ctx);
      ctx->getLoadedDialect<linalg::LinalgDialect>()
          ->getCanonicalizationPatterns(patterns);
      ctx->getLoadedDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
  }
};

} // end namespace
