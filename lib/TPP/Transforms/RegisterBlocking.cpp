//===- RegisterBlocking.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REGISTERBLOCKING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Attribute name used as a tile-and-fuse marker.
constexpr const static llvm::StringLiteral tiledFusedAttrName =
    "reg_tiled_fused";

// Extracts a vector of integers out of an array attribute.
template <typename IntType>
static SmallVector<IntType> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

// Returns register blocks for the innermost dims: [M, N, K]
static SmallVector<int64_t> getRegisterBlocks(Operation *op) {
  auto res = dlti::query(op, {"CPU", "reg_blocks"});
  if (failed(res))
    return {};
  auto vals = llvm::dyn_cast<ArrayAttr>(*res);
  if (!vals)
    return {};
  return extractVector<int64_t>(vals);
}

// Returns position of a dimension corresponding to the given iteration map
// and an iterator.
static std::optional<unsigned> mapIteratorToDimPos(PatternRewriter &rewriter,
                                                   AffineMap map,
                                                   unsigned iterPos) {
  return map.getResultPosition(rewriter.getAffineDimExpr(iterPos));
}

// Propagate the tile specification from producer to consumer. Example,
// Tile spec producer:  (1,  0, 0,  0, 1,  0)
// Output map producer: (i, ii, k, kk, j, jj) -> (i, ii, j, jj)
// Assuming an eltwise consumer, with map:
// (i, ii, j, jj) -> (i, ii, j, jj) the tiling specification will be:
// (1, 0, 1, 0).
static SmallVector<OpFoldResult>
remapTilesToEltwiseConsumer(linalg::LinalgOp consumer,
                            linalg::LinalgOp producer,
                            SmallVector<OpFoldResult> tilesProducer) {
  if (consumer == producer)
    return tilesProducer;

  assert(linalg::isElementwise(cast<linalg::LinalgOp>(consumer)) &&
         "Require eltwise consumer");

  assert(producer.getNumDpsInits() == 1);
  AffineMap outputMap =
      producer.getMatchingIndexingMap(&producer.getDpsInitsMutable()[0]);
  assert(outputMap.isProjectedPermutation());
  assert(outputMap.getNumDims() == tilesProducer.size());
  SmallVector<OpFoldResult> eltWiseTiles;
  for (auto expr : outputMap.getResults()) {
    eltWiseTiles.push_back(
        tilesProducer[cast<AffineDimExpr>(expr).getPosition()]);
  }
  return eltWiseTiles;
}

// Apply loop peeling to split tail iterations and allow for
// canonicalization to ensure all blocked ops operate on static values.
// Peeling is applied in reverse order from the innermost loop to ensure
// that all tiling loops are affected.
//
// Result is ignored as peeling can fail when tiling cleanly divides
// a dimension which means there is no need for peeling anyway.
static void peelTiledLoops(ArrayRef<Operation *> loops,
                           PatternRewriter &rewriter) {
  for (Operation *loop : llvm::reverse(loops)) {
    auto forOp = dyn_cast<scf::ForOp>(loop);
    assert(forOp && "requires scf.for operation");
    scf::ForOp partialIteration;
    (void)scf::peelForLoopAndSimplifyBounds(
        rewriter, dyn_cast<scf::ForOp>(loop), partialIteration);
  }
}

// Tile and fuse contraction ops.
// Uses a contraction op as a root and greedily fuses its elementwise consumers.
// Additionally, data initialization producers like fills are also fused.
//
// Peeling is applied to tiling loops to eliminate dynamic shapes in cases
// when original operands' shapes are not perfectly divisible by the tiles.
//
// Currently, only the innermost GEMM dimensions are tiled.
// If present, other parallel dimensions are tiled by one.
struct TileAndFuseContraction : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  TileAndFuseContraction(MLIRContext *ctx, tpp::RegisterBlockingOptions options)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx), options(options) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp contractOp,
                                PatternRewriter &rewriter) const override {
    if (!contractOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(contractOp,
                                         "expects tensor semantics");
    if (contractOp.hasDynamicShape())
      return rewriter.notifyMatchFailure(contractOp, "expects static shape");
    if (llvm::any_of(contractOp.getIndexingMapsArray(), [](AffineMap map) {
          return !map.isProjectedPermutation();
        }))
      return rewriter.notifyMatchFailure(contractOp,
                                         "expects projected permutation maps");

    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(contractOp);
    if (failed(dims))
      return rewriter.notifyMatchFailure(contractOp, "not a contraction");

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
          contractOp, "expects only 2 parallel (M and N) non-batch dimensions");

    SmallVector<int64_t> regBlocks = options.blocks;
    if (regBlocks.empty())
      regBlocks = getRegisterBlocks(contractOp);
    if (regBlocks.size() != 3)
      return rewriter.notifyMatchFailure(contractOp,
                                         "invalid register blocking");

    // Tile only along parallel dimensions to allow for fusion.
    //
    // The register blocking is applied to the remaining innermost dimension.
    // Scalarize batch and other parallel dimensions - it is a fallback option,
    // ideally user should've preprocessed them earlier.
    SmallVector<int64_t> parallelTileSizes(contractOp.getNumLoops(), 1);
    for (auto dim : dims->k)
      parallelTileSizes[dim] = 0;
    parallelTileSizes[dims->m[0]] = regBlocks[0];
    parallelTileSizes[dims->n[0]] = regBlocks[1];

    // Greedily fuse elementwise consumers.
    auto isFusable = [](linalg::LinalgOp producer) -> linalg::LinalgOp {
      // Producer constraints.
      if (producer->getNumResults() != 1)
        return nullptr;
      auto producerRes = producer->getResult(0);
      // Multiple uses even within the same user introduce recomputation.
      // To avoid worst-case GEMM duplication, fusion is more conservative here.
      if (!producerRes.hasOneUse())
        return nullptr;
      // Consumer constraints.
      auto consumer = dyn_cast<linalg::LinalgOp>(*producerRes.user_begin());
      if (!consumer || consumer->getNumResults() != 1 ||
          !linalg::isElementwise(consumer))
        return nullptr;
      // Require same iteration space.
      if (producer.getNumParallelLoops() != consumer.getNumParallelLoops())
        return nullptr;
      // Require same shapes to avoid any dimension permutations.
      // TODO: Relax this constraint.
      if (producer.getShape(&producer.getDpsInitsMutable()[0]) !=
          consumer.getShape(&consumer.getDpsInitsMutable()[0]))
        return nullptr;
      return consumer;
    };
    // Get the last fusable consumer to use for tiling and fusion root.
    linalg::LinalgOp consumer = contractOp;
    while (auto nextConsumer = isFusable(consumer)) {
      consumer = nextConsumer;
    }
    // Map original contraction tile sizes to consumer which has only
    // parallel iterators.
    SmallVector<OpFoldResult> consumerTiles = remapTilesToEltwiseConsumer(
        consumer, contractOp,
        getAsOpFoldResult(rewriter.getI64ArrayAttr(parallelTileSizes)));

    scf::SCFTilingOptions options;
    options.setTileSizes(consumerTiles);
    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(options);
    // Fuse only linalg eltwise ops and the original contraction.
    // However, its profitability is unclear at register (nanokernel) level and
    // it would complicate post-processing logic, thus, disable for now.
    DominanceInfo domInfo(contractOp);
    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand)
        -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
      auto candidateOp =
          dyn_cast_or_null<linalg::LinalgOp>(originalProducer.getOwner());
      if (!candidateOp)
        return std::nullopt;
      if (candidateOp != contractOp && !linalg::isElementwise(candidateOp))
        return std::nullopt;
      // Fuse only contraction epilogue and data initialization ops.
      if (!isa<linalg::FillOp, linalg::BroadcastOp>(candidateOp) &&
          !domInfo.dominates(contractOp, candidateOp))
        return std::nullopt;
      scf::SCFTileAndFuseOptions::ControlFnResult res;
      res.yieldProducerReplacement = false;
      return res;
    };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
        scf::tileConsumerAndFuseProducersUsingSCF(
            rewriter, cast<TilingInterface>(consumer.getOperation()),
            tileAndFuseOptions);
    if (failed(tileAndFuseResult))
      return rewriter.notifyMatchFailure(
          consumer, "failed to tile and fuse with op as root");

    rewriter.replaceOp(consumer,
                       tileAndFuseResult->replacements[consumer->getResult(0)]);

    // Mark all preprocessed ops to allow easier matching for further passes.
    for (auto *op : tileAndFuseResult->tiledAndFusedOps)
      op->setDiscardableAttr(tiledFusedAttrName, rewriter.getUnitAttr());

    // Tiling cleanup.
    // It is easier to post-process loops now without need for complex matching.
    peelTiledLoops(SmallVector<Operation *>(tileAndFuseResult->loops.begin(),
                                            tileAndFuseResult->loops.end()),
                   rewriter);

    return success();
  }

private:
  tpp::RegisterBlockingOptions options;
};

// Tile reduction dimensions of previously tile-and-fused contraction ops.
// It is a follow-up pattern to finalize tiling of a root contraction op
// processed by the tile and fuse pattern.
//
// Peeling is applied to tiling loops to eliminate dynamic shapes in cases
// when original operands' shapes are not perfectly divisible by the tiles.
//
// Currently, only the GEMM K-dimension is tiled.
// If present, VNNI dimensions remains untiled and other reduction dimensions
// are tiled by one.
struct TileContractionReductionDims
    : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  TileContractionReductionDims(MLIRContext *ctx,
                               tpp::RegisterBlockingOptions options)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(ctx), options(options) {}

  LogicalResult matchAndRewrite(linalg::LinalgOp contractOp,
                                PatternRewriter &rewriter) const override {
    // Only target previously parallel dim tiled and fused operations.
    auto tiledFusedAttr = contractOp->getDiscardableAttr(tiledFusedAttrName);
    if (!tiledFusedAttr)
      return rewriter.notifyMatchFailure(
          contractOp, "only targets previously register blocked ops");

    if (!contractOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(contractOp,
                                         "expects tensor semantics");
    if (contractOp.hasDynamicShape())
      return rewriter.notifyMatchFailure(contractOp, "expects static shape");

    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(contractOp);
    if (failed(dims))
      return rewriter.notifyMatchFailure(contractOp, "not a contraction");

    // Matching stil requires parallel dimensions to allow for VNNI detection.
    // TODO: Relax constraints.
    if (dims->m.size() != 1 || dims->n.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "expects only 2 parallel (M and N) non-batch dimensions");

    SmallVector<int64_t> regBlocks = options.blocks;
    if (regBlocks.empty())
      regBlocks = getRegisterBlocks(contractOp);
    if (regBlocks.size() != 3)
      return rewriter.notifyMatchFailure(contractOp,
                                         "invalid register blocking");

    auto matA = contractOp->getOperand(0);
    unsigned rankA = dyn_cast<ShapedType>(matA.getType()).getRank();
    AffineMap mapA =
        contractOp.getMatchingIndexingMap(&contractOp->getOpOperand(0));

    // Find the innermost reduction dimension for tiling.
    // NOTE: It is assumed that all batch-reduce dimensions are outer w.r.t.
    //       K-dim reduce dimensions.
    // In case of VNNI, take the second inner dimension as the VNNI
    // dimension is guaranteed to be the innermost.
    bool isVnni = vnni::utils::isInVnniLayout(contractOp);
    std::optional<unsigned> dimVnni = std::nullopt;
    if (isVnni)
      dimVnni =
          dyn_cast<AffineDimExpr>(mapA.getResult(rankA - 1)).getPosition();
    unsigned dimK = 0;
    int innermostDim = -1;
    for (auto pos : dims->k) {
      auto dimPos = mapIteratorToDimPos(rewriter, mapA, pos);
      assert(dimPos && "failed to map iterator to dim");
      if (static_cast<int>(*dimPos) > innermostDim &&
          (!isVnni || pos != *dimVnni)) {
        innermostDim = *dimPos;
        dimK = pos;
      }
    }

    // Tile only the innermost K-dim reduction dimension.
    // Do not tile the VNNI dimension if present.
    // Disable tiling along parallel dimensions to avoid unnecessary work.
    //
    // Scalarize other reduction dimensions - it is a fallback option,
    // ideally user should've preprocessed them earlier.
    SmallVector<int64_t> reductionTileSizes(contractOp.getNumLoops(), 0);
    for (auto dim : dims->k)
      reductionTileSizes[dim] = 1;
    if (isVnni)
      reductionTileSizes[*dimVnni] = 0;
    reductionTileSizes[dimK] = regBlocks[2];

    // Place parallel dimensions first as outer loops.
    // Move batch-reduce dimensions inside, then K-dim reductions.
    // Keep VNNI innermost if present.
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
    tilingOptions.setTileSizes(reductionTileSizes);
    tilingOptions.setInterchange(interchange);

    FailureOr<linalg::TiledLinalgOp> tiledOp =
        linalg::tileLinalgOp(rewriter, contractOp, tilingOptions);
    if (failed(tiledOp))
      return rewriter.notifyMatchFailure(contractOp,
                                         "failed to tile reduction dims");
    rewriter.replaceOp(contractOp, tiledOp->tensorResults);

    // Tiling cleanup.
    // It is easier to post-process loops now without need for complex matching.
    peelTiledLoops(tiledOp->loops, rewriter);

    return success();
  }

private:
  tpp::RegisterBlockingOptions options;
};

struct RegisterBlocking
    : public tpp::impl::RegisterBlockingBase<RegisterBlocking> {
  using RegisterBlockingBase::RegisterBlockingBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto op = getOperation();

    tpp::RegisterBlockingOptions options;
    options.blocks = SmallVector<int64_t>{*blocks};

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);

    // First, tile and fuse contraction along parallel dimensions.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<TileAndFuseContraction>(ctx, options);
      if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
        return signalPassFailure();
    }
    // Canonicalization patterns to remove dynamic dimensions after loop
    // peeling.
    FrozenRewritePatternSet cleanupPatterns;
    {
      RewritePatternSet patterns(ctx);
      ctx->getLoadedDialect<linalg::LinalgDialect>()
          ->getCanonicalizationPatterns(patterns);
      ctx->getLoadedDialect<tensor::TensorDialect>()
          ->getCanonicalizationPatterns(patterns);
      tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
      tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
      tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
      scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
      cleanupPatterns = std::move(patterns);
    }
    GreedyRewriteConfig cleanupConfig;
    if (failed(applyPatternsGreedily(op, cleanupPatterns, cleanupConfig)))
      return signalPassFailure();

    // Then tile reduction dimensions.
    {
      RewritePatternSet patterns(ctx);
      patterns.add<TileContractionReductionDims>(ctx, options);
      if (failed(applyPatternsGreedily(op, std::move(patterns), config)))
        return signalPassFailure();
    }
    if (failed(applyPatternsGreedily(op, cleanupPatterns, cleanupConfig)))
      return signalPassFailure();

    // TODO: Add tiling and fusion for remaining ops like unfused eltwise
    //       operations.
  }
};
} // namespace
