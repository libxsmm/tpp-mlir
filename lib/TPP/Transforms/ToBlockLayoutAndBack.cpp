//===- ConvertToBlockLayoutAndBack.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_PACKVNNI
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_PACKMATMUL
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_PROPAGATEPACKUNPACK
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_SIMPLIFYANDCANONICALIZEPACK
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// Helper function to create the pack operation.
static Value toPackLayoutImpl(OpBuilder &builder, Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles,
                              ArrayRef<int64_t> innerDimsPos,
                              ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(tiles, dynamicTiles, staticTiles);
  RankedTensorType result =
      linalg::PackOp::inferPackedType(cast<RankedTensorType>(input.getType()),
                                      staticTiles, innerDimsPos, outerDimsPerm);
  auto inputType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> shape = result.getShape();
  Value output =
      builder.create<tensor::EmptyOp>(loc, shape, inputType.getElementType());
  return builder.create<linalg::PackOp>(loc, input, output, innerDimsPos, tiles,
                                        /*paddingValue=*/std::nullopt,
                                        outerDimsPerm);
}

static Value handleLayout_VNNI(OpBuilder &builder, Location loc, Value input,
                               ArrayRef<OpFoldResult> tiles, int64_t kDimPos) {
  assert(tiles.size() == 1 && "expect 1 block for VNNI");
  return toPackLayoutImpl(builder, loc, input, tiles,
                          SmallVector<int64_t>{kDimPos},
                          /*outerDimsPerm=*/{});
}

static Value handleBRGemmLayout_VNNI(OpBuilder &builder, Location loc,
                                     Value input, ArrayRef<OpFoldResult> tiles,
                                     int64_t kDimPos) {
  assert(tiles.size() == 1 && "expect 1 block for VNNI");
  return toPackLayoutImpl(builder, loc, input, tiles,
                          SmallVector<int64_t>{kDimPos},
                          /*outerDimsPerm=*/{});
}

// Helper function to pack from [outer][K][inner] to [outer][K/2][inner][2].
static Value toPackLayout_VNNI(OpBuilder &builder, Location loc, Value input,
                               ArrayRef<OpFoldResult> tiles, int64_t kDimPos) {
  return handleLayout_VNNI(builder, loc, input, tiles, kDimPos);
}

// Helper function to pack from [outer][K][inner] to [outer][K/2][inner][2].
static Value toPackBRGemmLayout_VNNI(OpBuilder &builder, Location loc,
                                     Value input, ArrayRef<OpFoldResult> tiles,
                                     int64_t kDimPos) {
  return handleBRGemmLayout_VNNI(builder, loc, input, tiles, kDimPos);
}

//===----------------------------------------------------------------------===//
// MatmulOp (VNNI packing)
//===----------------------------------------------------------------------===//
// Original layout:
//      [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb][jb]
// New      layout:
//      [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb/VNNI][jb][VNNI]
FailureOr<linalg::GenericOp>
mlir::linalgx::packVNNIMatmulOp(RewriterBase &rewriter,
                                linalg::GenericOp matmulOp) {
  if (matmulOp.getInputs().size() > 0) {
    auto elementType = getElementTypeOrSelf(matmulOp.getInputs()[0].getType());
    if (!elementType.isBF16())
      return rewriter.notifyMatchFailure(matmulOp, "require bf16 type");
  }

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  auto dims = linalgx::utils::isContraction(matmulOp);
  if (failed(dims))
    return rewriter.notifyMatchFailure(matmulOp, "require matmul semantics");

  OpOperand &operandB = matmulOp->getOpOperand(1);
  auto blockingFactor =
      vnni::utils::getVnniBlockingFactor(operandB.get().getType(), matmulOp);
  if (!blockingFactor) {
    return rewriter.notifyMatchFailure(matmulOp,
                                       "unsupported blocking factor for type");
  }

  if (vnni::utils::isInVnniLayout(matmulOp)) {
    return rewriter.notifyMatchFailure(matmulOp, "already packed to VNNI");
  }

  Location loc = matmulOp.getLoc();
  SmallVector<OpFoldResult> tilesOnSmallK = {
      rewriter.getI64IntegerAttr(blockingFactor)};
  SmallVector<std::pair<Value, unsigned>> kOperands;
  matmulOp.mapIterationSpaceDimToAllOperandDims(dims->k.back(), kOperands);
  if (kOperands.size() != 2)
    return rewriter.notifyMatchFailure(matmulOp,
                                       "Invalid reduction dim operands");
  // Reshape input A.
  Value packedMatrixA =
      toPackLayout_VNNI(rewriter, loc, matmulOp.getInputs()[0], tilesOnSmallK,
                        kOperands[0].second);
  // Reshape input B.
  Value packedMatrixB = toPackLayout_VNNI(rewriter, loc, operandB.get(),
                                          tilesOnSmallK, kOperands[1].second);

  MLIRContext *ctx = matmulOp.getContext();
  AffineExpr p1, p2, r1, p3, p4, r2, r3;
  SmallVector<Value> packedInputs = {packedMatrixA, packedMatrixB};
  AffineMap mapA, mapB, mapC;
  Value matrixC = matmulOp.getOutputs()[0];

  //            IB  JB  KB  ib  jb  kb  VNNI
  bindDims(ctx, p1, p2, r1, p3, p4, r2, r3);
  mapA = AffineMap::get(/*dims=*/7, /*symbols=*/0, {p1, r1, p3, r2, r3}, ctx);
  mapB = AffineMap::get(/*dims=*/7, /*symbols=*/0, {p2, r1, r2, p4, r3}, ctx);
  mapC = AffineMap::get(/*dims=*/7, /*symbols=*/0, {p1, p2, p3, p4}, ctx);
  auto replacementOp = rewriter.create<linalg::GenericOp>(
      loc, matrixC.getType(), packedInputs, ValueRange{matrixC},
      ArrayRef<AffineMap>{mapA, mapB, mapC},
      ArrayRef<mlir::utils::IteratorType>{mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::reduction,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::reduction,
                                          mlir::utils::IteratorType::reduction},
      /*doc=*/"", /*libraryCall=*/"");

  rewriter.inlineRegionBefore(matmulOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  rewriter.replaceOp(matmulOp, replacementOp.getResult(0));
  return replacementOp;
}

//===----------------------------------------------------------------------===//
// BrgemmOp (VNNI layout)
//===----------------------------------------------------------------------===//
// Original layout: [I][J] += [R][I][K] * [R][K][J]
// New      layout: [I][J] += [R][I][K] * [R][K/VNNI][J][VNNI]
FailureOr<linalg::GenericOp>
mlir::linalgx::packVNNIBRGemmOp(RewriterBase &rewriter,
                                linalg::BatchReduceMatmulOp brgemmOp) {
  auto elementType = getElementTypeOrSelf(brgemmOp.getInputs()[0].getType());
  if (!elementType.isBF16())
    return rewriter.notifyMatchFailure(brgemmOp, "require bf16 type");

  if (brgemmOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(brgemmOp, "require static shape");

  if (brgemmOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(brgemmOp, "require tensor semantics");

  Value operandB = brgemmOp.getInputs()[1];
  // Blocking factor on the `k` dimension.
  auto blockingFactor =
      vnni::utils::getVnniBlockingFactor(operandB.getType(), brgemmOp);
  if (!blockingFactor) {
    return rewriter.notifyMatchFailure(brgemmOp,
                                       "unsupported blocking factor for type");
  }
  SmallVector<OpFoldResult> tilesOnK = {
      rewriter.getI64IntegerAttr(blockingFactor)};

  Location loc = brgemmOp.getLoc();
  // Reshape input A.
  Value packedMatrixA = toPackBRGemmLayout_VNNI(
      rewriter, loc, brgemmOp.getInputs()[0], tilesOnK, 2);
  // Reshape input B.
  Value packedMatrixB =
      toPackBRGemmLayout_VNNI(rewriter, loc, operandB, tilesOnK, 1);

  MLIRContext *ctx = brgemmOp.getContext();
  AffineExpr r1, p1, p2, r3, r4;
  AffineMap mapA, mapB, mapC;
  bindDims(ctx, r1, p1, p2, r3, r4);
  mapA = AffineMap::get(/*dims=*/5, /*symbols=*/0, {r1, p1, r3, r4}, ctx);
  mapB = AffineMap::get(/*dims=*/5, /*symbols=*/0, {r1, r3, p2, r4}, ctx);
  mapC = AffineMap::get(/*dims=*/5, /*symbols=*/0, {p1, p2}, ctx);

  auto replacementOp = rewriter.create<linalg::GenericOp>(
      loc, brgemmOp.getOutputs()[0].getType(),
      ValueRange{packedMatrixA, packedMatrixB},
      ValueRange{brgemmOp.getOutputs()[0]},
      ArrayRef<AffineMap>{mapA, mapB, mapC},
      ArrayRef<mlir::utils::IteratorType>{
          mlir::utils::IteratorType::reduction,  // b
          mlir::utils::IteratorType::parallel,   // i
          mlir::utils::IteratorType::parallel,   // j
          mlir::utils::IteratorType::reduction,  // k
          mlir::utils::IteratorType::reduction}, // k/VNNI
      /*doc=*/"", /*libraryCall=*/"");

  rewriter.inlineRegionBefore(brgemmOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  rewriter.replaceOp(brgemmOp, replacementOp.getResult(0));
  return replacementOp;
}

namespace {

static SmallVector<int64_t>
getDefaultBlockingFactors(linalg::LinalgOp linalgOp) {
  assert(linalgOp && "expect a valid linalgOp");
  auto *op = linalgOp.getOperation();
  assert(isa<linalg::MatmulOp>(op) ||
         isa<linalg::BatchMatmulOp>(op) ||
         isa<linalg::MatmulTransposeAOp>(op) ||
         isa<linalg::MatmulTransposeBOp>(op));
  return {32, 32, 32};
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

// Entry point for packing a matmul operation.
// Pack MatmulOp as following:
// [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
// CB = batch reduce dimension.
// Pack a BatchMatmulOp as following:
// [B][IB][JB][ib][jb] += [B][IB][KB][ib][kb] * [B][JB][KB][kb][jb]
// KB is the batch reduce dimension.
struct PackMatmul : public tpp::impl::PackMatmulBase<PackMatmul> {
  using PackMatmulBase::PackMatmulBase;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);

    // TODO: Add a cost function that decides whether to pack at all.
    auto packControlFn = [&](linalg::LinalgOp linalgOp)
        -> std::optional<linalg::BlockPackMatmulOptions> {
      linalg::BlockPackMatmulOptions options;
      auto *op = linalgOp.getOperation();

      // Pack only these named matmul variants.
      if (!(isa<linalg::MatmulOp>(op) ||
            isa<linalg::MatmulTransposeAOp>(op) ||
            isa<linalg::MatmulTransposeBOp>(op) ||
            isa<linalg::BatchMatmulOp>(op))) {
        return std::nullopt;
      }

      // Enforce user defined blocking factors or use defaults.
      if (!blockingFactors.empty()) {
        SmallVector<int64_t, 3> blockFactors{*blockingFactors};
        options.blockFactors = blockFactors;
      } else {
        options.blockFactors = getDefaultBlockingFactors(linalgOp);
      }

      // Allow padding to avoid double checks.
      options.allowPadding = true;

      // Adjust block factors to smaller dimensions.
      // If a dimension is smaller than the blocking factor, then
      // try to block by the dimension size.
      auto dims = linalg::inferContractionDims(linalgOp);
      if (failed(dims))
        return std::nullopt;

      OpBuilder builder(linalgOp);
      auto tileOp = cast<TilingInterface>(linalgOp.getOperation());
      SmallVector<Range> iterationDomain = tileOp.getIterationDomain(builder);

      if (std::optional<int64_t> dimM =
              linalgx::utils::getConstantRange(iterationDomain[dims->m.back()]))
        options.blockFactors[0] = std::min(*dimM, options.blockFactors[0]);
      if (std::optional<int64_t> dimN =
              linalgx::utils::getConstantRange(iterationDomain[dims->n.back()]))
        options.blockFactors[1] = std::min(*dimN, options.blockFactors[1]);
      if (std::optional<int64_t> dimK =
              linalgx::utils::getConstantRange(iterationDomain[dims->k.back()]))
        options.blockFactors[2] = std::min(*dimK, options.blockFactors[2]);

      // Apply more restrictive packing validation.
      SmallVector<OpFoldResult> tiles =
          getAsOpFoldResult(builder.getI64ArrayAttr(options.blockFactors));
      OpFoldResult tileOnI = tiles[0];
      OpFoldResult tileOnJ = tiles[1];
      OpFoldResult tileOnK = tiles[2];
      bool isBatchMatmulOp = isa<linalg::BatchMatmulOp>(linalgOp);
      size_t inc = isBatchMatmulOp ? 1 : 0;
      size_t posI = 0 + inc;
      size_t posJ = 1 + inc;
      size_t posK = 2 + inc;
      if (!linalgx::utils::validateFullTilesOnDims(
              cast<TilingInterface>(linalgOp.getOperation()),
              {tileOnI, tileOnJ, tileOnK}, {posI, posJ, posK},
              /*minTileFactor=*/1)) {
        return std::nullopt;
      }

      // Apply XSMM packing with block transpose only.
      options.lhsTransposeOuterBlocks = false;
      options.lhsTransposeInnerBlocks = false;
      options.rhsTransposeOuterBlocks = true;
      options.rhsTransposeInnerBlocks = false;

      return options;
    };
    linalg::populateBlockPackMatmulPatterns(patterns, packControlFn);
    linalg::populateLinalgDeGeneralizationPatterns(patterns);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

// Pack MatmulOp to VNNI.
struct VNNIOnMatmul : public OpRewritePattern<linalg::GenericOp> {
  VNNIOnMatmul(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit) {}
  LogicalResult matchAndRewrite(linalg::GenericOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> packedMatmul =
        mlir::linalgx::packVNNIMatmulOp(rewriter, matmulOp);
    if (failed(packedMatmul))
      return failure();
    return success();
  }
};

// Pack BRGemmOp to VNNI.
struct VNNIOnBRGemm : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  VNNIOnBRGemm(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::BatchReduceMatmulOp>(context, benefit) {}
  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> packedBRGemm =
        mlir::linalgx::packVNNIBRGemmOp(rewriter, brgemmOp);
    if (failed(packedBRGemm))
      return failure();
    return success();
  }
};

// Entry point for packing a matmul/brgemm operation to vnni format.
struct PackVNNI : public tpp::impl::PackVNNIBase<PackVNNI> {

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    patterns.add<VNNIOnMatmul, VNNIOnBRGemm>(ctx);
    linalg::populateSimplifyPackAndUnpackPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

struct PropagatePackUnPack
    : public tpp::impl::PropagatePackUnPackBase<PropagatePackUnPack> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    linalg::populateDataLayoutPropagationPatterns(
        patterns, [](OpOperand *operand) { return true; });
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

// Fold a linalg.unpack into an scf.parallel_insert.
//
// The pattern looks like:
//
// %p = linalg.pack %a into %b
// %l = scf.forall ... iter_args(%0 = %p) {
// ...
// }
// %u = linalg.unpack %l into %c
//
// We will rewrite as:
//
// %l = scf.forall ... iter_args(%0 = %a) {
// ...
// }
struct FoldUnPackIntoInsertSlice : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern<linalg::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    if (!unPackOp.getOuterDimsPerm().empty())
      return failure();
    SmallVector<int64_t> innerDimsPos =
        llvm::to_vector(unPackOp.getInnerDimsPos());
    SmallVector<int64_t> expectedDimsPos = llvm::to_vector(
        llvm::seq<int64_t>(0, unPackOp.getDestType().getRank()));
    if (innerDimsPos != expectedDimsPos)
      return failure();

    Operation *loop = unPackOp.getSource().getDefiningOp();
    if (!isa_and_nonnull<scf::ForallOp>(loop))
      return failure();
    auto forallOp = cast<scf::ForallOp>(loop);
    if (!forallOp->hasOneUse() || forallOp->getNumResults() != 1)
      return failure();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(forallOp);

    // Create a new scf.forall operation, updating its output.
    Value loopOperand =
        forallOp.getTiedOpOperand(forallOp->getResult(0))->get();
    linalg::PackOp packOp =
        dyn_cast_or_null<linalg::PackOp>(loopOperand.getDefiningOp());
    if (!packOp)
      return failure();
    Value newLoopOperand = packOp.getSource();
    SmallVector<Value> newOuts(forallOp.getOutputs());
    if (newOuts.size() != 1)
      return failure();

    newOuts.push_back(newLoopOperand);
    auto newForallOp = rewriter.create<scf::ForallOp>(
        forallOp.getLoc(), forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(), forallOp.getMixedStep(), newOuts,
        forallOp.getMapping());
    rewriter.eraseBlock(newForallOp.getBody());
    newForallOp.getRegion().takeBody(forallOp.getRegion());
    newForallOp.getBody()->addArgument(newOuts.back().getType(),
                                       newOuts.back().getLoc());

    ArrayRef<BlockArgument> bbArgs = newForallOp.getRegionIterArgs();
    assert(bbArgs.size() == 2);

    rewriter.setInsertionPointToStart(newForallOp.getBody());
    AffineExpr dim0;
    bindDims(rewriter.getContext(), dim0);
    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    auto mulMap = AffineMap::get(1, 1, {dim0 * s0});
    SmallVector<OpFoldResult> newMixedOffsets;
    for (auto ivs : llvm::enumerate(newForallOp.getInductionVars())) {
      OpFoldResult applied = affine::makeComposedFoldedAffineApply(
          rewriter, newForallOp.getLoc(), mulMap,
          {ivs.value(), unPackOp.getMixedTiles()[ivs.index()]});
      newMixedOffsets.push_back(applied);
    }

    for (Operation *operation : bbArgs.front().getUsers()) {
      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(operation)) {
        rewriter.setInsertionPoint(extractSliceOp);

        int64_t rank = unPackOp.getDestType().getRank();
        auto mixedStrides = extractSliceOp.getMixedStrides();
        auto newMixedStrides = SmallVector<OpFoldResult>(
            mixedStrides.begin() + rank, mixedStrides.end());

        auto mixedSizes = extractSliceOp.getMixedSizes();
        auto newMixedSizes = SmallVector<OpFoldResult>(
            mixedSizes.begin() + rank, mixedSizes.end());

        auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
            extractSliceOp.getLoc(), bbArgs.back(), newMixedOffsets,
            newMixedSizes, newMixedStrides);

        rewriter.replaceAllUsesWith(extractSliceOp->getResults(),
                                    newExtractSliceOp->getResults());
        continue;
      }
      if (auto parallelInsertSlice =
              dyn_cast<tensor::ParallelInsertSliceOp>(operation)) {
        rewriter.setInsertionPoint(parallelInsertSlice);

        int64_t rank = unPackOp.getDestType().getRank();
        auto mixedStrides = parallelInsertSlice.getMixedStrides();
        auto newMixedStrides = SmallVector<OpFoldResult>(
            mixedStrides.begin() + rank, mixedStrides.end());

        auto mixedSizes = parallelInsertSlice.getMixedSizes();
        auto newMixedSizes = SmallVector<OpFoldResult>(
            mixedSizes.begin() + rank, mixedSizes.end());

        auto newInsertSliceOp = rewriter.create<tensor::ParallelInsertSliceOp>(
            parallelInsertSlice.getLoc(), parallelInsertSlice.getSource(),
            bbArgs.back(), newMixedOffsets, newMixedSizes, newMixedStrides);
        rewriter.replaceAllUsesWith(parallelInsertSlice->getResults(),
                                    newInsertSliceOp->getResults());
        rewriter.eraseOp(parallelInsertSlice);
        continue;
      }
      return failure();
    }

    rewriter.replaceOp(unPackOp, newForallOp->getResults()[1]);
    return success();
  }
};

struct SimplifyAndCanonicalizePack
    : public tpp::impl::SimplifyAndCanonicalizePackBase<
          SimplifyAndCanonicalizePack> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    tpp::populateSimplifyPacking(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace

void mlir::tpp::populateSimplifyPacking(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  linalg::populateSimplifyPackAndUnpackPatterns(patterns);
  linalg::populateFoldPackUnpackIntoTensorEmptyPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  linalg::PackOp::getCanonicalizationPatterns(patterns, ctx);
  linalg::UnPackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ParallelInsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
  // Propagate packs/unpacks only through expand shapes at this point.
  // This captures the transformation scope of the replaced downstream pass.
  linalg::populateDataLayoutPropagationPatterns(
      patterns, [](OpOperand *operand) {
        return isa<tensor::ExpandShapeOp>(operand->get().getDefiningOp());
      });
  ctx->getLoadedDialect<linalg::LinalgDialect>()->getCanonicalizationPatterns(
      patterns);
  ctx->getLoadedDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
      patterns);
  patterns.add<FoldUnPackIntoInsertSlice>(ctx);
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
}
