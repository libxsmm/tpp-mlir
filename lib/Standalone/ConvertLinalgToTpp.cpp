//===- LinalgConvertToTpp.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppOps.h"
#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "Standalone/Transforms.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::tpp;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

#define DEBUG_TYPE "linalg-convert-to-tpp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace {

// TODO: evaluate transform dialect.

// Tiling function to remove all but the zero and first dimension.
// Tile of zero means no tiling on this dimension. The other
// dimensions are materialized as loops by tiling with a factor
// of 1.
static SmallVector<Value, 4> getTileSizes(OpBuilder &builder,
                                          linalg::LinalgOp linalgOp) {
  SmallVector<Value, 4> tppTiles;
  size_t numberOfLoops = linalgOp.getNumLoops();
  for (size_t i = 0; i < numberOfLoops; i++)
    tppTiles.push_back(
        builder.createOrFold<arith::ConstantIndexOp>(linalgOp.getLoc(), 1));
  Value zeroVal =
      builder.createOrFold<arith::ConstantIndexOp>(linalgOp.getLoc(), 0);
  tppTiles[numberOfLoops - 1] = zeroVal;
  tppTiles[numberOfLoops - 2] = zeroVal;
  return tppTiles;
}

static MemRefType dropUnitDims(MemRefType inputType, ArrayRef<int64_t> offsets,
                               ArrayRef<int64_t> sizes,
                               ArrayRef<int64_t> strides) {
  SmallVector<int64_t> targetShape = llvm::to_vector(
      llvm::make_filter_range(sizes, [](int64_t sz) { return sz != 1; }));
  Type rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      targetShape, inputType, offsets, sizes, strides);
  return canonicalizeStridedLayout(rankReducedType.cast<MemRefType>());
}

// Reduce rank for 'input' by dropping unit dimension.
static Value rankReducingSubviewDroppingUnitDims(OpBuilder &builder,
                                                 Location loc, Value input) {
  MemRefType inputType = input.getType().cast<MemRefType>();
  assert(inputType.hasStaticShape() && "expect static shape");
  SmallVector<int64_t> subViewOffsets(inputType.getRank(), 0);
  SmallVector<int64_t> subViewStrides(inputType.getRank(), 1);
  ArrayRef<int64_t> subViewSizes = inputType.getShape();
  MemRefType resultType =
      dropUnitDims(inputType, subViewOffsets, subViewSizes, subViewStrides);
  if (canonicalizeStridedLayout(resultType) ==
      canonicalizeStridedLayout(inputType))
    return input;
  return builder.create<memref::SubViewOp>(
      loc, resultType, input, subViewOffsets, subViewSizes, subViewStrides);
}

// Expand rank by adding unit dimensions.
// Example, memref<3xf32> -> memref<1x3xf32>
// static Value rankExpandUnitDims(OpBuilder &builder, Location loc, Value
// input) {
//  MemRefType memrefOrig = input.getType().cast<MemRefType>();
//  assert(memrefOrig.getShape().size() == 1 && "expect 1d memref");
//  MemRefType newShape = MemRefType::get({1, memrefOrig.getShape()[0]},
//                                        memrefOrig.getElementType());
//  ArrayAttr rZero = builder.getI64ArrayAttr({0, 1});
//  ArrayAttr r = ArrayAttr::get(builder.getContext(), {rZero});
//  return builder.create<memref::ExpandShapeOp>(loc, newShape, input, r);
//}

// Make the generic operation mappable to tpp by preserving
// the last and first dimension only.
LogicalResult reshape2D(linalg::GenericOp linalgOp, bool useParallelLoops) {
  if (!linalgOp.hasBufferSemantics())
    return linalgOp->emitError("Expect linalgOp with buffer semantics");

  // bail-out if we don't need to do tiling or all the dimensions
  // are not parallel.
  // TODO: restrict to only the tiling ones.
  if (linalgOp.getNumLoops() <= 2)
    return success();
  ArrayAttr iteratorTypes = linalgOp.iterator_types();
  if (!llvm::all_of(iteratorTypes, [](Attribute type) {
        return linalg::isParallelIterator(type);
      }))
    return success();

  OpBuilder builder(linalgOp);
  OpBuilder::InsertionGuard guard(builder);
  linalg::LinalgTilingOptions linalgTilingOptions;
  linalg::LinalgTilingLoopType loopsTypes =
      (useParallelLoops) ? linalg::LinalgTilingLoopType::ParallelLoops
                         : linalg::LinalgTilingLoopType::Loops;
  linalgTilingOptions.setLoopType(loopsTypes)
      .setTileSizeComputationFunction(getTileSizes);
  IRRewriter rewriter(builder);
  FailureOr<linalg::TiledLinalgOp> tiledOp =
      linalg::tileLinalgOp(rewriter, linalgOp, linalgTilingOptions);
  if (failed(tiledOp))
    return linalgOp->emitError("Failed to tile linalgOp");

  linalgOp->erase();
  return success();
}

// Tile sizes selection specific for matmul.
static SmallVector<Value>
getTileSizesForOptimalMappingMatmulImpl(OpBuilder &builder,
                                        linalg::LinalgOp linalgOp) {
  SmallVector<int64_t> dims = linalgOp.computeStaticLoopSizes();
  int64_t m = dims[0];
  int64_t n = dims[1];
  int64_t k = dims[2];

  int64_t bestTileN = n;
  if (n % 16 == 0) {
    bestTileN = n - (n % 16);
    if (bestTileN > 64)
      bestTileN = 64;
  }
  int64_t bestTileK = k;
  int64_t bestTileM = (m % 32 == 0) ? 32 : m;

  Location loc = linalgOp.getLoc();
  SmallVector<Value> tppTiles(3,
                              builder.create<arith::ConstantIndexOp>(loc, 0));

  // do not tile.
  if ((bestTileM == m) && (bestTileK == k) && (bestTileN == n))
    return tppTiles;

  tppTiles[0] = builder.create<arith::ConstantIndexOp>(loc, bestTileM);
  tppTiles[1] = builder.create<arith::ConstantIndexOp>(loc, bestTileN);
  tppTiles[2] = builder.create<arith::ConstantIndexOp>(loc, bestTileK);
  return tppTiles;
}

// Tile sizes selection for all the other tpp ops.
static SmallVector<Value>
getTileSizesForOptimalMappingImpl(OpBuilder &builder,
                                  linalg::LinalgOp linalgOp) {
  Location loc = linalgOp.getLoc();
  SmallVector<int64_t> dims = linalgOp.computeStaticLoopSizes();
  arith::ConstantIndexOp index0 =
      builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> tppTiles(dims.size(), index0);

  arith::ConstantIndexOp index32 =
      builder.create<arith::ConstantIndexOp>(loc, 32);
  for (size_t idx = 0; idx < dims.size(); idx++) {
    if (dims[idx] % 32 == 0) {
      if (dims[idx] == 32)
        tppTiles[idx] = index0;
      else
        tppTiles[idx] = index32;
    }
    // do not tile.
    else
      tppTiles[idx] = index0;
  }
  return tppTiles;
}

// Try to select optimal tile sizes.
static SmallVector<Value>
getTileSizesForOptimalMapping(OpBuilder &builder, linalg::LinalgOp linalgOp) {
  if (isMarkedWithTpp(linalgOp, "tpp.matmul"))
    return getTileSizesForOptimalMappingMatmulImpl(builder, linalgOp);
  return getTileSizesForOptimalMappingImpl(builder, linalgOp);
}

// Tile the generic operation such that we can select the best micro-kernel.
LogicalResult tileLinalgOp(linalg::GenericOp linalgOp,
                           ArrayRef<int64_t> tileSizes) {
  if (!linalgOp.hasBufferSemantics())
    return linalgOp->emitError("Expect linalgOp with buffer semantics");
  if (!hasTppMark(linalgOp))
    return failure();

  OpBuilder builder(linalgOp);
  OpBuilder::InsertionGuard guard(builder);
  linalg::LinalgTilingOptions linalgTilingOptions;
  linalgTilingOptions.setLoopType(
      linalg::LinalgTilingLoopType::/*Parallel*/ Loops);

  if (tileSizes.size())
    linalgTilingOptions.setTileSizes(tileSizes);
  else
    linalgTilingOptions.setTileSizeComputationFunction(
        getTileSizesForOptimalMapping);

  IRRewriter rewriter(builder);
  FailureOr<linalg::TiledLinalgOp> tiledOp =
      linalg::tileLinalgOp(rewriter, linalgOp, linalgTilingOptions);
  if (failed(tiledOp))
    return linalgOp->emitError("Failed to tile linalgOp");
  linalgOp->erase();
  return success();
}

// Given an operand 'operand' returns the updated operand to be used when
// building a TPP operation.  Scalar or shaped type with rank <= 2 are ok,
// while shaped type with rank > 2 are rank reduced by dropping unit
// dimensions.  Note that the rank-reduce may fail thus the caller needs to
// check if the returned operand is valid using 'checkOperandForTpp'.
Value getOperandForTpp(Value operand, PatternRewriter &rewriter, Location loc) {
  Type operandType = operand.getType();
  // Scalar value.
  if (!operandType.isa<ShapedType>())
    return operand;
  // Shaped type.
  if (operandType.cast<ShapedType>().getRank() <= 2)
    return operand;
  // Attempt to rank reduce, it may fail.
  return rankReducingSubviewDroppingUnitDims(rewriter, loc, operand);
}

// Given an operand 'operand' check if it is a scalar
// or a shape type with rank <= 2.
LogicalResult checkOperandForTpp(Value operand) {
  Type operandType = operand.getType();
  if (!operandType.isa<ShapedType>())
    return success();
  if (operandType.isa<ShapedType>()) {
    unsigned rank = operandType.cast<ShapedType>().getRank();
    if (rank <= 2)
      return success();
  }
  return failure();
}

struct ConvertGenericOpToTpp : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult rewriteToTppOp(linalg::GenericOp linalgOp,
                               ArrayRef<Value> operands,
                               PatternRewriter &rewriter) const {
    std::string libraryCall = linalgOp.getLibraryCallName();
    if (libraryCall.compare("tpp.identity") == 0) {
      assert(operands.size() == 2 && "Expect two operands");
      rewriter.replaceOpWithNewOp<tpp::IdentityOp>(linalgOp, operands[0],
                                                   operands[1]);
      return success();
    }
    if (libraryCall.compare("tpp.relu") == 0) {
      if (linalgOp.getNumInputs() == 2)
        rewriter.replaceOpWithNewOp<tpp::ReluOp>(linalgOp, operands[0],
                                                 operands[1]);
      else
        rewriter.replaceOpWithNewOp<tpp::ReluOp>(linalgOp, operands[0],
                                                 operands[0]);
      return success();
    }
    if (libraryCall.compare("tpp.add") == 0) {
      rewriter.replaceOpWithNewOp<tpp::AddOp>(linalgOp, operands[0],
                                              operands[1], operands[2]);
      return success();
    }
    if (libraryCall.compare("tpp.matmul") == 0) {
      rewriter.replaceOpWithNewOp<tpp::MatmulOp>(linalgOp, operands[0],
                                                 operands[1], operands[2]);
      return success();
    }
    return failure();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasBufferSemantics() || !linalgOp.library_callAttr() ||
        !hasTppMark(linalgOp))
      return failure();

    if (linalgOp->getNumResults() != 0)
      return failure();

    Location loc = linalgOp.getLoc();
    SmallVector<Value, 4> newOperands;
    for (Value operand : linalgOp->getOperands()) {
      Value newOperand = getOperandForTpp(operand, rewriter, loc);
      if (failed(checkOperandForTpp(newOperand)))
        return failure();
      newOperands.push_back(newOperand);
    }
    return rewriteToTppOp(linalgOp, newOperands, rewriter);
  }
};

struct ConvertBrgemmToTpp
    : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  using OpRewritePattern<linalg::BatchReduceMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasBufferSemantics())
      return failure();
    // TODO: provide more convenient builder.
    rewriter.replaceOpWithNewOp<tpp::BrgemmOp>(
        linalgOp, linalgOp.getInputOperands()[0]->get(),
        linalgOp.getInputOperands()[1]->get(),
        linalgOp.getOutputOperands()[0]->get());
    return failure();
  }
};

// Given the following pattern:
// %0 = memref.subview %i : memref<64x32x32> -> memref<1x32x32>
// %1 = memref.subview %0 : memref<1x32x32> -> memref<32x32>
// simplify to:
// %0 = memref.subview %i : memref<64x32x32> -> memref<32x32>
struct SubViewOfSubViewWithUnitDims
    : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp subViewOp,
                                PatternRewriter &rewriter) const override {
    Value source = subViewOp.getSource();
    MemRefType sourceType = source.getType().cast<MemRefType>();
    // bail out if the memref is dynamic.
    if (!sourceType.hasStaticShape())
      return failure();
    SmallVector<int64_t> numberOfOnes = llvm::to_vector(llvm::make_filter_range(
        sourceType.getShape(), [](int64_t sz) { return sz == 1; }));
    // no work to do.
    if (numberOfOnes.size() == 0)
      return failure();

    // the producer of the current memref should
    // be another subview.
    memref::SubViewOp producer = source.getDefiningOp<memref::SubViewOp>();
    if (!producer)
      return failure();

    memref::SubViewOp rankReduced = rewriter.create<memref::SubViewOp>(
        subViewOp.getLoc(), subViewOp.getResult().getType().cast<MemRefType>(),
        producer.getSource(), producer.getMixedOffsets(),
        producer.getMixedSizes(), producer.getMixedStrides());
    rewriter.replaceOp(subViewOp, rankReduced.getResult());
    return success();
  }
};

void populateSubViewFoldingPatterns(RewritePatternSet &patterns) {
  patterns.add<SubViewOfSubViewWithUnitDims>(patterns.getContext());
}

// TODO: PatternRwriter does not work well with tiling. I suspect
// because the builder is not properly propagated. But investigate more.
struct ConvertLinalgToTpp : public ConvertLinalgToTppBase<ConvertLinalgToTpp> {
  ConvertLinalgToTpp() = default;
  ConvertLinalgToTpp(bool enabledPreconditions, bool useParallelLoops,
                     ArrayRef<int64_t> tileSizes) {
    this->enableTiling = enableTiling;
    this->useParallelLoops = useParallelLoops;
    this->tileSizes = tileSizes;
  }
  void runOnOperation() override {
    getOperation().walk([&](linalg::GenericOp linalgOp) {
      if (failed(reshape2D(linalgOp, this->useParallelLoops)))
        return signalPassFailure();
    });
    if (enableTiling || tileSizes.size())
      getOperation().walk([&](linalg::GenericOp linalgOp) {
        (void)tileLinalgOp(linalgOp, tileSizes);
      });
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    tpp::populateConvertLinalgToTppPatterns(patterns);
    populateSubViewFoldingPatterns(patterns);
    linalg::populateFoldUnitExtentDimsPatterns(patterns);
    memref::SubViewOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

void mlir::tpp::populateConvertLinalgToTppPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<ConvertGenericOpToTpp,
               ConvertBrgemmToTpp>(patterns.getContext());
  // clang-format on
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToTppPass() {
  return std::make_unique<ConvertLinalgToTpp>();
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createConvertLinalgToTppPass(bool enableTiling,
                                        bool useParallelLoops,
                                        ArrayRef<int64_t> tileSizes) {
  return std::make_unique<ConvertLinalgToTpp>(enableTiling, useParallelLoops,
                                              tileSizes);
}
