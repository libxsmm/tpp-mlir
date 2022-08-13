//===- DecomposeConvToMatmul.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "Standalone/TransformUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

struct DecomposeConv : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  bool
  preOptimizeByInterchangeIteratorsConv(linalg::GenericOp genericOp) const {
    ArrayAttr iteratorTypes = genericOp.getIteratorTypes();
    if (iteratorTypes.size() != 7)
      return false;
    bool match = isParallelIterator(iteratorTypes[0]) &&
                 isParallelIterator(iteratorTypes[1]) &&
                 isReductionIterator(iteratorTypes[2]) &&
                 isReductionIterator(iteratorTypes[3]) &&
                 isParallelIterator(iteratorTypes[4]) &&
                 isParallelIterator(iteratorTypes[5]) &&
                 isReductionIterator(iteratorTypes[6]);
    return match;
  }

  bool hasFilterRSEqualOne(OpOperand *filter) const {
    ShapedType filterType = filter->get().getType().cast<ShapedType>();
    ArrayRef<int64_t> filterShape = filterType.getShape();
    bool tmp = ((filterShape[0] == 1) && (filterShape[1] == 1));
    return tmp;
  }

  SmallVector<int64_t> computeGemmSizeFrom(OpOperand *filter,
                                           OpOperand *output) const {
    ShapedType filterType = filter->get().getType().cast<ShapedType>();
    ShapedType outputType = output->get().getType().cast<ShapedType>();
    assert(filterType.getRank() == 4);
    assert(outputType.getRank() == 4);
    return SmallVector<int64_t>{outputType.getShape()[2],
                                filterType.getShape()[2]};
  }

  SmallVector<OpFoldResult> getSizesForImage(OpBuilder &builder,
                                             linalg::LinalgOp linalgOp,
                                             unsigned desiredResultRank) const {
    OpOperand *image = linalgOp.getInputOperands()[0];
    ShapedType operandType = image->get().getType().cast<ShapedType>();
    OpOperand *filter = linalgOp.getInputOperands()[1];
    OpOperand *output = linalgOp.getOutputOperands()[0];
    unsigned rank = image->get().getType().cast<ShapedType>().getRank();
    SmallVector<OpFoldResult> sizes;
    sizes.reserve(rank);

    for (size_t idx = 0, e = rank - desiredResultRank; idx < e; idx++)
      sizes.push_back(builder.getIndexAttr(1));
    if (!hasFilterRSEqualOne(filter)) {
      SmallVector<int64_t> gemmSizes = computeGemmSizeFrom(filter, output);
      for (int64_t s : gemmSizes)
        sizes.push_back(builder.getIndexAttr(s));
    } else {
      for (size_t idx = rank - desiredResultRank, e = rank; idx < e; idx++)
        sizes.push_back(builder.getIndexAttr(operandType.getShape()[idx]));
    }

    llvm::errs() << "------\n";
    for (OpFoldResult r : sizes)
      r.dump();
    llvm::errs() << "------\n";

    return sizes;
  }

  FailureOr<Value> getSlicedImg(OpBuilder &builder, linalg::LinalgOp linalgOp,
                                ValueRange ivs, ValueRange valuesToUse) const {
    OpOperand *image = linalgOp.getInputOperands()[0];
    unsigned rank = image->get().getType().cast<ShapedType>().getRank();
    Location loc = linalgOp.getLoc();
    FailureOr<SmallVector<Value>> ivsImage =
        utils::getInvolvedLocalDimsForOperand(
            builder, loc, image, linalgOp.getTiedIndexingMap(image), ivs);
    if (failed(ivsImage))
      return failure();

    ivs = *ivsImage;
    SmallVector<OpFoldResult> offsets;

    // offset into the tensor is the induction var or 0.
    for (size_t idx = 0, e = ivs.size(); idx < e; idx++)
      offsets.push_back(ivs[idx]);
    for (size_t idx = ivs.size(), e = rank; idx < e; idx++)
      offsets.push_back(builder.getIndexAttr(0));

    unsigned desiredResultRank = 2;
    SmallVector<OpFoldResult> sizes =
        getSizesForImage(builder, linalgOp, desiredResultRank);
    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    Value operandToUse = valuesToUse[image->getOperandNumber()];
    return utils::getSlicedOperand(builder, linalgOp, operandToUse, offsets,
                                   sizes, strides, desiredResultRank);
  }

  // TODO: make this util?
  Value getSliceOperandImpl(OpBuilder &builder, linalg::LinalgOp linalgOp,
                            OpOperand *operand, ValueRange ivs,
                            ValueRange valuesToUse,
                            unsigned desiredResultRank) const {
    Value operandToUse = valuesToUse[operand->getOperandNumber()];
    ShapedType operandType = operandToUse.getType().cast<ShapedType>();
    assert(operandType.hasStaticShape() && "tensor must have static shape");
    size_t rank = operandType.getRank();

    SmallVector<OpFoldResult> offsets, sizes;
    offsets.reserve(rank);
    sizes.reserve(rank);

    // offset into the tensor is the induction var or 0.
    for (size_t idx = 0, e = ivs.size(); idx < e; idx++)
      offsets.push_back(ivs[idx]);
    for (size_t idx = ivs.size(), e = rank; idx < e; idx++)
      offsets.push_back(builder.getIndexAttr(0));

    // sizes are 1 in [0 to rank - desiredResultRank)
    // and 'full' in [rank - desiredResultRank to rank).
    for (size_t idx = 0, e = rank - desiredResultRank; idx < e; idx++)
      sizes.push_back(builder.getIndexAttr(1));
    for (size_t idx = rank - desiredResultRank, e = rank; idx < e; idx++)
      sizes.push_back(builder.getIndexAttr(operandType.getShape()[idx]));

    SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
    return utils::getSlicedOperand(builder, linalgOp, operandToUse, offsets,
                                   sizes, strides, desiredResultRank);
  }

  // TODO: make this util?
  FailureOr<Value> getSliceOperand(OpBuilder &builder, OpOperand *operand,
                                   linalg::LinalgOp linalgOp, ValueRange ivs,
                                   ValueRange valuesToUse,
                                   unsigned desiredResultRank) const {
    Location loc = linalgOp.getLoc();
    FailureOr<SmallVector<Value>> involvedDimForOperand =
        utils::getInvolvedLocalDimsForOperand(
            builder, loc, operand, linalgOp.getTiedIndexingMap(operand), ivs);
    if (failed(involvedDimForOperand))
      return failure();
    return getSliceOperandImpl(builder, linalgOp, operand,
                               *involvedDimForOperand, valuesToUse,
                               desiredResultRank);
  }

  FailureOr<SmallVector<Value>>
  getSlicedOperands(OpBuilder &builder, Location loc, ValueRange localIvs,
                    linalg::LinalgOp linalgOp, ValueRange valuesToUse) const {
    assert(linalgOp.getNumInputsAndOutputs() == 3 &&
           "expect 3 input/output operands");
    assert(linalgOp.getInputOperands().size() == 2 &&
           "expect 2 input operands");

    SmallVector<Value> slicedOperands;

    FailureOr<Value> slicedImg =
        getSlicedImg(builder, linalgOp, localIvs, valuesToUse);
    if (failed(slicedImg))
      return failure();
    slicedOperands.push_back(*slicedImg);

    OpOperand *filter = linalgOp.getInputOperands()[1];
    FailureOr<Value> slicedFilter =
        getSliceOperand(builder, filter, linalgOp, localIvs, valuesToUse, 2);
    if (failed(slicedFilter))
      return failure();
    slicedOperands.push_back(*slicedFilter);

    OpOperand *output = linalgOp.getOutputOperands()[0];
    FailureOr<Value> slicedOutput =
        getSliceOperand(builder, output, linalgOp, localIvs, valuesToUse, 2);
    if (failed(slicedOutput))
      return failure();
    slicedOperands.push_back(*slicedOutput);

    return slicedOperands;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    if (!tpp::isMarkedWithTpp(genericOp, "tpp.Conv2DNhwcHwcfOp"))
      return failure();

    // Make sure we did loop re-ordering.
    if (!preOptimizeByInterchangeIteratorsConv(genericOp))
      return failure();

    // peel-out N, P, R, S and map Q, K and C to GEMM.
    Location loc = genericOp.getLoc();
    SmallVector<OpFoldResult> allShapeSizes =
        cast<linalg::LinalgOp>(genericOp.getOperation())
            .createFlatListOfOperandDims(rewriter, loc);
    AffineMap map = genericOp.getShapesToLoopsMap();
    if (!map)
      return failure();

    SmallVector<OpFoldResult> domain = makeComposedFoldedMultiResultAffineApply(
        rewriter, loc, map, allShapeSizes);
    SmallVector<Range> loopRanges;
    unsigned outerLoops = 3;
    for (unsigned idx = 0, e = domain.size() - outerLoops; idx < e; idx++)
      loopRanges.push_back(Range{rewriter.getIndexAttr(0), domain[idx],
                                 rewriter.getIndexAttr(1)});

    SmallVector<Value, 4> ivs, tensorResults;
    auto gemmBuilder = [&](OpBuilder &builder, Location loc,
                           ValueRange localIvs,
                           ValueRange operandsValuesToUse) -> scf::ValueVector {
      assert(localIvs.size() == 4);
      assert(operandsValuesToUse.size() ==
                 static_cast<size_t>(genericOp.getNumInputsAndOutputs()) &&
             "expect the number of operands and inputs and outputs to match");
      ivs.assign(localIvs.begin(), localIvs.end());
      FailureOr<SmallVector<Value>> maybeSlicedOperands = getSlicedOperands(
          builder, loc, localIvs, genericOp, operandsValuesToUse);
      if (failed(maybeSlicedOperands)) {
        // TODO: Can I just return?
        assert(0 && "failed to generate loops for op");
        return {};
      }
      SmallVector<Value> slicedOperands = *maybeSlicedOperands;
      assert(slicedOperands.size() == 3 && "expect three operands");

      linalg::MatmulOp matmul =
          (genericOp.hasTensorSemantics())
              ? builder.create<linalg::MatmulOp>(
                    loc, slicedOperands[2].getType(),
                    ValueRange{slicedOperands[0], slicedOperands[1]},
                    slicedOperands[2])
              : builder.create<linalg::MatmulOp>(
                    loc, ValueRange{slicedOperands[0], slicedOperands[1]},
                    slicedOperands[2]);
      tensorResults = insertSlicesBack(builder, loc, genericOp, slicedOperands,
                                       matmul->getResults());

      return scf::ValueVector(tensorResults.begin(), tensorResults.end());
    };

    linalg::GenerateLoopNest<scf::ForOp>::doit(
        rewriter, loc, loopRanges, genericOp, genericOp.getIteratorTypes(),
        gemmBuilder);

    // see: `Tiling.cpp` in Linalg/Transforms
    // Gather the newly created loops and return them with the new op.
    SmallVector<Operation *, 8> loops;
    loops.reserve(ivs.size());
    for (Value iv : ivs) {
      if (iv.isa<BlockArgument>()) {
        loops.push_back(iv.cast<BlockArgument>().getOwner()->getParentOp());
        assert(loops.back() && "no owner found for induction variable!");
      } else {
        loops.push_back(nullptr);
      }
    }

    // Get the tensor results from the outermost loop.
    Operation *outermostLoop = nullptr;
    for (Operation *loop : loops)
      if ((outermostLoop = loop))
        break;

    rewriter.replaceOp(genericOp, outermostLoop ? outermostLoop->getResults()
                                                : tensorResults);
    return success();
  }
};

// Interchange iterators for a tpp.Conv2DNhwcHwcfOp.
struct InterchangeIteratorsConv : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::isMarkedWithTpp(genericOp, "tpp.Conv2DNhwcHwcfOp"))
      return failure();

    // clang-format off
    // N        [parallel]
    //  P       [parallel]
    //   Q      [parallel]
    //    K     [parallel]
    //     R    [reduction]
    //      S   [reduction]
    //       C  [reduction]
    //        output[N][P][Q][K] += image[N][H][W][C] * filter[R][S][C][K]

    // expose the matmul by interchange:

    // N        [parallel]
    //  P       [parallel]
    //   R      [reduction]
    //    S     [reduction]
    //     Q    [parallel]
    //      K   [parallel]
    //       C  [reduction]
    //        output[N][P][Q][K] += image[N][H][W][C] * filter[R][S][C][K]
    // clang-format on

    SmallVector<unsigned> interchangeVector = {0, 1, 4, 5, 2, 3, 6};
    FailureOr<linalg::GenericOp> maybeInterchange =
        interchangeGenericOp(rewriter, genericOp, interchangeVector);
    if (failed(maybeInterchange))
      return failure();
    return success();
  }
};

// Generalize a linalg::Conv2DNhwcHwcfOp. Mark the operation
// with tpp.Conv2DNhwcHwcfOp such that later pattern can pick it up.
struct GeneralizeConv : OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    // do not handle convolutions with dilation and strides.
    if (DenseIntElementsAttr dilations = convOp.dilations()) {
      auto values = dilations.getValues<APInt>();
      if (llvm::any_of(values, [](const APInt &value) {
            return value.getSExtValue() != 1;
          })) {
        return failure();
      }
    }
    if (DenseIntElementsAttr strides = convOp.strides()) {
      auto values = strides.getValues<APInt>();
      if (llvm::any_of(values, [](const APInt &value) {
            return value.getSExtValue() != 1;
          })) {
        return failure();
      }
    }
    // [N][H][W][C]
    Value image = convOp.image();
    // [R][S][C][K]
    Value filter = convOp.filter();
    // [N][P][Q][K]
    Value output = convOp.outputs()[0];

    ShapedType imageType = image.getType().cast<ShapedType>();
    ShapedType filterType = filter.getType().cast<ShapedType>();
    ShapedType outputType = output.getType().cast<ShapedType>();

    // only static dimensions.
    if ((!imageType.hasStaticShape()) || (!filterType.hasStaticShape()) ||
        (!outputType.hasStaticShape()))
      return failure();

    FailureOr<linalg::GenericOp> maybeGeneric =
        generalizeNamedOp(rewriter, convOp);
    if (failed(maybeGeneric))
      return failure();
    linalg::GenericOp generic = *maybeGeneric;
    generic.library_callAttr(rewriter.getStringAttr("tpp.Conv2DNhwcHwcfOp"));
    return success();
  }
};

void populateConvDecomposePatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.insert<GeneralizeConv, 
                  DecomposeConv, 
                  InterchangeIteratorsConv>(
      patterns.getContext());
  // clang-format on
}

struct DecomposeConvToMatmul
    : public DecomposeConvToMatmulBase<DecomposeConvToMatmul> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    populateConvDecomposePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createDecomposeConvToMatmulPass() {
  return std::make_unique<DecomposeConvToMatmul>();
}
