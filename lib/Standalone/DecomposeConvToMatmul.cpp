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
#include "Standalone/Transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

// return true if all operands have static shape.
static bool hasStaticShape(Value image, Value filter, Value output) {
  ShapedType imageType = image.getType().cast<ShapedType>();
  ShapedType filterType = filter.getType().cast<ShapedType>();
  ShapedType outputType = output.getType().cast<ShapedType>();
  if ((!imageType.hasStaticShape()) || (!filterType.hasStaticShape()) ||
      (!outputType.hasStaticShape()))
    return false;
  return true;
}

// return true if the conv has stride != 1.
template <typename CONVOP>
static bool hasStride(CONVOP convOp) {
  if (DenseIntElementsAttr strides = convOp.strides()) {
    auto values = strides.getValues<APInt>();
    if (llvm::any_of(values, [](const APInt &value) {
          return value.getSExtValue() != 1;
        })) {
      return true;
    }
  }
  return false;
}

// return true if the conv has dilation != 1.
template <typename CONVOP>
static bool hasDilation(CONVOP convOp) {
  if (DenseIntElementsAttr dilations = convOp.dilations()) {
    auto values = dilations.getValues<APInt>();
    if (llvm::any_of(values, [](const APInt &value) {
          return value.getSExtValue() != 1;
        })) {
      return true;
    }
  }
  return false;
}

static bool hasFilterWithRandSEqualOne(OpOperand *filter, unsigned i,
                                       unsigned j) {
  ShapedType filterType = filter->get().getType().cast<ShapedType>();
  if (!filterType.hasStaticShape())
    return false;
  ArrayRef<int64_t> filterShape = filterType.getShape();
  assert(i >= 0 && i < filterShape.size() && "out of bound");
  assert(j >= 0 && j < filterShape.size() && "out of bound");
  return ((filterShape[i] == 1) && (filterShape[j] == 1));
}

struct DecomposeConv2DNhwcHwcf : OpRewritePattern<linalg::GenericOp> {
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
    if (!hasFilterWithRandSEqualOne(filter, /*Rpos=*/0, /*Spos=*/1)) {
      SmallVector<int64_t> gemmSizes = computeGemmSizeFrom(filter, output);
      for (int64_t s : gemmSizes)
        sizes.push_back(builder.getIndexAttr(s));
    } else {
      for (size_t idx = rank - desiredResultRank, e = rank; idx < e; idx++)
        sizes.push_back(builder.getIndexAttr(operandType.getShape()[idx]));
    }
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
    FailureOr<Value> slicedFilter = utils::getSliceOperand(
        builder, filter, linalgOp, localIvs, valuesToUse, 2);
    if (failed(slicedFilter))
      return failure();
    slicedOperands.push_back(*slicedFilter);

    OpOperand *output = linalgOp.getOutputOperands()[0];
    FailureOr<Value> slicedOutput = utils::getSliceOperand(
        builder, output, linalgOp, localIvs, valuesToUse, 2);
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
struct InterchangeIteratorsConv2DNhwcHwcf
    : OpRewritePattern<linalg::GenericOp> {
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
    //
    // You can now see the matmul: image[*][*][W][C] * filter[*][*][C][K]
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
struct GeneralizeConv2DNhwcHwcf : OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    // do not handle convolutions with dilation and strides.
    if (hasStride<linalg::Conv2DNhwcHwcfOp>(convOp) ||
        hasDilation<linalg::Conv2DNhwcHwcfOp>(convOp))
      return failure();

    // [N][H][W][C]
    Value image = convOp.image();
    // [R][S][C][K]
    Value filter = convOp.filter();
    // [N][P][Q][K]
    Value output = convOp.outputs()[0];

    if (!hasStaticShape(image, filter, output))
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

// Block a Conv2DNchwFchw. The pattern returns a generic operation
// marked as 'tpp.blocked.Conv2DNchwFchwOp' on success.
struct BlockConv2DNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult
  blockConv2DNchwFchwPreconditions(linalg::Conv2DNchwFchwOp convOp) const {
    // do not handle convolutions with dilation and strides.
    if (hasStride<linalg::Conv2DNchwFchwOp>(convOp) ||
        hasDilation<linalg::Conv2DNchwFchwOp>(convOp))
      return failure();

    // [N][C][H][W]
    Value image = convOp.image();
    // [K][C][R][S]
    Value filter = convOp.filter();
    // [N][K][P][Q]
    Value output = convOp.outputs()[0];

    // static shapes.
    if (!hasStaticShape(image, filter, output))
      return failure();

    // tensor semantics.
    if (convOp.hasBufferSemantics())
      return failure();

    return success();
  }

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(blockConv2DNchwFchwPreconditions(convOp)))
      return failure();
    // TODO: hardcoded values.
    FailureOr<linalg::GenericOp> maybeGeneric =
        mlir::tpp::BlockConv2DNchwFchwOp(rewriter, convOp, {32, 32});
    if (failed(maybeGeneric))
      return failure();
    linalg::GenericOp generic = *maybeGeneric;
    generic.library_callAttr(
        rewriter.getStringAttr("tpp.BlockedConv2DNchwFchwOp"));
    return failure();
  }
};

struct InterchangeIteratorsConv2DNchwFchw
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::isMarkedWithTpp(genericOp, "tpp.BlockedConv2DNchwFchwOp"))
      return failure();

    // clang-format off
    // N                [parallel]
    //  K               [parallel - blocked]
    //    P             [parallel]
    //      Q           [parallel]
    //        k         [parallel - block of K]
    //          C       [reduction - blocked]
    //            R     [reduction]
    //              S   [reduction]
    //                c [reduction - block of C]
    
    // expose matmul by interchange

    // N                [parallel]
    //  K               [parallel - blocked]
    //    P             [parallel]
    //      C           [reduction - blocked]
    //        R         [reduction]
    //          S       [reduction]
    //            Q     [parallel]
    //              k   [parallel - block of K]
    //                c [reduction - block of C]
    //
    // Matmul: m = %Q, n = %k and k = %c
    // clang-format on

    SmallVector<unsigned> interchangeVector = {0, 1, 2, 5, 6, 7, 3, 4, 8};
    FailureOr<linalg::GenericOp> maybeInterchange =
        interchangeGenericOp(rewriter, genericOp, interchangeVector);
    if (failed(maybeInterchange))
      return failure();
    return success();
  }
};

struct DecomposeConv2DNchwFchw : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  bool hasInterchangedDims(linalg::GenericOp linalgOp) const {
    ArrayAttr iteratorTypes = linalgOp.getIteratorTypes();
    if (iteratorTypes.size() != 9)
      return false;
    bool match = isParallelIterator(iteratorTypes[0]) &&
                 isParallelIterator(iteratorTypes[1]) &&
                 isParallelIterator(iteratorTypes[2]) &&
                 isReductionIterator(iteratorTypes[3]) &&
                 isReductionIterator(iteratorTypes[4]) &&
                 isReductionIterator(iteratorTypes[5]) &&
                 isParallelIterator(iteratorTypes[6]) &&
                 isParallelIterator(iteratorTypes[7]) &&
                 isReductionIterator(iteratorTypes[8]);
    return match;
  }

  // Make sure we are dealing with a generic 'Conv2DNchwFchwOp' blocked
  // with interchanged dimensions.
  LogicalResult
  decomposeConv2DNchwFchwPreconditions(linalg::GenericOp linalgOp) const {
    if (!tpp::isMarkedWithTpp(linalgOp, "tpp.BlockedConv2DNchwFchwOp"))
      return failure();
    if (!hasInterchangedDims(linalgOp))
      return failure();
    return success();
  }

  FailureOr<SmallVector<Value>>
  getSlicedOperands(OpBuilder &builder, Location loc, ValueRange localIvs,
                    linalg::LinalgOp linalgOp, ValueRange valuesToUse) const {
    assert(linalgOp.getNumInputsAndOutputs() == 3 &&
           "expect 3 input/output operands");
    assert(linalgOp.getInputOperands().size() == 2 &&
           "expect 2 input operands");
    SmallVector<Value> slicedOperands;
    for (OpOperand *operand : linalgOp.getInputOperands()) {
      FailureOr<Value> maybeSliced = utils::getSliceOperand(
          builder, operand, linalgOp, localIvs, valuesToUse, /*GEMM dims=*/2);
      if (failed(maybeSliced))
        return failure();
      slicedOperands.push_back(*maybeSliced);
    }
    FailureOr<Value> maybeSliced = utils::getSliceOperand(
        builder, linalgOp.getOutputOperands()[0], linalgOp, localIvs,
        valuesToUse, /*GEMM dims=*/2);
    if (failed(maybeSliced))
      return failure();
    slicedOperands.push_back(*maybeSliced);
    return slicedOperands;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(decomposeConv2DNchwFchwPreconditions(linalgOp)))
      return failure();

    SmallVector<OpOperand *> inputOperands = linalgOp.getInputOperands();
    if (!hasFilterWithRandSEqualOne(inputOperands[1], /*Rpos=*/2, /*Spos=*/3))
      return failure();

    // peelout {N, K, P, C, R, S} and map {Q, k, c} to GEMM.
    unsigned upTo = linalgOp.getNumLoops() - /*GEMM loops=*/3;
    FailureOr<SmallVector<Range>> maybeLoopRanges =
        mlir::utils::getLoopsToMaterialize(rewriter, linalgOp, upTo);
    if (failed(maybeLoopRanges))
      return failure();
    SmallVector<Range> loopRanges = *maybeLoopRanges;

    SmallVector<Value, 4> ivs, tensorResults;
    auto gemmBuilder = [&](OpBuilder &builder, Location loc,
                           ValueRange localIvs,
                           ValueRange operandsValuesToUse) -> scf::ValueVector {
      assert(localIvs.size() == 6);
      assert(operandsValuesToUse.size() ==
                 static_cast<size_t>(linalgOp.getNumInputsAndOutputs()) &&
             "expect the number of operands and inputs and outputs to match");
      ivs.assign(localIvs.begin(), localIvs.end());
      FailureOr<SmallVector<Value>> maybeSlicedOperands = getSlicedOperands(
          builder, loc, localIvs, linalgOp, operandsValuesToUse);
      if (failed(maybeSlicedOperands))
        return {};
      SmallVector<Value> slicedOperands = *maybeSlicedOperands;
      assert(slicedOperands.size() == 3 && "expect three operands");

      linalg::MatmulOp matmul =
          (linalgOp.hasTensorSemantics())
              ? builder.create<linalg::MatmulOp>(
                    loc, slicedOperands[2].getType(),
                    ValueRange{slicedOperands[0], slicedOperands[1]},
                    slicedOperands[2])
              : builder.create<linalg::MatmulOp>(
                    loc, ValueRange{slicedOperands[0], slicedOperands[1]},
                    slicedOperands[2]);
      tensorResults = insertSlicesBack(builder, loc, linalgOp, slicedOperands,
                                       matmul->getResults());

      return scf::ValueVector(tensorResults.begin(), tensorResults.end());
    };

    linalg::GenerateLoopNest<scf::ForOp>::doit(
        rewriter, linalgOp.getLoc(), loopRanges, linalgOp,
        linalgOp.getIteratorTypes(), gemmBuilder);

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

    rewriter.replaceOp(linalgOp, outermostLoop ? outermostLoop->getResults()
                                               : tensorResults);
    return success();
  }
};

// Prepare for BRGEMM. Requires R = S = 1. The pattern collapses
// H and W on the image and P and Q on the output.
struct CollapseFilterAndImage : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult CollapseFilterPreconditions(linalg::GenericOp linalgOp) const {
    if (!tpp::isMarkedWithTpp(linalgOp, "tpp.BlockedConv2DNchwFchwOp"))
      return failure();
    OpOperand *filter = linalgOp.getInputOperands()[1];
    if (!hasFilterWithRandSEqualOne(filter, /*Rpos=*/2, /*Spos=*/3))
      return failure();
    return success();
  }

  SmallVector<ReassociationExprs, 2>
  convertAffineMapArrayToExprs(ArrayAttr affineMapArrayAttr) const {
    SmallVector<ReassociationExprs, 2> reassociationExprs;
    for (auto attr : affineMapArrayAttr)
      reassociationExprs.push_back(llvm::to_vector<4>(
          attr.cast<AffineMapAttr>().getValue().getResults()));
    return reassociationExprs;
  }

  // Return the original value if the type is unchanged, or collapse it. Assert
  // if this is an unsupported type.
  Value collapse(Value operand, Type newOperandType, ArrayAttr reassociationMap,
                 Location loc, PatternRewriter &rewriter) const {
    Type operandType = operand.getType();
    if (operandType == newOperandType)
      return operand;
    if (operandType.isa<MemRefType>()) {
      return rewriter.create<memref::CollapseShapeOp>(
          loc, newOperandType, operand, reassociationMap);
    }
    if (operandType.isa<RankedTensorType>()) {
      return rewriter.create<tensor::CollapseShapeOp>(
          loc, newOperandType, operand, reassociationMap);
    }
    llvm_unreachable("expect tensor or memref");
  }

  // Return the original value if the type is unchanged, or expand it. Assert
  // if this is an unsupported type.
  Value expand(Value result, Type origResultType, ArrayAttr reassociationMap,
               Location loc, PatternRewriter &rewriter) const {
    if (origResultType == result.getType())
      return result;
    if (origResultType.isa<RankedTensorType>()) {
      return rewriter.create<tensor::ExpandShapeOp>(loc, origResultType, result,
                                                    reassociationMap);
    }
    if (origResultType.isa<MemRefType>()) {
      return rewriter.create<memref::ExpandShapeOp>(loc, origResultType, result,
                                                    reassociationMap);
    }
    llvm_unreachable("expect tensor or memref");
  }

  // Collapse dimension at index 'startCollapse' to 'endCollapse'.
  Type getCollapsedType(OpOperand *operand, size_t startCollapse,
                        size_t endCollapse) const {
    assert(endCollapse > startCollapse && "expect >");
    ShapedType operandType = operand->get().getType().cast<ShapedType>();
    size_t rank = operandType.getRank();
    ArrayRef<int64_t> oldShape = operandType.getShape();
    SmallVector<int64_t> newShape;

    size_t idx = 0;
    while (idx < rank) {
      int64_t collapsedDim = oldShape[idx];
      if (idx == startCollapse)
        while (idx < endCollapse)
          collapsedDim *= oldShape[++idx];
      newShape.push_back(collapsedDim);
      idx++;
    }
    if (operandType.isa<MemRefType>())
      return MemRefType::get(newShape, operandType.getElementType());
    if (operandType.isa<RankedTensorType>())
      return RankedTensorType::get(newShape, operandType.getElementType());
    llvm_unreachable("expect tensor or memref");
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(CollapseFilterPreconditions(linalgOp)))
      return failure();

    // [original] = N K P Q k C R S c (R and S are 1)
    // [drop R and S] = N K P Q k C c
    // [collapse P and Q ] = N K P0 k C c
    // [new maps]:
    //  - filter (N K P0 k C c) -> (K C c k)
    //  - image  (N K P0 k C c) -> (N C P0 c)
    //  - output (N K P0 k C c) -> (N K P0 k)
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr N, K, P0, k, C, c;
    bindDims(linalgOp.getContext(), N, K, P0, k, C, c);
    SmallVector<AffineMap> newIndexingMaps =
        infer({{N, C, P0, c}, {K, C, c, k}, {N, K, P0, k}});

    // We dropped R and S (two reductions) and collapse
    // parallel loops P and Q (aka H and W on the image).
    SmallVector<StringRef> newIteratorTypes = {
        getParallelIteratorTypeName(),  getParallelIteratorTypeName(),
        getParallelIteratorTypeName(),  getParallelIteratorTypeName(),
        getReductionIteratorTypeName(), getReductionIteratorTypeName()};

    Location loc = linalgOp.getLoc();
    OpOperand *image = linalgOp.getInputOperands()[0];
    Type newImageType = getCollapsedType(image, 2, 3);
    auto reassociationImage = getReassociationIndicesForCollapse(
        image->get().getType().cast<ShapedType>().getShape(),
        newImageType.cast<ShapedType>().getShape());
    if (!reassociationImage)
      return failure();

    OpOperand *filter = linalgOp.getInputOperands()[1];
    Type newFilterType = getCollapsedType(filter, 1, 3);
    auto reassociationFilter = getReassociationIndicesForCollapse(
        filter->get().getType().cast<ShapedType>().getShape(),
        newFilterType.cast<ShapedType>().getShape());
    if (!reassociationFilter)
      return failure();

    OpOperand *output = linalgOp.getOutputOperands()[0];
    Type newOutputType = getCollapsedType(output, 2, 3);
    auto reassociationOutput = getReassociationIndicesForCollapse(
        output->get().getType().cast<ShapedType>().getShape(),
        newOutputType.cast<ShapedType>().getShape());
    if (!reassociationOutput)
      return failure();

    Value collapsedImage = collapse(
        image->get(), newImageType,
        getReassociationIndicesAttribute(rewriter, *reassociationImage), loc,
        rewriter);

    Value collapsedFilter = collapse(
        filter->get(), newFilterType,
        getReassociationIndicesAttribute(rewriter, *reassociationFilter), loc,
        rewriter);

    Value collapsedOutput = collapse(
        output->get(), newOutputType,
        getReassociationIndicesAttribute(rewriter, *reassociationOutput), loc,
        rewriter);

    linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
        loc, newOutputType, ValueRange{collapsedImage, collapsedFilter},
        collapsedOutput, newIndexingMaps, newIteratorTypes, /*docs=*/"",
        /*library_call=*/"tpp.BlockedandCollapsedConv2DNchwFchwOp");
    rewriter.inlineRegionBefore(linalgOp->getRegion(0),
                                replacementOp.getRegion(),
                                replacementOp.getRegion().begin());
    Value res = replacementOp->getResult(0);
    reassociationOutput = getReassociationIndicesForReshape(
        res.getType().cast<ShapedType>(),
        output->get().getType().cast<ShapedType>());
    if (!reassociationOutput)
      return failure();
    Value resExpanded =
        expand(res, output->get().getType(),
               getReassociationIndicesAttribute(rewriter, *reassociationOutput),
               loc, rewriter);
    rewriter.replaceOp(linalgOp, resExpanded);
    return success();
  }
};

struct InterchangeAfterBlockingAndCollapsing
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::isMarkedWithTpp(linalgOp,
                              "tpp.BlockedandCollapsedConv2DNchwFchwOp"))
      return failure();

    // clang-format off
    // N       [parallel]
    //  K'     [parallel]
    //   P + Q [parallel]
    //    k    [parallel]
    //     C'  [reduction]
    //      c  [reduction]
    //        output[N][K'][P + Q][k] += image[N][C'][H + W][c] * filter[K'][C'][c][k]
    
    // expose BRGEMM by interchange:

    // N       [parallel]
    //  K'     [parallel]
    //   C'    [reduction] // BRGEMM red dimension
    //   /* GEMM */
    //   P + Q [parallel]
    //    k    [parallel]
    //      c  [reduction]
    //        output[N][K'][P + Q][k] += image[N][C'][H + W][c] * filter[K'][C'][c][k]
    // clang-format on

    SmallVector<unsigned> interchangeVector = {0, 1, 4, 2, 3, 5};
    if (linalgOp.getNumLoops() != interchangeVector.size())
      return failure();
    FailureOr<linalg::GenericOp> maybeInterchange =
        interchangeGenericOp(rewriter, linalgOp, interchangeVector);
    if (failed(maybeInterchange))
      return failure();
    StringAttr name =
        rewriter.getStringAttr("tpp.BlockedCollapsedAndInterConv2DNchwFchwOp");
    (*maybeInterchange).library_callAttr(name);
    return success();
  }
};

struct MapToBRGEMM : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::isMarkedWithTpp(linalgOp,
                              "tpp.BlockedCollapsedAndInterConv2DNchwFchwOp"))
      return failure();
    FailureOr<SmallVector<Value>> maybeLoopsOrGenericRes =
        mlir::tpp::MapToBRGEMMOp(rewriter, linalgOp);
    if (failed(maybeLoopsOrGenericRes))
      return failure();
    return success();
  }
};

// patterns for mapping a Conv2DNhwcHwcfOp to a GEMM operation.
void populateConv2DNhwcHwcfOpDecomposePatterns(RewritePatternSet &patterns) {
  patterns.insert<GeneralizeConv2DNhwcHwcf, DecomposeConv2DNhwcHwcf,
                  InterchangeIteratorsConv2DNhwcHwcf>(patterns.getContext());
}

// patterns for mapping a Conv2DNchwFchwOp to a GEMM operation.
void populateconv2DNchwFchwOpDecomposePatterns(RewritePatternSet &patterns) {
  // This is for GEMM
  patterns.insert<BlockConv2DNchwFchw, DecomposeConv2DNchwFchw,
                   InterchangeIteratorsConv2DNchwFchw>(patterns.getContext());

  // clang-format off
  // This is for BRGEMM
  // init: [N][K][P][Q] = [N][C][H][W] * [K][C][R][S]
  // blocking: [N][K'][P][Q][k] = [N][C'][H][W][c] * [K'][C'][R][S][c][k]
  // if (R = S = 1) -> [P + Q] == [H + W]
  // collapsing: [N][K'][P + Q][k] = [N][C'][H + W][c] * [K'][C'][c][k]
  // [*][* ][P + Q][k] = [*][* ][H + W][c] * [* ][* ][c][k] // GEMM with c as red.
  // [*][* ][P + Q][k] = [*][C'][H + W][c] * [* ][C'][c][k] // BRGEMM with C' as red.
  // clang-format on
  //patterns.insert<BlockConv2DNchwFchw, CollapseFilterAndImage,
  //                InterchangeAfterBlockingAndCollapsing, MapToBRGEMM>(
  //    patterns.getContext());
}

struct DecomposeConvToMatmul
    : public DecomposeConvToMatmulBase<DecomposeConvToMatmul> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    populateConv2DNhwcHwcfOpDecomposePatterns(patterns);
    populateconv2DNchwFchwOpDecomposePatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createDecomposeConvToMatmulPass() {
  return std::make_unique<DecomposeConvToMatmul>();
}
