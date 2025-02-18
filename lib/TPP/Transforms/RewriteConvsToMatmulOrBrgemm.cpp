//===- RewriteConvToMatmul.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REWRITECONVTOMATMULORBRGEMM
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

bool isMarkedWithTpp(linalg::LinalgOp linalgOp, const std::string &target) {
  return isa<linalg::GenericOp>(linalgOp) &&
         linalgOp.getLibraryCallName() == target;
}

// return true if all operands have static shape.
static bool hasStaticShape(Value image, Value filter, Value output) {
  ShapedType imageType = cast<ShapedType>(image.getType());
  ShapedType filterType = cast<ShapedType>(filter.getType());
  ShapedType outputType = cast<ShapedType>(output.getType());
  return !((!imageType.hasStaticShape()) || (!filterType.hasStaticShape()) ||
           (!outputType.hasStaticShape()));
}

// return true if the conv has stride != 1.
template <typename CONVOP> static bool hasStride(CONVOP convOp) {
  if (DenseIntElementsAttr strides = convOp.getStrides()) {
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
template <typename CONVOP> static bool hasDilation(CONVOP convOp) {
  if (DenseIntElementsAttr dilations = convOp.getDilations()) {
    auto values = dilations.getValues<APInt>();
    if (llvm::any_of(values, [](const APInt &value) {
          return value.getSExtValue() != 1;
        })) {
      return true;
    }
  }
  return false;
}

// Check dimension at index 'i' and 'j'. If both are '1' return true
// otherwise false. The operand is expected to have static shape.
static bool hasFilterWithRandSEqualOne(OpOperand *filter, unsigned i,
                                       unsigned j) {
  ShapedType filterType = cast<ShapedType>(filter->get().getType());
  if (!filterType.hasStaticShape())
    return false;
  ArrayRef<int64_t> filterShape = filterType.getShape();
  assert(i < filterShape.size() && "out of bound");
  assert(j < filterShape.size() && "out of bound");
  return ((filterShape[i] == 1) && (filterShape[j] == 1));
}

struct RewriteConv2DNhwcHwcfToMatmul : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  bool
  preOptimizeByInterchangeIteratorsConv(linalg::GenericOp genericOp) const {
    auto iteratorTypes = genericOp.getIteratorTypesArray();
    if (iteratorTypes.size() != 7)
      return false;
    bool match = linalg::isParallelIterator(iteratorTypes[0]) &&
                 linalg::isParallelIterator(iteratorTypes[1]) &&
                 linalg::isReductionIterator(iteratorTypes[2]) &&
                 linalg::isReductionIterator(iteratorTypes[3]) &&
                 linalg::isParallelIterator(iteratorTypes[4]) &&
                 linalg::isParallelIterator(iteratorTypes[5]) &&
                 linalg::isReductionIterator(iteratorTypes[6]);
    return match;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    if (!isMarkedWithTpp(genericOp, "tpp.Conv2DNhwcHwcfOp"))
      return failure();

    // Make sure we did loop re-ordering.
    if (!preOptimizeByInterchangeIteratorsConv(genericOp))
      return failure();

    FailureOr<linalg::MatmulOp> matmul =
        mlir::linalgx::rewriteConvToMatmul(rewriter, genericOp);
    if (failed(matmul))
      return rewriter.notifyMatchFailure(genericOp,
                                         "failed to map convolution to matmul");
    return success();
  }
};

// Interchange iterators for a tpp.Conv2DNhwcHwcfOp.
struct InterchangeIteratorsConv2DNhwcHwcf
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!isMarkedWithTpp(genericOp, "tpp.Conv2DNhwcHwcfOp"))
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
    // Do not handle convolutions with dilation.
    if (hasDilation<linalg::Conv2DNhwcHwcfOp>(convOp))
      return failure();

    // [N][H][W][C]
    Value image = convOp.image();
    // [R][S][C][K]
    Value filter = convOp.filter();
    // [N][P][Q][K]
    Value output = convOp.getOutputs()[0];

    if (!hasStaticShape(image, filter, output))
      return failure();

    FailureOr<linalg::GenericOp> maybeGeneric =
        generalizeNamedOp(rewriter, convOp);
    if (failed(maybeGeneric))
      return failure();
    linalg::GenericOp generic = *maybeGeneric;
    generic.setLibraryCallAttr(rewriter.getStringAttr("tpp.Conv2DNhwcHwcfOp"));
    return success();
  }
};

// Block a Conv2DNchwFchw. The pattern returns a generic operation
// marked as 'tpp.blocked.Conv2DNchwFchwOp' on success.
struct BlockConv2DNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  BlockConv2DNchwFchw(MLIRContext *context, ArrayRef<int64_t> blockingFactors,
                      PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit),
        blockingFactors(blockingFactors) {}

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
    Value output = convOp.getOutputs()[0];

    // static shapes.
    if (!hasStaticShape(image, filter, output))
      return failure();

    // tensor semantics.
    if (convOp.hasPureBufferSemantics())
      return failure();

    // blocking factors.
    if (blockingFactors.empty())
      return failure();

    return success();
  }

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(blockConv2DNchwFchwPreconditions(convOp)))
      return failure();
    FailureOr<linalg::GenericOp> maybeGeneric =
        mlir::linalgx::packConv2DNchwFchwOp(
            rewriter, convOp,
            getAsOpFoldResult(rewriter.getI64ArrayAttr(blockingFactors)));
    if (failed(maybeGeneric))
      return failure();
    linalg::GenericOp generic = *maybeGeneric;
    generic.setLibraryCallAttr(
        rewriter.getStringAttr("tpp.BlockedConv2DNchwFchwOp"));
    return success();
  }

private:
  SmallVector<int64_t> blockingFactors;
};

// Interchange a blocked convolutions to expose a linalg.matmul.
struct InterchangeIteratorsConv2DNchwFchw
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgx::utils::isBlockedConvolution(genericOp))
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

// Given a blocked convolution where the dimensions have been interchanged
// to expose a GEMM. Materialize outer loops and map the three innermost
// ones to a linalg.matmul operation.
struct RewriteConv2DNchwFchwToMatmul : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  bool hasInterchangedDims(linalg::GenericOp linalgOp) const {
    auto iteratorTypes = linalgOp.getIteratorTypesArray();
    if (iteratorTypes.size() != 9)
      return false;
    bool match = linalg::isParallelIterator(iteratorTypes[0]) &&
                 linalg::isParallelIterator(iteratorTypes[1]) &&
                 linalg::isParallelIterator(iteratorTypes[2]) &&
                 linalg::isReductionIterator(iteratorTypes[3]) &&
                 linalg::isReductionIterator(iteratorTypes[4]) &&
                 linalg::isReductionIterator(iteratorTypes[5]) &&
                 linalg::isParallelIterator(iteratorTypes[6]) &&
                 linalg::isParallelIterator(iteratorTypes[7]) &&
                 linalg::isReductionIterator(iteratorTypes[8]);
    return match;
  }

  // Make sure we are dealing with a generic 'Conv2DNchwFchwOp' blocked
  // with interchanged dimensions.
  LogicalResult
  rewriteConv2DNchwFchwPreconditions(linalg::GenericOp linalgOp) const {
    if (!hasInterchangedDims(linalgOp))
      return failure();
    return success();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(rewriteConv2DNchwFchwPreconditions(linalgOp)))
      return failure();
    FailureOr<linalg::MatmulOp> matmul =
        mlir::linalgx::rewriteConvToMatmul(rewriter, linalgOp);
    if (failed(matmul))
      return failure();
    return success();
  }
};

// Prepare for BRGEMM. Requires R = S = 1. The pattern collapses
// H and W on the image and P and Q on the output.
struct CollapseFilterAndImage : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult collapseFilterPreconditions(linalg::GenericOp linalgOp) const {
    if (!isMarkedWithTpp(linalgOp, "tpp.BlockedConv2DNchwFchwOp"))
      return failure();
    OpOperand *filter = linalgOp.getDpsInputOperands()[1];
    if (!hasFilterWithRandSEqualOne(filter, /*Rpos=*/2, /*Spos=*/3))
      return failure();
    return success();
  }

  SmallVector<ReassociationExprs, 2>
  convertAffineMapArrayToExprs(ArrayAttr affineMapArrayAttr) const {
    SmallVector<ReassociationExprs, 2> reassociationExprs;
    for (auto attr : affineMapArrayAttr)
      reassociationExprs.push_back(llvm::to_vector<4>(
          cast<AffineMapAttr>(attr).getValue().getResults()));
    return reassociationExprs;
  }

  // Collapse dimension at index 'startCollapse' to 'endCollapse'.
  Type getCollapsedType(OpOperand *operand, size_t startCollapse,
                        size_t endCollapse) const {
    assert(endCollapse > startCollapse && "expect >");
    ShapedType operandType = cast<ShapedType>(operand->get().getType());
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
    if (isa<MemRefType>(operandType))
      return MemRefType::get(newShape, operandType.getElementType());
    if (isa<RankedTensorType>(operandType))
      return RankedTensorType::get(newShape, operandType.getElementType());
    assert(false && "expect tensor or memref");
    abort();
  }

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(collapseFilterPreconditions(linalgOp)))
      return failure();

    // [original] = N K P Q k C R S c (R and S are 1)
    // [drop R and S] = N K P Q k C c
    // [collapse P and Q ] = N K P0 k C c
    // [new maps]:
    //  - filter (N K P0 k C c) -> (K C c k)
    //  - image  (N K P0 k C c) -> (N C P0 c)
    //  - output (N K P0 k C c) -> (N K P0 k)
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [&](MapList m) {
      return AffineMap::inferFromExprList(m, linalgOp.getContext());
    };
    AffineExpr tileN, tileK, P0, k, tileC, c;
    bindDims(linalgOp.getContext(), tileN, tileK, P0, k, tileC, c);
    SmallVector<AffineMap> newIndexingMaps = infer(
        {{tileN, tileC, P0, c}, {tileK, tileC, c, k}, {tileN, tileK, P0, k}});

    // We dropped R and S (two reductions) and collapse
    // parallel loops P and Q (aka H and W on the image).
    auto newIteratorTypes = {
        utils::IteratorType::parallel,  utils::IteratorType::parallel,
        utils::IteratorType::parallel,  utils::IteratorType::parallel,
        utils::IteratorType::reduction, utils::IteratorType::reduction};

    Location loc = linalgOp.getLoc();
    OpOperand *image = linalgOp.getDpsInputOperands()[0];
    Type newImageType = getCollapsedType(image, 2, 3);
    auto reassociationImage = getReassociationIndicesForCollapse(
        cast<ShapedType>(image->get().getType()).getShape(),
        cast<ShapedType>(newImageType).getShape());
    if (!reassociationImage)
      return failure();

    OpOperand *filter = linalgOp.getDpsInputOperands()[1];
    Type newFilterType = getCollapsedType(filter, 1, 3);
    auto reassociationFilter = getReassociationIndicesForCollapse(
        cast<ShapedType>(filter->get().getType()).getShape(),
        cast<ShapedType>(newFilterType).getShape());
    if (!reassociationFilter)
      return failure();

    OpOperand &output = linalgOp.getDpsInitsMutable()[0];
    Type newOutputType = getCollapsedType(&output, 2, 3);
    auto reassociationOutput = getReassociationIndicesForCollapse(
        cast<ShapedType>(output.get().getType()).getShape(),
        cast<ShapedType>(newOutputType).getShape());
    if (!reassociationOutput)
      return failure();

    Value collapsedImage = linalgx::utils::collapse(
        rewriter, loc, image->get(), newImageType, *reassociationImage);

    Value collapsedFilter = linalgx::utils::collapse(
        rewriter, loc, filter->get(), newFilterType, *reassociationFilter);

    Value collapsedOutput = linalgx::utils::collapse(
        rewriter, loc, output.get(), newOutputType, *reassociationOutput);

    linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
        loc, newOutputType, ValueRange{collapsedImage, collapsedFilter},
        collapsedOutput, newIndexingMaps, newIteratorTypes, /*docs=*/"",
        /*library_call=*/"tpp.BlockedandCollapsedConv2DNchwFchwOp");
    rewriter.inlineRegionBefore(linalgOp->getRegion(0),
                                replacementOp.getRegion(),
                                replacementOp.getRegion().begin());
    Value res = replacementOp->getResult(0);
    reassociationOutput = getReassociationIndicesForReshape(
        cast<ShapedType>(res.getType()),
        cast<ShapedType>(output.get().getType()));
    if (!reassociationOutput)
      return failure();
    Value resExpanded = linalgx::utils::expand(
        rewriter, loc, res, output.get().getType(), *reassociationOutput);
    rewriter.replaceOp(linalgOp, resExpanded);
    return success();
  }
};

struct InterchangeAfterBlockingAndCollapsing
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isMarkedWithTpp(linalgOp, "tpp.BlockedandCollapsedConv2DNchwFchwOp"))
      return failure();

    // clang-format off
    // N       [parallel]
    //  K'     [parallel]
    //   P * Q [parallel]
    //    k    [parallel]
    //     C'  [reduction]
    //      c  [reduction]
    //        output[N][K'][P * Q][k] += image[N][C'][H * W][c] * filter[K'][C'][c][k]

    // expose BRGEMM by interchange:

    // N       [parallel]
    //  K'     [parallel]
    //   C'    [reduction] // BRGEMM red dimension
    //   /* GEMM */
    //   P * Q [parallel]
    //    k    [parallel]
    //      c  [reduction]
    //        output[N][K'][P * Q][k] += image[N][C'][H * W][c] * filter[K'][C'][c][k]
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
    (*maybeInterchange).setLibraryCallAttr(name);
    return success();
  }
};

struct MapToBRGEMM : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!isMarkedWithTpp(linalgOp,
                         "tpp.BlockedCollapsedAndInterConv2DNchwFchwOp"))
      return failure();
    FailureOr<SmallVector<Value>> maybeLoopsOrGenericRes =
        mlir::linalgx::rewriteToBRGemmOp(rewriter, linalgOp);
    if (failed(maybeLoopsOrGenericRes))
      return failure();
    return success();
  }
};

// patterns for mapping a Conv2DNhwcHwcfOp to a GEMM operation.
void populateRewrite2DNhwcHwcfConvPatterns(RewritePatternSet &patterns) {
  patterns.insert<GeneralizeConv2DNhwcHwcf, RewriteConv2DNhwcHwcfToMatmul,
                  InterchangeIteratorsConv2DNhwcHwcf>(patterns.getContext());
}

// patterns for mapping a blocked convolutions to a GEMM/BRGEMM operations.
void populateRewriteBlockedConvPatterns(RewritePatternSet &patterns,
                                        bool enableBrgemm) {
  // clang-format off
  //
  // blocked conv: [N][K'][P][Q][k] = [N][C'][H][W][c] * [K'][C'][R][S][c][k]
  //
  // if (R = S = 1) -> [P * Q] == [H * W]
  //
  // collapsing: [N][K'][P * Q][k] = [N][C'][H * W][c] * [K'][C'][c][k]
  // [*][* ][P * Q][k] = [*][* ][H * W][c] * [* ][* ][c][k] // GEMM with c as red.
  // [*][* ][P * Q][k] = [*][C'][H * W][c] * [* ][C'][c][k] // BRGEMM with C' as red.
  //
  // clang-format on

  // Rewrite to GEMM.
  if (!enableBrgemm) {
    patterns.insert<RewriteConv2DNchwFchwToMatmul,
                    InterchangeIteratorsConv2DNchwFchw>(patterns.getContext());
  }
  // Rewrite to BRGEMM.
  else {
    patterns.insert<CollapseFilterAndImage,
                    InterchangeAfterBlockingAndCollapsing, MapToBRGEMM>(
        patterns.getContext());
  }
}

struct RewriteConvToMatmulOrBrgemm
    : public tpp::impl::RewriteConvToMatmulOrBrgemmBase<
          RewriteConvToMatmulOrBrgemm> {
  using RewriteConvToMatmulOrBrgemmBase::RewriteConvToMatmulOrBrgemmBase;

  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    populateRewrite2DNhwcHwcfConvPatterns(patterns);
    populateRewriteBlockedConvPatterns(patterns, this->enableBrgemm);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace
