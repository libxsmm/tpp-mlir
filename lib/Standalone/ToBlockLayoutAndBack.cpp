//===- ConvertToBlockLayoutAndBack.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/LinalgXOps.h"
#include "Standalone/Dialect/Tpp/TppUtils.h"
#include "Standalone/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalgx;

#define GEN_PASS_CLASSES
#include "Standalone/Passes.h.inc"

namespace {

enum class BlockLayout { FORMAT_NCnc, FORMAT_KCck };

// Get affine maps from NC layout to NCnc layout.
static std::pair<AffineMap, AffineMap>
getMapsToBlockLayoutNC_NCnc(int64_t dims, int64_t blockFactor,
                            MLIRContext *ctx) {
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(dims, ctx);

  AffineExpr N, C, n, c;
  bindDims(ctx, N, C, n, c);
  AffineMap inputMap = AffineMap::get(
      dims, /*symbols=*/0, {{N * blockFactor + n}, {C * blockFactor + c}}, ctx);
  return {inputMap, outputMap};
}

// Get affine maps from KC to KCck layout.
static std::pair<AffineMap, AffineMap>
getMapsToBlockLayoutKC_KCck(int64_t dims, int64_t blockFactor,
                            MLIRContext *ctx) {
  AffineMap outputMap = AffineMap::getMultiDimIdentityMap(dims, ctx);

  AffineExpr K, C, c, k;
  bindDims(ctx, K, C, c, k);
  AffineMap inputMap = AffineMap::get(
      dims, /*symbols=*/0, {{C * blockFactor + c}, {K * blockFactor + k}}, ctx);
  return {inputMap, outputMap};
}

// Get affine maps fron NCnc layout to NC.
static std::pair<AffineMap, AffineMap>
getMapsFromBlockLayoutNCnc_NC(int64_t dims, int64_t blockFactor,
                              MLIRContext *ctx) {
  std::pair<AffineMap, AffineMap> toBlockTensor =
      getMapsToBlockLayoutNC_NCnc(dims, blockFactor, ctx);
  return {toBlockTensor.second, toBlockTensor.first};
}

// Given a tensor `tensor` a layout format and a block factor, relayout
// the tensor using the specified block factor and tensor relayout.
static FailureOr<Value> getReshapedTensor(Location loc, Value tensor,
                                          BlockLayout layout,
                                          int64_t blockFactor,
                                          PatternRewriter &rewriter) {
  MLIRContext *ctx = rewriter.getContext();
  RankedTensorType unBlockedTensorType =
      tensor.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> shape = unBlockedTensorType.getShape();

  // check if the 'blockFactor' perfectly divides shape.
  for (int64_t s : shape)
    if (s % blockFactor != 0)
      return failure();

  std::pair<AffineMap, AffineMap> maps;
  SmallVector<int64_t> shapeBlockedTensor;
  if (layout == BlockLayout::FORMAT_NCnc) {
    int64_t N = shape[0] / blockFactor;
    int64_t C = shape[1] / blockFactor;
    int64_t n = blockFactor, c = blockFactor;
    shapeBlockedTensor = {N, C, n, c};
    maps = getMapsToBlockLayoutNC_NCnc(shapeBlockedTensor.size(), blockFactor,
                                       ctx);
  } else {
    assert(layout == BlockLayout::FORMAT_KCck);
    int64_t K = shape[1] / blockFactor;
    int64_t C = shape[0] / blockFactor;
    int64_t k = blockFactor, c = blockFactor;
    shapeBlockedTensor = {K, C, c, k};
    maps = getMapsToBlockLayoutKC_KCck(shapeBlockedTensor.size(), blockFactor,
                                       ctx);
  }
  Value initTensor = rewriter.create<linalg::InitTensorOp>(
      loc, shapeBlockedTensor, unBlockedTensorType.getElementType());
  return rewriter
      .create<linalgx::Relayout>(loc, initTensor.getType(), tensor, initTensor,
                                 maps.first, maps.second)
      .getResult()[0];
}

// Relayout MatmulOp as following:
// [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
// CB = batch reduce dimension.
struct DoItOnMatmul : public OpRewritePattern<linalg::MatmulOp> {
  DoItOnMatmul(MLIRContext *context, int64_t blockingFactor,
               PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        blockingFactor(blockingFactor) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (matmulOp.hasDynamicShape() || matmulOp.hasBufferSemantics())
      return failure();

    Location loc = matmulOp.getLoc();
    MLIRContext *ctx = matmulOp.getContext();
    SmallVector<Value, 2> reshapedInputTensors;
    // reshape input A to NCnc and B to KCck.
    BlockLayout blockLayout = BlockLayout::FORMAT_NCnc;
    for (Value input : matmulOp.getInputs()) {
      FailureOr<Value> maybeReshaped =
          getReshapedTensor(loc, input, blockLayout, blockingFactor, rewriter);
      if (failed(maybeReshaped))
        return failure();
      reshapedInputTensors.push_back(*maybeReshaped);
      blockLayout = BlockLayout::FORMAT_KCck;
    }

    blockLayout = BlockLayout::FORMAT_NCnc;
    Value reshapedOutputTensor = nullptr;
    for (Value output : matmulOp.getOutputs()) {
      FailureOr<Value> maybeReshaped =
          getReshapedTensor(loc, output, blockLayout, blockingFactor, rewriter);
      if (failed(maybeReshaped))
        return failure();
      reshapedOutputTensor = *maybeReshaped;
    }

    // swap linalg.matmul with a linalg.generic.
    AffineExpr p1, p2, r1, p3, p4, r2;
    bindDims(ctx, p1, p2, r1, p3, p4, r2);
    AffineMap mapA =
        AffineMap::get(/*dims=*/6, /*symbols=*/0, {p1, r1, p3, r2}, ctx);
    AffineMap mapB =
        AffineMap::get(/*dims=*/6, /*symbols=*/0, {p2, r1, r2, p4}, ctx);
    AffineMap mapC =
        AffineMap::get(/*dims=*/6, /*symbols=*/0, {p1, p2, p3, p4}, ctx);
    linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
        loc, reshapedOutputTensor.getType(), reshapedInputTensors,
        ValueRange{reshapedOutputTensor}, ArrayRef<AffineMap>{mapA, mapB, mapC},
        ArrayRef<StringRef>{
            getParallelIteratorTypeName(), getParallelIteratorTypeName(),
            getReductionIteratorTypeName(), getParallelIteratorTypeName(),
            getParallelIteratorTypeName(), getReductionIteratorTypeName()},
        /*doc=*/"", /*libraryCall=*/"");
    rewriter.inlineRegionBefore(matmulOp.region(), replacementOp.region(),
                                replacementOp.region().begin());

    // convert back from block layout.
    Value outBlockedTensor = replacementOp.getResult(0);
    Value outUnBlockedTensor = matmulOp.getOutputs()[0];
    std::pair<AffineMap, AffineMap> maps = getMapsFromBlockLayoutNCnc_NC(
        outBlockedTensor.getType().cast<ShapedType>().getRank(), blockingFactor,
        ctx);
    Value outReplacement =
        rewriter
            .create<linalgx::Relayout>(loc, outUnBlockedTensor.getType(),
                                       outBlockedTensor, outUnBlockedTensor,
                                       maps.first, maps.second)
            .getResults()[0];
    rewriter.replaceOp(matmulOp, outReplacement);
    return success();
  }

private:
  int64_t blockingFactor = 0;
};

// Relayout relu as: [NB][KB][nb][kb]
struct SinkBlockLayoutAfterRelu : public OpRewritePattern<linalg::GenericOp> {
  SinkBlockLayoutAfterRelu(MLIRContext *context, int64_t blockingFactor,
                           PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit),
        blockingFactor(blockingFactor) {}

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!tpp::isMarkedWithTpp(linalgOp, "tpp.relu"))
      return failure();
    if (linalgOp.getNumInputs() == 0)
      return failure();
    Location loc = linalgOp.getLoc();
    Value operand = linalgOp.getInputOperand(0)->get();
    linalgx::Relayout fromBlockLayout =
        operand.getDefiningOp<linalgx::Relayout>();
    if (!fromBlockLayout || !fromBlockLayout->getResult(0).hasOneUse())
      return failure();

    Value blockTensor = fromBlockLayout.inputs()[0];
    ShapedType blockTensorType = blockTensor.getType().cast<ShapedType>();
    BlockLayout blockLayout = BlockLayout::FORMAT_NCnc;
    Value reluBuffer = *getReshapedTensor(
        loc, linalgOp.outputs()[0], blockLayout, blockingFactor, rewriter);

    AffineMap mapI =
        AffineMap::getMultiDimIdentityMap(/*dims=*/4, linalgOp.getContext());
    AffineMap mapO = mapI;
    linalg::GenericOp newReluOp = rewriter.create<linalg::GenericOp>(
        loc, blockTensorType, ValueRange{blockTensor}, ValueRange{reluBuffer},
        ArrayRef<AffineMap>{mapI, mapO},
        ArrayRef<StringRef>{
            getParallelIteratorTypeName(), getParallelIteratorTypeName(),
            getParallelIteratorTypeName(), getParallelIteratorTypeName()},
        /*doc=*/"", /*libraryCall=*/"tpp.relu");
    rewriter.inlineRegionBefore(linalgOp.region(), newReluOp.region(),
                                newReluOp.region().begin());

    Value outUnBlockedTensor = linalgOp.getOutputOperand(0)->get();
    Type outUnBlockedTensorType = outUnBlockedTensor.getType();
    std::pair<AffineMap, AffineMap> maps = getMapsFromBlockLayoutNCnc_NC(
        /*dims=*/4, blockingFactor, linalgOp.getContext());

    Value outReplacement =
        rewriter
            .create<linalgx::Relayout>(
                loc, outUnBlockedTensorType, newReluOp->getResult(0),
                outUnBlockedTensor, maps.first, maps.second)
            .getResult()[0];
    rewriter.replaceOp(linalgOp, outReplacement);
    return success();
  }

private:
  int64_t blockingFactor = 0;
};

// pattern to go from linalg.generic {tpp.matmul} to linalg.matmul.
struct DeGeneralizeMatmul : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasTensorSemantics())
      return failure();
    if (!tpp::isMarkedWithTpp(linalgOp, "tpp.matmul"))
      return failure();
    SmallVector<Value> inputOperands = linalgOp.getInputOperands();
    SmallVector<Value> outputOperands = linalgOp.getOutputOperands();
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        linalgOp, linalgOp.getResultTypes(), inputOperands, outputOperands);
    return success();
  }
};

struct ToBlockLayoutAndBack
    : public ToBlockLayoutAndBackBase<ToBlockLayoutAndBack> {
  ToBlockLayoutAndBack() = default;
  ToBlockLayoutAndBack(int64_t blockingFactor) {
    this->blockingFactor = blockingFactor;
  }
  void runOnOperation() override {
    if (blockingFactor == 0)
      return;
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DoItOnMatmul, SinkBlockLayoutAfterRelu>(ctx, blockingFactor);
    patterns.add<DeGeneralizeMatmul>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return;
  }
};

} // end namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::tpp::createToBlockLayoutAndBackPass(int64_t blockingFactor) {
  return std::make_unique<ToBlockLayoutAndBack>();
}
