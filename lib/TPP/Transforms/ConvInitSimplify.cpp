//===- ConvInitSimplify.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVINITSIMPLIFY
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

static bool isBroadCastOp(linalg::GenericOp linalgOp) {
  // clang-format off
  AffineMap inputMap;
  using namespace mlir::structured_match;
  auto bCastMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>()
    .operation(NumDpsInits(EqualsTo(1)))
    .operation(NumDpsInputs(EqualsTo(1)))
    .output(MatchAll(), HasMap(Identity()))
    .input(MatchOne(0), HasMap(BroadcastableProjectedPermutation(), &inputMap))
    .region(
      MatchOne(0), WithSingleOp<linalg::YieldOp>(/*operands=*/nullptr));
  // clan-format on
  SmallVector<unsigned> perm;
  return bCastMatcher.match(linalgOp) &&
    inputMap.isPermutationOfMinorIdentityWithBroadcasting(perm);
}

static std::optional<linalg::GenericOp> getBroadCastProdcuer(OpOperand *rhs) {
  linalg::GenericOp broadcastOp = rhs->get().getDefiningOp<linalg::GenericOp>();
  if (!broadcastOp || !isBroadCastOp(broadcastOp))
    return std::nullopt;
  return broadcastOp;
}

static bool isElemetWiseAdd(linalg::GenericOp linalgOp) {
  // clang-format off
  using namespace mlir::structured_match;
  auto addMatcher =
    StructuredOpMatcher::make<linalg::LinalgOp>()
    .input(MatchAll(), HasMap(Identity()))
    .output(MatchAll(), HasMap(Identity()))
    .dim(MatchAll(), mlir::utils::IteratorType::parallel)
    .region(
      MatchOne(0), WithSingleOp<arith::AddFOp>(/*operands=*/nullptr));
  // clang-format on
  return addMatcher.match(linalgOp);
}

// Instead of initializing the output of a convolution with zero and then add a
// bias, initialize the output of the convolution with the bias.
struct EliminateZeroInitAndAddBiasToInit
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (!linalgOp.hasPureTensorSemantics())
      return failure();
    if (linalgOp->getNumOperands() != 3 || linalgOp->getNumResults() != 1)
      return failure();

    if (!isElemetWiseAdd(linalgOp))
      return failure();

    OpOperand *lhs = linalgOp.getDpsInputOperand(0);
    OpOperand *rhs = linalgOp.getDpsInputOperand(1);

    auto convProducer = lhs->get().getDefiningOp<linalg::Conv2DNhwcHwcfOp>();
    auto broadCastProducer = getBroadCastProdcuer(rhs);
    if (!convProducer || !broadCastProducer ||
        !utils::isZeroTensor(convProducer.getDpsInitOperand(0)->get()) ||
        !convProducer.getTiedOpResult(convProducer.getDpsInitOperand(0)))
      return failure();

    SmallVector<Value> convInputs;
    for (OpOperand *operand : convProducer.getDpsInputOperands())
      convInputs.push_back(operand->get());
    Value broadCastOutput = broadCastProducer->getTiedOpResult(
        broadCastProducer->getDpsInitOperand(0));

    auto replOp = rewriter.create<linalg::Conv2DNhwcHwcfOp>(
        linalgOp.getLoc(), broadCastOutput.getType(), convInputs,
        broadCastOutput, convProducer.getStrides(),
        convProducer.getDilations());
    if (auto metadata = convProducer->getAttr("metadata"))
      replOp->setAttr("metadata", metadata);

    rewriter.replaceOp(linalgOp, replOp.getResults());
    return success();
  }
};

struct ConvInitSimplify
    : public tpp::impl::ConvInitSimplifyBase<ConvInitSimplify> {
  void runOnOperation() override {
    RewritePatternSet patterns(getOperation().getContext());
    patterns.add<EliminateZeroInitAndAddBiasToInit>(patterns.getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
