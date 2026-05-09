//===-ConvertLinalgToInplace.cpp ---------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTLINALGTOINPLACE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;

namespace {

struct ConvertAddInplace : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getBody()->getOperations().size() != 2)
      return failure();
    auto addf = dyn_cast<arith::AddFOp>(&op.getBody()->getOperations().front());
    if (!addf)
      return failure();
    if (op.getNumOperands() == 2)
      return failure();
    // TODO: This needs to be changed in the future to a detailed analysis that
    // checks if the second input is not used subsequently
    if (op.getInputs()[0] == op.getInputs()[1])
      return failure();
    SmallVector<AffineMap> indexingMaps;
    SmallVector<utils::IteratorType> iteratorTypes;
    for (auto iteratorTypesArray : op.getIteratorTypesArray()) {
      iteratorTypes.push_back(iteratorTypesArray);
    }

    Value inputs, outputs;
    // Check which input is marked as non-broadcastable
    if (op.getIndexingMapsArray()[1] ==
        rewriter.getMultiDimIdentityMap(
            op.getIndexingMapsArray()[1].getNumDims())) {
      indexingMaps.push_back(op.getIndexingMapsArray()[0]);
      indexingMaps.push_back(op.getIndexingMapsArray()[1]);
      inputs = op.getInputs()[0];
      outputs = op.getInputs()[1];
    } else {
      indexingMaps.push_back(op.getIndexingMapsArray()[1]);
      indexingMaps.push_back(op.getIndexingMapsArray()[0]);
      inputs = op.getInputs()[1];
      outputs = op.getInputs()[0];
    }
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op.getResultTypes(), inputs, outputs, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto scalarOp = arith::AddFOp::create(builder, loc, regionArgs);
          linalg::YieldOp::create(builder, loc, scalarOp.getResult());
        });
    return success();
  }
};

struct EltwiseUnaryGenericToInplace
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!genericOp.hasPureTensorSemantics())
      return rewriter.notifyMatchFailure(genericOp, "expects tensor semantics");

    if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
      return rewriter.notifyMatchFailure(genericOp, "not a unary operation");

    if (genericOp.getInputs()[0].getType() !=
        genericOp.getOutputs()[0].getType())
      return rewriter.notifyMatchFailure(
          genericOp, "input type does not match the output");

    // Elementwise operation guarantees that all output elements are updated.
    // The output initial values can be ignored and the output buffer can be
    // replaced if the output is not used (write only).
    if (!linalg::isElementwise(genericOp))
      return rewriter.notifyMatchFailure(genericOp,
                                         "not an elementwise operation");
    if (genericOp.payloadUsesValueFromOperand(genericOp.getDpsInitOperand(0)))
      return rewriter.notifyMatchFailure(genericOp,
                                         "expects output to be unused");

    // Elementwise operation still allows different indexing for its input e.g.,
    // one of the dimensions can be fixed for the input.
    // Ensure that indexing maps of both operands are be equal. Otherwise,
    // the input cannot replace the output buffer.
    SmallVector<AffineMap> maps = genericOp.getIndexingMapsArray();
    if (maps[0] != maps[1])
      return rewriter.notifyMatchFailure(genericOp,
                                         "expects matching indexing maps");

    // Use the input value directly as the output.
    ValueRange outputs = genericOp.getInputs();
    SmallVector<Type> resultTypes = TypeRange(ValueRange{outputs});
    SmallVector<AffineMap> indexingMaps{maps[1]};

    auto newGeneric = linalg::GenericOp::create(rewriter, 
        genericOp.getLoc(), resultTypes, /*inputs=*/ValueRange{}, outputs,
        indexingMaps, genericOp.getIteratorTypesArray());
    rewriter.inlineRegionBefore(genericOp->getRegion(0), newGeneric.getRegion(),
                                newGeneric.getRegion().begin());

    // Replace input block arguments usage with the output block argument.
    Block *body = newGeneric.getBody();
    rewriter.replaceAllUsesWith(body->getArguments()[0],
                                body->getArguments()[1]);
    body->eraseArgument(0);

    rewriter.replaceOp(genericOp, newGeneric->getResults());

    return success();
  }
};

struct ConvertLinalgToInplace
    : public tpp::impl::ConvertLinalgToInplaceBase<ConvertLinalgToInplace> {
  void populateCombinePatterns(RewritePatternSet &patterns) {
    patterns.add<ConvertAddInplace, EltwiseUnaryGenericToInplace>(
        patterns.getContext());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
