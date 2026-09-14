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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTLINALGTOINPLACE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;

namespace {

// An input may only replace the destination buffer if no other op consumes
// it. Shape-only queries (tensor.dim) do not read the buffer contents and
// remain valid after one-shot bufferization, so they are allowed.
static bool hasOnlySafeUses(Value value, Operation *owner) {
  return llvm::all_of(value.getUses(), [&](OpOperand &use) {
    return use.getOwner() == owner || isa<tensor::DimOp>(use.getOwner());
  });
}

struct ConvertAddInplace : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {

    // Only rewrite ops on tensors: with buffer semantics the destination swap
    // below would write into a buffer the surrounding IR does not expect.
    if (!op.hasPureTensorSemantics())
      return failure();

    if (op.getBody()->getOperations().size() != 2)
      return failure();
    auto addf = dyn_cast<arith::AddFOp>(&op.getBody()->getOperations().front());
    if (!addf)
      return failure();
    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1)
      return failure();
    if (op.getInputs()[0] == op.getInputs()[1])
      return failure();

    // If the destination is already one of the inputs (e.g. the accumulation
    // pattern `ins(%acc, %x) outs(%acc)`), the op is already in-place on the
    // intended buffer. Leave it untouched: picking a different input as the
    // destination would write the result into a buffer the surrounding IR
    // does not expect (e.g. a per-iteration temporary of a loop-carried
    // value), which one-shot bufferization then rejects.
    //
    // Example of the IR this guards against (loop-carried accumulation):
    //
    //   %accN = scf.for %i = %lb to %ub step %c1
    //       iter_args(%acc = %acc0) -> tensor<8x4xf32> {
    //     %x = ... : tensor<8x4xf32>  // per-iteration temporary
    //     %new = linalg.generic
    //         ins(%acc, %x : tensor<8x4xf32>, tensor<8x4xf32>)
    //         outs(%acc : tensor<8x4xf32>) {
    //       ^bb0(%a: f32, %b: f32, %out: f32):
    //         %s = arith.addf %a, %b : f32
    //         linalg.yield %s : f32
    //     } -> tensor<8x4xf32>
    //     scf.yield %new : tensor<8x4xf32>
    //   }
    //
    // Without the check below, the pattern would repoint the destination to
    // %x, yielding `ins(%acc) outs(%x)`. Then %new no longer aliases %acc's
    // buffer, breaking the loop-carried buffer identity: one-shot
    // bufferization cannot keep the iteration argument in place and rejects
    // the IR (or falls back to per-iteration copies).
    Value init = op.getDpsInits()[0];
    if (init == op.getInputs()[0] || init == op.getInputs()[1])
      return failure();

    // For the out-of-place form `ins(%a, %b) outs(%init)`, an input may only
    // serve as the in-place destination if:
    //   1. its indexing map is an identity (it covers the full result shape),
    //   2. it has no uses besides this op, so writing into it cannot corrupt
    //      any other consumer (the uses of this op's result are repointed).
    auto isIdentityMap = [&](unsigned idx) {
      return op.getIndexingMapsArray()[idx] ==
             rewriter.getMultiDimIdentityMap(
                 op.getIndexingMapsArray()[idx].getNumDims());
    };

    Value inputs, outputs;
    Type initType = init.getType();
    if (isIdentityMap(1) && op.getInputs()[1].getType() == initType &&
        hasOnlySafeUses(op.getInputs()[1], op)) {
      inputs = op.getInputs()[0];
      outputs = op.getInputs()[1];
    } else if (isIdentityMap(0) && op.getInputs()[0].getType() == initType &&
               hasOnlySafeUses(op.getInputs()[0], op)) {
      inputs = op.getInputs()[1];
      outputs = op.getInputs()[0];
    } else {
      // Neither input can safely become the destination; stay out-of-place.
      return failure();
    }

    SmallVector<AffineMap> indexingMaps;
    SmallVector<utils::IteratorType> iteratorTypes;
    for (auto iteratorTypesArray : op.getIteratorTypesArray()) {
      iteratorTypes.push_back(iteratorTypesArray);
    }
    if (outputs == op.getInputs()[1]) {
      indexingMaps.push_back(op.getIndexingMapsArray()[0]);
      indexingMaps.push_back(op.getIndexingMapsArray()[1]);
    } else {
      indexingMaps.push_back(op.getIndexingMapsArray()[1]);
      indexingMaps.push_back(op.getIndexingMapsArray()[0]);
    }
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op.getResultTypes(), inputs, outputs, indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto scalarOp = arith::AddFOp::create(builder, loc, regionArgs);
          // Preserve the fastmath flags of the original addf.
          scalarOp.setFastmath(addf.getFastmath());
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

    // The input may only replace the output buffer if no other op consumes
    // it; otherwise the in-place write would corrupt those consumers.
    // Shape-only queries (tensor.dim) are fine.
    if (!hasOnlySafeUses(genericOp.getInputs()[0], genericOp))
      return rewriter.notifyMatchFailure(genericOp,
                                         "expects input to have no other uses");

    // Use the input value directly as the output.
    ValueRange outputs = genericOp.getInputs();
    SmallVector<Type> resultTypes = TypeRange(ValueRange{outputs});
    SmallVector<AffineMap> indexingMaps{maps[1]};

    auto newGeneric = linalg::GenericOp::create(
        rewriter, genericOp.getLoc(), resultTypes, /*inputs=*/ValueRange{},
        outputs, indexingMaps, genericOp.getIteratorTypesArray());
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
