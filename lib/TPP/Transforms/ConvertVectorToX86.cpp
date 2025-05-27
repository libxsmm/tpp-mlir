//===-ConvertVectorToX86.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTVECTORTOX86
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

struct ContractionToFMA : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    auto loc = contractOp.getLoc();

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind");

    VectorType lhsTy = contractOp.getLhsType();
    // TODO: Extend to support VNNI.
    if (lhsTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only F32 lowering is supported now");

    // Constrain support to only one M and one N dimension.
    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(indexingMaps);
    assert(succeeded(dims) && "Failed to infer contraction");
    if (dims->m.size() != 1 || dims->n.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "expects at only 2 parallel non-batch dimensions");

    // TODO: Improve outerproduct detection.
    if (llvm::any_of(lhsTy.getShape(), [](int64_t dim) { return dim != 1; }))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects single element LHS");
    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    if (llvm::any_of(llvm::seq<int64_t>(0, rhsTy.getRank() - 2),
                     [&](int64_t i) { return rhsShape[i] != 1; }))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects contiguous 1D-like RHS");
    auto accTy = dyn_cast<VectorType>(contractOp.getAccType());
    assert(accTy && "Invalid accumulator");
    ArrayRef<int64_t> accShape = accTy.getShape();
    if (accShape[accTy.getRank() - 2] != 1 ||
        accShape.back() != rhsShape.back())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Unsupported accumulator shape");

    auto castLhs = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get(1, lhsTy.getElementType()), contractOp.getLhs());
    auto castRhs = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get(rhsShape.back(), rhsTy.getElementType()),
        contractOp.getRhs());
    auto castAcc = rewriter.create<vector::ShapeCastOp>(
        loc, VectorType::get(accShape.back(), accTy.getElementType()),
        contractOp.getAcc());
    auto broadcastLhs = rewriter.create<vector::BroadcastOp>(
        loc, castRhs.getResult().getType(), castLhs);
    auto fma =
        rewriter.create<vector::FMAOp>(loc, broadcastLhs, castRhs, castAcc);
    auto castFma = rewriter.create<vector::ShapeCastOp>(loc, accTy, fma);

    rewriter.replaceOp(contractOp, castFma);

    return success();
  }
};

struct ConvertVectorToX86
    : public impl::ConvertVectorToX86Base<ConvertVectorToX86> {
  using ConvertVectorToX86Base::ConvertVectorToX86Base;

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ContractionToFMA>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace tpp
} // namespace mlir
