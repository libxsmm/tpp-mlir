//===- ConvertVectorToX86.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTVECTORTOX86
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Implements outer product contraction as a sequence of broadcast and
// FMA operations.
//
// For example - for F32 type:
// ```
//   vector.contract <1x1xf32>, <1x16xf32> into <1x16xf32>
// ```
// to
// ```
//   vector.broadcast %lhs to <16xf32>
//   vector.fma vector<16xf32>
// ```
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
    if (!lhsTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only F32 lowering is supported now");

    // Constrain support to only one M and one N dimension.
    // TODO: Relax matching constraints.
    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
    FailureOr<linalg::ContractionDimensions> dims =
        linalg::inferContractionDims(indexingMaps);
    assert(succeeded(dims) && "Failed to infer contraction");
    if (dims->m.size() != 1 || dims->n.size() != 1)
      return rewriter.notifyMatchFailure(
          contractOp, "expects only 2 parallel (M and N) non-batch dimensions");

    // TODO: Improve outerproduct detection.
    //       Removing unit dims could simplify matching, however, it might
    //       impact current VNNI detection.
    if (llvm::any_of(lhsTy.getShape(), [](int64_t dim) { return dim != 1; }))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects single element LHS");
    VectorType rhsTy = contractOp.getRhsType();
    ArrayRef<int64_t> rhsShape = rhsTy.getShape();
    if (llvm::any_of(llvm::seq<int64_t>(0, rhsTy.getRank() - 2),
                     [&](int64_t i) { return rhsShape[i] != 1; }))
      return rewriter.notifyMatchFailure(contractOp, "Invalid RHS shape");
    auto accTy = dyn_cast<VectorType>(contractOp.getAccType());
    assert(accTy && "Invalid accumulator");
    ArrayRef<int64_t> accShape = accTy.getShape();
    if (accShape[accTy.getRank() - 2] != 1 ||
        accShape.back() != rhsShape.back())
      return rewriter.notifyMatchFailure(contractOp,
                                         "Unsupported accumulator shape");

    // Turn an outer product contraction into a broadcast+FMA sequence.
    auto castLhs = vector::ShapeCastOp::create(rewriter, 
        loc, VectorType::get(1, lhsTy.getElementType()), contractOp.getLhs());
    auto castRhs = vector::ShapeCastOp::create(rewriter, 
        loc, VectorType::get(rhsShape.back(), rhsTy.getElementType()),
        contractOp.getRhs());
    auto castAcc = vector::ShapeCastOp::create(rewriter, 
        loc, VectorType::get(accShape.back(), accTy.getElementType()),
        contractOp.getAcc());
    auto broadcastLhs = vector::BroadcastOp::create(rewriter, 
        loc, castRhs.getResult().getType(), castLhs);
    auto fma =
        vector::FMAOp::create(rewriter, loc, broadcastLhs, castRhs, castAcc);
    auto castFma = vector::ShapeCastOp::create(rewriter, loc, accTy, fma);

    rewriter.replaceOp(contractOp, castFma);

    return success();
  }
};

struct ConvertVectorToX86
    : public tpp::impl::ConvertVectorToX86Base<ConvertVectorToX86> {
  using ConvertVectorToX86Base::ConvertVectorToX86Base;

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    // TODO: Enable patterns based on available target extension.
    // TODO: Use benefit to control which patterns application priority.
    patterns.add<ContractionToFMA>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
