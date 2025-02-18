//===- ConvertCheckToLoops.cpp ----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Check/CheckOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::check;
using namespace mlir::cf;
using namespace mlir::arith;
using namespace mlir::math;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTCHECKTOLOOPS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Convert
// check.expect_almost_equals(%t1:tensor<2x2xf32>, %t2:tensor<2x2xf32>,
// %threshold:f32)
// to the pattern:
// scf.some_loop(%i, %j)
// %0 = load from %t1
// %1 = load from %t2
// %diff = math.subf %0, %1
// %abs = arith.absf %diff
// %compare = arith.cmpf le, %abs, %threshold
// cf.assert %compare, "Result mismatch"
struct ConvertAlmostEqualsOp
    : public OpRewritePattern<check::ExpectAlmostEqOp> {
  using OpRewritePattern<check::ExpectAlmostEqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(check::ExpectAlmostEqOp almostEqOp,
                                PatternRewriter &rewriter) const override {
    Location loc = almostEqOp.getLoc();
    SmallVector<Value> ubs;
    if (!isa<MemRefType>(almostEqOp.getLhs().getType())) {
      return failure();
    }
    size_t rank = cast<MemRefType>(almostEqOp.getLhs().getType()).getRank();
    for (size_t idx = 0; idx < rank; idx++) {
      auto dim = linalg::createOrFoldDimOp(rewriter, loc,
                                           almostEqOp.getOperand(0), idx);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);
    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value scalarLhs =
              b.create<memref::LoadOp>(loc, almostEqOp.getLhs(), localIvs);
          Value scalarRhs =
              b.create<memref::LoadOp>(loc, almostEqOp.getRhs(), localIvs);
          Value compare;
          if (isa<mlir::FloatType>(scalarLhs.getType())) {
            Value diff = b.create<arith::SubFOp>(loc, scalarLhs, scalarRhs);
            Value abs = b.create<math::AbsFOp>(loc, diff);
            compare = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE,
                                              abs, almostEqOp.getThreshold());
          } else {
            Value diff = b.create<arith::SubIOp>(loc, scalarLhs, scalarRhs);
            Value abs = b.create<math::AbsIOp>(loc, diff);
            compare = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                              abs, almostEqOp.getThreshold());
          }
          b.create<cf::AssertOp>(loc, compare,
                                 b.getStringAttr("Result mismatch"));
        });
    rewriter.eraseOp(almostEqOp);
    return success();
  }
};

// Converts check.expect_true %true to
// cf.assert %true, "Result mismatch"
struct ConvertExpectTrueOp : public OpRewritePattern<ExpectTrueOp> {
  using OpRewritePattern<ExpectTrueOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpectTrueOp expectTrueOp,
                                PatternRewriter &rewriter) const override {
    Location loc = expectTrueOp.getLoc();
    IntegerType i1 = IntegerType::get(rewriter.getContext(), 1);
    Value one = rewriter.create<arith::ConstantOp>(
        loc, i1, rewriter.getIntegerAttr(i1, 1));
    Value compare = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, expectTrueOp.getOperand(), one);
    rewriter.create<cf::AssertOp>(loc, compare,
                                  rewriter.getStringAttr("Result mismatch"));
    rewriter.eraseOp(expectTrueOp);
    return success();
  }
};

// Convert
// check.expect_sane(%t1:tensor<2x2xf32>)
// to the pattern:
// scf.some_loop(%i, %j)
// %0 = load from %t1
// %1 = arith.absf %0:f32
// %2 = arith.cmpf ord, %1, %nan : f32
// %3 = arith.cmpf one, %1, %inf : f32
// %compare = arith.andi %2, %3 : i1
// cf.assert %compare, "Buffer can't contain NaNs or Infinite values"
struct ConvertExpectSaneOp : public OpRewritePattern<ExpectSaneOp> {
  using OpRewritePattern<ExpectSaneOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpectSaneOp expectSaneOp,
                                PatternRewriter &rewriter) const override {
    Location loc = expectSaneOp.getLoc();
    auto operand = expectSaneOp.getOperand();
    auto operandType = operand.getType();
    auto elementType = operandType.getElementType();
    SmallVector<Value> ubs;
    if (!isa<MemRefType>(operandType) || !isa<mlir::FloatType>(elementType)) {
      return failure();
    }
    size_t rank = cast<MemRefType>(operandType).getRank();
    for (size_t idx = 0; idx < rank; idx++) {
      auto dim = linalg::createOrFoldDimOp(rewriter, loc, operand, idx);
      ubs.push_back(dim);
    }
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> lbs(rank, zero);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    SmallVector<Value> steps(rank, one);
    (void)scf::buildLoopNest(
        rewriter, loc, lbs, ubs, steps,
        [&](OpBuilder &b, Location loc, ValueRange localIvs) {
          Value scalarOperand =
              b.create<memref::LoadOp>(loc, operand, localIvs);
          Value scalarOperandAbsf = b.create<math::AbsFOp>(loc, scalarOperand);
          Value zeroVal = b.create<arith::ConstantOp>(
              loc, elementType, rewriter.getZeroAttr(elementType));

          Value inf = rewriter.create<arith::ConstantOp>(
              loc, elementType,
              b.getFloatAttr(
                  elementType,
                  APFloat::getInf(
                      cast<FloatType>(elementType).getFloatSemantics())));
          Value notNan = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ORD,
                                                 scalarOperandAbsf, zeroVal);
          Value notInf = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::ONE,
                                                 scalarOperandAbsf, inf);

          Value compare = b.create<arith::AndIOp>(loc, notNan, notInf);
          b.create<cf::AssertOp>(
              loc, compare,
              b.getStringAttr("Buffer can't contain NaNs or Infinite values"));
        });

    rewriter.eraseOp(expectSaneOp);
    return success();
  }
};

void populateCheckToLoopsPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertAlmostEqualsOp, ConvertExpectTrueOp, ConvertExpectSaneOp>(
      patterns.getContext());
}

struct ConvertCheckToLoops
    : public tpp::impl::ConvertCheckToLoopsBase<ConvertCheckToLoops> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCheckToLoopsPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
