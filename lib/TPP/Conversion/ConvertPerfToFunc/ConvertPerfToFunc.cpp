//===- ConvertPerfToFunc.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <utility>

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::perf;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTPERFTOFUNC
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Perf dialect type normalization helper function.
// Cast memref and tensor to their unranked versions and convert
// custom perf types into default primitive types.
// Leave all the other operands as they are.
static SmallVector<Type> extractNormalizedTypes(OpBuilder &b,
                                                ValueRange values) {
  SmallVector<Type> results;
  results.reserve(values.size());

  for (Value val : values) {
    TypeSwitch<Type>(val.getType())
        .Case<MemRefType>([&](Type t) {
          auto memrefType = cast<MemRefType>(t);
          auto unrankedMemref = UnrankedMemRefType::get(
              memrefType.getElementType(), memrefType.getMemorySpace());
          results.push_back(unrankedMemref);
        })
        .Case<TensorType>([&](Type t) {
          auto tensorType = cast<TensorType>(t);
          auto unrankedTensor =
              UnrankedTensorType::get(tensorType.getElementType());
          results.push_back(unrankedTensor);
        })
        .Case<TimerType>([&](Type t) {
          auto i64 = IntegerType::get(b.getContext(), 64);
          results.push_back(i64);
        })
        .Default([&](Type t) { results.push_back(t); });
  }

  return results;
}

// Similar to 'extractNormalizedTypes' but inserts conversions from original
// types to normalized ones when possible.
// The conversions aim to simplify runtime function signature generation by, for
// example, erasing explicit memref and tensor shapes.
static SmallVector<Value> getNormalizedOperands(OpBuilder &b, Location loc,
                                                ValueRange operands) {
  SmallVector<Value> res;
  res.reserve(operands.size());

  for (Value op : operands) {
    auto type = extractNormalizedTypes(b, op);
    TypeSwitch<Type>(op.getType())
        .Case<MemRefType>([&](Type t) {
          Value cast = memref::CastOp::create(b, loc, type, op);
          res.push_back(cast);
        })
        .Case<TensorType>([&](Type t) {
          Value cast = tensor::CastOp::create(b, loc, type, op);
          res.push_back(cast);
        })
        .Default([&](Type t) { res.push_back(op); });
  }

  return res;
}

// Create a perf function prototype.
static func::FuncOp createPerfFuncPrototype(Location loc, const std::string& funcName,
                                            Operation *op,
                                            PatternRewriter &rewriter) {
  // Insert before module terminator.
  ModuleOp module = op->getParentOfType<ModuleOp>();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));

  FlatSymbolRefAttr fnName = SymbolRefAttr::get(op->getContext(), funcName);
  auto libFnType = rewriter.getFunctionType(
      extractNormalizedTypes(rewriter, op->getOperands()),
      extractNormalizedTypes(rewriter, op->getResults()));

  auto funcOp =
      func::FuncOp::create(rewriter, loc, fnName.getValue(), libFnType);
  funcOp.setPrivate();

  return funcOp;
}

// Generate function implementation for perf.sink operation.
static LogicalResult buildPerfSinkFunc(Location loc,
                                       const std::string &funcName,
                                       Operation *op,
                                       PatternRewriter &rewriter) {
  auto funcOp = createPerfFuncPrototype(loc, std::move(funcName), op, rewriter);

  // Add function attributes which ensure that the passed data and its producers
  // operations cannot be optimized away such that the time measured by a
  // benchmark loop correctly represents the full workload.
  auto *ctx = rewriter.getContext();
  funcOp->setAttr("passthrough",
                  rewriter.getArrayAttr({StringAttr::get(ctx, "optnone"),
                                         StringAttr::get(ctx, "noinline")}));

  // Create function body.
  Block *block = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(block);

  // Insert empty return.
  func::ReturnOp::create(rewriter, loc, ValueRange{});

  return success();
}

// Create a perf runtime function prototype.
// The function implementation has to be provided externally by the end user.
static LogicalResult buildPerfRuntimeFunc(Location loc,
                                          const std::string &funcName,
                                          Operation *op,
                                          PatternRewriter &rewriter) {
  (void)createPerfFuncPrototype(loc, std::move(funcName), op, rewriter);
  return success();
}

// Insert calls to functions implementing corresponding perf op functionality.
// If a function is unavailable in the current module, the function's builder
// is called.
static LogicalResult buildPerfFuncCall(Location loc, std::string funcName,
                                       Operation *op,
                                       PatternRewriter &rewriter) {
  if (op->getNumResults() > 1)
    return op->emitError(
               "expected operation to have 0 or 1 result, but provided ")
           << op->getNumResults();

  FlatSymbolRefAttr fnName = SymbolRefAttr::get(op->getContext(), funcName);
  ModuleOp module = op->getParentOfType<ModuleOp>();

  // If a function is not available yet, call its builder.
  if (!module.lookupSymbol(fnName.getAttr())) {
    auto res = TypeSwitch<Operation *, LogicalResult>(op)
                   .Case<perf::SinkOp>([&](Operation *op) {
                     return buildPerfSinkFunc(loc, funcName, op, rewriter);
                   })
                   .Default([&](Operation *op) {
                     return buildPerfRuntimeFunc(loc, funcName, op, rewriter);
                   });
    if (failed(res))
      return res;
  }

  // Insert a function call.
  auto funcCall = func::CallOp::create(rewriter, 
      loc, fnName.getValue(),
      extractNormalizedTypes(rewriter, op->getResults()),
      getNormalizedOperands(rewriter, loc, op->getOperands()));
  op->replaceAllUsesWith(funcCall.getResults());

  return success();
}

struct ConvertStartTimerOp : public OpRewritePattern<perf::StartTimerOp> {
  using OpRewritePattern<perf::StartTimerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StartTimerOp startTimerOp,
                                PatternRewriter &rewriter) const override {
    auto res = buildPerfFuncCall(startTimerOp.getLoc(),
                                 startTimerOp.getLibraryCallName(),
                                 startTimerOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(startTimerOp);
    return res;
  }
};

struct ConvertStopTimerOp : public OpRewritePattern<perf::StopTimerOp> {
  using OpRewritePattern<perf::StopTimerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::StopTimerOp stopTimerOp,
                                PatternRewriter &rewriter) const override {
    auto res = buildPerfFuncCall(stopTimerOp.getLoc(),
                                 stopTimerOp.getLibraryCallName(), stopTimerOp,
                                 rewriter);
    if (succeeded(res))
      rewriter.eraseOp(stopTimerOp);
    return res;
  }
};

struct ConvertSinkOp : public OpRewritePattern<perf::SinkOp> {
  using OpRewritePattern<perf::SinkOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(perf::SinkOp sinkOp,
                                PatternRewriter &rewriter) const override {

    // perf.sink relies on the perf runtime to prevent complier from
    // optimizing away marked data. Name mangling is required as the op
    // accepts any kind of data. For simplicity, the mangling makes some
    // assumptions on data types supported by the perf dialect.
    auto res = buildPerfFuncCall(sinkOp.getLoc(), sinkOp.getLibraryCallName(),
                                 sinkOp, rewriter);
    if (succeeded(res))
      rewriter.eraseOp(sinkOp);
    return res;
  }
};

void populatePerfToFuncPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertStartTimerOp, ConvertStopTimerOp, ConvertSinkOp>(
      patterns.getContext());
}

struct ConvertPerfToFunc
    : public tpp::impl::ConvertPerfToFuncBase<ConvertPerfToFunc> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePerfToFuncPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
