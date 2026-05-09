//===- ConvertLinalgToFunc.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTLINALGTOFUNC
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// TODO: check if we can avoid passing the offset and do the computation in the
// compiler.
static SmallVector<Type> extractGemmOperandTypes(OpBuilder &builder,
                                                 OperandRange operands) {
  auto indexType = builder.getIndexType();
  auto *ctx = builder.getContext();
  SmallVector<Type> results;

  results.push_back(indexType); // M
  results.push_back(indexType); // N
  results.push_back(indexType); // K
  results.push_back(LLVM::LLVMPointerType::get(ctx));              // A
  results.push_back(builder.getIndexType());                       // offset A
  results.push_back(indexType);                                    // lda
  results.push_back(LLVM::LLVMPointerType::get(ctx));              // B
  results.push_back(builder.getIndexType());                       // offset B
  results.push_back(indexType);                                    // ldb
  results.push_back(LLVM::LLVMPointerType::get(ctx));              // C
  results.push_back(builder.getIndexType());                       // offset C
  results.push_back(indexType);                                    // ldc
  return results;
}

static SmallVector<Value> getGemmOperands(OpBuilder &builder, Location loc,
                                          ValueRange operands) {
  SmallVector<Value> results;

  Value m = linalg::createOrFoldDimOp(builder, loc, operands[2], /*dim=*/0);
  Value n = linalg::createOrFoldDimOp(builder, loc, operands[2], /*dim=*/1);
  Value k = linalg::createOrFoldDimOp(builder, loc, operands[0], /*dim=*/1);
  results.push_back(m);
  results.push_back(n);
  results.push_back(k);
  auto [ptrA, offsetA] = utils::getPtrAndOffset(builder, operands[0], loc);
  results.push_back(ptrA);
  results.push_back(offsetA);
  results.push_back(m); // lda
  auto [ptrB, offsetB] = utils::getPtrAndOffset(builder, operands[1], loc);
  results.push_back(ptrB);
  results.push_back(offsetB);
  results.push_back(k); // ldb
  auto [ptrC, offsetC] = utils::getPtrAndOffset(builder, operands[2], loc);
  results.push_back(ptrC);
  results.push_back(offsetC);
  results.push_back(m); // ldc
  return results;
}

static void buildInvokeCall(OpBuilder &builder, Operation *op) {
  std::string funcName(op->getName().getStringRef().str());
  std::replace(funcName.begin(), funcName.end(), '.', '_');
  funcName.append("_blas");
  Location loc = op->getLoc();

  FlatSymbolRefAttr fnName = SymbolRefAttr::get(op->getContext(), funcName);
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto libFnType = builder.getFunctionType(
      extractGemmOperandTypes(builder, op->getOperands()), {});

  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(builder);
    // Insert before module terminator.
    builder.setInsertionPoint(module.getBody(),
                              std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        func::FuncOp::create(builder, loc, fnName.getValue(), libFnType);
    funcOp.setPrivate();
  }

  func::CallOp::create(builder, 
      loc, fnName.getValue(), TypeRange(),
      getGemmOperands(builder, loc, op->getOperands()));
}

struct ConvertMatmulOp : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (!matmulOp.hasPureBufferSemantics())
      return failure();
    SmallVector<Value> operands = matmulOp.getDpsInputs();
    operands.push_back(matmulOp.getDpsInits()[0]);
    if (!llvm::all_of(operands, [](Value operand) {
          MemRefType memref = cast<MemRefType>(operand.getType());
          return memref.getLayout().isIdentity() &&
                 memref.getElementType().isF32();
        })) {
      return failure();
    }
    buildInvokeCall(rewriter, matmulOp);
    rewriter.eraseOp(matmulOp);
    return success();
  }
};

struct ConvertLinalgToFunc
    : public tpp::impl::ConvertLinalgToFuncBase<ConvertLinalgToFunc> {
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMatmulOp>(ctx);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace
