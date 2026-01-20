//===- IntelAMXTileConfig.cpp ------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file inserts tile configuration calls.
//
//===----------------------------------------------------------------------===//
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Dialect/Xsmm/XsmmUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_INTELAMXTILECONFIGINSERTIONPASS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;

namespace mlir {
namespace tpp {

template <typename InvokeOpTy, typename DispatchOpTy>
struct IntelAMXTileConfig : OpRewritePattern<InvokeOpTy> {
  using OpRewritePattern<InvokeOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(InvokeOpTy op,
                                PatternRewriter &rewriter) const override {
    if (xsmm::utils::getDataType(rewriter, op.getOperand(1).getType()) !=
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16))
      return failure();
    auto flags =
        dyn_cast<DispatchOpTy>(op.getOperand(0).getDefiningOp()).getFlags();
    for (auto flagItr : flags)
      if (flagItr == xsmm::GemmFlagsAttr::get(
                         rewriter.getContext(),
                         mlir::xsmm::GemmFlags::NO_RESET_TILECONFIG) ||
          flagItr == xsmm::GemmFlagsAttr::get(
                         rewriter.getContext(),
                         mlir::xsmm::GemmFlags::NO_SETUP_TILECONFIG))
        return failure();

    auto brgemmFlags = mlir::xsmm::utils::getBrgemmFlags<DispatchOpTy>(
        rewriter, dyn_cast<DispatchOpTy>(op.getOperand(0).getDefiningOp()),
        false);
    if (failed(brgemmFlags)) {
      return failure();
    }

    auto attributesSetup = *brgemmFlags;
    attributesSetup.push_back(xsmm::GemmFlagsAttr::get(
        rewriter.getContext(), xsmm::GemmFlags::NO_RESET_TILECONFIG));
    auto tileConfigSetup = xsmm::IntelAMXTileConfigDispatchOp::create(rewriter, 
        op.getLoc(), rewriter.getI64Type(),
        DenseI64ArrayAttr::get(
            rewriter.getContext(),
            dyn_cast<DispatchOpTy>(op.getOperand(0).getDefiningOp())
                .getInputs()),
        rewriter.getArrayAttr(attributesSetup),
        xsmm::utils::getDataType(rewriter, op.getOperand(1).getType()));

    SmallVector<Attribute> attributesReset = *brgemmFlags;
    attributesReset.push_back(xsmm::GemmFlagsAttr::get(
        rewriter.getContext(), xsmm::GemmFlags::NO_SETUP_TILECONFIG));
    auto tileConfigReset = xsmm::IntelAMXTileConfigDispatchOp::create(rewriter, 
        op.getLoc(), rewriter.getI64Type(),
        DenseI64ArrayAttr::get(
            rewriter.getContext(),
            dyn_cast<DispatchOpTy>(op.getOperand(0).getDefiningOp())
                .getInputs()),
        rewriter.getArrayAttr(attributesReset),
        xsmm::utils::getDataType(rewriter, op.getOperand(1).getType()));

    SmallVector<Attribute> attributesBrgemm = *brgemmFlags;
    attributesBrgemm.push_back(xsmm::GemmFlagsAttr::get(
        rewriter.getContext(), xsmm::GemmFlags::NO_RESET_TILECONFIG));
    attributesBrgemm.push_back(xsmm::GemmFlagsAttr::get(
        rewriter.getContext(), xsmm::GemmFlags::NO_SETUP_TILECONFIG));

    auto dispatch = dyn_cast<DispatchOpTy>(
        rewriter.clone(*op.getOperand(0).getDefiningOp()));
    dispatch.setFlagsAttr(rewriter.getArrayAttr(attributesBrgemm));

    auto alloca = memref::AllocaOp::create(rewriter, 
        op.getLoc(), MemRefType::get({64}, rewriter.getI8Type()));

    SmallVector<Value> tileConfigInputs{alloca};
    mlir::xsmm::IntelAMXTileConfigOp::create(rewriter, 
        op.getLoc(), tileConfigSetup, tileConfigInputs);

    SmallVector<Value> invokeOperands;
    invokeOperands.push_back(dispatch);
    auto opItr = op->getOperands().begin();
    std::advance(opItr, 1);
    invokeOperands.append(opItr, op->getOperands().end());
    InvokeOpTy::create(rewriter, 
        op.getLoc(),
        xsmm::utils::getDataType(rewriter, op.getOperand(1).getType()),
        invokeOperands);

    SmallVector<Value> tileResetInputs{alloca};
    mlir::xsmm::IntelAMXTileConfigOp::create(rewriter, 
        op.getLoc(), tileConfigReset, tileResetInputs);

    rewriter.eraseOp(op);
    if (op.getOperand(0).getDefiningOp()->getUsers().empty())
      rewriter.eraseOp(op.getOperand(0).getDefiningOp());
    return success();
  }
};

struct IntelAMXTileConfigInsertionPass
    : public impl::IntelAMXTileConfigInsertionPassBase<
          IntelAMXTileConfigInsertionPass> {
  void populateCombinePatterns(RewritePatternSet &patterns) {
    patterns.add<IntelAMXTileConfig<xsmm::BrgemmOp, xsmm::BrgemmDispatchOp>>(
        patterns.getContext());
    patterns.add<
        IntelAMXTileConfig<xsmm::FusedBrgemmOp, xsmm::FusedBrgemmDispatchOp>>(
        patterns.getContext());
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCombinePatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace tpp
} // namespace mlir