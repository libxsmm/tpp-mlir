//===- Bufferize.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"

#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_BUFFERIZE
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_DUPLICATEFILL
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct Bufferize : public tpp::impl::BufferizeBase<Bufferize> {
  using BufferizeBase::BufferizeBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry
        .insert<affine::AffineDialect,
                arith::ArithDialect,
                bufferization::BufferizationDialect,
                func::FuncDialect,
                linalg::LinalgDialect,
                memref::MemRefDialect,
                check::CheckDialect,
                perf::PerfDialect,
                scf::SCFDialect,
                tensor::TensorDialect>();
    // clang-format on
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);
  }
  void runOnOperation() override;
};

struct DuplicateFill : public tpp::impl::DuplicateFillBase<DuplicateFill> {
  void runOnOperation() override;
};

void DuplicateFill::runOnOperation() {
  IRRewriter rewriter(&getContext());

  (void)getOperation()->walk([&](linalg::FillOp fillOp) {
    if (!fillOp.hasPureTensorSemantics())
      return WalkResult::advance();
    Value fillVal = fillOp.getResult(0);
    // We can fold only zero initialization. We duplicate only
    // if the fill has multiple uses.
    if (!utils::isZeroTensor(fillVal) || fillOp->hasOneUse())
      return WalkResult::advance();
    SetVector<Operation *> forwardSlice;
    getForwardSlice(fillVal, &forwardSlice);
    for (size_t idx = /*Skip first user. Use the current fill*/ 1;
         idx < forwardSlice.size(); idx++)
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(forwardSlice[idx])) {
        if (failed(linalgx::utils::isContraction(linalgOp)))
          continue;
        assert(linalgOp.getNumDpsInits() == 1);
        Value outLinalg = linalgOp.getDpsInits()[0];
        if (outLinalg == fillVal) {
          rewriter.setInsertionPoint(linalgOp);
          Operation *clonedOp = rewriter.clone(*fillOp.getOperation());
          rewriter.replaceUsesWithIf(fillOp->getResults(),
                                       clonedOp->getResults(),
                                       [&](OpOperand &operand) {
                                         return operand.getOwner() == linalgOp;
                                       });
        }
      }
    return WalkResult::advance();
  });
}

static LogicalResult defaultMemCpyFn(OpBuilder &builder, Location loc,
                                     Value from, Value to) {
  builder.create<linalg::CopyOp>(loc, from, to);
  return success();
}

void Bufferize::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  OpPassManager passManager;

  // Pre-processing.
  if (this->duplicateFill)
    passManager.addNestedPass<func::FuncOp>(tpp::createDuplicateFill());
  passManager.addPass(bufferization::createEmptyTensorEliminationPass());
  passManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());

  // One-shot.
  bufferization::OneShotBufferizationOptions buffOpts;
  buffOpts.bufferizeFunctionBoundaries = true;
  buffOpts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  buffOpts.memCpyFn = defaultMemCpyFn;
  bool runOnlyAnalysis = this->testAnalysisOnly || this->printConflicts;
  if (runOnlyAnalysis) {
    buffOpts.printConflicts = this->printConflicts;
    buffOpts.testAnalysisOnly = this->testAnalysisOnly;
  }
  passManager.addPass(bufferization::createOneShotBufferizePass(buffOpts));

  if (!runOnlyAnalysis) {
    passManager.addPass(bufferization::createDropEquivalentBufferResultsPass());

    // Post-processing.
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    passManager.addNestedPass<func::FuncOp>(createCSEPass());
    // There are redundant memcpy (with linalg.generic form) ops created, which
    // can be deleted by canonicalizer. We have to run it again because the
    // memrefs are unified in CSE pass, so we can truly remove redundant memcpy.
    passManager.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  }
  passManager.addPass(bufferization::createDropEquivalentBufferResultsPass());

  if (dealloc) {
    bufferization::BufferDeallocationPipelineOptions options;
    bufferization::buildBufferDeallocationPipeline(passManager, options);
  }

  passManager.addPass(createConvertBufferizationToMemRefPass());
  if (failed(runPipeline(passManager, moduleOp)))
    return signalPassFailure();
}

} // namespace
