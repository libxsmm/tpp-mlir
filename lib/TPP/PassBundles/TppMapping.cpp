//===- TppMapping.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/PassUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_TPPMAPPING
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

// Apply collection of high-level passes that map operations to
// TPP-compatible forms.
struct TppMapping : public tpp::impl::TppMappingBase<TppMapping>,
                    PassBundle<ModuleOp> {
  using TppMappingBase::TppMappingBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module))) {
      llvm::dbgs() << "Failed tpp mapping\n";
      return signalPassFailure();
    }
  }

private:
  void constructPipeline() override {
    // Canonicalize.
    pm.addPass(createCleanup());

    // Convert ops to packed layouts.
    pm.addPass(createPackMatmul());

    if (!disableVnniPacking) {
      pm.addPass(createPackVNNI());
    }

    if (lowerPackUnpackWithoutTranspose) {
      pm.addPass(createLowerPacksAndUnpacksWithoutTranspose());
    }
    // Postprocess packing.
    // Run only canonicalizer at this stage as full cleanup (mostly CSE) can
    // mess up tensor producer-consumer chains used for analysis in the
    // following passes.
    pm.addPass(createPropagatePackUnPack());
    pm.addPass(createConstantFoldPack());
    pm.addPass(createSimplifyAndCanonicalizePack());

    pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
    pm.addPass(createCleanup());
    pm.addNestedPass<func::FuncOp>(
        createLinalgConvertCompareSelectToMaximumfPass());

    pm.addPass(createTileConsumerAndFuseProducers());
    pm.addPass(createSimplifyAndCanonicalizePack());
    pm.addPass(createCleanup());
  }
};
