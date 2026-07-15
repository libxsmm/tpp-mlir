//===- LinalgLowering.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassUtils.h"

#include <string>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGLOWERING
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

// Lower Linalg to into combination of standard and local dialects.
struct LinalgLowering : public tpp::impl::LinalgLoweringBase<LinalgLowering>,
                        PassBundle<func::FuncOp> {
  using LinalgLoweringBase::LinalgLoweringBase;

  void runOnOperation() override {
    auto module = getOperation();

    // Initialize the pipeline if needed.
    // Otherwise, just run the cached one.
    if (pm.empty())
      constructPipeline();

    if (failed(runPipeline(pm, module)))
      return signalPassFailure();
  }

private:
  void constructPipeline() override {
    pm.addPass(createConvertLinalgToXsmm());
    pm.addPass(createCombineXsmmOpPass());
    pm.addPass(createFoldXsmmFlags());
    pm.addPass(createVerifyXsmmCalls());
  }
};
