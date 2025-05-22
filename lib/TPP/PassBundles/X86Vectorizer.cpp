//===- X86Vectorizer.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/PassUtils.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_X86VECTORIZER
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

// Vectorize ops for x86 targets.
struct X86Vectorizer : public tpp::impl::X86VectorizerBase<X86Vectorizer>,
                        PassBundle<ModuleOp> {
  using X86VectorizerBase::X86VectorizerBase;

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
    // Reshape ops into hardware-friendly sizes.
    pm.addNestedPass<func::FuncOp>(createRegisterBlocking());
    pm.addPass(createCleanup());

    // Vectorize ops.
    pm.addNestedPass<func::FuncOp>(createLinalgVectorize());
    pm.addPass(createCleanup());

    // Cleanup after vectorization.
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<func::FuncOp>(createLoopInvariantSubsetHoistingPass());
    pm.addPass(createCleanup());
  }
};
