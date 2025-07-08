//===- X86Vectorizer.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    tpp::RegisterBlockingOptions blockingOpts;
    blockingOpts.blocks = SmallVector<int64_t>{*blocks};
    pm.addNestedPass<func::FuncOp>(createRegisterBlocking(blockingOpts));

    // Vectorize ops.
    pm.addNestedPass<func::FuncOp>(createLinalgVectorize());
    pm.addPass(createCleanup());

    // Hoist after vectorization.
    //
    // Hoisting allows for more opportunities to fold write-read pairs which
    // results in fewer transfers after unrolling.
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    pm.addNestedPass<func::FuncOp>(createLoopInvariantSubsetHoistingPass());
    pm.addPass(createCleanup());

    // Split vectors into register shapes.
    //
    // Current unrolling only targets contractions and relies on LLVM backend
    // to cleanup and unroll elementwise consumers.
    // TODO: Check if LLVM manages that correctly for all targets and
    //       extensions.
    tpp::RegisterUnrollOptions unrollOpts;
    unrollOpts.gemmUnroll = SmallVector<int64_t>{*gemmUnroll};
    pm.addNestedPass<func::FuncOp>(createRegisterUnroll(unrollOpts));
    pm.addPass(createCleanup());

    // Lower vector ops to x86 sequences.
    pm.addNestedPass<func::FuncOp>(createConvertVectorToX86());
    pm.addPass(createCleanup());

    // Cleanup vector shapes.
    //
    // Helps to expose more canonical vector forms, cancel out casts, and later
    // lower reads and writes directly to LLVM ops instead of SCF versions.
    pm.addPass(createVectorDropUnitDims());
    pm.addPass(createCleanup());
  }
};
