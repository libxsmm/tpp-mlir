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
    pm.addNestedPass<func::FuncOp>(createRegisterBlocking());
    pm.addPass(createCleanup());

    // TODO: Test alternative unrolling by additional tiling at linalg level.
    //
    // Unrolling could be achieved by tiling without fusion again and then
    // immediately unrolling created loops.
    // This alternative unrolling strategy offers potential benefits:
    //   - unrolling interleaves operations - potential lower register pressure
    //   - easier layout propagation - IR graph of linalg ops without explicit
    //     reads and writes means fewer ops to consider
    // This approach requires full layout propagation, otherwise, vectorized
    // write-read pairs will not cancel out.

    // Vectorize ops.
    pm.addNestedPass<func::FuncOp>(createLinalgVectorize());
    pm.addPass(createCleanup());

    // TODO: Unroll before hoisting.
    //
    // This can be beneficial only if full layout propagation is done and
    // consumers are also unrolled. It will allow hoisting and canonicalization
    // to cancel out unrolled write-read op pairs.

    // Hoist after vectorization.
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
    pm.addNestedPass<func::FuncOp>(createRegisterUnroll());
    pm.addPass(createCleanup());

    // Lower vector ops to x86 sequences.
    pm.addNestedPass<func::FuncOp>(createConvertVectorToX86());
    pm.addPass(createCleanup());

    // Cleanup vector shapes.
    // Helps to expose more canonical vector forms, cancel out casts, and later
    // lower reads and writes directly to LLVM ops instead of SCF versions.
    pm.addPass(createVectorDropUnitDims());
    pm.addPass(createCleanup());
  }
};
