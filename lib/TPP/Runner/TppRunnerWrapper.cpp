//===- TppRunnerWrapper.cpp ------------------------------------*----C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Runner/MLIRBench.h"
#include "TPP/Transforms/Utils/TensorInit.h"
#include "TPP/Transforms/Utils/TensorInitFloat.h"
#include "TPP/Transforms/Utils/TensorInitInt.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_TPPRUNNERWRAPPER
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Create runner wrapper around the main kernel function.
struct TppRunnerWrapper
    : public tpp::impl::TppRunnerWrapperBase<TppRunnerWrapper> {
  using TppRunnerWrapperBase::TppRunnerWrapperBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    auto tensorInitType = parseTensorInitType(initType);

    // Randon options need seed.
    if (!seed && (randomSplat || tensorInitType == TensorInitType::Random ||
                  tensorInitType == TensorInitType::Normal)) {
      seed = std::time(0);
    }

    // Benchmark object.
    MLIRBenchConfig config(seed, tensorInitType, identity, backend, offloadToDevice);
    MLIRBench bench(module, config);

    // Can only either print or run benchmarks, make this clear before we try to
    // run.
    if (printResult && numBenchLoops > 1) {
      (void)bench.emitError(
          "Cannot print while running benchmarks, pick one or the other");
      return;
    }

    // Basic checks.
    if (kernelType != "void") {
      (void)bench.emitError(
          "Main function has to be 'void', even if the kernel return's a "
          "value, "
          "because that's the type of the wrapper we create here");
      return;
    }

    if (failed(bench.findKernel(kernelName))) {
      (void)bench.emitError("Cannot find kernel '" + kernelName + "'");
      return;
    }

    if (failed(bench.checkKernelSignature())) {
      (void)bench.terminate();
      return;
    }

    if (randomSplat && failed(bench.replaceSplatWithRandom())) {
      (void)bench.emitError(
          "Error converting splat tensors with random values");
      return;
    }

    // Move the kernel to a local name, so we can create `main` with the same
    // name as the pre-defined entry point (since we can't change it).
    if (failed(bench.renameKernel())) {
      (void)bench.emitError("Cannot rename kernel function");
      return;
    }

    // Creates the main wrapper.
    if (failed(bench.createMainWrapper())) {
      (void)bench.emitError("Cannot create main wrapper");
      return;
    }

    // Creates the inputs for the kernel.
    if (failed(bench.createKernelArgs())) {
      (void)bench.emitError("Cannot create kernel inputs");
      return;
    }

    // Print the kernel's input arguments by iterating through kernelArgs
    if (printInput) {
      for (auto arg : bench.getKernelArgs()) {
        if (auto shapedType = dyn_cast<ShapedType>(arg.value.getType())) {
          if (shapedType.hasStaticShape())
            if (failed(bench.printShapedType(arg.value)))
              return;
        }
      }
    }

    // Either run once or run benchmarks
    if (numBenchLoops > 1) {
      if (benchWarmup) {
        // Warmup to 1% of the total runs, but no less than 1 and no more than
        // 50.
        int warmupIter = numBenchLoops / 100;
        warmupIter = std::max(warmupIter, 1);
        warmupIter = std::min(warmupIter, 50);

        // This is the warmup loop, if N > 1, ignore the result.
        (void)bench.createTimerLoop(warmupIter);
      }

      // This is the benchmark loop.
      auto delta = bench.createTimerLoop(numBenchLoops);
      auto stats = bench.getTimerStats(delta);
      (void)bench.printMean(stats);
    } else {
      // Call kernel only once.
      auto *call = bench.callKernel();
      if (!call) {
        (void)bench.emitError("Cannot generate a call to the kernel");
        return;
      }

      if (printResult) {
        if (!call->getResults().size()) {
          (void)bench.emitError("Cannot print functions with void return");
          return;
        }

        if (failed(bench.printResult(call))) {
          (void)bench.emitError("Cannot print result memref");
          return;
        }
      }
    }

    // Terminate the created wrapper.
    (void)bench.terminate();
  }
};

} // namespace
