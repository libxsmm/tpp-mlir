//===- GpuConversion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/PassUtils.h"

#include <optional>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUCONVERSION
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Map and lower operations to generic GPU ops.
struct GpuConversion : public tpp::impl::GpuConversionBase<GpuConversion>,
                       PassBundle<ModuleOp> {
  using GpuConversionBase::GpuConversionBase;

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
    // Map loops into GPU kernels.
    pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createConvertParallelLoopToGpuPass());
    pm.addPass(createCleanup());

    // First lower linalg using custom patterns then fall back to
    // the default lowering for any remaining ops.
    pm.addNestedPass<func::FuncOp>(createLinalgDeGeneralize());
    if (isIntel) {
      pm.addNestedPass<func::FuncOp>(createLinalgToXeGPU(LinalgToXeGPUOptions{
          kTile, stages, SmallVector<int64_t>{*dpasTile}}));
    }
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    pm.addPass(createCleanup());

    // Create GPU kernels.
    pm.addNestedPass<func::FuncOp>(createGpuInlineConstants());
    pm.addPass(createGpuKernelOutliningPass());

    // Generic cleanup.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

} // namespace
