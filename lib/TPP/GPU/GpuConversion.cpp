//===- GpuConversion.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/PassUtils.h"

#include <optional>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUCONVERSION
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Map and lower operations to generic GPU ops.
struct GpuConversion : public tpp::impl::GpuConversionBase<GpuConversion>,
                       UtilityPassBase<ModuleOp> {
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
    pm.clear();

    // First lower linalg using custom patterns then fall back to
    // the default lowering for any remaining ops.
    pm.addNestedPass<func::FuncOp>(createLinalgDeGeneralize());
    if (isIntel) {
      pm.addNestedPass<func::FuncOp>(
          createLinalgToXeGPU(LinalgToXeGPUOptions{kTile, stages}));
    } else {
      pm.addNestedPass<func::FuncOp>(
          createLinalgToGpu(LinalgToGpuOptions{useWmma, warpTile, kTile}));
    }
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToParallelLoopsPass());

    // Map loops into GPU kernels.
    pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());
    pm.addNestedPass<func::FuncOp>(createParallelLoopToGpuPass());

    pm.addNestedPass<func::FuncOp>(createCleanup());

    // Create GPU kernels.
    pm.addNestedPass<func::FuncOp>(createGpuInlineConstants());
    pm.addPass(createGpuKernelOutliningPass());

    // Generic cleanup.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
};

} // namespace
