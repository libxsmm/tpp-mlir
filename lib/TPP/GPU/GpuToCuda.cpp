//===- GpuToCuda.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUTOCUDA
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Lower generic GPU ops to CUDA backend.
struct GpuToCuda : public tpp::impl::GpuToCudaBase<GpuToCuda>,
                   PassBundle<ModuleOp> {
  using GpuToCudaBase::GpuToCudaBase;

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
#ifdef TPP_CUDA_ENABLE
    // Preprocess and lower standard ops.
    pm.addNestedPass<gpu::GPUModuleOp>(
        memref::createExpandStridedMetadataPass());
    pm.addNestedPass<gpu::GPUModuleOp>(arith::createArithExpandOpsPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createLowerAffinePass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToSCFPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createSCFToControlFlowPass());

    pm.addNestedPass<gpu::GPUModuleOp>(createConvertNVGPUToNVVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertGpuOpsToNVVMOps());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertVectorToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertNVVMToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertFuncToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createArithToLLVMConversionPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createConvertIndexToLLVMPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createUBToLLVMConversionPass());

    GpuNVVMAttachTargetOptions nvvmTargetOptions;
    nvvmTargetOptions.triple = gpuTriple;
    nvvmTargetOptions.chip = gpuChip;
    nvvmTargetOptions.features = gpuFeatures;
    pm.addPass(createGpuNVVMAttachTarget(nvvmTargetOptions));

    // Create CUDA kernels.
    pm.addNestedPass<gpu::GPUModuleOp>(createCanonicalizerPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createCSEPass());
    pm.addNestedPass<gpu::GPUModuleOp>(createReconcileUnrealizedCastsPass());

    // Cleanup IR.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
#endif // TPP_CUDA_ENABLE
  }
};

} // namespace
