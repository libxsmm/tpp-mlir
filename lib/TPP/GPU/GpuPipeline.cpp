//===- GpuPipeline.cpp -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/XeGPU/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <optional>

using namespace mlir;
using namespace mlir::tpp;

llvm::cl::list<int64_t>
    gpuBlockTile("gpu-block-tile", llvm::cl::desc("GPU block tile size"),
                 llvm::cl::list_init<int64_t>(SmallVector<int64_t>{128, 128}),
                 llvm::cl::CommaSeparated);

llvm::cl::list<int64_t>
    gpuThreadTile("gpu-thread-tile", llvm::cl::desc("GPU thread tile size"),
                  llvm::cl::list_init<int64_t>(SmallVector<int64_t>{32, 32}),
                  llvm::cl::CommaSeparated);

llvm::cl::opt<int64_t> kTile("k-tile", llvm::cl::desc("GEMM K dim tiling size"),
                             llvm::cl::init(32));

llvm::cl::opt<int64_t> stages("stages",
                              llvm::cl::desc("GEMM coop prefetch stages"),
                              llvm::cl::init(1));

// DPAS size defaults to PVC.
llvm::cl::list<int64_t>
    gpuDpasTile("dpas-tile", llvm::cl::desc("DPAS register block sizes MxNxK"),
                llvm::cl::list_init<int64_t>(SmallVector<int64_t>{8, 16, 16}),
                llvm::cl::CommaSeparated);

// Control GPU vectorization.
llvm::cl::opt<bool> gpuVector("gpu-vector",
                              llvm::cl::desc("Vectorize GPU kernel"),
                              llvm::cl::init(false));

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUPIPELINE
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

enum class GpuType {
  Cuda,
  Intel,
};

GpuType parseGpuOption(StringRef gpuStr) {
  auto type = llvm::StringSwitch<std::optional<GpuType>>(gpuStr)
                  .CaseLower("cuda", GpuType::Cuda)
                  .CaseLower("intel", GpuType::Intel)
                  .Default(std::nullopt);
  assert(type && "Unsupported GPU backend");

  return *type;
}

struct GpuOptions {
  std::string triple;
  std::string chip;
  std::string features;
};

GpuOptions getGpuOptions(GpuType gpuType) {
  GpuOptions options;

  switch (gpuType) {
  case GpuType::Cuda: {
    options.triple = "nvptx64-nvidia-cuda";
    options.chip = "sm_70";
    options.features = "+ptx60";
    break;
  }
  case GpuType::Intel: {
    // No options needed at the moment.
    break;
  }
  }

  return options;
}

// GPU pipeline - map and lower operations to enable execution on a GPU.
struct GpuPipeline : public tpp::impl::GpuPipelineBase<GpuPipeline>,
                     PassBundle<ModuleOp> {
  using GpuPipelineBase::GpuPipelineBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<gpu::GPUDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<NVVM::NVVMDialect>();
    registry.insert<nvgpu::NVGPUDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<spirv::SPIRVDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<perf::PerfDialect>();
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects to make the pipeline more robust with respect
    // to accepted input IR by preventing cryptic runtime crashes due to missing
    // dialect registrations.
    registerAllDialects(registry);
  }

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
    GpuType gpuType = parseGpuOption(this->gpuBackend);
    GpuOptions gpuOptions = getGpuOptions(gpuType);

    // Input preprocessing.
    pm.addPass(createCleanup());
    pm.addPass(createFoldIntoEltwise());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToInplace());

    // Tile to split the kernel into threads and blocks.
    // Use default tiling to handle both packed and unpacked ops.
    pm.addPass(createCleanup());
    // First split computation into grid with blocks of specified size.
    TileConsumerAndFuseProducersOptions blockTileOptions;
    if (!llvm::any_of(gpuBlockTile, [](int64_t tile) { return tile == -1; }))
      blockTileOptions.tileSizes =
          SmallVector<int64_t>{gpuBlockTile.begin(), gpuBlockTile.end()};
    blockTileOptions.minTileFactor = 1;
    pm.addPass(createTileConsumerAndFuseProducers(blockTileOptions));

    // Then try to further split computation into subtiles.
    // This allows to split larger computations across multiple
    // threads/workitems. For smaller workloads, it provides another
    // chance for outlining.
    TileConsumerAndFuseProducersOptions threadTileOptions;
    if (!llvm::any_of(gpuThreadTile, [](int64_t tile) { return tile == -1; }))
      threadTileOptions.tileSizes =
          SmallVector<int64_t>{gpuThreadTile.begin(), gpuThreadTile.end()};
    threadTileOptions.minTileFactor = 1;
    pm.addPass(createTileConsumerAndFuseProducers(threadTileOptions));
    pm.addPass(createCleanup());

    if (gpuVector) {
      // Early reduction dimension splitting is incompatible with
      // Linalg to XeGPU lowering that expects full GEMM.
      // For now, enable only with other vectorization passes.
      pm.addPass(createSplitReductionDim(SplitReductionDimOptions{kTile}));
      pm.addPass(createCleanup());

      // Vectorize at tensor-level to benefit from better cleanup utilities like
      // folding.
      // TODO: Enable vectorization when vector unrolling is added.
      //       When vector sizes exceed hardware supported lengths,
      //       pipeline gets stuck on GPU binary compilation step.
      //       The vectorization can only be enabled when a pass
      //       to resize vector operations is available.
      pm.addPass(createGpuVectorize());
      pm.addPass(createCleanup());
    }

    // Preprocess and bufferize as further conversion requires memref
    // abstraction.
    pm.addPass(createLowerPacksAndUnPacks());
    pm.addPass(createBufferize(BufferizeOptions{/*dealloc=*/false}));
    pm.addPass(createConvertForAllToParallelOp());
    pm.addPass(createCleanup());

    // Convert to generic GPU ops.
    pm.addPass(createGpuConversion(GpuConversionOptions{
        gpuType == GpuType::Intel, kTile, stages,
        SmallVector<int64_t>{gpuDpasTile.begin(), gpuDpasTile.end()}}));

    // Lower GPU ops to the chosen GPU backend.
    switch (gpuType) {
    case GpuType::Cuda: {
      // Perform explicit GPU data transfers only for CUDA as the unified
      // memory is not currently used here.
      pm.addNestedPass<func::FuncOp>(createGpuDataTransfer());
      pm.addPass(createGpuToCuda(GpuToCudaOptions{
          gpuOptions.triple, gpuOptions.chip, gpuOptions.features}));
      break;
    }
    case GpuType::Intel: {
      pm.addPass(xegpu::createXeGPUFoldAliasOps());

      std::string clientApi = "intel";
      SetSPIRVCapabilitiesOptions capabilitiesOptions{clientApi};
      pm.addPass(tpp::createSetSPIRVCapabilities(capabilitiesOptions));
      SetSPIRVAbiAttributeOptions abiAttrOptions{clientApi};
      pm.addPass(tpp::createSetSPIRVAbiAttribute(abiAttrOptions));

      break;
    }
    }

    // Covert all local dialects like perf.
    pm.addPass(createLocalDialectsLowering());

    // Clean up after the GPU pipeline.
    pm.addPass(createCleanup());
  }
};

} // namespace
