//===- DefaultTppPasses.cpp --------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/PassBundles.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassUtils.h"
#include "mlir/Transforms/Passes.h"

#include <string>

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_DEFAULTTPPPASSES
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// The default pipeline for TPP.
struct DefaultTppPasses
    : public tpp::impl::DefaultTppPassesBase<DefaultTppPasses>,
      PassBundle<ModuleOp> {
  using DefaultTppPassesBase::DefaultTppPassesBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom TPP dialects.
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<perf::PerfDialect>();
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects as the default TPP passes may contain any
    // combination of other passes.
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
  // Vectorization is required by the explicit `linalg-to-vector` path as well
  // as the `vector-to-kernel` and `nano-kernel` paths, which are built on top
  // of it. Any of these flags enables the vectorization stage.
  bool shouldVectorize() const {
    return linalgToVector || vectorToKernel || nanoKernel;
  }

  // Lower linalg directly to loops, skipping all TPP transformations.
  void addLinalgToLoopsPasses() {
    // Generalize linalg.pack and linalg.unpack.
    pm.addPass(createLowerPacksAndUnPacks());
    pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOps());
    pm.addPass(createBufferize());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    pm.addPass(createCleanup());
  }

  // Vectorize the remaining Linalg operations and, optionally, lower vector
  // patterns to micro-kernels (`vector-to-kernel`) or nano-kernels
  // (`nano-kernel`).
  void addVectorizationPasses() {
    if (nanoKernel) {
      pm.addNestedPass<func::FuncOp>(createLinalgGeneralizeNamedOpsPass());
      pm.addNestedPass<func::FuncOp>(
          createConvertLinalgGenericTo32BitAccumulation());
    }

    // Vectorizes the remaining Linalg operations
    pm.addNestedPass<func::FuncOp>(createBrgemmLinalgTiling(
        BrgemmLinalgTilingOptions{SmallVector<unsigned>{*registerBlocking}}));
    pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
    if (!disableTileElementwiseOps)
      pm.addNestedPass<func::FuncOp>(createTileElementWiseOps());
    pm.addNestedPass<func::FuncOp>(createVectorizationPass());

    if (nanoKernel) {
      tpp::RegisterUnrollOptions unrollOpts;
      unrollOpts.gemmUnroll = SmallVector<int64_t>{*gemmUnroll};
      pm.addNestedPass<func::FuncOp>(createRegisterUnroll(unrollOpts));
      pm.addNestedPass<func::FuncOp>(createHoistLoopInvariantSubsets());
      pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
      pm.addPass(createBufferize());
      pm.addNestedPass<func::FuncOp>(createVectorContractToNanoKernels());
      pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      pm.addNestedPass<func::FuncOp>(createFlattenVectorOps());
    }

    // Please note, canonicalizer should be after hoisting pass because
    // it fuses outer tiling loops and it results in no pattern
    // matching for hoisting pass. Moved inside VectorToKernel Path.
    // This path will be soon replaced by the nanoKernel path.
    if (vectorToKernel) {
      VectorToKernelOptions options;
      options.vecBundleCpuTargetFeature = defBundleCpuTargetFeature;
      pm.addPass(createVectorToKernel(options));
    }
  }

  // Default TPP lowering: map and pack at the linalg level, bufferize, lower to
  // XSMM and optionally vectorize.
  void addDefaultLoweringPasses() {
    pm.addPass(createFoldIntoEltwise());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToInplace());
    // Convert linalg.batch_matmul to linalg.matmul.
    pm.addPass(createRewriteBatchMatmulToMatmul());

    // Applies a set of passes at the linalg level to fuse and pack.
    TppMappingOptions tppMappingOptions{lowerPackUnpackWithoutTranspose,
                                        disableVnniPacking};
    pm.addPass(createTppMapping(tppMappingOptions));

    // Generalize linalg.pack and linalg.unpack.
    pm.addPass(createLowerPacksAndUnPacks());
    pm.addPass(createCleanup());

    // Decompose Aggregated operations. These ops currently do not
    // bufferize. Once this is possible we can move this pass after
    // bufferization.
    pm.addNestedPass<func::FuncOp>(createDecomposeAggregatedOps());

    // Flatten 2D scf.forall loops using space-filling curve before
    // bufferization.
    if (sfcOrder)
      pm.addPass(createSCFForAllLoopFlattenSFC());

    // Bufferize: tensor->memref.
    if (!nanoKernel)
      pm.addPass(createBufferize());

    // Replicate benchmark kernel arguments for cold-cache timing. Runs on
    // bufferized memrefs so replicas are plain subviews (no allocs/copies).
    // No-op unless the benchmark producer requested replication.
    pm.addPass(createReplicateBenchArgs());

    // Lower Linalg to XSMM.
    pm.addNestedPass<func::FuncOp>(createLinalgLowering());

    if (shouldVectorize())
      addVectorizationPasses();

    // Final cleanup.
    pm.addPass(createCleanup());
  }

  // Convert to parallel loops and apply low-level parallelization. The
  // `linalg-to-vector` path lowers vector to SCF, while the XSMM path (also
  // used by `vector-to-kernel` and `nano-kernel`) applies AMX tile
  // configuration and lowers XSMM to function calls.
  void addParallelizationPasses() {
    // Convert forAll to parallel loops should run after bufferization
    // as scf.parallel does not handle tensor.
    pm.addPass(createConvertForAllToParallelOp());
    LowLevelParallelizationOptions LowLevelParallelization{
        SmallVector<unsigned>{*parallelTaskGrid}};

    if (linalgToVector)
      pm.addPass(createConvertVectorToSCFPass());

    // Low level parallelization passes.
    if (!sfcOrder)
      pm.addPass(createLowLevelParallelization(LowLevelParallelization));

    if (!linalgToVector) {
      // TODO: These passes have been moved out of low level parallelization
      // pass since these apply on xsmm dialect. They'll be moved back in
      // subsequent commits.
      pm.addNestedPass<func::FuncOp>(createIntelAMXTileConfigInsertionPass());
      pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      pm.addNestedPass<func::FuncOp>(createLoopInvariantCodeMotionPass());
      pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      pm.addNestedPass<func::FuncOp>(createIntelAMXTileConfigHoistingPass());
      // TODO: This pass has been moved out of LocalDialectsLowering since it is
      // applicable to xsmm only. It'll be moved back in subsequent commits.
      pm.addPass(createConvertXsmmToFunc());
    }
  }

  void constructPipeline() override {
    // We currently have four branches:
    //  * Linalg-to-Loops: Enable with `linalg-to-loops`. Skips all TPP
    //    transformations and lowers linalg directly to loops.
    //  * Linalg-to-XSMM: the default path, no options needed.
    //  * Linalg-to-Vector: Enable with `linalg-to-vector`. Lowers straight to
    //    LLVM with no further changes to the IR.
    //  * Vector-to-Kernel / Nano-Kernel: Enable with `vector-to-kernel` or
    //    `nano-kernel`. Both require vectorization and lower vector patterns to
    //    libxsmm-like micro-/nano-kernels via specialized lowering.
    pm.addPass(createFoldAddIntoDest());

    if (linalgToLoops)
      addLinalgToLoopsPasses();
    else
      addDefaultLoweringPasses();

    addParallelizationPasses();

    // Covert all local TPP-related dialects.
    pm.addPass(createLocalDialectsLowering());

    // Clean up after the default pipeline.
    pm.addNestedPass<func::FuncOp>(createPostprocessing());
  }
};

} // namespace
