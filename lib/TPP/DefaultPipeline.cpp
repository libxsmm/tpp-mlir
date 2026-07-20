//===- DefaultPipeline.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassBundles.h"
#include "TPP/PassUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"

#include <string>

using namespace mlir;
using namespace mlir::tpp;

// Print MLIR before lowering
llvm::cl::opt<std::string>
    printMLIR("print-mlir",
              llvm::cl::desc("Print MLIR to stdout (early, mid, late, llvm)"),
              llvm::cl::init(""));

// Lower Linalg directly to loops without TPP (for validation purposes)
llvm::cl::opt<bool> linalgToLoops("linalg-to-loops",
                                  llvm::cl::desc("Lower linalg to loops"),
                                  llvm::cl::init(false));

// Control parallelism.
llvm::cl::opt<bool>
    defParallel("def-parallel",
                llvm::cl::desc("Default pipeline - enable parallel execution"),
                llvm::cl::init(false));

// Control scf.forall iteration ordering / flattening strategy.
llvm::cl::opt<bool> sfcOrder(
    "sfc-order",
    llvm::cl::desc("Use space-filling-curve-based iteration ordering / "
                   "flattening for scf.forall loops in the default pipeline"),
    llvm::cl::init(true));

// Control grid parallelism sizes.
llvm::cl::list<unsigned>
    parallelTaskGrid("parallel-task-grid",
                     llvm::cl::desc("Grid-sizes for parallel tasks"),
                     llvm::cl::list_init<unsigned>(SmallVector<unsigned>{2, 8}),
                     llvm::cl::CommaSeparated);

llvm::cl::opt<bool>
    vectorToKernel("vector-to-kernels",
                   llvm::cl::desc("Lower vector to micro-kernels"),
                   llvm::cl::init(false));

llvm::cl::opt<bool>
    nanoKernel("nano-kernels",
                   llvm::cl::desc("Lower vector.contract to nano-kernels"),
                   llvm::cl::init(false));

llvm::cl::opt<bool> lowerPackUnpackWithoutTranspose(
    "lower-pack-unpack-without-transpose",
    llvm::cl::desc("Lower packs and unpacks reverting any dim permutations"),
    llvm::cl::init(false));

llvm::cl::opt<bool>
    disableVnniPacking("disable-vnni-packing",
                       llvm::cl::desc("Disables VNNI packing for packed types"),
                       llvm::cl::init(false));

llvm::cl::opt<bool> disableTileElementwiseOps(
    "disable-tile-elementwise-ops",
    llvm::cl::desc("Disables tiling of elementwise operations"),
    llvm::cl::init(false));

llvm::cl::list<unsigned> registerBlocking(
    "registerBlocking",
    llvm::cl::desc("Register blocking tile sizes for brgemm operation"),
    llvm::cl::list_init<unsigned>(SmallVector<unsigned>{8, 32}),
    llvm::cl::CommaSeparated);

llvm::cl::list<int64_t> gemmUnroll(
    "gemm-unroll",
    llvm::cl::desc("GEMM register unroll sizes for the innermost dims: [M, N, "
                   "K]. Required by the nano-kernel path to reduce "
                   "vector.contract to register-sized shapes"),
    llvm::cl::CommaSeparated);

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_DEFAULTPIPELINE
#include "TPP/PassBundles.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Enum to control IR printing.
enum class PrintStage {
  None,
  Early, // After main generation, before optimization
  Mid,   // After initial TPP-related optimizations
  Late,  // After optimizaiton, before LLVM dialect
  LLVM,  // Final MLIR, in LLVM dialect
};

// Parses MLIR print stage
PrintStage parsePrintStage(StringRef stage) {
  return StringSwitch<PrintStage>(stage)
      .CaseLower("early", PrintStage::Early)
      .CaseLower("mid", PrintStage::Mid)
      .CaseLower("late", PrintStage::Late)
      .CaseLower("llvm", PrintStage::LLVM)
      .Default(PrintStage::None);
}

// The default lowering pipeline.
struct DefaultPipeline : public tpp::impl::DefaultPipelineBase<DefaultPipeline>,
                         PassBundle<ModuleOp> {
  using DefaultPipelineBase::DefaultPipelineBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    // Add all custom TPP dialects.
    registry.insert<xsmm::XsmmDialect>();
    registry.insert<check::CheckDialect>();
    registry.insert<perf::PerfDialect>();
    check::registerBufferizableOpInterfaceExternalModels(registry);
    perf::registerBufferizableOpInterfaceExternalModels(registry);

    // Add all core MLIR dialects as the default pipeline may contain any
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
  void constructPipeline() override {
    auto print = parsePrintStage(printMLIR);

    // Print IR of unoptimized kernel and main
    if (print == PrintStage::Early)
      pm.addPass(createPrintIRPass());

    addDefaultTppPasses();

    if (print == PrintStage::Mid)
      pm.addPass(createPrintIRPass());

    addPartialLoweringPasses();

    // Print IR of optimized kernel and main
    if (print == PrintStage::Late)
      pm.addPass(createPrintIRPass());

    addLowerToLLVMPasses();

    // Print IR of kernel and main in LLVM dialect
    if (print == PrintStage::LLVM)
      pm.addPass(createPrintIRPass());
  }

  // Apply the default TPP pass.
  void addDefaultTppPasses() {
    DefaultTppPassesOptions tppDefaultOptions;
    tppDefaultOptions.linalgToLoops = linalgToLoops;
    tppDefaultOptions.sfcOrder = sfcOrder;
    tppDefaultOptions.parallelTaskGrid = SmallVector<unsigned>{
        parallelTaskGrid.begin(), parallelTaskGrid.end()};
    tppDefaultOptions.lowerPackUnpackWithoutTranspose =
        lowerPackUnpackWithoutTranspose;
    tppDefaultOptions.disableVnniPacking = disableVnniPacking;
    tppDefaultOptions.disableTileElementwiseOps = disableTileElementwiseOps;
    tppDefaultOptions.registerBlocking = SmallVector<unsigned>{
        registerBlocking.begin(), registerBlocking.end()};
    tppDefaultOptions.gemmUnroll =
        SmallVector<int64_t>{gemmUnroll.begin(), gemmUnroll.end()};
    tppDefaultOptions.vectorToKernel = vectorToKernel;
    tppDefaultOptions.nanoKernel = nanoKernel;
    tppDefaultOptions.defBundleCpuTargetFeature = pipelineCpuTargetFeature;
    pm.addPass(createDefaultTppPasses(tppDefaultOptions));
  }

  // Partially lower the IR towards LLVM.
  void addPartialLoweringPasses() {
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    if (defParallel)
      pm.addPass(createConvertSCFToOpenMPPass());
    pm.addPass(createConvertVectorToSCFPass());
    mlir::arith::ArithExpandOpsPassOptions arithExpandOpsOptions;
    arithExpandOpsOptions.includeF8E8M0 = true;
    pm.addPass(arith::createArithExpandOpsPass(arithExpandOpsOptions));
    pm.addPass(createLowerAffinePass());
  }

  // Lower the remaining dialects to the LLVM dialect.
  void addLowerToLLVMPasses() {
    ConvertVectorToLLVMPassOptions options;
#if defined(__x86_64__)
    options.x86 = true;
#endif
    pm.addPass(createConvertVectorToLLVMPass(options));
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createSCFToControlFlowPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createAsyncToAsyncRuntimePass());
    pm.addPass(createAsyncRuntimeRefCountingPass());
    pm.addPass(createConvertAsyncToLLVMPass());
    pm.addPass(createConvertIndexToLLVMPass());
    pm.addPass(createConvertFuncToLLVMPass());

    // FIXME: Removing this line crashes two CPU quantization tests
    // even though neither are affected by GPU lowering.
    // There must be some pattern that this pass does that makes a difference.
    pm.addPass(createGpuToLLVMConversionPass());

    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    if (defParallel)
      pm.addPass(createConvertOpenMPToLLVMPass());
    pm.addPass(createUBToLLVMConversionPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createSymbolDCEPass());
  }
};

} // namespace
