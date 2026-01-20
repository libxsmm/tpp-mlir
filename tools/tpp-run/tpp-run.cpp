//===- tpp-run.cpp - TPP CPU Execution Driver--------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by translating MLIR to LLVM IR before JIT-compiling and executing the
// latter. Handles TPP/LIBXSMM include/library paths as well as benchmarking
// modes, with warmups, measurements, output comparison, etc.
//
//===----------------------------------------------------------------------===//

#include "TPP/Runner/MLIRBench.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "TPP/Transforms/Utils/TensorInit.h"
#include "libxsmm.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Dialect/Transform/TuneExtension/TuneExtensionOps.h"
#include "mlir/Dialect/Transform/TuneExtension/TuneExtension.h"

#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/GPU/Utils.h"
#include "TPP/PassBundles.h"
#include "TPP/Passes.h"

#include <algorithm>

using namespace mlir;

// Number of loops for benchmarks
llvm::cl::opt<unsigned>
    benchNumLoops("n", llvm::cl::desc("Number of loops for benchmarks"),
                  llvm::cl::value_desc("int"), llvm::cl::init(1));

// Print result
llvm::cl::opt<bool> printKernelResult("print",
                                      llvm::cl::desc("Print kernel result"),
                                      llvm::cl::init(false));

// Print input args
llvm::cl::opt<bool> printKernelInput("print-input",
                                     llvm::cl::desc("Print kernel inputs"),
                                     llvm::cl::init(false));

// Replace dense splat tensors with random dense
llvm::cl::opt<bool>
    splatRandom("splat-to-random",
                llvm::cl::desc("Replace splat dense tensors with random value"),
                llvm::cl::init(false));

// Random seed, if zero, don't emit randominputs
llvm::cl::opt<int> seed("seed",
                        llvm::cl::desc("Random seed, default 0 (no random)"),
                        llvm::cl::value_desc("int"), llvm::cl::init(0));

// Initializer type
// Default const if seed == 0, and normal otherwise
llvm::cl::opt<std::string> initType(
    "init-type",
    llvm::cl::desc("Initializer type (const, rand, normal)"),
    llvm::cl::init(""));

// Identity matrix
// Replace single square argument with identity matrix
// Note: Must have two arguments and the selected must be square
llvm::cl::opt<int> identity(
    "identity",
    llvm::cl::desc("Identity matrix on one argument (-1=none, 0=a, 1=b, ...)"),
    llvm::cl::init(-1));

// Speed optimization level
llvm::cl::opt<unsigned>
    optLevel("O", llvm::cl::desc("Speed optimization level (O0, O1, O2, O3)"),
             llvm::cl::value_desc("0-3"), llvm::cl::init(2));

// Print LLVM IR before lowering
llvm::cl::opt<bool> printLLVM("print-llvm",
                              llvm::cl::desc("print LLVM IR before lowering"),
                              llvm::cl::init(false));

// Select target GPU backend for the pipeline.
llvm::cl::opt<std::string>
    defGpuBackend("gpu", llvm::cl::desc("Target GPU backend for lowering"),
                  llvm::cl::value_desc("cuda,intel"), llvm::cl::init(""));

// Select target CPU feature for the pipeline.
llvm::cl::opt<std::string> runnerCpuTargetFeature(
    "target-feature", llvm::cl::desc("Specify CPU target feature for lowering"),
    llvm::cl::value_desc("avx, avx2, avx512f, avx512vnni, avx512bf16, amx, "
                         "amx_bf16, amx_tile, neon, sve"),
    llvm::cl::init(""));

// Kernel buffers - arguments and return values - are expected to be allocated
// on GPU.
llvm::cl::opt<bool>
    defGpuArgs("gpu-args",
               llvm::cl::desc("Kernel buffers are allocated on GPU"),
               llvm::cl::init(true));

struct TargetMachineOptions {
  std::string triple;
  std::string cpu;
  std::string features;
};

/// Returns the target machine options for the given CPU feature string.
/// Does not include full support for all CPU features, only the ones that are
/// relevant for now.
TargetMachineOptions getTargetMachineOptions(StringRef option) {
  std::string defaultCpu = "";
  std::string defaultFeature = "";
  std::string defaultTriple = "";
#if defined(__x86_64__)
  defaultTriple = "x86_64-linux-gnu";
  defaultCpu = "nehalem";
  defaultFeature = "+sse4.2";
#elif defined(__aarch64__)
  defaultTriple = "aarch64-linux-gnu";
  defaultCpu = "cortex-a53";
  defaultFeature = "+neon";
#else
#error Unsupported architecture
#endif
  return StringSwitch<TargetMachineOptions>(option)
      .Case("avx", {"x86_64-linux-gnu", "sandybridge", "+avx"})
      .Case("avx2", {"x86_64-linux-gnu", "haswell", "+avx2"})
      .Case("avx512f", {"x86_64-linux-gnu", "skylake-avx512", "+avx512f"})
      .Case("avx512vnni", {"x86_64-linux-gnu", "znver4", "+avx512vnni"})
      .Case("avx512bf16", {"x86_64-linux-gnu", "cooperlake", "+avx512bf16"})
      .Case("amx", {"x86_64-linux-gnu", "sapphirerapids", "+amx"})
      .Case("amx_bf16", {"x86_64-linux-gnu", "sapphirerapids", "+amx_bf16"})
      .Case("amx_tile", {"x86_64-linux-gnu", "sapphirerapids", "+amx_tile"})
      .Case("neon", {"armv8a-linux-gnu", "cortex-a53", "+neon"})
      .Case("sve", {"armv8a-linux-gnu", "a64fx", "+sve"})
      .Case("testfeature", {"x86_64-linux-gnu", "sandybridge", "+testfeature"})
      .Default({defaultTriple, defaultCpu, defaultFeature});
}

// This function will be called by the pass manager after parsing,
// so we can modify the IR with the needed wrappers
static LogicalResult prepareMLIRKernel(Operation *op,
                                       JitRunnerOptions &options) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitOpError("Expected a 'builtin.module' op");

  // A set of default passes that lower any input IR to LLVM
  PassManager passManager(module.getContext());

  // Propagate pass manager's command-line options.
  if (failed(applyPassManagerCLOptions(passManager)))
    return failure();

  tpp::TppRunnerWrapperOptions wrapperOpts;
  wrapperOpts.kernelName = options.mainFuncName;
  wrapperOpts.kernelType = options.mainFuncType;
  wrapperOpts.backend = defGpuBackend;
  wrapperOpts.wrapperCpuTargetFeature = runnerCpuTargetFeature;
  wrapperOpts.offloadToDevice = defGpuArgs;
  wrapperOpts.numBenchLoops = benchNumLoops;
  // Warmup on GPUs are currently breaking buffer allocation on GPUs
  wrapperOpts.benchWarmup = defGpuBackend.empty();
  wrapperOpts.printResult = printKernelResult;
  wrapperOpts.printInput = printKernelInput;
  wrapperOpts.randomSplat = splatRandom;
  wrapperOpts.seed = seed;
  wrapperOpts.initType = initType;
  wrapperOpts.identity = identity;
  passManager.addPass(tpp::createTppRunnerWrapper(wrapperOpts));

  tpp::DefaultPipelineOptions defPipelineOpts{defGpuBackend,
                                              runnerCpuTargetFeature};
  passManager.addPass(tpp::createDefaultPipeline(defPipelineOpts));

  auto result = passManager.run(module);
  if (failed(result)) {
    llvm::errs() << "ERROR: Failed to lower IR to LLVM dialect\n";
    module->print(llvm::errs());
    return result;
  }

  return success();
}

std::unique_ptr<llvm::Module> lowerToLLVMIR(Operation *module,
                                            llvm::LLVMContext &llvmContext) {
  // Default lowering for mlir-runner
  auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
  assert(llvmModule);

  // Target machine, null if not specified
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  TargetMachineOptions targetMachineOptStr =
      getTargetMachineOptions(runnerCpuTargetFeature);

  // Specify target machine
  std::string error;
  llvm::Triple triple(targetMachineOptStr.triple);
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);

  if (!target) {
    llvm::errs() << "Error while looking up target triple: ";
    llvm::errs() << error << "\n";
    return nullptr;
  }

  auto codeGenOpt = (llvm::CodeGenOptLevel)optLevel.getValue();

  // These options should force fused MLA, but they don't. :/
  // Adding unsafe math attribute to functions below do the trick.
  llvm::TargetOptions targetOptions;
  targetOptions.AllowFPOpFusion = llvm::FPOpFusion::FPOpFusionMode::Fast;
  targetMachine.reset(target->createTargetMachine(
      llvm::Triple(targetMachineOptStr.triple), targetMachineOptStr.cpu,
      targetMachineOptStr.features, targetOptions,
      /* reloc model */ std::nullopt,
      /* code model */ std::nullopt, codeGenOpt));

  if (!targetMachine) {
    llvm::errs() << "Error while looking up target CPU: ";
    llvm::errs() << targetMachineOptStr.cpu << "\n";
    return nullptr;
  }

  // Run the optimized pipeline
  int sizeLevel = 0;
  auto optPipeline =
      makeOptimizingTransformer(optLevel, sizeLevel, targetMachine.get());
  if (auto err = optPipeline(llvmModule.get())) {
    llvmModule->print(llvm::errs(), nullptr);
    llvm::errs() << "Error while passing through the LLVM pipeline: ";
    llvm::errs() << err << "\n";
    return nullptr;
  }

  // MLIR doesn't lower LLVM with fast-math flags, but we need that, so we
  // add for each function, to get FMAs and other goodies.
  for (auto &func : llvmModule->functions()) {
    func.addFnAttr("unsafe-fp-math", "true");
  }

  if (printLLVM)
    llvmModule->print(llvm::outs(), nullptr);

  return llvmModule;
}

LogicalResult emitError(StringRef msg) {
  llvm::errs() << "ERROR: " << msg << "\n";
  return failure();
}

// Input validation
LogicalResult validateInput() {
  // Parse tensor init
  auto init = parseTensorInitType(initType);
  if (init == TensorInitType::Invalid)
    return emitError("Invalid tensor init " + initType);

  return success();
}

int main(int argc, char **argv) {
  // Make sure the args are compatible
  if (failed(validateInput()))
    return 1;

  // Initialize the underlying platform
  // TODO: Move this to use the target information flags
  libxsmm_init();
  // Initialize the LLVM machinery
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Initialize GPU-related LLVM machinery
  tpp::initializeGpuTargets();

  // Register all passes to expose them for debugging
  mlir::registerAllPasses();
  mlir::tpp::registerTppCompilerPasses();
  mlir::tpp::registerTppPassBundlePasses();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  DialectRegistry registry;
  registry.insert<mlir::xsmm::XsmmDialect>();
  registry.insert<mlir::check::CheckDialect>();
  registry.insert<mlir::perf::PerfDialect>();
  mlir::transform::registerTuneExtension(registry);
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllToLLVMIRTranslations(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);

  // Add pass manager CLI debug options - exposes IR printing capabilities
  // same as in opt tool
  mlir::registerPassManagerCLOptions();

  // This is how we integrate with the pipeline
  JitRunnerConfig config;
  config.mlirTransformer = prepareMLIRKernel;
  config.llvmModuleBuilder = lowerToLLVMIR;

  // Call the main JIT function
  return JitRunnerMain(argc, argv, registry, config);
}
