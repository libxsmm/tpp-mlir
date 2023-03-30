//===- tpp-run.cpp - TPP CPU Execution Driver------------------------------===//
//
// Main entry point to a command line utility that executes an MLIR file on the
// CPU by translating MLIR to LLVM IR before JIT-compiling and executing the
// latter. Handles TPP/LIBXSMM include/library paths as well as benchmarking
// modes, with warmups, measurements, output comparison, etc.
//
//===----------------------------------------------------------------------===//

#include "MLIRBench.h"

#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "TPP/TensorInit.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Tpp/TppDialect.h"
#include "TPP/Dialect/Transform/LinalgXTransformOps.h"
#include "TPP/Dialect/VNNI/VNNIDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/Passes.h"

using namespace mlir;

// Number of loops for benchmarks
llvm::cl::opt<unsigned>
    benchNumLoops("n", llvm::cl::desc("Number of loops for benchmarks"),
                  llvm::cl::value_desc("int"), llvm::cl::init(1));

// Print result
llvm::cl::opt<bool> printKernelResult("print",
                                      llvm::cl::desc("Print kernel result"),
                                      llvm::cl::init(false));

// Lower TPP to loops (for validation purposes)
llvm::cl::opt<bool> tppToLoops("tpp-to-loops",
                               llvm::cl::desc("Lower TPP to loops"),
                               llvm::cl::init(false));

// Lower Linalg directly to loops without TPP (for validation purposes)
llvm::cl::opt<bool> linalgToLoops("linalg-to-loops",
                                  llvm::cl::desc("Lower linalg to loops"),
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

// Speed optimization level
llvm::cl::opt<unsigned>
    optLevel("O", llvm::cl::desc("Speed optimization level (O0, O1, O2, O3)"),
             llvm::cl::value_desc("0-3"), llvm::cl::init(2));

// Target Triple
// Default x86_64, can be changed to aarch64 on other arches
llvm::cl::opt<std::string> triple("triple", llvm::cl::desc("Target triple"),
                                  llvm::cl::init("x86_64-linux-gnu"));

// Target CPU name
// Default skylake is old enough to be relevant for most cases
llvm::cl::opt<std::string>
    cpuName("cpu", llvm::cl::desc("CPU name (sapphirerapids, alderlake, etc)"),
            llvm::cl::init("skylake"));

// Target FPU name
// Default avx2 is old enough to be relevant for most cases
llvm::cl::opt<std::string>
    fpuName("fpu", llvm::cl::desc("FPU name (avx, avx2, avx512bf16)"),
            llvm::cl::init("avx2"));

// Initializer type
// Default const if seed == 0, and normal otherwise
llvm::cl::opt<std::string> initType(
    "init-type",
    llvm::cl::desc("Initializer type (const, simple, cont, rand, normal)"),
    llvm::cl::init(""));

// Print MLIR before lowering
llvm::cl::opt<std::string> printMLIR(
    "print-mlir",
    llvm::cl::desc("Print MLIR to stdout (early, mid, late, llvm)"),
    llvm::cl::init(""));

// Print LLVM IR before lowering
llvm::cl::opt<bool> printLLVM("print-llvm",
                              llvm::cl::desc("print LLVM IR before lowering"),
                              llvm::cl::init(false));

// Parses MLIR print stage
MLIRBench::PrintStage parsePrintStage(StringRef stage) {
  return StringSwitch<MLIRBench::PrintStage>(stage)
      .Case("", MLIRBench::PrintStage::None)
      .Case("early", MLIRBench::PrintStage::Early)
      .Case("mid", MLIRBench::PrintStage::Mid)
      .Case("late", MLIRBench::PrintStage::Late)
      .Case("llvm", MLIRBench::PrintStage::LLVM)
      .Default(MLIRBench::PrintStage::Invalid);
}

// This function will be called by the pass manager after parsing,
// so we can modify the IR with the needed wrappers
static LogicalResult prepareMLIRKernel(Operation *op,
                                       JitRunnerOptions &options) {
  // Benchmark object
  MLIRBenchConfig config(seed, tppToLoops, linalgToLoops,
                         parseTensorInitType(initType));
  MLIRBench bench(op, config);

  // Basic checks
  if (options.mainFuncType != "void")
    return bench.emitError(
        "Main function has to be 'void', even if the kernel return's a value, "
        "because that's the type of the wrapper we create here");

  if (failed(bench.findKernel(options.mainFuncName)))
    return bench.emitError("Cannot find kernel '" + options.mainFuncName + "'");

  if (failed(bench.checkKernelSignature()))
    return bench.finalize(parsePrintStage(printMLIR));

  if (splatRandom && failed(bench.replaceSplatWithRandom()))
    return bench.emitError("Error converting splat tensors with random values");

  // Move the kernel to a local name, so we can create `main` with the same
  // name as the pre-defined entry point (since we can't change it)
  if (failed(bench.renameKernel()))
    return bench.emitError("Cannot rename kernel function");

  // Creates the main wrapper
  if (failed(bench.createMainWrapper()))
    return bench.emitError("Cannot create main wrapper");

  // Creates the inputs for the kernel
  if (failed(bench.createKernelArgs()))
    return bench.emitError("Cannot create kernel inputs");

  // Call kernel once, to bootstrap (JIT compile, warm up caches)
  auto *call = bench.callKernel();
  if (!call)
    return bench.emitError("Cannot generate a call to the kernel");

  // Print the result of the warming up, should be the same as any other
  if (printKernelResult && failed(bench.printResult(call)))
    return bench.emitError("Cannot print result memref");

  // This is the main loop, if N > 1
  if (benchNumLoops > 1) {
    auto acc = bench.createTimerLoop(benchNumLoops);
    if (!acc)
      return bench.emitError("Cannot create timer loop");
    auto stats = bench.getTimerStats(acc);
    if (!stats)
      return bench.emitError("Cannot get timer stats");
    bench.printVector(stats);
  }

  // Finally lower to LLVM Dialect
  return bench.finalize(parsePrintStage(printMLIR));
}

std::unique_ptr<llvm::Module> lowerToLLVMIR(Operation *module,
                                            llvm::LLVMContext &llvmContext) {
  // Default lowering for mlir-cpu-runner
  auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
  assert(llvmModule);

  // Target machine, null if not specified
  std::unique_ptr<llvm::TargetMachine> targetMachine;

  // Specify target machine
  if (!triple.empty() && !cpuName.empty()) {
    std::string error;
    const llvm::Target *target =
        llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
      llvm::errs() << "Error while looking up target triple: ";
      llvm::errs() << error << "\n";
      return nullptr;
    }

    llvm::CodeGenOpt::Level codeGenOpt =
        (llvm::CodeGenOpt::Level)optLevel.getValue();

    // These options should force fused MLA, but they don't. :/
    // Adding unsafe math attribute to functions below do the trick.
    llvm::TargetOptions targetOptions;
    targetOptions.UnsafeFPMath = true;
    targetOptions.AllowFPOpFusion = llvm::FPOpFusion::FPOpFusionMode::Fast;
    targetMachine.reset(target->createTargetMachine(
        triple, cpuName, "+" + fpuName, targetOptions,
        /* reloc model */ std::nullopt,
        /* code model */ std::nullopt, codeGenOpt));
    if (!targetMachine) {
      llvm::errs() << "Error while looking up target CPU: ";
      llvm::errs() << cpuName << "\n";
      return nullptr;
    }
  }

  // Run the optimized pipeline
  int sizeLevel = 0;
  auto optPipeline =
      makeOptimizingTransformer(optLevel, sizeLevel, targetMachine.get());
  if (auto err = optPipeline(llvmModule.get())) {
    llvmModule->dump();
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
  // Randon options need seed
  if (!seed) {
    if (splatRandom)
      return emitError("Cannot replace splats with random without seed");
    if (initType == "random" || initType == "normal")
      return emitError("Cannot init random tensors without seed");
  }

  // Parse tensor init
  auto init = parseTensorInitType(initType);
  if (init == TensorInitType::Invalid)
    return emitError("Invalid tensor init " + initType);

  // Parse print MLIR stage
  auto stage = parsePrintStage(printMLIR);
  if (stage == MLIRBench::PrintStage::Invalid)
    return emitError("Invalid print MLIR stage " + printMLIR);

  return success();
}

int main(int argc, char **argv) {
  // Make sure the args are compatible
  if (failed(validateInput()))
    return 1;

  // Initialize the LLVM machinery
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  DialectRegistry registry;
  registry.insert<mlir::tpp::TppDialect>();
  registry.insert<mlir::xsmm::XsmmDialect>();
  registry.insert<mlir::check::CheckDialect>();
  registry.insert<mlir::vnni::VNNIDialect>();
  registry.insert<mlir::perf::PerfDialect>();
  mlir::linalgx::registerTransformDialectExtension(registry);
  registerAllDialects(registry);
  registerAllToLLVMIRTranslations(registry);

  // This is how we integrate with the pipeline
  JitRunnerConfig config;
  config.mlirTransformer = prepareMLIRKernel;
  config.llvmModuleBuilder = lowerToLLVMIR;

  // Call the main JIT function
  return JitRunnerMain(argc, argv, registry, config);
}
