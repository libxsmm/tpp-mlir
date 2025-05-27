//===- tpp-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "TPP/Dialect/Check/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/BufferizableOpInterfaceImpl.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Tune/TuneTransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassBundles.h"
#include "TPP/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tpp::registerTppCompilerPasses();
  mlir::tpp::registerTppPassBundlePasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::xsmm::XsmmDialect>();
  registry.insert<mlir::check::CheckDialect>();
  registry.insert<mlir::perf::PerfDialect>();
  mlir::check::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::perf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tpp::registerTestStructuralMatchers();
  mlir::tpp::registerTestForToForAllRewrite();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated.
  registerAllDialects(registry);
  registerAllExtensions(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::tensor::registerTransformDialectExtension(registry);
  mlir::tune::registerTransformDialectExtension(registry);
  registerAllToLLVMIRTranslations(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TPP optimizer driver\n", registry));
}
