//===- mlir-gen MLIR Generator -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry-point to the MLIR generator. Creates an MLIR model with input,
// output and multiple hidden layers, different activation functions, etc.
// Handles multiple tensor sizes, conversion, broadcast, etc.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"

#include "MLIRGen.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

using namespace mlir;

// Kind of linalg Op, generic or nameed ops
llvm::cl::opt<std::string> outputOpKind(
    "output", llvm::cl::desc("Specifies linalg op kind generic, contract or named"),
    llvm::cl::value_desc("generic,contract,named"), llvm::cl::init("generic"));

// Type of kernel to be generated
llvm::cl::opt<std::string> kernel("kernel",
                                  llvm::cl::desc("Kernel type to be generated"),
                                  llvm::cl::value_desc("const,args"),
                                  llvm::cl::init("const"));

// Input layer
llvm::cl::opt<unsigned> batch("batch", llvm::cl::desc("Mini batch size"),
                              llvm::cl::value_desc("256"), llvm::cl::init(256));

// Hidden layers
llvm::cl::opt<std::string> layers(
    "layers",
    llvm::cl::desc("Comma-separated values of size of each layer (at least 2)"),
    llvm::cl::value_desc("128,256,512"), llvm::cl::init("128,256,512"));

// Tile sizes (N, C, K)
llvm::cl::opt<std::string>
    tiles("tiles",
          llvm::cl::desc("Comma-separated values of size of each tile (N,K,C)"),
          llvm::cl::value_desc("32,32,32"), llvm::cl::init(""));

// Tile sizes (N, C, K)
llvm::cl::opt<std::string>
    registerUnroll("registerUnroll",
          llvm::cl::desc("Comma-separated values of size of each tile (N,K,C)"),
          llvm::cl::value_desc("1,1,1"), llvm::cl::init(""));

// Data type for the arguments
llvm::cl::opt<std::string>
    dataType("data-type", llvm::cl::desc("Data type for arguments"),
              llvm::cl::value_desc(
                  "f32|f16|bf16|bf8|hf8|i8|i8-f32"),
              llvm::cl::init("f32"));

// Scale type flag to chose data type of scaling factor.For now it is kind of a
// place holder.
llvm::cl::opt<std::string>
    scaleType("scale-type", llvm::cl::desc("Data type of scaling factor"),
              llvm::cl::value_desc("f32|f8E8M0FNU"), llvm::cl::init(""));

// Enable quantization. The concrete behaviour (quantize, dequantize or
// requantize) is derived from the selected data type, mirroring what PyTorch
// does with quantized models.
llvm::cl::opt<bool> quant("quant",
                          llvm::cl::desc("Generate a quantized kernel with "
                                         "scaling factors"),
                          llvm::cl::value_desc("bool"), llvm::cl::init(false));

// Random seed
llvm::cl::opt<int> seed("seed", llvm::cl::desc("Random seed"),
                        llvm::cl::value_desc("int"), llvm::cl::init(0));

// Identity matrix
// Replace single square argument with identity matrix
// Note: Must have two arguments and the selected must be square
llvm::cl::opt<bool> identity("identity",
                             llvm::cl::desc("Identity matrix on weight (bias 1s)"),
                             llvm::cl::value_desc("bool"),
                             llvm::cl::init(false));

// Output filename
llvm::cl::opt<std::string> filename("o", llvm::cl::desc("Output filename"),
                                    llvm::cl::value_desc("stdout"),
                                    llvm::cl::init("-"));

// Enable bias add on every layer
llvm::cl::opt<bool> enableBias("bias",
                               llvm::cl::desc("Enable bias on every layer"),
                               llvm::cl::value_desc("bool"),
                               llvm::cl::init(false));

// Enable relu on every layer
llvm::cl::opt<bool> enableRelu("relu",
                               llvm::cl::desc("Enable relu on every layer"),
                               llvm::cl::value_desc("bool"),
                               llvm::cl::init(false));

// Enable softmax at the last layer
llvm::cl::opt<bool>
    enableSoftmax("softmax", llvm::cl::desc("Enable softmax on the last layer"),
                  llvm::cl::value_desc("bool"), llvm::cl::init(false));

// Set VNNI packing factor for BF16
llvm::cl::opt<int>
    vnni("vnni", llvm::cl::desc("VNNI packing factor (disabled if zero)"),
         llvm::cl::value_desc("0|2|4"), llvm::cl::init(0));

// Transpose the A (input) operand layout
llvm::cl::opt<bool>
    transposeA("transpose-a",
               llvm::cl::desc("Store the A (input) operand transposed"),
               llvm::cl::value_desc("bool"), llvm::cl::init(false));

// Transpose the B (weight) operand layout
llvm::cl::opt<bool>
    transposeB("transpose-b",
               llvm::cl::desc("Store the B (weight) operand transposed"),
               llvm::cl::value_desc("bool"), llvm::cl::init(false));

int main(int argc, char **argv) {
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "MLIR Generator");

  // Quantization needs an 8-bit integer operand; reject wider-only types here
  // instead of silently ignoring --quant during code generation.
  if (quant) {
    llvm::StringRef ty(dataType);
    if (!ty.equals_insensitive("i8") && !ty.equals_insensitive("i8-f32")) {
      llvm::errs() << "error: --quant requires a data type with an 8-bit "
                      "integer operand (i8 or i8-f32); got '"
                   << dataType << "'\n";
      return 1;
    }
  }

  MLIRGenerator gen(outputOpKind, kernel, batch, layers, tiles, registerUnroll, dataType,
                    scaleType, quant, seed, identity, enableBias,
                    enableRelu, enableSoftmax, vnni, transposeA, transposeB);
  return gen.generate(filename);
}
