//===- PassBundles.td --------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TPP_PASSBUNDLES
#define TPP_DIALECT_TPP_PASSBUNDLES

include "mlir/Pass/PassBase.td"

def DefaultPipeline : Pass<"default-pipeline", "ModuleOp"> {
  let summary = "The default compiler lowering pipeline";
  let description = [{
    A collection of passes that lower everything to MLIR LLVM IR.
  }];
  let options = [
    Option<"gpuBackend", "gpu", "std::string",
            /*default=*/"\"\"",
           "Optional target GPU backend.">,
    Option<"pipelineCpuTargetFeature", "target-feature", "std::string",
            /*default=*/"",
           "Optional CPU target feature.">,
  ];
}

def DefaultTppPasses : Pass<"default-tpp-passes", "ModuleOp"> {
  let summary = "Collection of default TPP passes";
  let description = [{
    A collection of passes that lower everything TPP-related
    to standard low-level dialects.
  }];
  let options= [
    Option<"linalgToLoops", "linalg-to-loops",
           "bool", /*default=*/"false",
           "Skip all TPP transformations. Lower linalg directly to loops.">,
    ListOption<"parallelTaskGrid", "parallel-task-grid",
           "unsigned", "Grid-sizes for parallel tasks.">,
    Option<"linalgToVector", "linalg-to-vector",
           "bool", /*default=*/"false",
           "Lower linalg directly to vector.">,
    Option<"vectorToXSMM", "vector-to-xsmm",
           "bool", /*default=*/"false",
           "Lower vector patterns to XSMM calls.">,
    Option<"vectorToKernel", "vector-to-kernel",
           "bool", /*default=*/"false",
           "Lower vector patterns to micro-kernels.">,
    Option<"defBundleCpuTargetFeature", "target-feature", "std::string", "",
                "Target feature for a given architecture">,
    Option<"lowerPackUnpackWithoutTranspose", "lower-pack-unpack-without-transpose",
           "bool", /*default=*/"false",
           "Lower non-constant packs and unpacks reverting any dim permutations.">,
    ListOption<"registerBlocking", "registerBlocking",
           "unsigned", "Register blocking tile sizes for brgemm operation.">,

  ];
}

def TppMapping : Pass<"tpp-mapping", "ModuleOp"> {
  let summary = "Map operations to be TPP compatible";
  let description = [{
    Apply collection of TPP rewriting passes to map eligble operations
    into equivalent TPP-compatible forms.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "memref::MemRefDialect",
                           "scf::SCFDialect",
                           "tensor::TensorDialect"];
  let options= [
    Option<"lowerPackUnpackWithoutTranspose", "lower-pack-unpack-without-transpose",
           "bool", /*default=*/"false",
           "Lower non-constant packs and unpacks reverting any dim permutations.">
  ];
}

def LinalgLowering : Pass<"linalg-lowering", "func::FuncOp"> {
  let summary = "Lower Linalg operations to XSMM operations.";
  let dependentDialects = ["xsmm::XsmmDialect",
                           "scf::SCFDialect",
                           "memref::MemRefDialect"];
  let options = [
    ListOption<"skipOperations", "skip-operations", "std::string",
           "Operations to skip lowering linalg-to-xsmm directly.">
  ];
}

def VectorToXSMM : Pass<"vector-to-xsmm", "ModuleOp"> {
  let summary = "Lower Vector operations to XSMM functions.";
  let dependentDialects = ["scf::SCFDialect",
			   "pdl::PDLDialect",
			   "pdl_interp::PDLInterpDialect",
                    	   "vector::VectorDialect", 
			   "func::FuncDialect",
                    	   "memref::MemRefDialect", 
			   "LLVM::LLVMDialect", 
			   "BuiltinDialect"];
}

def VectorToKernel : Pass<"vector-to-kernel", "ModuleOp"> {
  let summary = "Lower Vector operations to micro-kernel special lowering.";
  let dependentDialects = ["vector::VectorDialect",
                           "scf::SCFDialect"];
  let options= [
    Option<"vecBundleCpuTargetFeature", "target-feature", "std::string", "",
                "Target feature for a given architecture">
  ];
}

def LowLevelParallelization : Pass<"low-level-parallel", "ModuleOp"> {
  let summary = "Low level parallelization (multi-threading, AMX config).";
  let dependentDialects = ["affine::AffineDialect",
                           "arith::ArithDialect",
                           "func::FuncDialect",
                           "memref::MemRefDialect",
                           "scf::SCFDialect",
                           "xsmm::XsmmDialect",
                           "LLVM::LLVMDialect"];
  let options = [
    ListOption<"parallelTaskGrid", "parallel-task-grid",
           "unsigned", "Grid-sizes for parallel tasks.">

  ];
}

def LocalDialectsLowering : Pass<"lower-local-dialects", "ModuleOp"> {
  let summary = "Lower all local dialects (XSMM, check etc.).";
  let dependentDialects = ["affine::AffineDialect",
                           "arith::ArithDialect",
                           "func::FuncDialect",
                           "memref::MemRefDialect",
                           "check::CheckDialect",
                           "perf::PerfDialect",
                           "scf::SCFDialect",
                           "tensor::TensorDialect",
                           "xsmm::XsmmDialect",
                           "LLVM::LLVMDialect"];
}

def Postprocessing : Pass<"postprocess", "func::FuncOp"> {
  let summary = "IR postprocessing pass";
  let description = [{
    Apply various postprocessing passes such parallel loop fusion,
    buffer deallocation, general cleanup etc.
  }];
  let dependentDialects = ["bufferization::BufferizationDialect",
                           "scf::SCFDialect",
                           "memref::MemRefDialect"];
}

def Cleanup : Pass<"cleanup"> {
  let summary = "General IR cleanup e.g., canonicalization, CSE etc.";
}

def GpuPipeline : Pass<"gpu-pipeline", "ModuleOp"> {
  let summary = "Lower all eligible operations into GPU compatible IR";
  let options = [
    Option<"gpuBackend", "gpu", "std::string",
            /*default=*/"\"cuda\"",
           "Target GPU backend for lowering (cuda).">,
  ];
}

def GpuConversion : Pass<"gpu-conversion", "ModuleOp"> {
  let summary = "Convert operations to GPU";
  let description = [{
    Convert all eligble operations into generic GPU operations.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "gpu::GPUDialect",
                           "scf::SCFDialect",
                           "memref::MemRefDialect",
                           "xegpu::XeGPUDialect"];
  let options = [
    Option<"isIntel", "intel",
           "bool", /*default=*/"false",
           "Convert for Intel GPU">,
    Option<"kTile", "k-tile", "int64_t",
           /*default=*/"32",
           "GEMM tile size for reduction dimension.">,
    Option<"stages", "stages", "int64_t",
           /*default=*/"1",
           "Number of cooperative prefetch stages.">,
    ListOption<"dpasTile", "dpas-tile", "int64_t",
               "DPAS register block sizes MxNxK">,
  ];
}

def GpuToCuda : Pass<"gpu-to-cuda", "ModuleOp"> {
  let summary = "Lower generic GPU operations to CUDA backend";
  let dependentDialects = ["affine::AffineDialect",
                           "arith::ArithDialect",
                           "memref::MemRefDialect",
                           "scf::SCFDialect",
                           "gpu::GPUDialect",
                           "NVVM::NVVMDialect",
                           "nvgpu::NVGPUDialect"];
  let options = [
    Option<"gpuTriple", "triple", "std::string",
            /*default=*/"\"nvptx64-nvidia-cuda\"",
           "GPU target triple.">,
    Option<"gpuChip", "chip", "std::string",
            /*default=*/"\"sm_70\"",
           "GPU target architecture.">,
    Option<"gpuFeatures", "features", "std::string",
            /*default=*/"\"+ptx60\"",
           "GPU target features.">,
  ];
}

#endif // TPP_DIALECT_TPP_PASSBUNDLES
