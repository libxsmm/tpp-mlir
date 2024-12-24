//===- TppPasses.td ----------------------------------------*- Tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TPP_PASSES
#define TPP_DIALECT_TPP_PASSES

include "mlir/Pass/PassBase.td"

def LoadTppDialects : Pass<"load-tpp-dialects", "ModuleOp"> {
  let summary = "Pre-load all TPP-specific dialects";
  let description = [{
    Pre-load dialects that -transform-interpreter would try to load at runtime.

    The issue is that -transform-interpreter runs inside the multi-threaded
    passmanager. Hence when the interpreter dynamically tries to load dependent
    dialects this triggers an assert as loading during multi-threaded execution
    could lead to concurrency issues.
  }];
  let dependentDialects = ["xsmm::XsmmDialect",
                           "check::CheckDialect",
                           "perf::PerfDialect",
                           "omp::OpenMPDialect",
                           "async::AsyncDialect"];
}

def ConvertLinalgToXsmm : Pass<"convert-linalg-to-xsmm", "func::FuncOp"> {
  let summary = "Convert linalg to xsmm";
  let description = [{
    Convert linalg operations to XSMM operations.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "linalg::LinalgDialect",
                           "xsmm::XsmmDialect",
                           "tensor::TensorDialect"];
  let options = [
    ListOption<"skipOperations", "skip-operations", "std::string",
           "Operations to skip.">
  ];
}

def VerifyXsmmCalls : Pass<"verify-xsmm-calls", "func::FuncOp"> {
  let summary = "Verify XSMM calls (dispatch and invoke)";
  let description = [{
    Make sure XSMM dispatch and invoke call are in a consistent
    state and they do not contradict each others.
  }];
  let dependentDialects = [ "xsmm::XsmmDialect" ];
}

def ConvertLinalgToFunc : Pass<"convert-linalg-to-func", "ModuleOp"> {
  let summary = "Convert linalg to func";
  let description = [{
    Convert linalg named operations to function call using a BLAS-style
    API.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "linalg::LinalgDialect", "LLVM::LLVMDialect"];
}

def VectorizationPass : Pass<"vectorization-pass",
                                     "func::FuncOp"> {
  let summary = "convert linalg to memref/vector";
  let description = [{
    Vectorization pass to convert linalg to memref/vector instead of Xsmm.
  }];

  let dependentDialects = [ "memref::MemRefDialect", "linalg::LinalgDialect", "vector::VectorDialect" ];
}



def HoistVectorTransfers : Pass<"hoist-vector-transfer"> {
  let summary = "Hoist vector transfer operation outside of reduction and k loop";
  let description = [{
    Hoists the vector transfer read and write operations of the resultant  matrix outside the reduction and k loop for a brgemm operation. This pass should be applied after the BrgemmLinalgTiling Pass.
  }];
  let dependentDialects = [ "vector::VectorDialect", "scf::SCFDialect" ];
}



def VectorContractToOuterproduct : Pass<
    "vector-contract-to-outerproduct"> {
  let summary = "Perform outerproduct lowering of vector contraction ops";
  let dependentDialects = ["memref::MemRefDialect",
                           "scf::SCFDialect",
                           "tensor::TensorDialect",
                           "vector::VectorDialect"];
}

def VectorContractToFMA : Pass<
    "vector-contract-to-fma"> {
  let summary = "Perform vector fma lowering of vector contraction ops";
  let dependentDialects = ["memref::MemRefDialect",
                           "scf::SCFDialect",
                           "tensor::TensorDialect",
                           "vector::VectorDialect",
                           "arith::ArithDialect"];
}

def VectorContractToAMX : Pass<
    "vector-contract-to-amx"> {
  let summary = "Perform vector amx lowering of vector contraction ops";
  let dependentDialects = ["memref::MemRefDialect",
                           "scf::SCFDialect",
                           "tensor::TensorDialect",
                           "vector::VectorDialect",
                           "arith::ArithDialect",
                           "amx::AMXDialect",
                           "x86vector::X86VectorDialect"];
}


def BrgemmLinalgTiling : Pass<"tile-brgemm-linalg"> {
  let summary = "Tile bregmm  matmul and reduction dimension.";
  let description = [{
    Tiles the innermost dimensions of the batch reduce matmul operation to support perfect register allocation. Additionally, it swaps the reduction and k dimension loop. The final loop structure is as follows: M-loop->N-loop->reduction-loop->K-loop. For example: --tile-brgemm-linalg="registerBlocking=<mTileSize>, <nTileSize>".
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "memref::MemRefDialect",
                           "arith::ArithDialect"];
  let options = [
         ListOption<"registerTileShape", "registerBlocking", "unsigned", "Input for the register blocking tile shapes for a brgemm operation">,
  ];
}


def BF16DotProduct : Pass<"vector-contract-to-bf16dp"> {
   let summary = "Perform avx512bf16 dot product lowering of vector contraction operation for bf16 type with vnni layout=2";
   let description = [{
     Perform avx512bf16 dot product lowering of vector contraction operation for bf16 type with vnni layout=2.
   }];
   let dependentDialects = [ "vector::VectorDialect", "scf::SCFDialect", "x86vector::X86VectorDialect" ];
 }

def ConvertXsmmToFunc : Pass<"convert-xsmm-to-func", "ModuleOp"> {
  let summary = "Convert xsmm to func";
  let description = [{
    Convert XSMM operations to libXSMM function calls.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "xsmm::XsmmDialect",
                           "LLVM::LLVMDialect"];
}

def ConvertCheckToLoops : Pass<"convert-check-to-loops", "func::FuncOp"> {
  let summary = "Convert check to loops";
  let description = [{
    Convert check operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertPerfToLoops : Pass<"convert-perf-to-loops", "func::FuncOp"> {
  let summary = "Convert perf to loops";
  let description = [{
    Convert perf operations to SCF loops.
  }];
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertPerfToFunc : Pass<"convert-perf-to-func", "ModuleOp"> {
  let summary = "Convert perf to func";
  let description = [{
    Convert perf operations to function calls.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "math::MathDialect",
                           "memref::MemRefDialect",
                           "tensor::TensorDialect"];
}

def PackVNNI : Pass<"pack-vnni", "func::FuncOp"> {
  let summary = "Convert matmul/brgemm to vnni layout";
  let description = [{
    Relayout following matmuls and brgemm as following:
    - VNNI Matmul as: C[M][N]= A[M][K] * B[K/VNNI][N][VNNI]
    - VNNI Blocked Matmul as:
      [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb/VNNI][jb][VNNI]
    - VNNI BRGemm as: C[M][N]= A[R][M][K] * B[R][K/VNNI][N][VNNI]
  }];
  let dependentDialects = ["tensor::TensorDialect"];
}

def PackMatmul : Pass<"pack-matmul", "func::FuncOp"> {
  let summary = "Convert matmul to block layout and back";
  let description = [{
    Block a linalg.matmul
    as: [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb].
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t",
               "Blocking factor for relayout">
  ];
}

def PackConv2DNchwFchw : Pass<"pack-conv2DNchwFchw", "func::FuncOp"> {
  let summary = "Convert Conv2DNchwFchw to block layout and back";
  let description = [{
    Block Conv2DNchwFchw as: [N][BK][P][Q][bk] += [N][BC][H][W][bc] * [BK][BC][R][S][bk][bc]
                             output            += image             * filter
    Pack the image's channel with a block factor BC.
    Pack the filter's channels C and K with a block factor of BC and BK.
    Pack the output's channel K with a block factor BK.
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t",
               "Blocking factor for relayout">
  ];
}

def PackConv2DNhwcHwcf : Pass<"pack-conv2DNhwcHwcf", "func::FuncOp"> {
  let summary = "Pack and unpack Conv2DNhwcHwcf";
  let description = [{
    Pack Conv2DNhwcHwcf as [N][K'][P][Q][k] += [N][C'][H][W][c] * [K'][C'][R][S][c][k]
                           output           += image            * filter
    Pack the image and block the image's channel with a factor k.
    Pack the filter and block the filter's channels with k and c.
    Pack the output and block the output's channel with k.
  }];
  let options = [
    ListOption<"blockingFactors", "block-factors", "int64_t",
               "Blocking factor for pack and unpack operation">
  ];
}

def TileConsumerAndFuseProducers : Pass<"tile-consumer-and-fuse-producers",
                                        "func::FuncOp"> {
  let summary = "Tile consumers and fuse producers";
  let description = [{
    The pass uses `TileConsumerAndFuseProducersUsingSCFForOp` to tile the
    consumer and fuse the consumer with the producers. The fusion anchor to matmul
    or conv-like patterns allows two additional options to control how many
    producers fuse together with the latched operation and how many consumers.
    Precisely, `max-depth` controls how many producers should be considered, while
    `start-from-last-consumer` allows to move the anchor point to the last fusable
    consumer of the conv or matmul-like pattern.
  }];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes">,
    Option<"maxDepth", "max-depth", "int64_t", "5",
           "Get producers till maxDepth">,
    Option<"numIters", "num-iters", "int64_t", "3",
           "Run fusion for the given number of iterations">,
    Option<"useForAll", "use-for-all", "bool", "true", "Use parallel forAll">,
    Option<"minTileFactor", "min-tile-factor", "int64_t", "2",
           "Minimum factor between dimension size and a tile size">
  ];
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];
}

def LowerPacksAndUnPacks : Pass<"lower-packs-unpacks", "func::FuncOp"> {
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect"];
}

def LowerPacksAndUnpacksWithoutTranspose : Pass<"lower-packs-unpacks-without-transpose",
   "ModuleOp"> {
  let dependentDialects = ["linalg::LinalgDialect",
                           "tensor::TensorDialect"];
}

def RewriteConvToMatmulOrBrgemm : Pass<"rewrite-conv-to-matmul-or-brgemm",
                                       "func::FuncOp"> {
  let summary = "Rewrite Conv2DNhwcHwcfOp/Conv2DNchwFchwOp to Matmul or Brgemm.";
  let description = [{
    Rewrite a convolution to a matmul or brgemm operation.
  }];
  let options = [
    Option<"enableBrgemm", "enable-brgemm", "bool", "false",
           "Rewrite convolution to BRGEMM if possible">
  ];
  let dependentDialects = ["scf::SCFDialect", "linalg::LinalgDialect"];
}

def RewriteBatchMatmulToMatmul : Pass<"rewrite-batch-matmul-to-matmul",
                                      "func::FuncOp"> {
  let summary = "Rewrite a linalg.batch_matmul to linalg.matmul.";
  let dependentDialects = ["scf::SCFDialect", "linalg::LinalgDialect"];
}

def CombineXsmmOpPass : Pass<"combine-xsmm-op-optimization", "func::FuncOp"> {
  let summary = "Fuse brgemm-add-relu ops into a fused brgemm op";
  let description =
      [{Fuse brgemm-add-relu ops into a fused brgemm op}];

  let dependentDialects = ["xsmm::XsmmDialect"];

}

def PropagatePackUnPack : Pass<"propagate-pack-and-unpack", "func::FuncOp"> {
  let summary = "Propagate linalg.pack and linalg.unpack";
  let description = [{
    Attempt to push linalg.pack and linalg.unpack at the boundaries. Currently,
    it propagates through linalg element-wise operations. Only one operand in the
    generic must come from a linalg.pack/linalg.unpack.
  }];
}

def SimplifyAndCanonicalizePack : Pass<"simplify-pack", "func::FuncOp"> {
  let summary = "Simplify and canonicalize linalg.pack";
  let description = [{
    Apply `linalg.pack` and `linalg.unpack` canonicalization and simplification
    patterns.
  }];
}

def ConstantFoldPack : Pass<"constant-fold-pack", "ModuleOp"> {
  let summary = "Constant fold linalg.pack";
  let description = [{
    Reduce pack overhead by folding linalg.pack into constant tensors.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "tensor::TensorDialect",
                           "arith::ArithDialect"];
}

def FoldAddIntoDest : Pass<"fold-add-into-dest", "ModuleOp"> {
  let summary = "Fold linalg.add into dest of contraction op";
  let description = [{
    Replace a linalg.add with one operand the single user of a contraction,
    which has a zero-filled, "identity-mapped" destination and is dominated by
    the `other` operand, by the contraction with `other` as its dest.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "tensor::TensorDialect",
                           "arith::ArithDialect"];
}

def ElementWiseFusion : Pass<"element-wise-fusion", "func::FuncOp"> {
  let summary = "Run linalg element-wise fusion";
}

def ConvInitSimplify : Pass<"conv-init-simplify", "func::FuncOp"> {
  let summary = "Simplify initialization for convolution";
  let description = [{
    Perform a graph-rewrite to simplify initialization for a Conv2DNhwcHwcfOp
    operation. Specifically, instead of initializing the output of a convolution
    with zero and then adding the bias, initialize the output with the bias.
  }];
}

def Bufferize : Pass<"bufferize", "ModuleOp"> {
  let summary = "Bufferize tensor to memref for the entire module";
  let options = [
    Option<"dealloc", "dealloc", "bool",
            /*default=*/"true",
           "Enables automatic deallocation.">,
    Option<"testAnalysisOnly", "test-analysis-only", "bool",
            /*default=*/"false",
           "Only runs inplaceability analysis (for testing purposes only)">,
    Option<"printConflicts", "print-conflicts", "bool",
            /*default=*/"false",
           "Annotates IR with RaW conflicts. Requires test-analysis-only.">,
    Option<"duplicateFill", "duplicate-fill", "bool",
           /*default=*/"true",
           "Enable duplication of fill operation (for testing only).">
  ];
}

def DuplicateFill : Pass<"duplicate-fill", "func::FuncOp"> {
  let summary = "Duplicate fill operations";
  let description = [{
    Duplicate linalg.fill operations to avoid memref.copy after
    bufferization. This can trigger later folding of the fill.
    We duplicate only zero fill on contraction operations.
  }];
  let dependentDialects = [ "linalg::LinalgDialect" ];
}

def ConvertForAllToParallelOp : Pass<"convert-forall-to-parallel",
                                     "func::FuncOp"> {
  let summary = "Convert scf.forall to scf.parallel";
  let description = [{
    Rewrite an scf.forall to scf.parallel after bufferization.
  }];
}

def LinalgDeGeneralize : Pass<"linalg-degeneralize-generic-ops", "func::FuncOp"> {
  let summary = "Convert generic ops into named ops";
  let dependentDialects = ["linalg::LinalgDialect"];
}

def SetSPIRVCapabilities : Pass<"tpp-set-spirv-capabilities", "ModuleOp"> {
  let summary = "Set SPIR-V capabilities.";
  let options = [
    Option<"clientAPI", "client-api", "std::string",
            /*default=*/"\"opencl\"",
           "The client API to use for capabilities">,
  ];
}

def SetSPIRVAbiAttribute : Pass<"set-spirv-abi-attr", "gpu::GPUModuleOp"> {
  let summary = "Set SPIR-V ABI attribute.";
  let options = [
    Option<"clientAPI", "client-api", "std::string",
            /*default=*/"\"opencl\"",
           "The client API to use for ABI attribute">,
  ];
}

def DecomposeAggregatedOps : Pass<"decompose-aggregated-ops", "func::FuncOp"> {
  let summary = "Decompose aggregated operations.";
  let description = [{
    Decompose operations that implement the `AggregatedOpInterface`.
  }];
}

def GpuDataTransfer : Pass<"gpu-data-transfer", "func::FuncOp"> {
  let summary = "Transfer data to and from GPU.";
  let description = [{
    Make host data required by GPU kernels accessible by the device.
    It might involve data copies and/or movement.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "memref::MemRefDialect",
                           "gpu::GPUDialect"];
}

def FoldXsmmFlags : Pass<"fold-xsmm-flags", "func::FuncOp"> {
  let summary = "Attempt to fold dispatch op as flags in XSMM.";
  let description = [{
    Attempt to fold dispatch operations as flags in consumer dispatch
    operations, for example:
    ```mlir
      %alloc = memref.alloc
      xsmm.unary zero (%alloc)
      xsmm.gemm.dispatch (%alloc)
    ```
    the zero is folded as `beta_0` in `xsmm.gemm.dispatch`.
  }];
  let dependentDialects = [ "memref::MemRefDialect", "xsmm::XsmmDialect" ];
}


def SCFParallelLoopTiling : Pass<"scf-parallel-loop-tiling-pass"> {
  let summary = "Tile parallel loops";
  let options = [
    ListOption<"tileSizes", "parallel-loop-tile-sizes", "unsigned",
               "Factors to tile parallel loops by">,
    Option<"noMinMaxBounds", "no-min-max-bounds", "bool",
           /*default=*/"false",
           "Perform tiling with fixed upper bound with inbound check "
           "inside the internal loops">
  ];
  let dependentDialects = ["affine::AffineDialect", "scf::SCFDialect"];
}

def GpuInlineConstants : Pass<"gpu-inline-constants", "func::FuncOp"> {
  let summary = "Inlines constants into GPU launch.";
  let description = [{
    Inline constants into GPU launch body to reduce number of parameters
    and allow further constant propagation after kernel outlining.
    The pass should be used just before GPU kernel outlining.
  }];
  let dependentDialects = ["gpu::GPUDialect",
                           "arith::ArithDialect"];
}

def IntelAMXTileConfigInsertionPass : Pass<"intel-amx-tile-config-insertion-pass",
                                     "func::FuncOp"> {
  let summary = "Insert intel amx tile configuration xsmm calls";
  let description = [{
    Insert intel amx tile configuration xsmm calls.
  }];

  let dependentDialects = [ "memref::MemRefDialect", "xsmm::XsmmDialect" ];
}

def IntelAMXTileConfigHoistingPass : Pass<"intel-amx-tile-config-hoisting-pass",
                                     "func::FuncOp"> {
  let summary = "Hoist intel amx tile configuration invoke xsmm calls";
  let description = [{
    Run LICM on intel amx tile configuration invoke calls.
  }];

  let dependentDialects = [ "memref::MemRefDialect", "xsmm::XsmmDialect" ];
}

def LinalgConvertCompareSelectToMaximumfPass: Pass<"linalg-convert-compare-select-to-maximumf-pass",
					"func::FuncOp">{
  let summary = "Convert linalg compare-select generic operation to maximumf operation";
  let description = [{
    Convert linalg generic compare-select operation to maximumf operation.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "arith::ArithDialect"];
}

def ConvertLinalgToInplace: Pass<"convert-linalg-to-inplace",
					"func::FuncOp">{
  let summary = "Convert linalg ops to inplace operation";
  let description = [{
    Convert linalg ops to inplace update operation.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "arith::ArithDialect"];
}

def TppRunnerWrapper : Pass<"tpp-runner-wrapper", "ModuleOp">{
  let summary = "Create main function runner wrapper";
  let description = [{
    Creates a runner wrapper - maps the arguments and random initialize them.
    Optionally, inserts benchmark wrapper calling the main kernel repeatedly
    and taking measurements, or printing the result in the end.
  }];
  let dependentDialects = ["func::FuncDialect",
                           "tensor::TensorDialect",
                           "memref::MemRefDialect",
                           "gpu::GPUDialect",
                           "arith::ArithDialect",
                           "scf::SCFDialect",
                           "vector::VectorDialect",
                           "bufferization::BufferizationDialect",
                           "perf::PerfDialect"];
  let options = [
    Option<"kernelName", "kernel-name", "std::string",
            /*default=*/"\"entry\"",
           "The kernel function to be called.">,
    Option<"kernelType", "kernel-type", "std::string",
            /*default=*/"\"void\"",
           "The type of the kernel function.">,
    Option<"backend", "backend", "std::string",
            /*default=*/"\"cpu\"",
           "Kernel target device backend (cpu, cuda, intel).">,
    Option<"offloadToDevice", "offload-on-device", "bool",
            /*default=*/"true",
           "Offload kernel arguments to the target device.">,
    Option<"numBenchLoops", "bench-loops", "int64_t",
            /*default=*/"1",
           "Number of benchmarking loops.">,
    Option<"benchWarmup", "bench-warmup", "bool",
            /*default=*/"true",
           "Add benchmark warmup loops.">,
    Option<"printResult", "print", "bool",
            /*default=*/"false",
           "Print kernel results.">,
    Option<"randomSplat", "random-splat", "bool",
            /*default=*/"false",
           "Replace splat dense tensors with random values.">,
    Option<"seed", "seed", "int64_t",
            /*default=*/"0",
           "Initialization random seed.">,
    Option<"initType", "init-type", "std::string",
            /*default=*/"",
           "Initializer type (const, simple, cont, rand, normal).">,
  ];
}

def LinalgToXeGPU : Pass<"linalg-to-xegpu", "func::FuncOp"> {
  let summary = "Convert linalg dialect to XeGPU dialect.";
  let description = [{
    Lower linalg ops to XeGPU dialect.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "gpu::GPUDialect",
                           "xegpu::XeGPUDialect",
                           "scf::SCFDialect",
                           "memref::MemRefDialect",
                           "arith::ArithDialect",
                           "math::MathDialect",
                           "vector::VectorDialect"];
  let options = [
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

def FoldIntoEltwise : Pass<"fold-into-eltwise", "ModuleOp"> {
  let summary = "Fold operations into elementwise ops.";
  let description = [{
    Fold operations into Linalg elementwise ops.
    Results in linalg.generic representation.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "arith::ArithDialect",
                           "affine::AffineDialect"];
}

def SplitReductionDim : Pass<"split-reduction-dim", "func::FuncOp"> {
  let summary = "Split innermost reduction dimension.";
  let description = [{
    Split innermost reduction dimension and compute it sequentially
    using a serial loop and in-place accumulation.
  }];
  let dependentDialects = ["linalg::LinalgDialect",
                           "scf::SCFDialect",
                           "tensor::TensorDialect",
                           "memref::MemRefDialect",
                           "affine::AffineDialect",
                           "arith::ArithDialect"];
  let options = [
    Option<"tileSize", "tile", "int64_t",
           /*default=*/"0",
           "Reduction dimension tile size">,
  ];
}

def GpuVectorize : Pass<"gpu-vectorize", "ModuleOp"> {
  let summary = "Vectorize GPU kernel.";
  let description = [{
    Convert ops targeting GPU to vectorized representation.
  }];
  let dependentDialects = ["gpu::GPUDialect",
                           "scf::SCFDialect",
                           "memref::MemRefDialect",
                           "tensor::TensorDialect",
                           "math::MathDialect",
                           "arith::ArithDialect",
                           "vector::VectorDialect"];
}

#endif // TPP_DIALECT_TPP_PASSES
