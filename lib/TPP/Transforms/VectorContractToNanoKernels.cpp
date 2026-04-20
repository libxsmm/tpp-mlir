//===-VectorContractToNanoKernels.cpp -----------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/VNNIUtils.h"

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORCONTRACTTONANOKERNELS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

struct VectorContractToNanoKernels
    : public impl::VectorContractToNanoKernelsBase<VectorContractToNanoKernels> {
  using VectorContractToNanoKernelsBase::VectorContractToNanoKernelsBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    x86::populateVectorContractToFMAPatterns(patterns);

    auto cpuName = vnni::utils::getTargetArchName();
    if (cpuName == "SRF") {
       x86::populateVectorContractBF16ToFMAPatterns(patterns);
       x86::populateShuffleVectorFMAOpsPatterns(patterns);
    }

    if (cpuName == "CPX_SPR")
       x86::populateVectorContractToPackedTypeDotProductPatterns(patterns);

    if (vnni::utils::hasAMX())
      x86::populateVectorContractToAMXDotProductPatterns(patterns);

    x86::populateSinkVectorProducerOpsPatterns(patterns);

    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
