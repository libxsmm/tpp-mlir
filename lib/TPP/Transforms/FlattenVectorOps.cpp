//===-FlattenVectorOps.cpp ----------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_FLATTENVECTOROPS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

struct FlattenVectorOps
    : public impl::FlattenVectorOpsBase<FlattenVectorOps> {
  using FlattenVectorOpsBase::FlattenVectorOpsBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    vector::populateFlattenVectorTransferPatterns(patterns);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace tpp
} // namespace mlir
