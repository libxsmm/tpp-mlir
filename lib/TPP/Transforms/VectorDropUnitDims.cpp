//===-VectorDropUnitDims.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORDROPUNITDIMS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

struct VectorDropUnitDims
    : public impl::VectorDropUnitDimsBase<VectorDropUnitDims> {
  using VectorDropUnitDimsBase::VectorDropUnitDimsBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    vector::populateDropUnitDimWithShapeCastPatterns(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace tpp
} // namespace mlir
