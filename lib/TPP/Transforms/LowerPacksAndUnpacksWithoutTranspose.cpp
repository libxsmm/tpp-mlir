//===- LowerPacksAndUnpacksWithoutTranspose.cpp ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include <numeric>
using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LOWERPACKSANDUNPACKSWITHOUTTRANSPOSE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

/// Wrapper around linalg::lowerPack which undoes the transpose that might have
/// happened. Single user genericOp's indexing_maps is corrected accordingly.
void lowerPackAndFoldTranspose(linalg::PackOp packOp,
                               linalg::GenericOp genericOp, uint operandIdx,
                               PatternRewriter &rewriter) {
  mlir::PackingMetadata metadata;
  auto packInversionPerm = linalg::getPackInverseDestPerm(packOp, metadata);

  auto res = linalg::lowerPack(rewriter, packOp);

  if (res->transposeOp) {
    // Forget about the permutation of the dims on expandShapeOp.
    rewriter.replaceAllOpUsesWith(res->transposeOp, res->expandShapeOp);

    // Invert corresponding transposed accesses by the single user, genericOp.
    auto indexingMaps = genericOp.getIndexingMapsArray();
    auto packInverseMap =
        AffineMap::getPermutationMap(packInversionPerm, rewriter.getContext());
    auto normalizedIndexingMap =
        packInverseMap.compose(indexingMaps[operandIdx]);

    SmallVector<Attribute> maps = llvm::to_vector(genericOp.getIndexingMaps());
    maps[operandIdx] = AffineMapAttr::get(normalizedIndexingMap);
    genericOp.setIndexingMapsAttr(ArrayAttr::get(rewriter.getContext(), maps));
  }
}

struct LowerPackOnInputsFoldingTranspose
    : public OpRewritePattern<linalg::GenericOp> {
  // Is only called with single-user packOp operands, so callback can always
  // find the (use by the) linalg.generic that is the target of the pattern.
  using ControlFn = std::function<bool(linalg::PackOp)>;
  ControlFn controlFn;

  LowerPackOnInputsFoldingTranspose(MLIRContext *context,
                                    ControlFn controlFn = nullptr,
                                    PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    bool modifiedAnOperand = false;
    for (auto &&[operandIdx, inOperand] :
         llvm::enumerate(genericOp.getInputs())) {
      auto packOp =
          dyn_cast_if_present<linalg::PackOp>(inOperand.getDefiningOp());

      if (!packOp || !packOp->hasOneUse() || (controlFn && !controlFn(packOp)))
        continue;

      lowerPackAndFoldTranspose(packOp, genericOp, operandIdx, rewriter);

      modifiedAnOperand = true;
    }

    return modifiedAnOperand ? success() : failure();
  }
};

struct LowerPackUnpackOnOutputFoldingTranspose
    : public OpRewritePattern<linalg::GenericOp> {
  // Is only called with single-user packOp operands, so callback can always
  // find the (use by the) linalg.generic that is the target of the pattern.
  using ControlFn = std::function<bool(linalg::PackOp, linalg::UnPackOp)>;
  ControlFn controlFn;

  LowerPackUnpackOnOutputFoldingTranspose(MLIRContext *context,
                                          ControlFn controlFn = nullptr,
                                          PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit), controlFn(std::move(controlFn)) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    bool modifiedAnOperand = false;
    size_t numInputs = genericOp.getInputs().size();
    for (auto &&[outOperandIdx, outOperand] :
         llvm::enumerate(genericOp.getOutputs())) {
      size_t operandIdx = numInputs + outOperandIdx;
      auto result = genericOp->getResult(outOperandIdx);

      if (!result.hasOneUse())
        continue;

      auto packOp =
          dyn_cast_if_present<linalg::PackOp>(outOperand.getDefiningOp());
      auto unpackOp =
          llvm::dyn_cast<linalg::UnPackOp>(*(result.getUsers().begin()));

      if (!packOp || !packOp->hasOneUse() || !unpackOp)
        continue;

      // Normalize empty outer_dims_perm to its corresponding identity perm.
      auto packOuterDimsPerm = SmallVector<long>(packOp.getOuterDimsPerm());
      if (packOuterDimsPerm.empty()) {
        packOuterDimsPerm =
            SmallVector<long>(packOp.getSource().getType().getRank());
        std::iota(packOuterDimsPerm.begin(), packOuterDimsPerm.end(), 0);
      }
      auto unpackOuterDimsPerm = SmallVector<long>(unpackOp.getOuterDimsPerm());
      if (unpackOuterDimsPerm.empty()) {
        unpackOuterDimsPerm =
            SmallVector<long>(unpackOp.getResult().getType().getRank());
        std::iota(unpackOuterDimsPerm.begin(), unpackOuterDimsPerm.end(), 0);
      }

      if (unpackOp.getInnerDimsPos() != packOp.getInnerDimsPos() ||
          packOuterDimsPerm != unpackOuterDimsPerm ||
          (controlFn && !controlFn(packOp, unpackOp)))
        continue;

      auto unpackDest = unpackOp.getDest();
      bool destHasStaticShape = unpackDest.getType().hasStaticShape();

      lowerPackAndFoldTranspose(packOp, genericOp, operandIdx, rewriter);
      auto res = linalg::lowerUnPack(rewriter, unpackOp);

      // Set genericOp's result type to the adjusted type of the out parameter.
      result.setType(genericOp.getOperand(operandIdx).getType());

      if (auto transposeOp = res->transposeOp) {
        // Forget about the transpose introduced by lowerUnPack.
        rewriter.replaceAllOpUsesWith(transposeOp, transposeOp.getInput());
      }

      // lowerUnPack introduces a copy to maintain DPS w.r.t. unpackOp's dest.
      // As we ignore permutations and, in the static case, don't do padding,
      // we know the underlying buffer will be used as is and hence we do not
      // need to specify a dest to update into.
      auto extractSliceOp = res->extractSliceOp;
      if (destHasStaticShape && extractSliceOp && extractSliceOp->hasOneUse()) {
        auto copyOp =
            dyn_cast<linalg::CopyOp>(*extractSliceOp->getUsers().begin());
        if (copyOp && copyOp.getOutputs()[0] == unpackDest) {
          rewriter.replaceAllOpUsesWith(copyOp, copyOp.getInputs()[0]);
        }
      }
      modifiedAnOperand = true;
    }

    return modifiedAnOperand ? success() : failure();
  }
};

struct LowerPacksAndUnpacksWithoutTranspose
    : public tpp::impl::LowerPacksAndUnpacksWithoutTransposeBase<
          LowerPacksAndUnpacksWithoutTranspose> {

  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<LowerPackOnInputsFoldingTranspose>(
        ctx, [](linalg::PackOp packOp) {
          // Only lower packOps whose argument is not a constant.
          return !llvm::dyn_cast_if_present<arith::ConstantOp>(
              packOp.getOperand(0).getDefiningOp());
        });
    patterns.add<LowerPackUnpackOnOutputFoldingTranspose>(ctx);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
