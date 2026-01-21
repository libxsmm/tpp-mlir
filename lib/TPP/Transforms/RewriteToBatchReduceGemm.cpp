//===- RewriteToBatchReduceGemm.cpp ------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REWRITEBATCHMATMULTOMATMUL
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "mlir-rewrite-to-brgemm"

// Look for [p ... p] brgemm[r p p r]
static LogicalResult checkStructure(linalg::LinalgOp linalgOp) {
  SmallVector<utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (iteratorTypes.size() < 4)
    return failure();
  size_t size = iteratorTypes.size() - 1;
  bool match = linalg::isReductionIterator(iteratorTypes[size]) &&
               linalg::isParallelIterator(iteratorTypes[size - 1]) &&
               linalg::isParallelIterator(iteratorTypes[size - 2]) &&
               linalg::isReductionIterator(iteratorTypes[size - 3]);
  if (!match)
    return failure();
  size = size - /*BRGEMM loops=*/3;
  size_t idx = 0;
  while (idx < size) {
    if (!linalg::isParallelIterator(iteratorTypes[idx++]))
      return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << __func__ << " OK\n");
  return success();
}

// Check if the operand is an input to linalgOp.
static bool isInputOperand(linalg::LinalgOp linalgOp, OpOperand &operand) {
  return operand.getOperandNumber() < linalgOp.getNumDpsInputs();
}

// Check if the operand is an output to linalgOp.
[[maybe_unused]] static bool isOutputOperand(linalg::LinalgOp linalgOp, OpOperand &operand) {
  return !isInputOperand(linalgOp, operand);
}

// Check the access pattern that must match the one expected for BRGEMM.
// We extract the 3 innermost dimensions for the input and the 2 innermost
// dimensions for the output. We then check that they equal:
// [p3, p4] += [r1, p3, r2] * [r1, r2, p4].
static LogicalResult checkAccessPatterns(linalg::LinalgOp linalgOp) {
  SmallVector<AffineMap> maps;
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    AffineMap map = linalgOp.getMatchingIndexingMap(&operand);
    if (isInputOperand(linalgOp, operand)) {
      if (map.getNumResults() < 3)
        return failure();
      maps.push_back(map.getMinorSubMap(3));
    } else {
      assert(isOutputOperand(linalgOp, operand));
      if (map.getNumResults() < 2)
        return failure();
      maps.push_back(map.getMinorSubMap(2));
    }
  }
  SmallVector<AffineMap> compressedDimMaps = compressUnusedDims(maps);
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [&](MapList m) {
    return AffineMap::inferFromExprList(m, linalgOp.getContext());
  };
  AffineExpr r1, p3, p4, r2;
  SmallVector<AffineMap> expectedMaps;

  bindDims(linalgOp.getContext(), r1, p3, p4, r2);
  expectedMaps = infer({{r1, p3, r2}, {r1, r2, p4}, {p3, p4}});

  if (compressedDimMaps != expectedMaps)
    return failure();
  LLVM_DEBUG(llvm::dbgs() << __func__ << " OK\n");
  return success();
}

// single region block with add, mul and linalg::yield.
static LogicalResult checkBody(linalg::LinalgOp linalgOp) {
  // clang-format off
  using namespace mlir::structured_match;
  auto hasRightOpChain =
    StructuredOpMatcher::make<linalg::LinalgOp>()
      .region(MatchOne(0), WithOpChain<KindMul, KindAdd>(
                                     /*captures=*/nullptr));
  // clang-format on
  if (!hasRightOpChain.match(linalgOp))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << __func__ << " OK\n");
  return success();
}

static FailureOr<SmallVector<Value>>
getSlicedOperands(OpBuilder &builder, Location loc, ValueRange localIvs,
                  linalg::LinalgOp linalgOp, ValueRange valuesToUse,
                  bool isVNNILoop = false) {
  assert(linalgOp->getNumOperands() == 3 && "expect 3 input/output operands");
  assert(linalgOp.getDpsInputOperands().size() == 2 &&
         "expect 2 input operands");

  SmallVector<Value> slicedOperands;
  for (OpOperand *operand : linalgOp.getDpsInputOperands()) {
    FailureOr<Value> slicedOperand;
    // In VNNI layout the second operand has size '4'.
    if (isVNNILoop && operand->getOperandNumber() == 1) {
      slicedOperand = linalgx::utils::getSliceOperand(
          builder, operand, linalgOp, localIvs, valuesToUse, 4);
    } else {
      slicedOperand = linalgx::utils::getSliceOperand(
          builder, operand, linalgOp, localIvs, valuesToUse, 3);
    }
    if (failed(slicedOperand))
      return failure();

    slicedOperands.push_back(*slicedOperand);
  }
  for (OpOperand &operand : linalgOp.getDpsInitsMutable()) {
    FailureOr<Value> slicedOperand = linalgx::utils::getSliceOperand(
        builder, &operand, linalgOp, localIvs, valuesToUse, 2);
    if (failed(slicedOperand))
      return failure();
    slicedOperands.push_back(*slicedOperand);
  }
  return slicedOperands;
}

// Walk `ivs` and return the outermost loop.
static Operation *getOuterMostLoop(ArrayRef<Value> ivs) {
  // See: `Tiling.cpp` in Linalg/Transforms.
  // Gather the newly created loops and return them with the new op.
  // TODO: make utils:
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (Value iv : ivs) {
    if (isa<BlockArgument>(iv)) {
      loops.push_back(cast<BlockArgument>(iv).getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      loops.push_back(nullptr);
    }
  }

  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop))
      break;

  return outermostLoop;
}

// Rewrite a generic operation to BRGEMM. The following conditions apply:
// 1. The generic has a single region. The region performs a scalar GEMM
// operation.
// 2. The innermost dimensions for the generic must be [r, p, p, r]. r =
// reduction p = parallel. Outermost dimensions must be parallel.
// 3. Access pattern must be [p3, p4] += [r1, p3, r2] * [r1, r2, p4].
FailureOr<SmallVector<Value>>
mlir::linalgx::rewriteToBRGemmOp(RewriterBase &rewriter,
                                 linalg::LinalgOp linalgOp) {

  if (!isa<linalg::GenericOp>(linalgOp))
    return rewriter.notifyMatchFailure(linalgOp, "expects a linalg.generic");

  if (failed(checkBody(linalgOp)))
    return rewriter.notifyMatchFailure(linalgOp, "expects a GEMM-like body");

  if (failed(checkStructure(linalgOp))) {
    return rewriter.notifyMatchFailure(
        linalgOp, "failed to match structurally with BRGEMM");
  }

  if (failed(checkAccessPatterns(linalgOp))) {
    return rewriter.notifyMatchFailure(
        linalgOp, "failed to match BRGEMM access patterns");
  }

  // Materialize outer loops.
  unsigned upTo = linalgOp.getNumLoops() - /*BRGEMM loops=*/4;

  FailureOr<SmallVector<Range>> maybeLoopRanges =
      linalgx::utils::getLoopsToMaterialize(rewriter, linalgOp, upTo);
  if (failed(maybeLoopRanges))
    return failure();
  SmallVector<Range> loopRanges = *maybeLoopRanges;

  // Replace linalgOp with BRGEMM.
  SmallVector<Value> ivs, tensorResults;
  auto brgemmBuilder = [&](OpBuilder &builder, Location loc,
                           ValueRange localIvs,
                           ValueRange operandValuesToUse) -> scf::ValueVector {
    assert(operandValuesToUse.size() ==
               static_cast<size_t>(linalgOp->getNumOperands()) &&
           "expect the number of operands and inputs and outputs to match");
    ivs.assign(localIvs.begin(), localIvs.end());
    FailureOr<SmallVector<Value>> maybeSlicedOperands =
        getSlicedOperands(builder, loc, localIvs, linalgOp, operandValuesToUse);
    if (failed(maybeSlicedOperands)) {
      assert(0 && "failed to generate loops");
      return {};
    }
    SmallVector<Value> slicedOperands = *maybeSlicedOperands;
    assert(slicedOperands.size() == 3 && "expect three operands");

    linalg::BatchReduceMatmulOp brgemm =
        (linalgOp.hasPureTensorSemantics())
            ? linalg::BatchReduceMatmulOp::create(builder, 
                  loc, slicedOperands[2].getType(),
                  ValueRange{slicedOperands[0], slicedOperands[1]},
                  slicedOperands[2])
            : linalg::BatchReduceMatmulOp::create(builder, 
                  loc, ValueRange{slicedOperands[0], slicedOperands[1]},
                  slicedOperands[2]);
    tensorResults =
        (loopRanges.empty())
            ? brgemm->getResults()
            : insertSlicesBack(builder, loc, linalgOp, slicedOperands,
                               brgemm->getResults());

    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };

  if (linalgOp.hasPureBufferSemantics()) {
    // TODO: (lorenzo) this is legacy. It will be removed from linalg too.
    // Use the tiling interface.
    linalg::GenerateLoopNest<scf::ParallelOp>::doit(
        rewriter, linalgOp.getLoc(), loopRanges, linalgOp,
        linalgOp.getIteratorTypesArray(), brgemmBuilder);
  } else {
    linalg::GenerateLoopNest<scf::ForOp>::doit(
        rewriter, linalgOp.getLoc(), loopRanges, linalgOp,
        linalgOp.getIteratorTypesArray(), brgemmBuilder);
  }

  Operation *outermostLoop = getOuterMostLoop(ivs);
  if (outermostLoop && isa<scf::ForOp>(outermostLoop)) {
    outermostLoop->setAttr(linalgx::utils::kLoopParallel,
                           rewriter.getStringAttr(linalgx::utils::kLoopRoot));
  }

  rewriter.replaceOp(linalgOp, outermostLoop ? outermostLoop->getResults()
                                             : tensorResults);
  return outermostLoop ? outermostLoop->getResults() : tensorResults;
}
