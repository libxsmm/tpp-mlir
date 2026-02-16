//===- SCFForAllLoopFlattenSFC.cpp - Flatten 2D scf.forall ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements flattening of 2D forall loops into 1D with index
// vectors.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_SCFFORALLLOOPFLATTENSFC
#define GEN_PASS_DEF_SCFFORALLLOOPFLATTENSFC
#include "TPP/Passes.h.inc"

namespace sfc {

/// Free functions for calculating generalized hilbert index from multi-dimensional indices and bounds
/// The generalized hilbert functions are adopted from: https://github.com/jakubcerveny/gilbert

void gilbertD2xyRecursive(int64_t dstIdx, int64_t curIdx,
                          int64_t &xres, int64_t &yres,
                          int64_t ax, int64_t ay,
                          int64_t bx, int64_t by) {
  const int64_t w = std::abs(ax + ay);
  const int64_t h = std::abs(bx + by);

  const int64_t x = xres;
  const int64_t y = yres;

  // Unit major direction
  const int64_t dax = (ax > 0) - (ax < 0);
  const int64_t day = (ay > 0) - (ay < 0);

  // Unit orthogonal direction
  const int64_t dbx = (bx > 0) - (bx < 0);
  const int64_t dby = (by > 0) - (by < 0);

  const int64_t di = dstIdx - curIdx;

  if (h == 1) {
    xres = x + dax * di;
    yres = y + day * di;
    return;
  }

  if (w == 1) {
    xres = x + dbx * di;
    yres = y + dby * di;
    return;
  }

  // Floor function
  int64_t ax2 = ax >> 1;
  int64_t ay2 = ay >> 1;
  int64_t bx2 = bx >> 1;
  int64_t by2 = by >> 1;

  const int64_t w2 = std::abs(ax2 + ay2);
  const int64_t h2 = std::abs(bx2 + by2);

  if ((2 * w) > (3 * h)) {
    if ((w2 & 1) && (w > 2)) {
      // Prefer even steps
      ax2 += dax;
      ay2 += day;
    }

    // Long case: split in two parts only
    int64_t nxtIdx = curIdx + std::abs((ax2 + ay2) * (bx + by));
    if ((curIdx <= dstIdx) && (dstIdx < nxtIdx)) {
      xres = x;
      yres = y;
      gilbertD2xyRecursive(dstIdx, curIdx, xres, yres, ax2, ay2, bx, by);
      return;
    }
    curIdx = nxtIdx;

    xres = x + ax2;
    yres = y + ay2;
    gilbertD2xyRecursive(dstIdx, curIdx, xres, yres, ax - ax2, ay - ay2, bx, by);
    return;
  }

  if ((h2 & 1) && (h > 2)) {
    // Prefer even steps
    bx2 += dbx;
    by2 += dby;
  }

  // Standard case: one step up, one long horizontal, one step down
  int64_t nxtIdx = curIdx + std::abs((bx2 + by2) * (ax2 + ay2));
  if ((curIdx <= dstIdx) && (dstIdx < nxtIdx)) {
    xres = x;
    yres = y;
    gilbertD2xyRecursive(dstIdx, curIdx, xres, yres, bx2, by2, ax2, ay2);
    return;
  }
  curIdx = nxtIdx;

  nxtIdx = curIdx + std::abs((ax + ay) * ((bx - bx2) + (by - by2)));
  if ((curIdx <= dstIdx) && (dstIdx < nxtIdx)) {
    xres = x + bx2;
    yres = y + by2;
    gilbertD2xyRecursive(dstIdx, curIdx, xres, yres, ax, ay, bx - bx2, by - by2);
    return;
  }
  curIdx = nxtIdx;

  xres = x + (ax - dax) + (bx2 - dbx);
  yres = y + (ay - day) + (by2 - dby);
  gilbertD2xyRecursive(dstIdx, curIdx, xres, yres, -bx2, -by2, -(ax - ax2), -(ay - ay2));
}

void gilbertD2xy(int64_t &x, int64_t &y, int64_t idx, int64_t w, int64_t h) {
  x = 0;
  y = 0;

  if (w >= h) {
    gilbertD2xyRecursive(idx, 0, x, y, w, 0, 0, h);
  } else {
    gilbertD2xyRecursive(idx, 0, x, y, 0, h, w, 0);
  }
}

} // namespace sfc
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

/// Flatten a 2D forall loop of the form:
///   scf.forall (%i, %j) in (%ub0, %ub1) {
///     // body
///   }
///
/// into:
///   %iv_i = arith.constant dense<[...]> : vector<NxI16>
///   %iv_j = arith.constant dense<[...]> : vector<NxI16>
///   scf.forall (%idx) in (%cN) {
///     %i_i16 = vector.extract %iv_i[%idx] : i16 from vector<NxI16>
///     %j_i16 = vector.extract %iv_j[%idx] : i16 from vector<NxI16>
///     %i = arith.index_cast %i_i16 : i16 to index
///     %j = arith.index_cast %j_i16 : i16 to index
///     // original body using %i and %j
///   }
///
/// where N is the total iteration count ub0 * ub1
static LogicalResult flattenForallLoop(ForallOp op, OpBuilder &builder) {
  // Only handle 2D forall loops
  if (op.getRank() != 2)
    return failure();

  Location loc = op.getLoc();
  builder.setInsertionPoint(op);

  // Get loop bounds - forall uses mixed bounds (can be values or attributes)
  SmallVector<OpFoldResult> lowerBounds = op.getMixedLowerBound();
  SmallVector<OpFoldResult> upperBounds = op.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = op.getMixedStep();

  // Helper to extract constant int from OpFoldResult
  auto getConstant = [](OpFoldResult ofr) -> std::optional<int64_t> {
    if (auto attr = dyn_cast<Attribute>(ofr)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return intAttr.getInt();
    }
    if (auto val = dyn_cast<Value>(ofr)) {
      if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
        return constOp.value();
      if (auto constOp = val.getDefiningOp<arith::ConstantIntOp>())
        return constOp.value();
    }
    return std::nullopt;
  };

  // Extract constant values
  auto lb0 = getConstant(lowerBounds[0]);
  auto lb1 = getConstant(lowerBounds[1]);
  auto ub0 = getConstant(upperBounds[0]);
  auto ub1 = getConstant(upperBounds[1]);
  auto step0 = getConstant(steps[0]);
  auto step1 = getConstant(steps[1]);

  // We need constant bounds to generate the dense vectors
  if (!lb0 || !lb1 || !ub0 || !ub1 || !step0 || !step1)
    return failure();

  // Only support unit steps for simplicity (can be relaxed if needed)
  if (*step0 != 1 || *step1 != 1)
    return failure();

  // Only support lower bounds of 0 for simplicity (can be relaxed if needed)
  if (*lb0 != 0 || *lb1 != 0)
    return failure();

  // Calculate iteration counts
  int64_t count0 = (*ub0 - *lb0) / *step0;
  int64_t count1 = (*ub1 - *lb1) / *step1;
  int64_t totalCount = count0 * count1;

  if (totalCount <= 0)
    return failure();
  if (count0 > std::numeric_limits<int16_t>::max() || count1 > std::numeric_limits<int16_t>::max())
    return failure();

  // Build the flattened index vectors
  SmallVector<int16_t> iv0Values;
  SmallVector<int16_t> iv1Values;

  for (int64_t i = 0; i < count0; ++i) {
    for (int64_t j = 0; j < count1; ++j) {
      int64_t iv0val = 0;
      int64_t iv1val = 0;
      // Instead of simple flattening, e.g. 
      //   iv0val = *lb0 + i * *step0;
      //   iv1val = *lb1 + j * *step1;
      // we are using a generalized Hilbert curve to calculate the indices, which can improve locality for certain access patterns
      tpp::sfc::gilbertD2xy(iv0val, iv1val, i * count1 + j, count0, count1);

      iv0Values.push_back(static_cast<int16_t>(iv0val));
      iv1Values.push_back(static_cast<int16_t>(iv1val));
    }
  }

  // Create dense constant vectors
  auto vectorType = VectorType::get(ArrayRef<int64_t>{totalCount}, builder.getI16Type());
  auto iv0Attr = DenseElementsAttr::get(vectorType, ArrayRef<int16_t>(iv0Values));
  auto iv1Attr = DenseElementsAttr::get(vectorType, ArrayRef<int16_t>(iv1Values));

  Value iv0Vector = arith::ConstantOp::create(builder, loc, vectorType, iv0Attr);
  Value iv1Vector = arith::ConstantOp::create(builder, loc, vectorType, iv1Attr);

  // Create the new 1D forall loop
  SmallVector<OpFoldResult> newLowerBound = {builder.getIndexAttr(*lb0)};
  SmallVector<OpFoldResult> newUpperBound = {builder.getIndexAttr(totalCount)};
  SmallVector<OpFoldResult> newStep = {builder.getIndexAttr(*step0)};

  auto newLoop = ForallOp::create(builder, loc, newLowerBound, newUpperBound, newStep,
      op.getOutputs(), op.getMapping());

  // Build the body of the new loop
  builder.setInsertionPointToStart(newLoop.getBody());

  Value idx = newLoop.getInductionVars()[0];

  // Extract the original induction variable values using vector.extract with dynamic position
  Value i = vector::ExtractOp::create(builder, loc, iv0Vector, idx);
  Value j = vector::ExtractOp::create(builder, loc, iv1Vector, idx);

  // Convert extracted values to index type
  Value iIndex = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), i);
  Value jIndex = arith::IndexCastOp::create(builder, loc, builder.getIndexType(), j);

  // Clone the original loop body
  IRMapping mapper;
  mapper.map(op.getInductionVars()[0], iIndex);
  mapper.map(op.getInductionVars()[1], jIndex);

  // Map block arguments for outputs if any
  for (auto [oldArg, newArg] : llvm::zip(op.getRegionIterArgs(), newLoop.getRegionIterArgs())) {
    mapper.map(oldArg, newArg);
  }

  for (auto &bodyOp : op.getBody()->without_terminator()) {
    builder.clone(bodyOp, mapper);
  }

  // Handle the terminator (scf.forall.in_parallel)
  // Only clone terminator contents if there are outputs (shared_outs)
  if (!op.getOutputs().empty()) {
    auto oldInParallel = cast<scf::InParallelOp>(op.getBody()->getTerminator());
    auto newInParallel = cast<scf::InParallelOp>(newLoop.getBody()->getTerminator());
    // Clone the operations inside the in_parallel block
    builder.setInsertionPointToStart(newInParallel.getBody());
    for (auto &inParallelOp : oldInParallel.getBody()->without_terminator()) {
      builder.clone(inParallelOp, mapper);
    }
  }

  // Replace uses of the old forall with the new forall results
  op.replaceAllUsesWith(newLoop);

  // Erase the original forall loop
  op.erase();

  return success();
}

namespace {

// Helper to collect innermost forall loops with exactly 2 induction variables
static void getInnermostForallLoops(Operation *rootOp,
                                     SmallVectorImpl<ForallOp> &result) {
  rootOp->walk([&](ForallOp forallOp) {
    // Check if this forall contains any nested forall ops
    bool hasNestedForall = false;
    forallOp->walk([&](ForallOp nestedOp) {
      if (nestedOp != forallOp) {
        hasNestedForall = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    if (!hasNestedForall) {
      // Only consider 2D forall loops that are innermost (no nested forall)
      if (forallOp.getRank() != 2)
        return WalkResult::advance();
      
      // Check if the forall body contains any affine.apply operations
      bool hasAffineApply = false;
      forallOp->walk([&](affine::AffineApplyOp applyOp) {
        hasAffineApply = true;
        return WalkResult::interrupt();
      });

      // Only add if no affine.apply operations found
      if (!hasAffineApply)
        result.push_back(forallOp);
    }
    return WalkResult::advance();
  });
}

struct SCFForAllLoopFlattenSFC
    : public tpp::impl::SCFForAllLoopFlattenSFCBase<SCFForAllLoopFlattenSFC> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    
    // Collect all innermost forall loops with exactly 2 induction variables
    SmallVector<ForallOp, 20> innermostForalls;
    getInnermostForallLoops(parentOp, innermostForalls);

    OpBuilder builder(&getContext());

    // Process each innermost 2D forall loop
    for (ForallOp forallOp : innermostForalls) {
      // Flatten the loop if possible; failures are silently ignored
      // (e.g., non-constant bounds, non-unit steps, etc.)
      (void)succeeded(flattenForallLoop(forallOp, builder));
    }
  }
};
} // namespace
