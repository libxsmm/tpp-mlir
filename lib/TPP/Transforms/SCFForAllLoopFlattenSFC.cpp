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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/APInt.h"
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
///   memref.global "private" constant @iv_i : memref<NxiW> = dense<[...]>
///   memref.global "private" constant @iv_j : memref<NxiW> = dense<[...]>
///   %tbl_i = memref.get_global @iv_i : memref<NxiW>
///   %tbl_j = memref.get_global @iv_j : memref<NxiW>
///   scf.forall (%idx) in (%cN) {
///     %i_iw = memref.load %tbl_i[%idx] : memref<NxiW>
///     %j_iw = memref.load %tbl_j[%idx] : memref<NxiW>
///     %i = arith.index_cast %i_iw : iW to index
///     %j = arith.index_cast %j_iw : iW to index
///     // original body using %i and %j
///   }
///
/// where N is the total iteration count ub0 * ub1 and iW is the smallest
/// integer type (i8, i16, i32 or i64) whose range can index all N tiles. The
/// lookup tables are constant memref globals (rather than vector constants)
/// so that large tile counts do not overflow the SelectionDAG backend.
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

  // Select the smallest integer type whose value range can index the entire
  // linearized tile space. The flattened loop walks totalCount tiles, so the
  // index values stored in the lookup vectors range over [0, totalCount). For
  // large matrices this can exceed the i16 range that was previously hard-coded
  // here, so we widen the element type as needed (i8 -> i16 -> i32 -> i64).
  IntegerType elemType = [&]() -> IntegerType {
    int64_t maxIndex = totalCount - 1;
    if (maxIndex <= std::numeric_limits<int8_t>::max())
      return builder.getI8Type();
    if (maxIndex <= std::numeric_limits<int16_t>::max())
      return builder.getI16Type();
    if (maxIndex <= std::numeric_limits<int32_t>::max())
      return builder.getI32Type();
    return builder.getI64Type();
  }();
  unsigned elemBitWidth = elemType.getWidth();

  // Build the flattened index vectors using the selected integer width.
  SmallVector<APInt> iv0Values;
  SmallVector<APInt> iv1Values;
  iv0Values.reserve(totalCount);
  iv1Values.reserve(totalCount);

  for (int64_t i = 0; i < count0; ++i) {
    for (int64_t j = 0; j < count1; ++j) {
      int64_t iv0val = 0;
      int64_t iv1val = 0;
      // Instead of simple flattening, e.g. 
      //   iv0val = *lb0 + i * *step0;
      //   iv1val = *lb1 + j * *step1;
      // we are using a generalized Hilbert curve to calculate the indices, which can improve locality for certain access patterns
      tpp::sfc::gilbertD2xy(iv0val, iv1val, i * count1 + j, count0, count1);

      iv0Values.emplace_back(elemBitWidth, static_cast<uint64_t>(iv0val),
                             /*isSigned=*/true);
      iv1Values.emplace_back(elemBitWidth, static_cast<uint64_t>(iv1val),
                             /*isSigned=*/true);
    }
  }

  // Store the SFC index tables as module-level constant memref globals and look
  // them up with memref.load inside the loop. A previous implementation
  // materialized these as `vector<NxiW>` constants and used vector.extract with
  // a dynamic position; for large tile counts (e.g. a 256x256 block grid =
  // 65536 tiles) lowering that dynamic extract builds a BUILD_VECTOR SDNode
  // with one operand per element, which overflows SDNode's operand limit and
  // crashes the SelectionDAG backend. Constant globals + loads scale to any
  // table size.
  auto moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return failure();

  auto tableType = MemRefType::get(ArrayRef<int64_t>{totalCount}, elemType);
  auto tensorType =
      RankedTensorType::get(ArrayRef<int64_t>{totalCount}, elemType);
  auto iv0Attr = DenseElementsAttr::get(tensorType, iv0Values);
  auto iv1Attr = DenseElementsAttr::get(tensorType, iv1Values);

  // Create the constant globals at the top of the module with unique names.
  OpBuilder globalBuilder(op->getContext());
  globalBuilder.setInsertionPointToStart(moduleOp.getBody());
  auto uniqueName = [&](StringRef base) -> std::string {
    if (!moduleOp.lookupSymbol(base))
      return base.str();
    unsigned c = 0;
    std::string name;
    do {
      name = base.str() + "_" + std::to_string(c++);
    } while (moduleOp.lookupSymbol(name));
    return name;
  };

  std::string name0 = uniqueName("__sfc_iv0");
  auto g0 = memref::GlobalOp::create(
      globalBuilder, loc, name0, globalBuilder.getStringAttr("private"),
      tableType, iv0Attr, /*constant=*/true, /*alignment=*/IntegerAttr());
  std::string name1 = uniqueName("__sfc_iv1");
  auto g1 = memref::GlobalOp::create(
      globalBuilder, loc, name1, globalBuilder.getStringAttr("private"),
      tableType, iv1Attr, /*constant=*/true, /*alignment=*/IntegerAttr());

  // Materialize references to the globals in the function before the loop.
  Value iv0Table =
      memref::GetGlobalOp::create(builder, loc, tableType, g0.getSymName());
  Value iv1Table =
      memref::GetGlobalOp::create(builder, loc, tableType, g1.getSymName());

  // Create the new 1D forall loop
  SmallVector<OpFoldResult> newLowerBound = {builder.getIndexAttr(*lb0)};
  SmallVector<OpFoldResult> newUpperBound = {builder.getIndexAttr(totalCount)};
  SmallVector<OpFoldResult> newStep = {builder.getIndexAttr(*step0)};

  auto newLoop = ForallOp::create(builder, loc, newLowerBound, newUpperBound, newStep,
      op.getOutputs(), op.getMapping());

  // Build the body of the new loop
  builder.setInsertionPointToStart(newLoop.getBody());

  Value idx = newLoop.getInductionVars()[0];

  // Look up the original induction variable values from the constant tables.
  Value i = memref::LoadOp::create(builder, loc, iv0Table, ValueRange{idx});
  Value j = memref::LoadOp::create(builder, loc, iv1Table, ValueRange{idx});

  // Convert loaded values to index type
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
