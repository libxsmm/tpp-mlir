//===- SCFParallelLoopTiling.cpp - Tiles scf.parallel ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop tiling on parallel loops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
namespace mlir {
namespace tpp {
#define GEN_PASS_DECL_SCFPARALLELLOOPTILING
#define GEN_PASS_DEF_SCFPARALLELLOOPTILING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

/// Tile a parallel loop of the form
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4, %arg5)
///
/// into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (min(%arg4*tileSize[0], %arg2-%i0)
///                                          min(%arg5*tileSize[1], %arg3-%i1))
///                                      step (%arg4, %arg5)
///
/// or, when no-min-max-bounds is true, into
///   scf.parallel (%i0, %i1) = (%arg0, %arg1) to (%arg2, %arg3)
///                                            step (%arg4*tileSize[0],
///                                                  %arg5*tileSize[1])
///     scf.parallel (%j0, %j1) = (0, 0) to (%arg4*tileSize[0],
///                                          %arg5*tileSize[1])
///                                      step (%arg4, %arg5)
///        %inbound = (%j0 * %arg4 + %i0 < %arg2) &&
///                   (%j1 * %arg5 + %i1 < %arg3)
///        scf.if (%inbound)
///          ....
///
/// where the uses of %i0 and %i1 in the loop body are replaced by
/// %i0 + j0 and %i1 + %j1.
///
/// The old loop is replaced with the new one.
void tileParallelLoop(ParallelOp op, ArrayRef<unsigned> tileSizes,
                      bool noMinMaxBounds) {
  bool useParallelOp = false;
  /* TODO, need to implement this case */
  if (!useParallelOp && noMinMaxBounds) {
    return;
  }

  OpBuilder b(op);
  auto zero = b.create<arith::ConstantIndexOp>(op.getLoc(), 0);
  SmallVector<Value, 2> tileSizeConstants;
  tileSizeConstants.reserve(op.getUpperBound().size());
  for (size_t i = 0, end = op.getUpperBound().size(); i != end; ++i) {
    if (i < tileSizes.size())
      tileSizeConstants.push_back(
          b.create<arith::ConstantIndexOp>(op.getLoc(), tileSizes[i]));
    else
      // Just pick 1 for the remaining dimensions.
      tileSizeConstants.push_back(
          b.create<arith::ConstantIndexOp>(op.getLoc(), 1));
  }

  // Create the outer loop with adjusted steps.
  SmallVector<Value, 2> newSteps;
  newSteps.reserve(op.getStep().size());
  for (auto step : llvm::zip_equal(op.getStep(), tileSizeConstants)) {
    newSteps.push_back(b.create<arith::MulIOp>(op.getLoc(), std::get<0>(step),
                                               std::get<1>(step)));
  }
  auto outerLoop = b.create<ParallelOp>(op.getLoc(), op.getLowerBound(),
                                        op.getUpperBound(), newSteps);
  b.setInsertionPointToStart(outerLoop.getBody());

  // Compute min(size, dim - offset) to avoid out-of-bounds accesses.
  auto minMap = AffineMap::get(
      /*dimCount=*/3, /*symbolCount=*/0,
      {getAffineDimExpr(/*position=*/0, b.getContext()),
       getAffineDimExpr(/*position=*/1, b.getContext()) -
           getAffineDimExpr(/*position=*/2, b.getContext())},
      b.getContext());

  // Create the inner loop with adjusted bounds.
  SmallVector<Value, 2> newBounds;
  newBounds.reserve(op.getUpperBound().size());
  bool needInboundCheck = false;
  for (auto [lowerBound, upperBound, newStep, iv, step, tileSizeConstant] :
       llvm::zip(outerLoop.getLowerBound(), outerLoop.getUpperBound(),
                 outerLoop.getStep(), outerLoop.getInductionVars(),
                 op.getStep(), tileSizeConstants)) {
    // Collect the statically known loop bounds
    auto lowerBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(lowerBound.getDefiningOp());
    auto upperBoundConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(upperBound.getDefiningOp());
    auto stepConstant =
        dyn_cast_or_null<arith::ConstantIndexOp>(step.getDefiningOp());
    auto tileSize =
        cast<arith::ConstantIndexOp>(tileSizeConstant.getDefiningOp()).value();
    // If the loop bounds and the loop step are constant and if the number of
    // loop iterations is an integer multiple of the tile size, we use a static
    // bound for the inner loop.
    if (lowerBoundConstant && upperBoundConstant && stepConstant) {
      auto numIterations = llvm::divideCeil(upperBoundConstant.value() -
                                                lowerBoundConstant.value(),
                                            stepConstant.value());
      if (numIterations % tileSize == 0) {
        newBounds.push_back(newStep);
        continue;
      }
    }

    // For InboundCheck mode, just use the variable outer step
    if (noMinMaxBounds) {
      newBounds.push_back(newStep);
      needInboundCheck = true;
      continue;
    }

    // Otherwise, we dynamically compute the bound for
    // each iteration of the outer loop.
    newBounds.push_back(
        b.create<affine::AffineMinOp>(op.getLoc(), b.getIndexType(), minMap,
                                      ValueRange{newStep, upperBound, iv}));
  }

  SmallVector<scf::ForOp> innerLoops;
  ParallelOp innerLoop;
  if (useParallelOp) {
    innerLoop = b.create<ParallelOp>(
        op.getLoc(), SmallVector<Value, 2>(newBounds.size(), zero), newBounds,
        op.getStep());
  } else {
    for (size_t i = 0; i < newBounds.size(); i++) {
      auto innerForLoop = b.create<scf::ForOp>(op.getLoc(), zero, newBounds[i],
                                               op.getStep()[i]);
      b.setInsertionPointToStart(innerForLoop.getBody());
      innerLoops.push_back(innerForLoop);
    }
  }
  if (noMinMaxBounds && needInboundCheck && useParallelOp) {
    b.setInsertionPointToStart(innerLoop.getBody());
    // Insert in-bound check
    Value inbound =
        b.create<arith::ConstantIntOp>(op.getLoc(), 1, 1);
    for (auto [outerUpperBound, outerIV, innerIV, innerStep] :
         llvm::zip(outerLoop.getUpperBound(), outerLoop.getInductionVars(),
                   innerLoop.getInductionVars(), innerLoop.getStep())) {
      // %in_bound = %in_bound &&
      //             (%inner_iv * %inner_step + %outer_iv <
      //             %outer_upper_bound)
      Value index = b.create<arith::AddIOp>(
          op.getLoc(), b.create<arith::MulIOp>(op.getLoc(), innerIV, innerStep),
          outerIV);
      Value dimInbound = b.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ult, index, outerUpperBound);
      inbound = b.create<arith::AndIOp>(op.getLoc(), inbound, dimInbound);
    }
    auto ifInbound = b.create<IfOp>(op.getLoc(),
                                    /*resultTypes*/ ArrayRef<Type>{}, inbound,
                                    /*hasElseRegion*/ false);
    ifInbound.getThenRegion().takeBody(op.getRegion());
    Block &thenBlock = ifInbound.getThenRegion().front();
    // Replace the scf.reduce terminator with an scf.yield terminator.
    Operation *reduceOp = thenBlock.getTerminator();
    b.setInsertionPointToEnd(&thenBlock);
    b.create<scf::YieldOp>(reduceOp->getLoc());
    reduceOp->erase();
    b.setInsertionPointToStart(innerLoop.getBody());
    for (const auto &ivs : llvm::enumerate(llvm::zip(
             innerLoop.getInductionVars(), outerLoop.getInductionVars()))) {
      auto newIndex = b.create<arith::AddIOp>(
          op.getLoc(), std::get<0>(ivs.value()), std::get<1>(ivs.value()));
      thenBlock.getArgument(ivs.index())
          .replaceAllUsesExcept(newIndex, newIndex);
    }
    thenBlock.eraseArguments(0, thenBlock.getNumArguments());
  } else {

    if (useParallelOp) {
      innerLoop.getRegion().takeBody(op.getRegion());
      b.setInsertionPointToStart(innerLoop.getBody());
      for (auto ivs : llvm::zip_equal(innerLoop.getInductionVars(),
                                      outerLoop.getInductionVars())) {
        Value innerIndex = std::get<0>(ivs);
        auto newIndex = b.create<arith::AddIOp>(op.getLoc(), std::get<0>(ivs),
                                                std::get<1>(ivs));
        innerIndex.replaceAllUsesExcept(newIndex, newIndex);
      }
    }

    else {
      b.setInsertionPointToStart(innerLoops[innerLoops.size() - 1].getBody());
      IRMapping mapper;
      // Copy instruction by instruction and replace uses
      for (auto opItr = op.getRegion().op_begin();
           opItr != op.getRegion().op_end(); opItr++) {
        if (!dyn_cast<scf::ReduceOp>(*opItr)) {
          auto instr = b.clone(*opItr, mapper);
          opItr->replaceAllUsesWith(instr);
        }
      }
      b.setInsertionPointToStart(innerLoops[innerLoops.size() - 1].getBody());
      SmallVector<arith::AddIOp> indices;

      for (size_t index = 0; index < innerLoops.size(); index++) {
        Value innerIndex = innerLoops[index].getInductionVar();
        auto newIndex = b.create<arith::AddIOp>(
            op.getLoc(), innerIndex, outerLoop.getInductionVars()[index]);
        // Cache the newly created index variables
        indices.push_back(newIndex);
      }

      auto inductionVars =
          dyn_cast<ParallelOp>(op.getRegion().getParentOp()).getInductionVars();

      // Replace induction variables with newly created index variables
      for (size_t i = 0; i < inductionVars.size(); i++) {
        auto inductionVar = inductionVars[i];
        inductionVar.replaceAllUsesWith(indices[i]);
      }
    }
  }

  op.erase();
}

namespace {
struct SCFParallelLoopTiling
    : public tpp::impl::SCFParallelLoopTilingBase<SCFParallelLoopTiling> {
  SCFParallelLoopTiling(){};
  SCFParallelLoopTiling(ArrayRef<unsigned> tileSizes,
                        bool noMinMaxBounds = false) {
    this->tileSizes = tileSizes;
    this->noMinMaxBounds = noMinMaxBounds;
  };

  SCFParallelLoopTiling(const tpp::SCFParallelLoopTilingOptions &options) {
    tileSizes = options.tileSizes;
    noMinMaxBounds = options.noMinMaxBounds;
  };

  void runOnOperation() override {
    for (auto tileSize : tileSizes)
      if (tileSize == 0) {
        mlir::emitError(mlir::UnknownLoc::get(&Pass::getContext()),
                        "tile size cannot be 0");
        return signalPassFailure();
      }
    auto *parentOp = getOperation();
    SmallVector<ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(parentOp, innermostPloops);
    for (ParallelOp ploop : innermostPloops) {
      // FIXME: Add reduction support.
      if (ploop.getNumReductions() == 0)
        tileParallelLoop(ploop, tileSizes, noMinMaxBounds);
    }
  }
};
} // namespace
