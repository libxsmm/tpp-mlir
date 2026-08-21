//===- ConvertToStreamingStore.cpp -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTTOSTREAMINGSTORE
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Walk view-like ops back to the underlying root buffer that the given memref
// value aliases.
static Value getRootMemref(Value memref) {
  while (auto view = memref.getDefiningOp<ViewLikeOpInterface>())
    memref = view.getViewSource();
  return memref;
}

// Returns true if the root buffer, and every memref value that aliases it
// through view-like ops, is never read. Conservatively returns false whenever
// a user cannot be proven read-free (unknown memory effects, escaping uses).
static bool isWriteOnly(Value root) {
  SmallVector<Value> worklist{root};
  SmallPtrSet<Value, 8> visited;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    for (Operation *user : cur.getUsers()) {
      // A view-like user only creates a new alias; keep tracing it.
      if (isa<ViewLikeOpInterface>(user)) {
        for (Value res : user->getResults())
          if (isa<MemRefType>(res.getType()))
            worklist.push_back(res);
        continue;
      }
      // Any user with unknown effects might read the buffer.
      auto memEffects = dyn_cast<MemoryEffectOpInterface>(user);
      if (!memEffects)
        continue;

      SmallVector<MemoryEffects::EffectInstance> effects;
      memEffects.getEffects(effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (!isa<MemoryEffects::Read>(effect.getEffect()))
          continue;
        // A read we cannot pin to another value might touch this buffer.
        Value effectValue = effect.getValue();
        if (!effectValue || effectValue == cur)
          return false;
      }
    }
  }
  return true;
}

// Rewrites a vector.transfer_write whose destination buffer is never read back
// into a vector.store with the nontemporal = true attribute.
struct TransferWriteToStreamingStore
    : OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // A plain vector.store can only replace a write into a memref.
    if (!isa<MemRefType>(writeOp.getShapedType()))
      return rewriter.notifyMatchFailure(writeOp,
                                         "expects a memref destination");

    // Masked or non-contiguous transfers cannot be lowered to a plain store.
    if (writeOp.getMask())
      return rewriter.notifyMatchFailure(writeOp, "masked transfer write");
    if (!writeOp.getPermutationMap().isMinorIdentity())
      return rewriter.notifyMatchFailure(writeOp,
                                         "non minor-identity permutation map");
    if (!llvm::all_of(writeOp.getInBoundsValues(),
                      [](bool inBounds) { return inBounds; }))
      return rewriter.notifyMatchFailure(writeOp,
                                         "out-of-bounds transfer write");

    // Only rank-1 vector.store ops lower to LLVM; expect the transfers to have
    // already been flattened.
    if (writeOp.getVectorType().getRank() != 1)
      return rewriter.notifyMatchFailure(writeOp, "expects a 1-D vector write");

    // Only stream stores whose destination buffer is never read back.
    if (!isWriteOnly(getRootMemref(writeOp.getBase())))
      return rewriter.notifyMatchFailure(
          writeOp, "destination buffer is read; not a streaming store");

    rewriter.replaceOpWithNewOp<vector::StoreOp>(
        writeOp, writeOp.getVector(), writeOp.getBase(), writeOp.getIndices(),
        /*nontemporal=*/true);
    return success();
  }
};

struct ConvertToStreamingStore
    : public tpp::impl::ConvertToStreamingStoreBase<ConvertToStreamingStore> {
  using ConvertToStreamingStoreBase::ConvertToStreamingStoreBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TransferWriteToStreamingStore>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
