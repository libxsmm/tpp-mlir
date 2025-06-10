//===-RegisterUnroll.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REGISTERUNROLL
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

template <typename IntType>
static SmallVector<IntType> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector(llvm::map_range(
      arrayAttr.getAsRange<IntegerAttr>(),
      [](IntegerAttr attr) { return static_cast<IntType>(attr.getInt()); }));
}

// Returns register unroll shapes for innermost dims: [M, N, K]
static SmallVector<int64_t> getRegisterGemmUnroll(Operation *op) {
  auto res = dlti::query(op, {"CPU", "reg_gemm_unroll"});
  if (failed(res))
    return {};
  auto vals = dyn_cast<ArrayAttr>(*res);
  if (!vals)
    return {};
  return extractVector<int64_t>(vals);
}

static std::optional<unsigned> mapIteratorToDim(AffineMap map,
                                                unsigned iterPos) {
  return map.getResultPosition(getAffineDimExpr(iterPos, map.getContext()));
}

static std::optional<SmallVector<int64_t>> getContractionShape(vector::ContractionOp contractOp){
  SmallVector<int64_t> regUnroll = getRegisterGemmUnroll(contractOp);
  if (regUnroll.size() != 3)
    return std::nullopt;

  if (contractOp.getKind() != vector::CombiningKind::ADD)
    return std::nullopt;

  SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(indexingMaps);
  if (failed(dims))
    return std::nullopt;
  // Constrain support to only one M and one N dimension.
  // TODO: Generalize when getting unroll shape logic is smarter.
  if (dims->m.size() != 1 || dims->n.size() != 1)
    return std::nullopt;

  // TODO: Generalize 'isInVnniLayout' to work on vector.
  //       For now assume that BF16 implies VNNI layout.
  TypedValue<VectorType> lhs = contractOp.getLhs();
  VectorType lhsTy = lhs.getType();
  unsigned rankLhs = lhsTy.getRank();
  bool isVnni = lhsTy.getElementType().isBF16() && rankLhs >= 3;

  // Find the innermost reduction dimension for unrolling.
  // In case of VNNI, take the second inner dimension as the VNNI
  // dimension is guaranteed to be the innermost.
  std::optional<unsigned> dimVnni = std::nullopt;
  AffineMap mapLhs = indexingMaps[0];
  if (isVnni)
    dimVnni =
        dyn_cast<AffineDimExpr>(mapLhs.getResult(rankLhs - 1)).getPosition();
  unsigned dimK = 0;
  unsigned innermostDim = 0;
  for (auto pos : dims->k) {
    auto dimPos = mapIteratorToDim(mapLhs, pos);
    assert(dimPos && "failed to map iterator to dim");
    if (*dimPos > innermostDim && (!isVnni || pos != *dimVnni)) {
      innermostDim = *dimPos;
      dimK = pos;
    }
  }

  // The register unrolling is applied to the remaining innermost dimensions.
  // NOTE: It is assumed that all batch-reduce dimensions are outer w.r.t.
  //       K-dim reduce dimension.
  //
  // Scalarize batch dimensions - it is a fallback option, ideally
  // user should've preprocessed batch dimension earlier or it might have
  // remained present as a unit dimension. Same for batch-reduce dims.
  // Do not unroll the VNNI dimension if present.
  SmallVector<int64_t> unrollShapes(contractOp.getIteratorTypes().size(), 1);
  if (isVnni)
    unrollShapes[*dimVnni] = lhsTy.getShape().back();
  unrollShapes[dims->m[0]] = regUnroll[0];
  unrollShapes[dims->n[0]] = regUnroll[1];
  unrollShapes[dimK] = regUnroll[2];

  return unrollShapes;
}

void selectUnrollSizes(Operation *op) {
  MLIRContext *ctx = op->getContext();

  auto contractOp = dyn_cast<vector::ContractionOp>(op);
  if (!contractOp)
    return;

  std::optional<SmallVector<int64_t>> unrollShape =
      getContractionShape(contractOp);
  if (!unrollShape)
    return;

  std::string unrollAttrName = "unroll_shape";
  contractOp->setDiscardableAttr(unrollAttrName,
                                 DenseI64ArrayAttr::get(ctx, *unrollShape));

  // Map contraction unroll shape to its operands.
  SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
  auto getOperandUnrollShape = [&](AffineMap map) -> SmallVector<int64_t> {
    SmallVector<int64_t> operandShape;
    for (AffineExpr dim : map.getResults()) {
      unsigned dimPos = dyn_cast<AffineDimExpr>(dim).getPosition();
      operandShape.push_back((*unrollShape)[dimPos]);
    }
    return operandShape;
  };

  // Only propagate layout to reads and writes for now.
  // All other ops will default to extract/insert vector slices.
  if (auto read = dyn_cast_or_null<vector::TransferReadOp>(
          contractOp.getLhs().getDefiningOp())) {
    SmallVector<int64_t> shape = getOperandUnrollShape(indexingMaps[0]);
    read->setDiscardableAttr(unrollAttrName,
                             DenseI64ArrayAttr::get(ctx, shape));
  }
  if (auto read = dyn_cast_or_null<vector::TransferReadOp>(
          contractOp.getRhs().getDefiningOp())) {
    SmallVector<int64_t> shape = getOperandUnrollShape(indexingMaps[1]);
    read->setDiscardableAttr(unrollAttrName,
                             DenseI64ArrayAttr::get(ctx, shape));
  }

  // Set the same unroll for accumulator and writes.
  SmallVector<int64_t> accUnroll = getOperandUnrollShape(indexingMaps[2]);
  if (auto read = dyn_cast_or_null<vector::TransferReadOp>(
          contractOp.getAcc().getDefiningOp())) {
    read->setDiscardableAttr(unrollAttrName,
                             DenseI64ArrayAttr::get(ctx, accUnroll));
  }
  for (Operation *user : contractOp->getUsers()) {
    if (auto write = dyn_cast_or_null<vector::TransferWriteOp>(user)) {
      write->setDiscardableAttr(unrollAttrName,
                                DenseI64ArrayAttr::get(ctx, accUnroll));
    }
  }
}

static std::optional<SmallVector<int64_t>> getVectorShape(Operation *op) {
  auto unrollAttr = dyn_cast_or_null<DenseI64ArrayAttr>(
      op->getDiscardableAttr("unroll_shape"));
  if (!unrollAttr)
    return std::nullopt;
  return SmallVector<int64_t>(unrollAttr.asArrayRef());
}

struct RegisterUnroll
    : public impl::RegisterUnrollBase<RegisterUnroll> {
  using RegisterUnrollBase::RegisterUnrollBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    // TODO: Replace with proper layout and propagation analysis like
    //       'SparseBackwardDataFlowAnalysis'.
    getOperation()->walk([&](Operation *op) { selectUnrollSizes(op); });

    // TODO: Propagate and unroll through loop iter_args.
    RewritePatternSet patterns(ctx);
    vector::populateVectorUnrollPatterns(
        patterns,
        vector::UnrollVectorOptions().setNativeShapeFn(getVectorShape));

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace tpp
} // namespace mlir
