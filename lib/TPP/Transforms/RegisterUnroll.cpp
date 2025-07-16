//===- RegisterUnroll.cpp ----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REGISTERUNROLL
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Attribute name used as marker to assign vector unroll shapes.
constexpr const static llvm::StringLiteral unrollShapeAttrName = "unroll_shape";

// Extracts a vector of integers out of an array attribute.
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

// Returns position of a dimension corresponding to the given iteration map
// and an iterator.
static std::optional<unsigned> mapIteratorToDimPos(AffineMap map,
                                                   unsigned iterPos) {
  return map.getResultPosition(getAffineDimExpr(iterPos, map.getContext()));
}

// Returns unrolling shape for a contraction op or nullopt if no unrolling
// should take place.
static std::optional<SmallVector<int64_t>>
getContractionUnrollShape(vector::ContractionOp contractOp,
                          SmallVector<int64_t> userShape) {
  SmallVector<int64_t> regUnroll = userShape;
  if (regUnroll.empty())
    regUnroll = getRegisterGemmUnroll(contractOp);
  // Invalid unrolling shape.
  if (regUnroll.size() != 3)
    return std::nullopt;
  // Invalid contraction kind.
  if (contractOp.getKind() != vector::CombiningKind::ADD)
    return std::nullopt;

  SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
  FailureOr<linalg::ContractionDimensions> dims =
      linalg::inferContractionDims(indexingMaps);
  assert(succeeded(dims) && "failed to infer contraction dims");
  // Constrain support to only one M and one N dimension.
  // TODO: Generalize when getting unroll shape logic is smarter.
  if (dims->m.size() != 1 || dims->n.size() != 1)
    return std::nullopt;

  VectorType lhsTy = contractOp.getLhs().getType();
  unsigned rankLhs = lhsTy.getRank();
  AffineMap mapLhs = indexingMaps[0];

  // Find the innermost reduction dimension for unrolling.
  // In case of VNNI, take the second inner dimension as the VNNI
  // dimension is guaranteed to be the innermost.
  bool isVnni = vnni::utils::isInVnniLayout(contractOp, indexingMaps);
  std::optional<unsigned> dimVnni = std::nullopt;
  if (isVnni)
    dimVnni =
        dyn_cast<AffineDimExpr>(mapLhs.getResult(rankLhs - 1)).getPosition();
  unsigned dimK = 0;
  unsigned innermostDim = 0;
  for (auto pos : dims->k) {
    auto dimPos = mapIteratorToDimPos(mapLhs, pos);
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

// A naive strategy that focuses on unrolling contractions and their data
// transfers when possible.
// Elementwise operation unrolling is left to the LLVM backend. Earlier
// transformations should ensure that their shapes are SIMD-friendly.
//
// TODO: Unroll more ops.
//       It can be particularly beneficial to unroll transfer ops into
//       contiguous 1D chunks to avoid extra stack allocations.
void selectUnrollSizes(Operation *op,
                       const tpp::RegisterUnrollOptions &options) {
  MLIRContext *ctx = op->getContext();

  auto contractOp = dyn_cast<vector::ContractionOp>(op);
  if (!contractOp)
    return;

  std::optional<SmallVector<int64_t>> unrollShape =
      getContractionUnrollShape(contractOp, options.gemmUnroll);
  if (!unrollShape)
    return;

  // Assign target unroll shape.
  contractOp->setDiscardableAttr(unrollShapeAttrName,
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
    read->setDiscardableAttr(unrollShapeAttrName,
                             DenseI64ArrayAttr::get(ctx, shape));
  }
  if (auto read = dyn_cast_or_null<vector::TransferReadOp>(
          contractOp.getRhs().getDefiningOp())) {
    SmallVector<int64_t> shape = getOperandUnrollShape(indexingMaps[1]);
    read->setDiscardableAttr(unrollShapeAttrName,
                             DenseI64ArrayAttr::get(ctx, shape));
  }

  // Set the same unroll for accumulator and writes.
  SmallVector<int64_t> accUnroll = getOperandUnrollShape(indexingMaps[2]);
  if (auto read = dyn_cast_or_null<vector::TransferReadOp>(
          contractOp.getAcc().getDefiningOp())) {
    read->setDiscardableAttr(unrollShapeAttrName,
                             DenseI64ArrayAttr::get(ctx, accUnroll));
  }
  for (Operation *user : contractOp->getUsers()) {
    if (auto write = dyn_cast_or_null<vector::TransferWriteOp>(user)) {
      write->setDiscardableAttr(unrollShapeAttrName,
                                DenseI64ArrayAttr::get(ctx, accUnroll));
    }
  }
}

// Helper callback that controls unrolling.
// Returns desired unrolled vector shape if unrolling should happen.
static std::optional<SmallVector<int64_t>> getVectorShape(Operation *op) {
  auto unrollAttr = dyn_cast_or_null<DenseI64ArrayAttr>(
      op->getDiscardableAttr(unrollShapeAttrName));
  if (!unrollAttr)
    return std::nullopt;
  return SmallVector<int64_t>(unrollAttr.asArrayRef());
}

// Vector unroll driver pass designed to split operations into smaller
// hardware-compatible shapes in preparation for target-specific code
// generation.
//
// First, unroll shapes are assigned and propagated throughout the graph as
// attribute annotations. Then the unroll driver consumes these annotations
// to apply rewrites.
//
// For example - in prepation for further FMA lowering:
// ```
//   vector.contract {unroll_shape = [1, 16, 1]}
//     : vector<4x1xf32>, vector<1x32xf32> into vector<4x32xf32>
// ```
// the contractions is split into eight smaller ops:
// ```
//   vector.contract : vector<1x1xf32>, vector<1x16xf32> into vector<1x16xf32>
// ```
struct RegisterUnroll
    : public tpp::impl::RegisterUnrollBase<RegisterUnroll> {
  using RegisterUnrollBase::RegisterUnrollBase;

  void runOnOperation() override {
    auto *ctx = &getContext();

    tpp::RegisterUnrollOptions options;
    options.gemmUnroll = SmallVector<int64_t>{*gemmUnroll};

    // Assign and propagate unroll shapes.
    //
    // TODO: Replace with proper layout and propagation analysis like
    //       'SparseBackwardDataFlowAnalysis'.
    getOperation()->walk(
        [&](Operation *op) { selectUnrollSizes(op, options); });

    // TODO: Propagate unrolling through scf iter_args.
    RewritePatternSet patterns(ctx);
    vector::populateVectorUnrollPatterns(
        patterns,
        vector::UnrollVectorOptions().setNativeShapeFn(getVectorShape));

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
