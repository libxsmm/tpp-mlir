//===- TransformUtils.h ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_TRANSFORMUTILS_H
#define TPP_TRANSFORMS_UTILS_TRANSFORMUTILS_H

#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {

class Operation;
class OpBuilder;
struct Range;
class RewriterBase;
class TilingInterface;

namespace linalg {
class LinalgOp;
struct ContractionDimensions;
} // namespace linalg

namespace linalgx {
namespace utils {

// Given an opOperand and a range of ivs return the one used by the operands.
FailureOr<SmallVector<Value>>
getInvolvedLocalDimsForOperand(OpBuilder &builder, Location loc,
                               OpOperand *operand, AffineMap mapOperand,
                               ValueRange localIvs);

// Extract and return a slice for operand. Offsets are the induction variable
// touched by the operand. Sizes are: '1' in [0 to rank - desiredResultRank]
// while the full chunk in [rank - desiredResultRank to rank). Strides are
// assumed to be always 1. The methods effectively peel out the outermost [0 to
// rank - desiredResultRank] dimensions that are materialized as loops.
FailureOr<Value> getSliceOperand(OpBuilder &builder, OpOperand *operand,
                                 linalg::LinalgOp linalgOp, ValueRange ivs,
                                 ValueRange valuesToUse,
                                 unsigned desiredResultRank);

// Extract a slice of `operand` based on `offset`, `sizes` and
// `strides`.
Value getSliceOperand(OpBuilder &builder, linalg::LinalgOp linalgOp,
                      Value operand, ArrayRef<OpFoldResult> offset,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides,
                      unsigned desiredResultRank);

// Return the loop range to materialize as loops from '0' to 'upTo'.
// '0' is the outermost loop.
FailureOr<SmallVector<Range>> getLoopsToMaterialize(RewriterBase &rewriter,
                                                    linalg::LinalgOp linalgOp,
                                                    unsigned upTo);

// Return true if `op` is a blocked matmul.
bool isBlockedMatmul(Operation *op);

// Return true if the `op` is a contraction defined as:
// - 2 input operands (LHS and RHS), and 1 output operand OUT.
// - The body is matmul-like
// - We have at least 1 m dimension involved in an outer-product along LHS.
// - We have at lest 1 n dimension involved in an outer-product along RHS.
// - We have at least 1 k dimension as a permutation on LHS and RHS.
// - The output map is a permutation map, while not gurantee is given on the
// input maps.
FailureOr<linalg::ContractionDimensions>
isContraction(linalg::LinalgOp linalgOp);

// Return constant range span or nullopt, otherwise.
std::optional<int64_t> getConstantRange(const Range &range);

// Validate a tile configuration for a linalgOp when we can statically do that.
// Specific dims can be passed using 'dims'. If dims is empty the validation
// will start from the outermost dimension, moving to innermost ones up to the
// number of tiles.
// Tiling application can restricted based on the workload dimension size.
// The tiling is applied only to when all dimensions fulfill the predicate:
// '(dimSize[i] / tiles[i]) >= minTileFactor'.
bool validateFullTilesOnDims(TilingInterface tileOp,
                             ArrayRef<OpFoldResult> tiles,
                             ArrayRef<size_t> dims = {},
                             int64_t minTileFactor = 2);

// Rewrite scf.for to scf.forall. Assumes the loop to be parallel and
// marked with `kLoopId`.
constexpr const static llvm::StringLiteral kLoopParallel = "parallel";
constexpr const static llvm::StringLiteral kLoopRoot = "root";
void populateScfForToForAllRewritePattern(RewritePatternSet &patterns);

// Given a value `val` expand its shape based on `reassociationMap`.
Value expand(OpBuilder &builder, Location loc, Value val, Type newType,
             ArrayRef<ReassociationIndices> reassociationMap);

// Given a value `val` collapse its shape based on  `reassociationMap`.
Value collapse(OpBuilder &builder, Location loc, Value val, Type newType,
               ArrayRef<ReassociationIndices> reassociationMap);

} // namespace utils
} // namespace linalgx
} // namespace mlir

#endif
