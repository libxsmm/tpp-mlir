//===- Transforms.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_TRANSFORMS_H
#define TPP_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewriterBase;
class Value;

namespace linalg {
class GenericOp;
class LinalgOp;
class MatmulOp;
class BatchReduceMatmulOp;
class BatchMatmulOp;
} // namespace linalg

namespace linalgx {

// Attempt to map the current linalgOp to a BRGEMM.
// On success the returned values are the materialzed loops with BRGEMM inside.
FailureOr<SmallVector<Value>> rewriteToBRGemmOp(RewriterBase &rewriter,
                                                linalg::LinalgOp linalgOp);

// Attempt to block a MatmulOp to VNNI format.
FailureOr<linalg::GenericOp> packVNNIMatmulOp(RewriterBase &rewriter,
                                              linalg::GenericOp linalgOp);

// Attempt to block a Brgemm to VNNI format.
FailureOr<linalg::GenericOp>
packVNNIBRGemmOp(RewriterBase &rewriter, linalg::BatchReduceMatmulOp linalgOp);

// Collapse iterators in a linalg.generic based on 'reassociation'.
FailureOr<linalg::GenericOp>
collapseIterators(RewriterBase &rewriter, linalg::GenericOp genericOp,
                  ArrayRef<SmallVector<int64_t, 2>> reassociation);

} // namespace linalgx

namespace tpp {
void populateLinalgToXsmmPatterns(RewritePatternSet &patterns, ArrayRef<StringRef> skipPatterns);
void populateSimplifyPacking(RewritePatternSet &patterns);
void populateSinkPackPatterns(RewritePatternSet &patterns);
} // namespace tpp
namespace linalg {
void populateLinalgDeGeneralizationPatterns(RewritePatternSet &patterns);
} // namespace linalg
} // namespace mlir

#endif
