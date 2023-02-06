//===- Transforms.h ----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_H
#define TPP_TRANSFORMS_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class RewriterBase;
class Value;

namespace linalg {
class GenericOp;
class LinalgOp;
class Conv2DNchwFchwOp;
class Conv2DNhwcHwcfOp;
class MatmulOp;
class BatchReduceMatmulOp;
} // namespace linalg

namespace vnni {
class MatmulOp;
class BRGemmOp;
} // namespace vnni

namespace linalgx {

// Attempt to map the current linalgOp to a BRGEMM.
// On success the returned values are the materialzed loops with BRGEMM inside.
FailureOr<SmallVector<Value>> rewriteToBRGEMMOp(RewriterBase &rewriter,
                                                linalg::LinalgOp linalgOp);

// Rewrite a convolution to a matmul operation. We support the following
// formats:
// 1. [N][P][Q][K] += [N][H][W][C] * [R][S][C][K]
// 2. [N][K’][P][Q][k] += [N][C’][H][W][c] * [K’][C’][R][S][c][k] (blocked)
FailureOr<linalg::MatmulOp> rewriteConvToMatmul(RewriterBase &rewriter,
                                                linalg::LinalgOp linalgOp);

// Attempt to block a Conv2DNchwFchwOp.
FailureOr<linalg::GenericOp>
packConv2DNchwFchwOp(RewriterBase &rewriter, linalg::Conv2DNchwFchwOp linalgOp,
                     ArrayRef<OpFoldResult> tiles);

// Attempt to pack a Conv2DNhwcHwcfOp.
FailureOr<linalg::GenericOp>
packConv2DNhwcHwcfOp(RewriterBase &rewriter, linalg::Conv2DNhwcHwcfOp linalgOp,
                     ArrayRef<OpFoldResult> tiles);

// Attempt to block a MatmulOp.
FailureOr<linalg::GenericOp> packMatmulOp(RewriterBase &rewriter,
                                          linalg::MatmulOp linalgOp,
                                          ArrayRef<OpFoldResult> tiles);

// Attempt to block a MatmulOp to VNNI format.
FailureOr<linalg::GenericOp> packVNNIMatmulOp(RewriterBase &rewriter,
                                              linalg::GenericOp linalgOp);

// Attempt to block a BRGemmOp to VNNI format.
FailureOr<vnni::BRGemmOp>
packVNNIBRGemmOp(RewriterBase &rewriter, linalg::BatchReduceMatmulOp linalgOp);

// Collapse iterators in a linalg.generic based on 'reassociation'.
FailureOr<linalg::GenericOp>
collapseIterators(RewriterBase &rewriter, linalg::GenericOp genericOp,
                  ArrayRef<SmallVector<int64_t, 2>> reassociation);

// Annotate a linalg.generic with a possible mapping for tpp operations.
// The annotation uses the library_call attribute in linalg.generic.
// TODO: We may not want to fail here.
FailureOr<linalg::GenericOp> mapLinalgToTpp(RewriterBase &rewriter,
                                            linalg::GenericOp linalgOp);

} // namespace linalgx

namespace tpp {
void populateConvertLinalgToTppPatterns(RewritePatternSet &patterns,
                                        bool useParallelLoops);
void populateMapLinalgToTppPatterns(RewritePatternSet &patterns);
void populateTppToXsmmPatterns(RewritePatternSet &patterns);
void populateXsmmToFuncPatterns(RewritePatternSet &patterns,
                                bool useExtractMetaData);
void populateCheckToFuncPatterns(RewritePatternSet &patterns);
void populateSinkPackPatterns(RewritePatternSet &patterns);
} // namespace tpp
} // namespace mlir

#endif
