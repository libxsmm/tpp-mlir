//===--------------- VectorContractToFMA.cpp ------------*- C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction to vector fma.
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "vector-contract-to-fma"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORCONTRACTTOFMA
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace {

/// Returns the target vector length based on target features avx2/avx512 for
/// FP32 data type.
static unsigned getTargetVectorLengthForFP32(llvm::StringRef targetFeatureStr) {
  unsigned vecElemTypeSizeInBits = 32;
  unsigned vecRegSizeInBits = StringSwitch<unsigned>(targetFeatureStr)
                                  .Case("avx2", 256)
                                  .Case("avx512", 512)
                                  .Default(0);
  if (vecRegSizeInBits > 0)
    return vecRegSizeInBits / vecElemTypeSizeInBits;

  vecRegSizeInBits = vnni::utils::hasAVX512() ? 512
                     : vnni::utils::hasAVX2() ? 256
                                              : 0;
  return vecRegSizeInBits / vecElemTypeSizeInBits;
}
/// Returns true if the \p map is transposed.
static bool isTransposed(AffineMap map) {
  auto results = map.getResults();
  // Assert if the map does not have 3 or 4 inputs ([] m, n, k).
  assert((map.getNumInputs() == 3 || map.getNumInputs() == 4) &&
         "3 or 4 input dim expected");
  // Assert if the result is not 2D.
  assert(map.getNumResults() == 2 && "Only 2 output dim expected");

  // Check the last two dimensions for transposition.
  auto dimExpr0 = dyn_cast<AffineDimExpr>(results[0]);
  auto dimExpr1 = dyn_cast<AffineDimExpr>(results[1]);
  assert((dimExpr0 && dimExpr1) && "Unexpected dim expression");

  // Exclude output map result.
  bool isOutputResultMap =
      dimExpr0 ==
          mlir::getAffineDimExpr(map.getNumInputs() - 3, map.getContext()) &&
      dimExpr1 ==
          mlir::getAffineDimExpr(map.getNumInputs() - 2, map.getContext());
  assert(!isOutputResultMap && "Output result map not expected");

  // It's transposed if result found as (k, m) or (n, k), else not transposed.
  if ((dimExpr0 ==
           mlir::getAffineDimExpr(map.getNumInputs() - 1, map.getContext()) &&
       dimExpr1 ==
           mlir::getAffineDimExpr(map.getNumInputs() - 3, map.getContext())) ||
      (dimExpr0 ==
           mlir::getAffineDimExpr(map.getNumInputs() - 2, map.getContext()) &&
       dimExpr1 ==
           mlir::getAffineDimExpr(map.getNumInputs() - 1, map.getContext())))
    return true;
  return false;
}
} // namespace

namespace mlir {
namespace tpp {

// Structure to hold transformation context
struct TransformationContext {
  scf::ForOp innerForOp;
  scf::ForOp outerForOp;
  scf::ForOp outermostLoop;
};

enum class MatMulType { Standard, Batch, BatchReduce };

struct VectorContractToFMA
    : public tpp::impl::VectorContractToFMABase<VectorContractToFMA> {

  using VectorContractToFMABase::VectorContractToFMABase;

  void runOnOperation() override;

private:
  TransformationContext ctx;
};

struct VectorContractToFMAPattern
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  VectorContractToFMAPattern(MLIRContext *context,
                             VectorContractToFMAOptions options,
                             TransformationContext &ctx)
      : OpRewritePattern<vector::ContractionOp>(context), options(options),
        ctx(ctx) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(
          op, "Unsupported combining kind, only supports ADD at the moment)");

    auto maskableOp = cast<vector::MaskableOpInterface>(op.getOperation());
    if (maskableOp.isMasked())
      return rewriter.notifyMatchFailure(op, "Masked contractOp not supported");

    SmallVector<AffineMap, 3> maps = op.getIndexingMapsArray();
    if (llvm::any_of(
            maps, [](AffineMap map) { return !map.isProjectedPermutation(); }))
      return rewriter.notifyMatchFailure(op, "Unexpected map");

    // Check for the variant of matrix multiply.
    auto iteratorTypes = op.getIteratorTypesArray();
    MatMulType matmulType;
    unsigned outerDimIndex = 0;
    if (iteratorTypes.size() > 3) {
      outerDimIndex = iteratorTypes.size() - 4;
      matmulType =
          iteratorTypes[outerDimIndex] == vector::IteratorType::parallel
              ? MatMulType::Batch
              : MatMulType::BatchReduce;
      outerDimIndex++;
    } else if (iteratorTypes.size() == 3) {
      matmulType = MatMulType::Standard;
    } else {
      return rewriter.notifyMatchFailure(op, "Not a gemm");
    }

    if (matmulType == MatMulType::Batch)
      return rewriter.notifyMatchFailure(op, "Batch matmul not supported");
    if (iteratorTypes[outerDimIndex] != vector::IteratorType::parallel ||
        iteratorTypes[outerDimIndex + 1] != vector::IteratorType::parallel ||
        iteratorTypes[outerDimIndex + 2] != vector::IteratorType::reduction)
      return rewriter.notifyMatchFailure(op, "Not a gemm");

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto acc = op.getAcc();
    auto lhsDefiningOp = lhs.getDefiningOp<vector::TransferReadOp>();
    auto rhsDefiningOp = rhs.getDefiningOp<vector::TransferReadOp>();
    auto accDefiningOp = acc.getDefiningOp<vector::TransferReadOp>();
    if (!lhsDefiningOp || !rhsDefiningOp)
      return failure();

    // Accumulator can be a TransferReadOp but must be coming from the chain of
    // iterargs of nested loop.
    if (accDefiningOp)
      return failure();

    // Make sure the inputs being read are whole tensor or subview.
    if (!llvm::all_of(lhsDefiningOp.getIndices(), isZeroInteger) ||
        !llvm::all_of(rhsDefiningOp.getIndices(), isZeroInteger)) {
      return failure();
    }

    auto lhsType = cast<ShapedType>(lhsDefiningOp.getType());
    auto rhsType = cast<ShapedType>(rhsDefiningOp.getType());
    // auto accType = acc.getType();
    //  auto accType = cast<ShapedType>(accDefiningOp.getType());

    if (matmulType == MatMulType::BatchReduce &&
        (lhsType.getRank() != 3 || rhsType.getRank() != 3))
      return failure();

    if (matmulType == MatMulType::Standard &&
        (lhsType.getRank() != 2 || rhsType.getRank() != 2))
      return failure();

    // Check for non-transposed matrices.
    auto mapLHS = maps[0];
    auto mapRHS = maps[1];
    if (matmulType == MatMulType::BatchReduce) {
      mapLHS = mapLHS.dropResult(0);
      mapRHS = mapRHS.dropResult(0);
    }
    if (isTransposed(mapLHS) || isTransposed(mapRHS))
      return rewriter.notifyMatchFailure(
          op, "Transposed matrices are not expected");

    // Verify that the accumulator is coming through a chain of iterargs of
    // nested loop and it is define by 'TransferReadOp'.
    ctx.innerForOp = op->getParentOfType<scf::ForOp>();
    if (!ctx.innerForOp)
      return failure();
    ctx.outerForOp = ctx.innerForOp->getParentOfType<scf::ForOp>();
    if (!ctx.outerForOp)
      return failure();
    ctx.outermostLoop = ctx.outerForOp->getParentOfType<scf::ForOp>();
    if (!ctx.outermostLoop)
      return failure();

    // Verify original inner loop has only one iterarg.
    auto origIterArgs = ctx.innerForOp.getRegionIterArgs();
    if (origIterArgs.size() != 1)
      return failure();

    // Verify chain, accumulator must be inner loop's iterarg.
    auto bbArg = dyn_cast<BlockArgument>(acc);
    if (!bbArg)
      return failure();

    // This block arg must be init arg, not induction variable.
    if (bbArg.getOwner() != ctx.innerForOp.getBody() ||
        bbArg.getArgNumber() == 0) {
      return failure();
    }

    // This iterarg must be intialized by outer loop's iterarg.
    auto innerInitValue =
        ctx.innerForOp.getInitArgs()[bbArg.getArgNumber() - 1];
    auto outerBBArg = dyn_cast<BlockArgument>(innerInitValue);
    if (!outerBBArg)
      return failure();

    // This block arg must be init arg, not induction variable.
    if (outerBBArg.getOwner() != ctx.outerForOp.getBody() ||
        outerBBArg.getArgNumber() == 0) {
      return failure();
    }

    // Outer loop's iterarg initializer must be a TransferReadOp.
    acc = ctx.outerForOp.getInitArgs()[outerBBArg.getArgNumber() - 1];

    //  This must be defined by vector.transfer_read
    if (!acc.getDefiningOp<vector::TransferReadOp>())
      return failure();

    accDefiningOp = acc.getDefiningOp<vector::TransferReadOp>();
    if (!accDefiningOp)
      return failure();

    // Only 2-D output expected.
    auto accType = cast<ShapedType>(accDefiningOp.getType());
    if (accType.getRank() != 2)
      return failure();

    int64_t M = accType.getDimSize(0);
    int64_t N = accType.getDimSize(1);
    int64_t K = lhsType.getDimSize(lhsType.getRank() - 1);

    // K must be 1.
    if (K != 1)
      return failure();

    unsigned vecLen = getTargetVectorLengthForFP32(options.targetFeature);
    if (vecLen == 0)
      return failure();

    SmallVector<Value, 12> results;
    SmallVector<Value, 12> argResults;
    auto accSubview = accDefiningOp.getBase();
    Location loc = op.getLoc();

    // Create M different <1xN> subviews.
    auto memrefType = cast<MemRefType>(accSubview.getType());
    auto elementType = memrefType.getElementType();
    SmallVector<OpFoldResult> mixedSizes = {rewriter.getIndexAttr(K),
                                            rewriter.getIndexAttr(N)};
    SmallVector<OpFoldResult> mixedStrides = {rewriter.getIndexAttr(1),
                                              rewriter.getIndexAttr(1)};

    rewriter.setInsertionPoint(
        ctx.outermostLoop.getBody(),
        std::prev(ctx.outermostLoop.getBody()->end(), 1));

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value, 4> subview_2_splits;
    for (int i = 0; i < M; i++) {
      SmallVector<OpFoldResult> mixedOffsets = {
          rewriter.getIndexAttr(i),
          rewriter.getIndexAttr(0),
      };
      auto split = memref::SubViewOp::create(rewriter, 
          loc, accSubview, mixedOffsets, mixedSizes, mixedStrides);
      subview_2_splits.push_back(split);
    }

    // Intialize each accumulator with a vector of size vecLen
    SmallVector<Value, 4> initAccs;
    for (auto subview : subview_2_splits) {
      for (unsigned j = 0; j < N; j += vecLen) {
        auto acc = vector::LoadOp::create(rewriter, 
            loc, VectorType::get({vecLen}, elementType), subview,
            ValueRange{c0, arith::ConstantIndexOp::create(rewriter, loc, j)});
        initAccs.push_back(acc);
      }
    }

    // Create new outer loop with M different accumulators.
    auto newOuterForOp = scf::ForOp::create(rewriter, 
        loc, ctx.outerForOp.getLowerBound(), ctx.outerForOp.getUpperBound(),
        ctx.outerForOp.getStep(), initAccs,
        [&](OpBuilder &nestedBuilder, Location loc, Value iv,
            ValueRange iterArgs) {
          // Create new inner loop with M accumulators.
          auto newInnerForOp = scf::ForOp::create(nestedBuilder, 
              loc, ctx.innerForOp.getLowerBound(),
              ctx.innerForOp.getUpperBound(), ctx.innerForOp.getStep(),
              iterArgs,
              [&](OpBuilder &innerBuilder, Location loc, Value innerIv,
                  ValueRange innerIterArgs) {
                IRMapping mapping;
                mapping.map(
                    lhsDefiningOp.getBase().getDefiningOp()->getOperand(1),
                    iv);
                mapping.map(
                    lhsDefiningOp.getBase().getDefiningOp()->getOperand(3),
                    innerIv);
                auto lhsClone = innerBuilder.clone(
                    *lhsDefiningOp.getBase().getDefiningOp(), mapping);

                // Load and broadcast individual elements
                SmallVector<Value, 4> broadcasts;
                for (int i = 0; i < M; i++) {
                  auto elem = memref::LoadOp::create(innerBuilder, 
                      loc, lhsClone->getResult(0),
                      ValueRange{
                          c0,
                          arith::ConstantIndexOp::create(innerBuilder, loc, i),
                          c0});
                  auto bcast = vector::BroadcastOp::create(innerBuilder, 
                      loc, VectorType::get({vecLen}, elem.getType()), elem);
                  broadcasts.push_back(bcast);
                }

                IRMapping rhsMapping;
                rhsMapping.map(
                    rhsDefiningOp.getBase().getDefiningOp()->getOperand(1),
                    iv);
                rhsMapping.map(
                    rhsDefiningOp.getBase().getDefiningOp()->getOperand(2),
                    innerIv);

                // Create Mx(N/vecLen) different FMAs using broadcasts and
                // current accumulator values.
                auto rhsClone = innerBuilder.clone(
                    *rhsDefiningOp.getBase().getDefiningOp(), rhsMapping);
                if (vecLen == 8) {
                  for (unsigned j = 0; j < N; j += vecLen) {
                    auto rowVec = vector::LoadOp::create(innerBuilder, 
                        loc, VectorType::get({vecLen}, elementType),
                        rhsClone->getResult(0),
                        ValueRange{c0, c0,
                                   arith::ConstantIndexOp::create(innerBuilder, 
                                       loc, j)});
                    unsigned iterArgAccessStride = N / vecLen;
                    unsigned offset = j / vecLen;
                    for (int i = 0; i < M; i++) {
                      auto fma = vector::FMAOp::create(innerBuilder, 
                          loc, broadcasts[i], rowVec,
                          innerIterArgs[offset + iterArgAccessStride * i]);
                      argResults.push_back(fma);
                    }
                  }

                  // Perform strided circular copy of elements from argResults
                  // to results.
                  unsigned stride = (N / vecLen);
                  unsigned totalElements = argResults.size();
                  results.resize(totalElements);
                  for (unsigned i = 0; i < totalElements; ++i) {
                    unsigned circularIndex =
                        (i % stride) * (stride - 1) + (i / stride);
                    results[i] = argResults[circularIndex];
                  }

                } else {
                  for (int i = 0; i < M; i++) {
                    unsigned iterArgAccessStride = (i) * ((N / vecLen));
                    for (unsigned j = 0; j < N; j += vecLen) {
                      auto rowVec = vector::LoadOp::create(innerBuilder, 
                          loc, VectorType::get({vecLen}, elementType),
                          rhsClone->getResult(0),
                          ValueRange{
                              c0, c0,
                              arith::ConstantIndexOp::create(innerBuilder, loc,
                                                                          j)});
                      unsigned offset = (j / vecLen);
                      auto fma = vector::FMAOp::create(innerBuilder, 
                          loc, broadcasts[i], rowVec,
                          innerIterArgs[offset + iterArgAccessStride]);
                      results.push_back(fma);
                    }
                  }
                }

                // Yield all M results
                scf::YieldOp::create(innerBuilder, loc, results);
              });

          // Yield results from inner loop to outer loop
          scf::YieldOp::create(nestedBuilder, loc, newInnerForOp.getResults());
        });

    Value matResult = ctx.outerForOp.getResult(0);
    Operation *writeOp;
    for (auto user : matResult.getUsers()) {
      writeOp = dyn_cast<vector::TransferWriteOp>(user);
      if (writeOp)
        break;
    }

    // Store final results back to original locations.
    if (writeOp) {
      for (int i = 0; i < M; i++) {
        unsigned iterArgAccessStride = i * (N / vecLen);
        for (unsigned j = 0; j < N; j += vecLen) {
          unsigned offset = j / vecLen;
          vector::StoreOp::create(rewriter, 
              loc, newOuterForOp.getResult(offset + iterArgAccessStride),
              subview_2_splits[i],
              ValueRange{c0, arith::ConstantIndexOp::create(rewriter, loc, j)});
        }
      }
    }

    // Erase original write.
    if (writeOp)
      rewriter.eraseOp(writeOp);

    return success();
  }

private:
  VectorContractToFMAOptions options;
  TransformationContext &ctx;
};

void VectorContractToFMA::runOnOperation() {
  VectorContractToFMAOptions options;
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();
  options.targetFeature = targetFeature;
  RewritePatternSet patterns(context);
  patterns.add<VectorContractToFMAPattern>(context, options, ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace tpp
} // namespace mlir
