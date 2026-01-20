//===- VectorContractToBF16DotProduct.cpp -----------------------*- C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction to x86vector::DotBF16Op.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_BF16DOTPRODUCT
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

static FailureOr<SmallVector<vector::TransferReadOp>>
getContractProducers(vector::ContractionOp contractOp) {
  SmallVector<vector::TransferReadOp> list;
  for (Value operand : contractOp->getOperands()) {
    auto vectorReadOp = operand.getDefiningOp<vector::TransferReadOp>();
    if (!vectorReadOp)
      return failure();
    list.push_back(vectorReadOp);
  }
  return list;
}

static FailureOr<SmallVector<memref::SubViewOp>>
getReadOperands(SmallVector<vector::TransferReadOp> readOps) {
  SmallVector<memref::SubViewOp> list;
  for (vector::TransferReadOp readOp : readOps) {
    auto subViewOp = readOp.getOperand(0).getDefiningOp<memref::SubViewOp>();
    if (!subViewOp)
      return failure();
    list.push_back(subViewOp);
  }
  return list;
}

static FailureOr<SmallVector<scf::ForOp>>
getNestedLoop(vector::ContractionOp contractOp) {
  SmallVector<scf::ForOp> list;
  Operation *current = contractOp;
  // It is register tiled loop structure on batch reduce matmul
  // (M->N->Batch-reduce->K).
  // TODO: support for matmul and batch matmul
  for (int i = 0; i < 4; i++) {
    Operation *parent = current->getParentOfType<scf::ForOp>();
    if (!parent)
      return failure();
    list.push_back(dyn_cast<scf::ForOp>(parent));
    current = parent;
  }
  return list;
}

static bool isTransposedMatrix(vector::ContractionOp contractOp) {
  SmallVector<AffineMap, 3> contractMaps = contractOp.getIndexingMapsArray();
  AffineMap mapA = contractMaps[0];
  AffineMap mapB = contractMaps[1];

  auto resultsMapA = mapA.getNumResults();
  auto resultsMapB = mapB.getNumResults();
  assert(resultsMapA == 4 && resultsMapB == 4 &&
         "Result dim map for A and B should be 4");

  auto inputsMapA = mapA.getNumInputs();
  auto inputsMapB = mapB.getNumInputs();
  assert(inputsMapA == 5 && inputsMapB == 5 &&
         "Input dim map for A and B should be 5");

  auto vnniDim = dyn_cast<AffineDimExpr>(mapA.getResult(3));
  auto dimBR = dyn_cast<AffineDimExpr>(mapA.getResult(0));

  SmallVector<AffineDimExpr> listMxNxK;
  for (unsigned int i = 0; i < inputsMapA; i++) {
    auto affineExpr =
        dyn_cast<AffineDimExpr>(mlir::getAffineDimExpr(i, mapA.getContext()));
    if (affineExpr != vnniDim && affineExpr != dimBR)
      listMxNxK.push_back(affineExpr);
  }
  auto dimM = listMxNxK[0];
  auto dimN = listMxNxK[1];
  auto dimK = listMxNxK[2];

  // Transpose if the mapA is kxm
  if (dyn_cast<AffineDimExpr>(mapA.getResult(1)) == dimK &&
      dyn_cast<AffineDimExpr>(mapA.getResult(2)) == dimM)
    return true;
  // Transpose if the mapB is nxk
  if (dyn_cast<AffineDimExpr>(mapB.getResult(1)) == dimN &&
      dyn_cast<AffineDimExpr>(mapB.getResult(2)) == dimK)
    return true;

  return false;
}

static bool permutationCheck(vector::ContractionOp contractOp) {
  SmallVector<AffineMap, 3> contractMaps = contractOp.getIndexingMapsArray();
  AffineMap mapA = contractMaps[0];
  AffineMap mapB = contractMaps[1];

  auto inputsMapA = mapA.getNumInputs();
  SmallVector<AffineDimExpr> inputDims;
  for (unsigned int i = 0; i < inputsMapA; i++) {
    auto affineExpr =
        dyn_cast<AffineDimExpr>(mlir::getAffineDimExpr(i, mapA.getContext()));
    inputDims.push_back(affineExpr);
  }

  bool flag = true;
  // mapA result dims
  auto resultsMapA = mapA.getNumResults();
  SmallVector<AffineDimExpr> outputDimsA;
  for (unsigned int i = 0; i < resultsMapA; i++) {
    auto affineExpr = dyn_cast<AffineDimExpr>(mapA.getResult(i));
    outputDimsA.push_back(affineExpr);
  }

  // We match the pattern {Batch-reduction, vnni, M, N, K} or {Batch-reduction,
  // M, N, K, vnni} -> {Batch-reduction, M, K, vnni}
  auto c1 = inputDims[0] == outputDimsA[0];
  auto c2 = (inputDims[1] == outputDimsA[3]) &&
            (inputDims[2] == outputDimsA[1]) &&
            (inputDims[4] == outputDimsA[2]);
  auto c3 = (inputDims[1] == outputDimsA[1]) &&
            (inputDims[3] == outputDimsA[2]) &&
            (inputDims[4] == outputDimsA[3]);
  flag = flag && (c1 && (c2 || c3));

  // mapB result dims
  auto resultsMapB = mapB.getNumResults();
  SmallVector<AffineDimExpr> outputDimsB;
  for (unsigned int i = 0; i < resultsMapB; i++) {
    auto affineExpr = dyn_cast<AffineDimExpr>(mapB.getResult(i));
    outputDimsB.push_back(affineExpr);
  }

  // We match the pattern {Batch-reduction, vnni, M, N, K} or {Batch-reduction,
  // M, N, K, vnni} -> {Batch-reduction, K, N, vnni}
  auto c4 = inputDims[0] == outputDimsB[0];
  auto c5 = (inputDims[1] == outputDimsB[3]) &&
            (inputDims[4] == outputDimsB[1]) &&
            (inputDims[3] == outputDimsB[2]);
  auto c6 = (inputDims[2] == outputDimsB[2]) &&
            (inputDims[3] == outputDimsB[1]) &&
            (inputDims[4] == outputDimsB[3]);
  flag = flag && (c4 && (c5 || c6));

  return flag;
}

static LogicalResult checkNestedLoop(SmallVector<scf::ForOp> loops,
                                     SmallVector<memref::SubViewOp> subviews) {
  auto loopSize = loops.size();
  assert(loopSize == 4 && subviews.size() == 3);

  auto subviewOpLhsOffsets = subviews[0].getOffsets();
  auto subviewOpRhsOffsets = subviews[1].getOffsets();
  auto subviewOpAccOffsets = subviews[2].getOffsets();

  SmallVector<Value> list;
  for (size_t i = 0; i < loopSize; i++) {
    list.push_back(loops[i].getInductionVar());
  }

  if (list[0] != subviewOpLhsOffsets[2] || list[0] != subviewOpRhsOffsets[1])
    return failure();

  if (list[1] != subviewOpLhsOffsets[0] || list[1] != subviewOpRhsOffsets[0])
    return failure();

  if (list[2] != subviewOpAccOffsets[1] || list[2] != subviewOpRhsOffsets[2])
    return failure();

  if (list[3] != subviewOpLhsOffsets[1] || list[3] != subviewOpAccOffsets[0])
    return failure();

  return success();
}

/// This pass lowers vector.contract (linalg.batch_reduce_matmul) for bf16
/// (vnni=2) type into sequence of x86vector::DotBF16Op.
///
/// As an example, the following pseudo-code will be rewritten
/// scf.for // m-tile
///  scf.for // n-tile
///   subview // C matrix
///   scf.for // batch-reduce
///   scf.for // k-tile
///    subview // A and B matrix
///    vector.read // A, B, and C matrix
///    vector.contract
///    vector.write // to C matrix
///
/// to:
///
/// scf.for // m-tile
///  scf.for // n-tile
///   vector.load // load C matrix in chunks of <16xbf16>
///   arith.shli + vector.bitcast // upconvert to f32 and pass them as iterargs
///   scf.for (iterargs = C matrix load as f32) // batch-reduce
///    scf.for (iterargs = batch-reduce iterArgs) // k-tile
///     vector.load // load 2 elements of A matrix and broadcast them into <32xbf16>
///     vector.load // load elements of B matrix into <32xbf16>
///     x86vector.avx512.dot %iterargs, %Ax, %Bx // accumulate in f32 (via iterargs)
///     x86vector.avx512.dot %iterargs, %Ax, %By // accumulate in f32 (via iterargs)
///     ..............
///     ..............
///    scf.yield // yield dpbf16 results
///   scf.yield // yield results of scf.for k-tile
///  arith.truncate // downconvert accumulator value from f32 to bf16
///  vector.store // store back into C matrix
///  .............
///  ............

struct BF16DotProductOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(
          contractOp, "Only combining kind 'ADD' is supported now.");

    // Check the vector contract operation satisfies the required pattern.
    // Check the Acc, Lhs, and Rhs of contract operation
    auto operands = getContractProducers(contractOp);
    if (failed(operands))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Invalid operands for contract op");
    auto readOps = *operands;
    auto vectorReadOpAcc = readOps[2];
    auto vectorReadOpLhs = readOps[0];
    auto vectorReadOpRhs = readOps[1];

    // Check whether the operand of vector transfer read is a subview
    auto readOpSubviews = getReadOperands(readOps);
    if (failed(readOpSubviews))
      return rewriter.notifyMatchFailure(
          contractOp, "Vector read op operands are not a subview");

    auto subviews = *readOpSubviews;
    auto subviewOpAcc = subviews[2];
    auto elementType =
        (cast<MemRefType>(subviewOpAcc.getType())).getElementType();

    if (!elementType.isBF16())
      return rewriter.notifyMatchFailure(contractOp, "The type is not BF16");

    // Check the operation type MatMul, B-MatMul, or BR-MatMul (FP32/BF16)
    SmallVector<vector::IteratorType> contractIteratorTypes =
        contractOp.getIteratorTypesArray();
    int reductionCount =
        std::count(contractIteratorTypes.begin(), contractIteratorTypes.end(),
                   vector::IteratorType::reduction);

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          contractOp, "Batch matmul operation not supported yet");

    if (reductionCount == 2)
      return rewriter.notifyMatchFailure(
          contractOp, "Batch reduce matmul operation without vnni layout");

    if (reductionCount > 3)
      return rewriter.notifyMatchFailure(
          contractOp, "The vector contract operation is not a gemm");

    auto lhsType = dyn_cast<ShapedType>(vectorReadOpLhs.getType());
    auto rhsType = dyn_cast<ShapedType>(vectorReadOpRhs.getType());
    auto accType = dyn_cast<ShapedType>(vectorReadOpAcc.getType());

    if (reductionCount == 3 &&
        (lhsType.getRank() != 4 || rhsType.getRank() != 4))
      return rewriter.notifyMatchFailure(
          contractOp,
          "Invalid rank for batch reduce operation with vnni layout");

    int64_t M = accType.getDimSize(0);
    int64_t N = accType.getDimSize(1);
    int64_t K = lhsType.getDimSize(lhsType.getRank() - 2);
    int64_t vnni = lhsType.getDimSize(lhsType.getRank() - 1);

    // K tile should be equal to vnni layout
    if (K != (vnni / 2))
      return rewriter.notifyMatchFailure(
          contractOp, "K tile size should be equal to VNNI layout");

    if (N != 32)
      return rewriter.notifyMatchFailure(
          contractOp,
          "N tile size should be equal to 32 to ensure avx512bf16 dp");

    if (vnni != 2)
      return rewriter.notifyMatchFailure(
          contractOp, "Only VNNI layout=2 is supported, now");

    if (isTransposedMatrix(contractOp))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Matrices shoudn't be transposed.");

    if (!permutationCheck(contractOp))
      return rewriter.notifyMatchFailure(
          contractOp, "Affine map permutation not supported.");

    auto loops = getNestedLoop(contractOp);
    if (failed(loops))
      return rewriter.notifyMatchFailure(
          contractOp, "Invalid loop nest in contract pattern");

    auto checkLoops = checkNestedLoop(*loops, subviews);
    if (failed(checkLoops))
      return rewriter.notifyMatchFailure(
          contractOp, "Loops doesn't match the iv in subviews");

    auto nestedLoops = *loops;
    auto kForOp = nestedLoops[0];
    auto reductionForOp = nestedLoops[1];

    rewriter.setInsertionPoint(reductionForOp);
    Value c0 =
        arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), 0);

    // Creating further subviews from the C matrix subview
    llvm::SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(K),
                                             rewriter.getIndexAttr(N)};
    llvm::SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1),
                                               rewriter.getIndexAttr(1)};
    llvm::SmallVector<Value, 8> subviewCMatrix;
    llvm::SmallVector<Value, 8> loopItrArgs;
    for (int i = 0; i < M; i++) {
      SmallVector<OpFoldResult> offsets = {
          rewriter.getIndexAttr(i),
          rewriter.getIndexAttr(0),
      };
      auto newSubview = memref::SubViewOp::create(rewriter, 
          reductionForOp.getLoc(), subviewOpAcc, offsets, sizes, strides);
      subviewCMatrix.push_back(newSubview);

      // vector <16xf32> for iterargs to accumulate results in fp32
      for (int j = 0; j < vnni; j++) {
        Value indexOp = arith::ConstantIndexOp::create(rewriter, 
            reductionForOp.getLoc(), j * (N / 2));
        auto valueCRow = vector::LoadOp::create(rewriter, 
            reductionForOp.getLoc(), VectorType::get({N / 2}, elementType),
            newSubview, ValueRange{c0, indexOp});
        auto bitcast_i16 = vector::BitCastOp::create(rewriter, 
            reductionForOp.getLoc(),
            VectorType::get({N / 2}, rewriter.getIntegerType(16)), valueCRow);
        auto extend_i32 = arith::ExtUIOp::create(rewriter, 
            reductionForOp.getLoc(),
            VectorType::get({N / 2}, rewriter.getIntegerType(32)), bitcast_i16);
        auto cst16 = arith::ConstantOp::create(rewriter, 
            reductionForOp.getLoc(),
            rewriter.getIntegerAttr(rewriter.getIntegerType(32), N / 2));
        auto vectType = VectorType::get({N / 2}, rewriter.getIntegerType(32));
        auto shiftOp = arith::ShLIOp::create(rewriter, 
            reductionForOp.getLoc(), vectType, extend_i32,
            vector::BroadcastOp::create(rewriter, reductionForOp.getLoc(),
                                                 vectType, cst16));
        auto f32CVector = vector::BitCastOp::create(rewriter, 
            reductionForOp.getLoc(),
            VectorType::get({N / 2}, rewriter.getF32Type()), shiftOp);

        loopItrArgs.push_back(f32CVector);
      }
    }

    SmallVector<Value, 8> bf16DP;

    // Code to re-create the reduction and k loop with iter args
    auto newReductionForOp = scf::ForOp::create(rewriter, 
        reductionForOp.getLoc(), reductionForOp.getLowerBound(),
        reductionForOp.getUpperBound(), reductionForOp.getStep(), loopItrArgs,
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp,
            Value ivNewReductionForOp, ValueRange iterArgsNewReductionForOp) {
          auto newKForOp = scf::ForOp::create(rewriter, 
              kForOp.getLoc(), kForOp.getLowerBound(), kForOp.getUpperBound(),
              kForOp.getStep(), iterArgsNewReductionForOp,
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                IRMapping mapping;
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(3),
                    ivNewKForOp);
                auto lhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

                // Memory access for A Matrix into <32xbf16>
                llvm::SmallVector<Value, 8> vectorA;

                for (int i = 0; i < M; i++) {
                  Value indexOp = arith::ConstantIndexOp::create(rewriter, 
                      reductionForOp.getLoc(), i);
                  auto valueA = vector::LoadOp::create(rewriter, 
                      kForOp.getLoc(), VectorType::get({vnni}, elementType),
                      lhsClone->getResult(0), ValueRange{c0, indexOp, c0, c0});
                  auto bitcastValueA =
                      vector::BitCastOp::create(rewriter, 
                          kForOp.getLoc(),
                          VectorType::get({1}, rewriterNewKForOp.getI32Type()),
                          valueA);
                  auto broadcastValueA =
                      vector::BroadcastOp::create(rewriter, 
                          kForOp.getLoc(),
                          VectorType::get(16, rewriterNewKForOp.getI32Type()),
                          bitcastValueA);
                  auto bitcastValueA_32 =
                      vector::BitCastOp::create(rewriter, 
                          kForOp.getLoc(),
                          VectorType::get({N}, rewriterNewKForOp.getBF16Type()),
                          broadcastValueA);

                  vectorA.push_back(bitcastValueA_32);
                }

                IRMapping rhsMapping;
                rhsMapping.map(
                  vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1),
                  ivNewReductionForOp);
                rhsMapping.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(2),
                    ivNewKForOp);
                auto rhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

                // Memory access for B Matrix into <32xbf16>
                llvm::SmallVector<Value, 8> vectorB;
                for (int i = 0, j = 0; i < vnni; i++, j = j + 16) {
                  Value indexOp = arith::ConstantIndexOp::create(rewriter, 
                      reductionForOp.getLoc(), j);
                  auto valueBRow = vector::LoadOp::create(rewriter, 
                      kForOp.getLoc(), VectorType::get({N}, elementType),
                      rhsClone->getResult(0), ValueRange{c0, c0, indexOp, c0});
                  vectorB.push_back(valueBRow);
                }

                // Code for x86vector.avx512.dot
                mlir::VectorType dstType =
                    mlir::VectorType::get({N / 2}, rewriter.getF32Type());
                for (int i = 0, k = 0; i < M; i++, k = k + vnni) {
                  for (int j = 0; j < vnni; j++) {
                    auto dp = mlir::x86vector::DotBF16Op::create(rewriter, 
                        kForOp.getLoc(), dstType, iterArgsNewKForOp[j + k],
                        vectorA[i], vectorB[j]);
                    bf16DP.push_back(dp);
                  }
                }

                scf::YieldOp::create(rewriter, locNewKForOp, bf16DP);
              });

          scf::YieldOp::create(rewriter, 
              locNewReductionForOp, newKForOp.getResults());
        });

    // Downconvert <16xf32> to <16xbf16> and store into C Matrix
    for (int i = 0, k = 0; i < M; i++) {
      for (int j = 0; j < vnni; j++) {
        Value indexOp = arith::ConstantIndexOp::create(rewriter, 
            reductionForOp.getLoc(), j * 16);
        auto bf16vec = arith::TruncFOp::create(rewriter, 
            reductionForOp.getLoc(),
            VectorType::get({16}, rewriter.getBF16Type()),
            newReductionForOp.getResult(k));
        vector::StoreOp::create(rewriter, reductionForOp.getLoc(), bf16vec,
                                         subviewCMatrix[i],
                                         ValueRange{c0, indexOp});
        k++;
      }
    }

    // Delete the contract operation
    for (auto result : contractOp->getResults()) {
      for (auto *userOp : result.getUsers()) {
        rewriter.eraseOp(userOp);
      }
    }
    rewriter.eraseOp(contractOp);
    return success();
  }
};

void populateBF16DotProductPatterns(RewritePatternSet &patterns) {
  patterns.add<BF16DotProductOp>(patterns.getContext());
}

struct BF16DotProduct : public impl::BF16DotProductBase<BF16DotProduct> {
  using BF16DotProductBase::BF16DotProductBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBF16DotProductPatterns(patterns);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
