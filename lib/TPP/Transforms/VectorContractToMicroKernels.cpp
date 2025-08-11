//===- VectorContractToMicroKernels.cpp --------------------------*- C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction using x86vector ops
// to micro kernels.
// Target types: f32, bf16, i8, and f16
//
//===----------------------------------------------------------------------===//
#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
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
#define GEN_PASS_DEF_MICROKERNELS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace mlir {
namespace tpp {

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

static bool isTransposedMatrix(vector::ContractionOp contractOp,
                               Type elementType, bool isSplat) {
  SmallVector<AffineMap, 3> contractMaps = contractOp.getIndexingMapsArray();
  AffineMap mapA = contractMaps[0];
  AffineMap mapB = contractMaps[1];

  bool isUnPackedType = elementType.isF32() || isSplat;
  bool isPackedType = (elementType.isF16() || elementType.isBF16() ||
                       elementType.isSignlessInteger(8)) &&
                      !isSplat;

  auto resultsMapA = mapA.getNumResults();
  auto resultsMapB = mapB.getNumResults();

  if (isUnPackedType) {
    assert(resultsMapA == 3 && resultsMapB == 3 &&
           "Result dim map for A and B should be 3");
  }

  if (isPackedType) {
    assert(resultsMapA == 4 && resultsMapB == 4 &&
           "Result dim map for A and B should be 4");
  }

  auto inputsMapA = mapA.getNumInputs();
  auto inputsMapB = mapB.getNumInputs();

  if (isUnPackedType) {
    assert(inputsMapA == 4 && inputsMapB == 4 &&
           "Input dim map for A and B should be 4");
  }

  if (isPackedType) {
    assert(inputsMapA == 5 && inputsMapB == 5 &&
           "Input dim map for A and B should be 5");
  }

  auto dimBR = dyn_cast<AffineDimExpr>(mapA.getResult(0));

  SmallVector<AffineDimExpr> listMxNxK;
  for (unsigned int i = 0; i < inputsMapA; i++) {
    auto affineExpr =
        dyn_cast<AffineDimExpr>(mlir::getAffineDimExpr(i, mapA.getContext()));

    if (isPackedType) {
      auto vnniDim = dyn_cast<AffineDimExpr>(mapA.getResult(3));
      if (affineExpr != vnniDim && affineExpr != dimBR)
        listMxNxK.push_back(affineExpr);
    }

    if (isUnPackedType && (affineExpr != dimBR)) {
      listMxNxK.push_back(affineExpr);
    }
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

static bool permutationCheck(vector::ContractionOp contractOp, Type elementType,
                             bool isSplat) {
  SmallVector<AffineMap, 3> contractMaps = contractOp.getIndexingMapsArray();
  AffineMap mapA = contractMaps[0];
  AffineMap mapB = contractMaps[1];

  bool isUnPackedType = elementType.isF32() || isSplat;
  bool isPackedType = (elementType.isF16() || elementType.isBF16() ||
                       elementType.isSignlessInteger(8)) &&
                      !isSplat;

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

  if (isPackedType) {
    // We match the pattern {Batch-reduction, vnni, M, N, K} or
    // {Batch-reduction, M, N, K, vnni} -> {Batch-reduction, M, K, vnni}
    auto c1 = inputDims[0] == outputDimsA[0];
    auto c2 = (inputDims[1] == outputDimsA[3]) &&
              (inputDims[2] == outputDimsA[1]) &&
              (inputDims[4] == outputDimsA[2]);
    auto c3 = (inputDims[1] == outputDimsA[1]) &&
              (inputDims[3] == outputDimsA[2]) &&
              (inputDims[4] == outputDimsA[3]);
    flag = flag && (c1 && (c2 || c3));
  }

  if (isUnPackedType) {
    // We match the pattern {Batch-reduction, M, N, K}
    // -> {Batch-reduction, M, K}
    auto c1 = inputDims[0] == outputDimsA[0];
    auto c2 =
        (inputDims[1] == outputDimsA[1]) && (inputDims[3] == outputDimsA[2]);
    flag = flag && (c1 && c2);
  }

  // mapB result dims
  auto resultsMapB = mapB.getNumResults();
  SmallVector<AffineDimExpr> outputDimsB;
  for (unsigned int i = 0; i < resultsMapB; i++) {
    auto affineExpr = dyn_cast<AffineDimExpr>(mapB.getResult(i));
    outputDimsB.push_back(affineExpr);
  }

  if (isPackedType) {
    // We match the pattern {Batch-reduction, vnni, M, N, K} or
    // {Batch-reduction, M, N, K, vnni} -> {Batch-reduction, K, N, vnni}
    auto c4 = inputDims[0] == outputDimsB[0];
    auto c5 = (inputDims[1] == outputDimsB[3]) &&
              (inputDims[4] == outputDimsB[1]) &&
              (inputDims[3] == outputDimsB[2]);
    auto c6 = (inputDims[2] == outputDimsB[2]) &&
              (inputDims[3] == outputDimsB[1]) &&
              (inputDims[4] == outputDimsB[3]);
    flag = flag && (c4 && (c5 || c6));
  }

  if (isUnPackedType) {
    // We match the pattern {Batch-reduction, M, N, K}
    // -> {Batch-reduction, K, N}
    auto c4 = inputDims[0] == outputDimsB[0];
    auto c5 =
        (inputDims[2] == outputDimsB[2]) && (inputDims[3] == outputDimsB[1]);
    flag = flag && (c4 && c5);
  }

  return flag;
}

static memref::AllocOp createMask(Location loc, PatternRewriter &rewriter,
                                  Value indexOp_c0, int64_t sizeFactor,
                                  Type i32Type) {

  auto intAttr = rewriter.getI32IntegerAttr(0xFFFF0000);
  auto maskConst =
      rewriter.create<mlir::arith::ConstantOp>(loc, i32Type, intAttr);
  auto mBcst = rewriter.create<vector::BroadcastOp>(
      loc, VectorType::get(sizeFactor, i32Type), maskConst);
  auto memrefMask = rewriter.create<mlir::memref::AllocOp>(
      loc, MemRefType::get({1}, VectorType::get(sizeFactor, i32Type)));
  rewriter.create<memref::StoreOp>(loc, mBcst, memrefMask,
                                   ValueRange{indexOp_c0});
  return memrefMask;
}

static Value performBroadcast(Location loc, PatternRewriter &rewriter,
                              Value vector, int64_t sizeFactor, int64_t vnni,
                              Type elementType, Type i32Type) {

  auto bitcastValue_i32 = rewriter.create<vector::BitCastOp>(
      loc, VectorType::get({1}, i32Type), vector);
  auto bcst_i32 = rewriter.create<vector::BroadcastOp>(
      loc, VectorType::get(sizeFactor, i32Type), bitcastValue_i32);
  auto value = rewriter.create<vector::BitCastOp>(
      loc, VectorType::get({sizeFactor * vnni}, elementType), bcst_i32);
  return value;
}

static Value performBitcast(Location loc, PatternRewriter &rewriter,
                            Value vector, int64_t sizeFactor, int64_t vnni,
                            Type elementType, Type i32Type, Type i16Type,
                            Value cst16) {

  auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
      loc, VectorType::get(sizeFactor, i16Type), vector);
  auto extend_i32 = rewriter.create<arith::ExtUIOp>(
      loc, VectorType::get(sizeFactor, i32Type), bitcast_i16);
  auto vectType = VectorType::get(sizeFactor, i32Type);
  auto shiftOp = rewriter.create<arith::ShLIOp>(
      loc, vectType, extend_i32,
      rewriter.create<vector::BroadcastOp>(loc, vectType, cst16));
  auto value = rewriter.create<vector::BitCastOp>(
      loc, VectorType::get(sizeFactor, rewriter.getF32Type()), shiftOp);

  return value;
}

static SmallVector<Value> performShuffle(Location loc,
                                         PatternRewriter &rewriter, Value vec1,
                                         Value vec2, int64_t elementSize,
                                         int64_t sizeFactor, int64_t vnni) {
  SmallVector<Value> vectors;
  if (elementSize == 16) {
    auto shuffle = rewriter.create<vector::ShuffleOp>(
        loc, VectorType::get({sizeFactor * vnni}, rewriter.getBF16Type()), vec1,
        vec2, ArrayRef<int64_t>{0,  16, 1,  17, 2,  18, 3,  19, 4,  20, 5,
                                21, 6,  22, 7,  23, 8,  24, 9,  25, 10, 26,
                                11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

    vectors.push_back(shuffle);
  }

  if (elementSize == 32) {
    auto shuffle1 = rewriter.create<vector::ShuffleOp>(
        loc, VectorType::get({sizeFactor * vnni}, rewriter.getBF16Type()), vec1,
        vec2, ArrayRef<int64_t>{0,  32, 1,  33, 2,  34, 3,  35, 8,  40, 9,
                                41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50,
                                19, 51, 24, 56, 25, 57, 26, 58, 27, 59});
    vectors.push_back(shuffle1);
    auto shuffle2 = rewriter.create<vector::ShuffleOp>(
        loc, VectorType::get({sizeFactor * vnni}, rewriter.getBF16Type()), vec1,
        vec2, ArrayRef<int64_t>{4,  36, 5,  37, 6,  38, 7,  39, 12, 44, 13,
                                45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54,
                                23, 55, 28, 60, 29, 61, 30, 62, 31, 63});
    vectors.push_back(shuffle2);
  }

  return vectors;
}

// We perform lowering based on the target architecture and type.
// (1) f32 - lowering is decided based on avx512 (1st preference) or
// avx2 support by the machine
// (2) bf16 - We support three different lowerings (a) avx512bf16dp - machine
// that has `avx512_bf16` support (both vnni packed and splat layout),
// (b) machine that has vcvtneebf162ps, vcvtneobf162ps, and
// vbcstnebf162ps instructions, and (c) fallback case - where we
// up-convert bf16 to f32, do FMAs, down-convert the accumulation back to bf16.
// (3) f16 - we support lowering only for machine that has vcvtneeph2ps,
// vcvtneoph2ps, and vbcstnesh2ps.
// (4) i8 - we support lowering only for the machine that has vpdpbssd.
struct MicroKernelsOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  MicroKernelsOp(MLIRContext *ctx, MicroKernelsOptions options)
      : OpRewritePattern<vector::ContractionOp>(ctx), options(options) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(
          contractOp, "Only combining kind 'ADD' is supported now.");

    // Check the vector contract operation satisfies the required loop
    // pattern.
    auto loops = getNestedLoop(contractOp);
    if (failed(loops))
      return rewriter.notifyMatchFailure(
          contractOp, "Invalid loop nest in contract pattern");

    auto nestedLoops = *loops;
    auto kForOp = nestedLoops[0];
    auto reductionForOp = nestedLoops[1];
    auto nForOp = nestedLoops[2];
    auto mForOp = nestedLoops[3];

    // Get the Lhs, Rhs, and Acc of the contract operation.
    auto vectorReadOpAcc =
        reductionForOp.getInitArgs()[0].getDefiningOp<vector::TransferReadOp>();
    auto vectorReadOpLhs =
        contractOp.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto vectorReadOpRhs =
        contractOp.getRhs().getDefiningOp<vector::TransferReadOp>();

    // Retrive the element type (f32 or bf16 or f16)
    auto subviewOpAcc =
        vectorReadOpAcc.getOperand(0).getDefiningOp<memref::SubViewOp>();
    auto subviewOpLhs =
        vectorReadOpLhs.getOperand(0).getDefiningOp<memref::SubViewOp>();

    auto elementType =
        (cast<MemRefType>(subviewOpLhs.getType())).getElementType();
    auto outsElementType =
        (cast<MemRefType>(subviewOpAcc.getType())).getElementType();

    // We get target architecture and decide on uKernel lowering using flags
    bool avx512 = vnni::utils::hasAVX512();
    bool avx2 = vnni::utils::hasAVX2();
    bool sve256 = vnni::utils::hasSVE256();
    bool sve512 = vnni::utils::hasSVE512();

    // disable avx512, if target feature is avx2
    if (options.targetFeature == "avx2")
      avx512 = false;

    int64_t sizeFactor = (avx512 || sve512) ? 16 : (avx2 || sve256) ? 8 : 0;

    bool isF32 = elementType.isF32();
    bool isF16 = elementType.isF16();
    bool isBF16 = elementType.isBF16();
    bool isI8 = elementType.isSignlessInteger(8);

    bool isPackedType = isF16 || isBF16 || isI8;

    if (sizeFactor == 0)
      return rewriter.notifyMatchFailure(
          contractOp, "AVX512 or AVX2 or SVE512/256 instruction set is not available or "
                      "lowering is not available for this target machine.");

    if ((sve256 || sve512) && isPackedType)
      return rewriter.notifyMatchFailure(
            contractOp,
            "only FP32 type lowering is supported for AARCH64(ARM) machines.");

    int64_t vnniFactor = (isBF16 || isF16) ? 2 : isI8 ? 4 : 1;
    bool isSplat = false;

    // Check the operation type MatMul, B-MatMul, or BR-MatMul
    SmallVector<vector::IteratorType> contractIteratorTypes =
        contractOp.getIteratorTypesArray();
    int reductionCount =
        std::count(contractIteratorTypes.begin(), contractIteratorTypes.end(),
                   vector::IteratorType::reduction);
    auto lhsType = dyn_cast<ShapedType>(vectorReadOpLhs.getType());
    auto rhsType = dyn_cast<ShapedType>(vectorReadOpRhs.getType());

    if (isPackedType && reductionCount == 2 && lhsType.getRank() == 3 &&
        rhsType.getRank() == 3)
      isSplat = true;

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          contractOp, "Batch matmul operation not supported yet");

    if (isPackedType && !isSplat) {
      if (reductionCount == 2)
        return rewriter.notifyMatchFailure(
            contractOp, "Batch reduce matmul operation without vnni layout");

      if (reductionCount > 3)
        return rewriter.notifyMatchFailure(
            contractOp, "The vector contract operation is not a gemm");

      if (reductionCount == 3 &&
          (lhsType.getRank() != 4 || rhsType.getRank() != 4))
        return rewriter.notifyMatchFailure(
            contractOp,
            "Invalid rank for batch reduce operation with vnni layout");
    }

    if (isF32) {
      if (reductionCount > 2)
        return rewriter.notifyMatchFailure(
            contractOp, "The vector contract operation is not a gemm");

      if (reductionCount == 2 &&
          (lhsType.getRank() != 3 || rhsType.getRank() != 3))
        return rewriter.notifyMatchFailure(
            contractOp, "Invalid rank for batch reduce operation");
    }

    if (!(isF32 || isPackedType))
      return rewriter.notifyMatchFailure(
          contractOp, "The type is not F32 or F16 or BF16 or I8");

    bool bf16dp = false;
    bool srf = false;
    bool fallback = false;

    if (isPackedType) {
      auto cpuName = vnni::utils::getTargetArchName();
      if (cpuName == "SRF")
        srf = true;

      if (cpuName == "CPX_SPR" && avx512)
        bf16dp = true;

      if (!(srf || bf16dp))
        fallback = true;
    }

    if ((isF16 || isI8) && !(srf))
      return rewriter.notifyMatchFailure(
          contractOp, "F16/I8 type is supported only for SRF kind of machines");

    if (isSplat && !(bf16dp || srf))
      return rewriter.notifyMatchFailure(
          contractOp, "Only Splat-bf16 avx512-dp + SRF lowering is supported");

    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;
    int64_t vnni = 0;

    if (isPackedType && !isSplat) {
      M = lhsType.getDimSize(lhsType.getRank() - 3);
      N = rhsType.getDimSize(lhsType.getRank() - 2);
      K = lhsType.getDimSize(lhsType.getRank() - 2);
      vnni = lhsType.getDimSize(lhsType.getRank() - 1);

      // TODO: We need the N tile size to be divisible by 16 for avx2
      // fallback case. So that it ensures, LLVM find a pattern and lowers to
      // assembly without `vpinsrt`. This issue has to be fixed.
      if (((N % 16) != 0) && fallback && !avx512)
        return rewriter.notifyMatchFailure(
            contractOp, "N tile size divisible by 16 are only supported");

      if (vnni != 2 && isBF16)
        return rewriter.notifyMatchFailure(
            contractOp, "Only VNNI layout=2 is supported for bf16, now");

      if (vnni != 4 && isI8)
        return rewriter.notifyMatchFailure(
            contractOp, "Only VNNI layout=4 is supported for i8, now");

      if (K != (vnni / vnniFactor))
        return rewriter.notifyMatchFailure(
            contractOp, "K tile size should be equal to VNNI layout");
    }

    if (isF32 || isSplat) {
      M = lhsType.getDimSize(lhsType.getRank() - 2);
      N = rhsType.getDimSize(lhsType.getRank() - 1);
      K = lhsType.getDimSize(lhsType.getRank() - 1);
      vnni = vnniFactor;

      if (K != vnni && bf16dp)
        return rewriter.notifyMatchFailure(
            contractOp, "K tile size should be equal to vnni");

      if (K != 1 && (srf || isF32))
        return rewriter.notifyMatchFailure(
            contractOp, "K tile size should be equal to one");
    }

    if (avx512 && (N < 16))
      return rewriter.notifyMatchFailure(
          contractOp, "N tile size should be >= 16 for avx512 targets");

    if (!avx512 && avx2 && (N < 8))
      return rewriter.notifyMatchFailure(
          contractOp, "N tile size should be >= 8 for avx2 targets");

    if (isTransposedMatrix(contractOp, elementType, isSplat))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Matrices shoudn't be transposed.");

    if (!permutationCheck(contractOp, elementType, isSplat))
      return rewriter.notifyMatchFailure(
          contractOp, "Affine map permutation not supported.");

    // Lowering is done based on M and N tile sizes. If M >= N: load all B
    // matrix then broadcast A ony-by-one + FMA.
    // If N > M: perform opposite. Broadcast A matrix then load B one-by-
    // one + FMA.
    // Following this kind of lowering, we reduce the register loads by
    // stacking the less B loads or less A broadcasts and do the larger B
    // loads or A broadcast in a LIFO manner. Finally, it helps in reducing
    // the probablity of register spills.
    bool mDriven = true;
    int64_t nBlock = N / sizeFactor;

    if (nBlock > M)
      mDriven = false;

    rewriter.setInsertionPoint(mForOp);
    auto i32Type = rewriter.getIntegerType(32);
    auto i16Type = rewriter.getIntegerType(16);
    Value indexOp_c0 =
        rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), 0);
    Value indexOp_c1 =
        rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), 1);
    auto cst16 = rewriter.create<arith::ConstantOp>(
        reductionForOp.getLoc(), rewriter.getIntegerAttr(i32Type, 16));

    // Creating the mask for doing bitwise `and` operation + store them
    // in memory(target: fallback)
    auto memrefMask = createMask(reductionForOp.getLoc(), rewriter, indexOp_c0,
                                 sizeFactor, i32Type);

    rewriter.setInsertionPoint(reductionForOp);
    llvm::SmallVector<Value> loopItrArgs;

    // C matrix load:
    // f32 - just load the matrix as f32 type
    // bf16 and f16 - load the matrix up-convert to f32
    if (outsElementType.isBF16() && !isSplat) {
      for (int j = 0; j < N; j = j + sizeFactor) {
        for (int i = 0; i < M; i++) {
          Value indexOp_A = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), i);
          Value indexOp_B = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j);
          auto valueCRow = rewriter.create<vector::LoadOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
              ValueRange{indexOp_A, indexOp_B});
          auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, i16Type),
              valueCRow);
          auto extend_i32 = rewriter.create<arith::ExtUIOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
              bitcast_i16);
          auto vectType = VectorType::get(sizeFactor, i32Type);
          auto shiftOp = rewriter.create<arith::ShLIOp>(
              reductionForOp.getLoc(), vectType, extend_i32,
              rewriter.create<vector::BroadcastOp>(reductionForOp.getLoc(),
                                                   vectType, cst16));
          auto f32CVector = rewriter.create<vector::BitCastOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, rewriter.getF32Type()), shiftOp);
          loopItrArgs.push_back(f32CVector);
        }
      }
    }

    if (outsElementType.isF16() && !isSplat) {
      for (int j = 0; j < N; j = j + sizeFactor) {
        for (int i = 0; i < M; i++) {
          Value indexOp_A = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), i);
          Value indexOp_B = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j);
          auto valueCRow = rewriter.create<vector::LoadOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
              ValueRange{indexOp_A, indexOp_B});
          auto f32CVector = rewriter.create<arith::ExtFOp>(
              reductionForOp.getLoc(),
              VectorType::get({8}, rewriter.getF32Type()),
              valueCRow.getResult());
          loopItrArgs.push_back(f32CVector);
        }
      }
    }

    if ((outsElementType.isF32() || outsElementType.isSignlessInteger(32)) &&
        !isSplat) {
      for (int j = 0; j < N; j = j + sizeFactor) {
        for (int i = 0; i < M; i++) {
          Value indexOp_A = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), i);
          Value indexOp_B = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j);
          auto valueCRow = rewriter.create<vector::LoadOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
              ValueRange{indexOp_A, indexOp_B});
          loopItrArgs.push_back(valueCRow);
        }
      }
    }

    // For splat layout we init the iter args with zero vector for the
    // equivalent number of C loads.
    if (isSplat) {
      for (int j = 0; j < N; j = j + sizeFactor) {
        for (int i = 0; i < M; i++) {
          auto zeroAttr = DenseElementsAttr::get(
              VectorType::get({sizeFactor}, rewriter.getF32Type()), 0.0f);
          auto cst = rewriter.create<arith::ConstantOp>(
              reductionForOp.getLoc(),
              VectorType::get({sizeFactor}, rewriter.getF32Type()), zeroAttr);
          loopItrArgs.push_back(cst);
        }
      }
    }

    SmallVector<Value> evenFMAs;
    SmallVector<Value> oddFMAs;
    SmallVector<Value> matf32;

    // Code to re-create the reduction and k loop with iter args
    auto newReductionForOp = rewriter.create<scf::ForOp>(
        reductionForOp.getLoc(), reductionForOp.getLowerBound(),
        reductionForOp.getUpperBound(), reductionForOp.getStep(), loopItrArgs,
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp,
            Value ivNewReductionForOp, ValueRange iterArgsNewReductionForOp) {
          auto newKForOp = rewriter.create<scf::ForOp>(
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

                IRMapping rhsMapping;
                rhsMapping.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                rhsMapping.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(2),
                    ivNewKForOp);
                auto rhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

                // Load i32 mask that is created and stored in memory earlier
                auto maskBcst = rewriter.create<memref::LoadOp>(
                    kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                    memrefMask, ValueRange{indexOp_c0});

                // i1 type mask
                auto boolAttr_16 = DenseElementsAttr::get(
                    VectorType::get({16}, rewriter.getI1Type()),
                    ArrayRef<bool>{false, true, false, true, false, true, false,
                                   true, false, true, false, true, false, true,
                                   false, true});
                auto i1Mask_16 = rewriter.create<arith::ConstantOp>(
                    kForOp.getLoc(),
                    VectorType::get({16}, rewriter.getI1Type()), boolAttr_16);
                auto boolAttr_2 = DenseElementsAttr::get(
                    VectorType::get(2, rewriter.getI1Type()),
                    ArrayRef<bool>{false, true});
                auto i1Mask_2 = rewriter.create<arith::ConstantOp>(
                    kForOp.getLoc(), VectorType::get(2, rewriter.getI1Type()),
                    boolAttr_2);

                // ZeroAttr is not needed for i8 type lowering on ARL machine,
                // may be need in future for lowering on other machine.
                FloatAttr zeroAttr;
                if (!isI8) {
                  zeroAttr = rewriter.getFloatAttr(elementType, 0.0);
                }

                // Destination type
                mlir::VectorType dstType =
                    mlir::VectorType::get(sizeFactor, rewriter.getF32Type());

                if (isI8)
                  dstType =
                      mlir::VectorType::get(sizeFactor, rewriter.getI32Type());

                llvm::SmallVector<OpFoldResult> strides = {
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
                llvm::SmallVector<OpFoldResult> sizes = {
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                    rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
                if (!isSplat) {
                  // uKernel lowering for f32 type. M -> N.
                  if (isF32 && mDriven) {
                    // Load elements of B matrix and store in a DS
                    for (int j = 0; j < N; j = j + sizeFactor) {
                      Value indexOp_j = rewriter.create<arith::ConstantIndexOp>(
                          reductionForOp.getLoc(), j);
                      auto valueRow = rewriterNewKForOp.create<vector::LoadOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor, elementType),
                          rhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_c0, indexOp_j});
                      matf32.push_back(valueRow);
                    }

                    // Load elements of A matrix, do FMA, and store FMA in a DS
                    for (int i = 0; i < M; i++) {
                      Value indexOp_i = rewriter.create<arith::ConstantIndexOp>(
                          reductionForOp.getLoc(), i);
                      auto valueRow = rewriterNewKForOp.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get({vnni}, elementType),
                          lhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_i, indexOp_c0});
                      auto bcst_i32 =
                          rewriterNewKForOp.create<vector::BroadcastOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor,
                                              rewriterNewKForOp.getF32Type()),
                              valueRow);

                      // Iterate through the stored elements of B and do FMA
                      for (int j = 0; j < (N / sizeFactor); j++) {
                        auto fmaOdd = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), bcst_i32, matf32[j],
                            iterArgsNewKForOp[i + (j * M)]);
                        oddFMAs.push_back(fmaOdd);
                      }
                    }

                    // Re-arrange the stored FMAs in order of N -> M.
                    // We load C matrix with N -> M. For example: c[0][0],
                    // c[1][0] We do FMA as M -> N order { [0][0], [0][16] ...}.
                    // So, shuffling the M -> N to N -> M order
                    for (int j = 0; j < (N / sizeFactor); j++) {
                      for (int i = 0; i < M; i++) {
                        evenFMAs.push_back(oddFMAs[j + (i * (N / sizeFactor))]);
                      }
                    }
                  } else if (isF32 && !mDriven) { // N -> M.
                    // Load elements of A matrix and store in a DS
                    for (int i = 0; i < M; i++) {
                      Value indexOp_i = rewriter.create<arith::ConstantIndexOp>(
                          reductionForOp.getLoc(), i);
                      auto valueRow = rewriterNewKForOp.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get({vnni}, elementType),
                          lhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_i, indexOp_c0});
                      auto bcst_i32 =
                          rewriterNewKForOp.create<vector::BroadcastOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor,
                                              rewriterNewKForOp.getF32Type()),
                              valueRow);
                      matf32.push_back(bcst_i32);
                    }

                    // Load elements of B matrix, do FMA, and store FMA in a DS
                    for (int j = 0, k = 0; j < N; j = j + sizeFactor) {
                      Value indexOp_j = rewriter.create<arith::ConstantIndexOp>(
                          reductionForOp.getLoc(), j);
                      auto valueRow = rewriterNewKForOp.create<vector::LoadOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor, elementType),
                          rhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_c0, indexOp_j});
                      matf32.push_back(valueRow);

                      // Iterate through the stored elements of B and do FMA
                      for (int i = 0; i < M; i++) {
                        auto fmaOdd = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), matf32[i], valueRow,
                            iterArgsNewKForOp[k]);
                        k++;
                        evenFMAs.push_back(fmaOdd);
                      }
                    }
                  }

                  // bf16 type + avx512. uKernel lowering for machines like
                  // cpx (zen5) to target avx512bf16dp.
                  if (bf16dp || isI8) {

                    if (mDriven) { // M -> N
                      // Load elements of B matrix and store in a DS
                      for (int j = 0; j < N; j = j + sizeFactor) {
                        Value indexOp_j =
                            rewriter.create<arith::ConstantIndexOp>(
                                reductionForOp.getLoc(), j);
                        auto valueRow =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get({sizeFactor * vnni},
                                                elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c0, indexOp_j,
                                           indexOp_c0});
                        matf32.push_back(valueRow);
                      }

                      // Load elements of A matrix, do FMA, and store in a DS
                      for (int i = 0; i < M; i++) {
                        Value indexOp_i =
                            rewriter.create<arith::ConstantIndexOp>(
                                reductionForOp.getLoc(), i);
                        auto valueRow =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get({vnni}, elementType),
                                lhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_i, indexOp_c0,
                                           indexOp_c0});
                        auto bitcastValue_i32 =
                            rewriterNewKForOp.create<vector::BitCastOp>(
                                kForOp.getLoc(), VectorType::get({1}, i32Type),
                                valueRow);
                        auto bcst_i32 =
                            rewriterNewKForOp.create<vector::BroadcastOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor, i32Type),
                                bitcastValue_i32);
                        auto valuef32 =
                            rewriterNewKForOp.create<vector::BitCastOp>(
                                kForOp.getLoc(),
                                VectorType::get({sizeFactor * vnni},
                                                elementType),
                                bcst_i32);

                        if (isBF16) {
                          for (int j = 0; j < (N / sizeFactor); j++) {
                            auto dp =
                                rewriter.create<mlir::x86vector::DotBF16Op>(
                                    kForOp.getLoc(), dstType,
                                    iterArgsNewKForOp[i + (j * M)], valuef32,
                                    matf32[j]);
                            oddFMAs.push_back(dp);
                          }
                        }

                        if (isI8) {
                          for (int j = 0; j < (N / sizeFactor); j++) {
                            auto dp =
                                rewriter.create<mlir::x86vector::DotInt8Op>(
                                    kForOp.getLoc(), dstType,
                                    iterArgsNewKForOp[i + (j * M)], valuef32,
                                    matf32[j]);
                            oddFMAs.push_back(dp);
                          }
                        }
                      }

                      // Re-arrange the stored FMAs in order of N -> M.
                      // We load C matrix with N -> M. For example: c[0][0],
                      // c[1][0] We do dp as M -> N order { [0][0], [0][16]
                      // ...}. So, shuffling the M -> N to N -> M order
                      for (int j = 0; j < (N / sizeFactor); j++) {
                        for (int i = 0; i < M; i++) {
                          evenFMAs.push_back(
                              oddFMAs[j + (i * (N / sizeFactor))]);
                        }
                      }

                    } else { // N -> M
                      for (int i = 0; i < M; i++) {
                        Value indexOp_i =
                            rewriter.create<arith::ConstantIndexOp>(
                                reductionForOp.getLoc(), i);
                        auto valueRow =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get({vnni}, elementType),
                                lhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_i, indexOp_c0,
                                           indexOp_c0});
                        auto bitcastValue_i32 =
                            rewriterNewKForOp.create<vector::BitCastOp>(
                                kForOp.getLoc(), VectorType::get({1}, i32Type),
                                valueRow);
                        auto bcst_i32 =
                            rewriterNewKForOp.create<vector::BroadcastOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor, i32Type),
                                bitcastValue_i32);
                        auto valuef32 =
                            rewriterNewKForOp.create<vector::BitCastOp>(
                                kForOp.getLoc(),
                                VectorType::get({sizeFactor * vnni},
                                                elementType),
                                bcst_i32);
                        matf32.push_back(valuef32);
                      }

                      for (int j = 0, k = 0; j < N; j = j + sizeFactor) {
                        Value indexOp_j =
                            rewriter.create<arith::ConstantIndexOp>(
                                reductionForOp.getLoc(), j);
                        auto valueRow =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get({sizeFactor * vnni},
                                                elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c0, indexOp_j,
                                           indexOp_c0});

                        if (isBF16) {
                          for (int i = 0; i < M; i++) {
                            auto dp =
                                rewriter.create<mlir::x86vector::DotBF16Op>(
                                    kForOp.getLoc(), dstType,
                                    iterArgsNewKForOp[k], matf32[i], valueRow);
                            k++;
                            evenFMAs.push_back(dp);
                          }
                        }

                        if (isI8) {
                          for (int i = 0; i < M; i++) {
                            auto dp =
                                rewriter.create<mlir::x86vector::DotInt8Op>(
                                    kForOp.getLoc(), dstType,
                                    iterArgsNewKForOp[k], matf32[i], valueRow);
                            k++;
                            evenFMAs.push_back(dp);
                          }
                        }
                      }
                    }
                  }

                  // bf16 type + fallback + avx512. uKernel lowering for
                  // machines that has no support for avx512bf16dp and AMX
                  if (fallback && avx512) {
                    // Load odd elements of B Matrix and store in a DS
                    for (int j = 0; j < N; j = j + sizeFactor) {
                      Value indexOp_cj =
                          rewriter.create<arith::ConstantIndexOp>(
                              reductionForOp.getLoc(), j);
                      auto valueRow = rewriter.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(32, elementType),
                          rhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_c0, indexOp_cj,
                                     indexOp_c0});
                      auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                          valueRow);
                      auto andOpB = rewriter.create<arith::AndIOp>(
                          kForOp.getLoc(), bitcast_i32, maskBcst);
                      auto oddB = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor, rewriter.getF32Type()),
                          andOpB);
                      matf32.push_back(oddB);
                    }

                    // Load odd elements of A Matrix, perform fma (odd), and
                    // store to a DS
                    for (int i = 0; i < M; i++) {
                      Value indexOp_ci =
                          rewriter.create<arith::ConstantIndexOp>(
                              reductionForOp.getLoc(), i);
                      auto denseAttr = DenseElementsAttr::get(
                          VectorType::get(vnni, elementType), zeroAttr);
                      auto passThru = rewriter.create<arith::ConstantOp>(
                          kForOp.getLoc(), VectorType::get(vnni, elementType),
                          denseAttr);
                      auto maskedLoad = rewriter.create<vector::MaskedLoadOp>(
                          kForOp.getLoc(), VectorType::get(vnni, elementType),
                          lhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_ci, indexOp_c0,
                                     indexOp_c0},
                          i1Mask_2, passThru);
                      auto bitcast_f32 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get(1, rewriter.getF32Type()),
                          maskedLoad);
                      auto oddA = rewriterNewKForOp.create<vector::BroadcastOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor,
                                          rewriterNewKForOp.getF32Type()),
                          bitcast_f32);

                      // Odd FMAs
                      for (int j = 0; j < (N / sizeFactor); j++) {
                        auto fmaOdd = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), oddA, matf32[j],
                            iterArgsNewKForOp[i + (j * M)]);
                        oddFMAs.push_back(fmaOdd);
                      }
                    }

                    // Clear B matrix odd loads to save even loads
                    matf32.clear();

                    // Load even elements of B Matrix and store in a DS
                    for (int j = 0; j < N; j = j + sizeFactor) {
                      Value indexOp_cj =
                          rewriter.create<arith::ConstantIndexOp>(
                              reductionForOp.getLoc(), j);
                      auto valueRow = rewriter.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(32, elementType),
                          rhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_c0, indexOp_cj,
                                     indexOp_c0});
                      auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                          valueRow);
                      auto shiftOpB = rewriter.create<arith::ShLIOp>(
                          kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                          bitcast_i32,
                          rewriter.create<vector::BroadcastOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor, i32Type), cst16));
                      auto evenB = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor, rewriter.getF32Type()),
                          shiftOpB);
                      matf32.push_back(evenB);
                    }

                    SmallVector<Value> evenFMAs_swap;
                    // Load even elements of A Matrix, perform fma (even), and
                    // store to a DS
                    for (int i = 0, k = 0; i < M; i++) {
                      Value indexOp_ci =
                          rewriter.create<arith::ConstantIndexOp>(
                              reductionForOp.getLoc(), i);
                      auto valueRow = rewriter.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(vnni, elementType),
                          lhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_ci, indexOp_c0,
                                     indexOp_c0});
                      auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(), VectorType::get(1, i32Type),
                          valueRow);
                      auto bcstValue =
                          rewriterNewKForOp.create<vector::BroadcastOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor, i32Type),
                              bitcast_i32);
                      auto shiftOp = rewriter.create<arith::ShLIOp>(
                          kForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                          bcstValue,
                          rewriter.create<vector::BroadcastOp>(
                              kForOp.getLoc(),
                              VectorType::get(sizeFactor, i32Type), cst16));
                      auto evenA = rewriter.create<vector::BitCastOp>(
                          kForOp.getLoc(),
                          VectorType::get(sizeFactor, rewriter.getF32Type()),
                          shiftOp);

                      // Even FMAs
                      for (int j = 0; j < (N / sizeFactor); j++) {
                        auto fmaEven = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), evenA, matf32[j], oddFMAs[k]);
                        evenFMAs_swap.push_back(fmaEven);
                        k++;
                      }
                    }

                    //  Re-arrange the stored FMAs in order
                    for (int j = 0; j < (N / sizeFactor); j++) {
                      for (int i = 0; i < M; i++) {
                        evenFMAs.push_back(
                            evenFMAs_swap[j + (i * (N / sizeFactor))]);
                      }
                    }
                  }

                  // uKernel lowering for AVX2  machines
                  // Target: (a) f16 and bf16 for srf kind of machines
                  // (b) bf16 fallback + avx2 instructions.
                  // TODO: update lowering based on M & N. Now it is
                  // default to M -> N
                  if ((srf && !isI8) || (fallback && avx2 && !avx512)) {
                    // Load odd elements of A Matrix and store in a DS
                    for (int i = 0; i < M; i++) {
                      Value oddA;

                      if (fallback) {
                        Value indexOp_ci =
                            rewriter.create<arith::ConstantIndexOp>(
                                reductionForOp.getLoc(), i);
                        auto valueRow = rewriter.create<vector::LoadOp>(
                            kForOp.getLoc(), VectorType::get(2, elementType),
                            lhsClone->getResult(0),
                            ValueRange{indexOp_c0, indexOp_ci, indexOp_c0,
                                       indexOp_c0});
                        auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                            kForOp.getLoc(), VectorType::get(1, i32Type),
                            valueRow);
                        auto bcst_i32 =
                            rewriterNewKForOp.create<vector::BroadcastOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor, i32Type),
                                bitcast_i32);
                        auto andOpA = rewriter.create<arith::AndIOp>(
                            kForOp.getLoc(), bcst_i32, maskBcst);
                        oddA = rewriter.create<vector::BitCastOp>(
                            kForOp.getLoc(),
                            VectorType::get(sizeFactor, rewriter.getF32Type()),
                            andOpA);
                      }

                      if (srf) {
                        SmallVector<OpFoldResult> offsets = {
                            rewriter.getIndexAttr(0),
                            rewriter.getIndexAttr(i),
                            rewriter.getIndexAttr(0),
                            rewriter.getIndexAttr(1),
                        };
                        auto subview = rewriter.create<memref::SubViewOp>(
                            kForOp.getLoc(), lhsClone->getResult(0), offsets,
                            sizes, strides);
                        oddA =
                            rewriter.create<mlir::x86vector::BcstToPackedF32Op>(
                                kForOp.getLoc(), dstType, subview);
                      }

                      matf32.push_back(oddA);
                    }

                    // Load odd elements of B-Matrix, perform fma (odd), and
                    // store to a DS
                    for (int j = 0, k = 0; j < N; j = j + sizeFactor) {
                      Value oddB;

                      if (fallback) {
                        Value indexOp_cj =
                            rewriter.create<arith::ConstantIndexOp>(
                                reductionForOp.getLoc(), j);
                        auto valueRow = rewriter.create<vector::LoadOp>(
                            kForOp.getLoc(), VectorType::get(16, elementType),
                            rhsClone->getResult(0),
                            ValueRange{indexOp_c0, indexOp_c0, indexOp_cj,
                                       indexOp_c0});
                        auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
                            kForOp.getLoc(), VectorType::get(16, i16Type),
                            valueRow);
                        auto bitcast_i16_andOpA =
                            rewriter.create<vector::BitCastOp>(
                                kForOp.getLoc(), VectorType::get(16, i16Type),
                                matf32[0]);
                        auto selectOp = rewriter.create<arith::SelectOp>(
                            kForOp.getLoc(), VectorType::get({16}, i16Type),
                            i1Mask_16, bitcast_i16, bitcast_i16_andOpA);
                        oddB = rewriter.create<vector::BitCastOp>(
                            kForOp.getLoc(),
                            VectorType::get(sizeFactor, rewriter.getF32Type()),
                            selectOp);
                      }

                      if (srf) {
                        SmallVector<OpFoldResult> offsets = {
                            rewriter.getIndexAttr(0),
                            rewriter.getIndexAttr(0),
                            rewriter.getIndexAttr(j),
                            rewriter.getIndexAttr(0),
                        };
                        llvm::SmallVector<OpFoldResult> sizes = {
                            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                            rewriter.getIndexAttr(8), rewriter.getIndexAttr(2)};
                        auto subview = rewriter.create<memref::SubViewOp>(
                            kForOp.getLoc(), rhsClone->getResult(0), offsets,
                            sizes, strides);
                        oddB = rewriter.create<
                            mlir::x86vector::CvtPackedOddIndexedToF32Op>(
                            kForOp.getLoc(), dstType, subview);
                      }

                      // Odd FMAs
                      for (int i = 0; i < M; i++) {
                        auto fmaOdd = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), matf32[i], oddB,
                            iterArgsNewKForOp[k]);
                        k++;
                        oddFMAs.push_back(fmaOdd);
                      }
                    }

                    // Clear A matrix  odd loads to save even loads
                    matf32.clear();

                    // Load even elements of A Matrix and store in a DS
                    for (int i = 0; i < M; i++) {
                      Value evenA;

                      if (fallback) {
                        Value indexOp_ci =
                            rewriter.create<arith::ConstantIndexOp>(
                                reductionForOp.getLoc(), i);
                        auto valueRow = rewriter.create<vector::LoadOp>(
                            kForOp.getLoc(), VectorType::get(2, elementType),
                            lhsClone->getResult(0),
                            ValueRange{indexOp_c0, indexOp_ci, indexOp_c0,
                                       indexOp_c0});
                        auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                            kForOp.getLoc(), VectorType::get(1, i32Type),
                            valueRow);
                        auto shift = rewriter.create<arith::ShLIOp>(
                            kForOp.getLoc(), VectorType::get(1, i32Type),
                            bitcast_i32,
                            rewriter.create<vector::BroadcastOp>(
                                kForOp.getLoc(), VectorType::get(1, i32Type),
                                cst16));
                        auto bcstShift =
                            rewriterNewKForOp.create<vector::BroadcastOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor, i32Type), shift);
                        evenA = rewriter.create<vector::BitCastOp>(
                            kForOp.getLoc(),
                            VectorType::get(sizeFactor, rewriter.getF32Type()),
                            bcstShift);
                      }

                      if (srf) {
                        SmallVector<OpFoldResult> offsets = {
                            rewriter.getIndexAttr(0),
                            rewriter.getIndexAttr(i),
                            rewriter.getIndexAttr(0),
                            rewriter.getIndexAttr(0),
                        };

                        auto subview = rewriter.create<memref::SubViewOp>(
                            kForOp.getLoc(), lhsClone->getResult(0), offsets,
                            sizes, strides);
                        evenA =
                            rewriter.create<mlir::x86vector::BcstToPackedF32Op>(
                                kForOp.getLoc(), dstType, subview);
                      }

                      matf32.push_back(evenA);
                    }

                    // Load even elements of B-Matrix, perform fma (even), and
                    // store to a DS.
                    // TODO: We increment `j` value by 16 so that LLVM finds
                    // correct pattern for lowering to assembly. To be fixed.
                    if (fallback) {
                      for (int j = 0, k = 0; j < N; j = j + 16) {

                        Value indexOp_cj =
                            rewriter.create<arith::ConstantIndexOp>(
                                reductionForOp.getLoc(), j);
                        auto valueRow = rewriter.create<vector::LoadOp>(
                            kForOp.getLoc(), VectorType::get(32, elementType),
                            rhsClone->getResult(0),
                            ValueRange{indexOp_c0, indexOp_c0, indexOp_cj,
                                       indexOp_c0});
                        auto bitcast_i32 = rewriter.create<vector::BitCastOp>(
                            kForOp.getLoc(), VectorType::get(16, i32Type),
                            valueRow);
                        auto shift = rewriter.create<arith::ShLIOp>(
                            kForOp.getLoc(), VectorType::get(16, i32Type),
                            bitcast_i32,
                            rewriter.create<vector::BroadcastOp>(
                                kForOp.getLoc(), VectorType::get(16, i32Type),
                                cst16));
                        auto evenB_16 = rewriter.create<vector::BitCastOp>(
                            kForOp.getLoc(),
                            VectorType::get(16, rewriter.getF32Type()), shift);

                        auto offsetsAttr = rewriter.getI64ArrayAttr({0});
                        auto offsetsAttr_8 =
                            rewriter.getI64ArrayAttr({sizeFactor});
                        auto sizesAttr = rewriter.getI64ArrayAttr({sizeFactor});
                        auto stridesAttr = rewriter.getI64ArrayAttr({1});
                        auto evenB1 =
                            rewriter.create<vector::ExtractStridedSliceOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor,
                                                rewriter.getF32Type()),
                                evenB_16, offsetsAttr, sizesAttr, stridesAttr);
                        auto evenB2 =
                            rewriter.create<vector::ExtractStridedSliceOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor,
                                                rewriter.getF32Type()),
                                evenB_16, offsetsAttr_8, sizesAttr,
                                stridesAttr);

                        // Even FMAs
                        for (int i = 0; i < M; i++) {
                          auto fmaEven = rewriter.create<vector::FMAOp>(
                              kForOp.getLoc(), matf32[i], evenB1, oddFMAs[k]);
                          k++;
                          evenFMAs.push_back(fmaEven);
                        }

                        for (int i = 0; i < M; i++) {
                          auto fmaEven = rewriter.create<vector::FMAOp>(
                              kForOp.getLoc(), matf32[i], evenB2, oddFMAs[k]);
                          k++;
                          evenFMAs.push_back(fmaEven);
                        }
                      }
                    }

                    if (srf) {
                      for (int j = 0, k = 0; j < N; j = j + sizeFactor) {

                        SmallVector<OpFoldResult> offsets = {
                            rewriter.getIndexAttr(0),
                            rewriter.getIndexAttr(0),
                            rewriter.getIndexAttr(j),
                            rewriter.getIndexAttr(0),
                        };
                        llvm::SmallVector<OpFoldResult> sizes = {
                            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                            rewriter.getIndexAttr(8), rewriter.getIndexAttr(2)};
                        auto subview = rewriter.create<memref::SubViewOp>(
                            kForOp.getLoc(), rhsClone->getResult(0), offsets,
                            sizes, strides);
                        auto evenB = rewriter.create<
                            mlir::x86vector::CvtPackedEvenIndexedToF32Op>(
                            kForOp.getLoc(), dstType, subview);

                        // Even FMAs
                        for (int i = 0; i < M; i++) {
                          auto fmaEven = rewriter.create<vector::FMAOp>(
                              kForOp.getLoc(), matf32[i], evenB, oddFMAs[k]);
                          k++;
                          evenFMAs.push_back(fmaEven);
                        }
                      }
                    }
                  }
                }

                if (isSplat && srf) {
                  llvm::SmallVector<OpFoldResult> strides_splat = {
                      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                      rewriter.getIndexAttr(1)};
                  llvm::SmallVector<OpFoldResult> sizes_splat = {
                      rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                      rewriter.getIndexAttr(1)};

                  if (mDriven) { // M -> N
                    // Load B-Matrix even+odd store to a DS
                    for (int j = 0; j < N; j = j + (sizeFactor * 2)) {
                      SmallVector<OpFoldResult> offsets = {
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(j),
                      };

                      // For case B-matrix with one vector<8xbf16>, we do xmm
                      // load of even + odd elements followed by a shuffle
                      if ((N - j) <= 8) {
                        llvm::SmallVector<OpFoldResult> sizes = {
                            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                            rewriter.getIndexAttr(8)};

                        auto subview = rewriter.create<memref::SubViewOp>(
                            kForOp.getLoc(), rhsClone->getResult(0), offsets,
                            sizes, strides_splat);

                        mlir::VectorType dstType = mlir::VectorType::get(
                            {sizeFactor / 2}, rewriter.getF32Type());

                        auto evenB = rewriter.create<
                            mlir::x86vector::CvtPackedEvenIndexedToF32Op>(
                            kForOp.getLoc(), dstType, subview);

                        auto oddB = rewriter.create<
                            mlir::x86vector::CvtPackedOddIndexedToF32Op>(
                            kForOp.getLoc(), dstType, subview);

                        auto shuffle = rewriter.create<vector::ShuffleOp>(
                            kForOp.getLoc(),
                            VectorType::get({sizeFactor},
                                            rewriter.getF32Type()),
                            evenB, oddB,
                            ArrayRef<int64_t>{0, 4, 1, 5, 2, 6, 3, 7});

                        matf32.push_back(shuffle);
                        continue;
                      }

                      llvm::SmallVector<OpFoldResult> sizes = {
                          rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                          rewriter.getIndexAttr(16)};
                      auto subview = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), rhsClone->getResult(0), offsets,
                          sizes, strides_splat);

                      auto evenB = rewriter.create<
                          mlir::x86vector::CvtPackedEvenIndexedToF32Op>(
                          kForOp.getLoc(), dstType, subview);

                      matf32.push_back(evenB);

                      auto subview1 = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), rhsClone->getResult(0), offsets,
                          sizes, strides_splat);

                      auto oddB = rewriter.create<
                          mlir::x86vector::CvtPackedOddIndexedToF32Op>(
                          kForOp.getLoc(), dstType, subview1);

                      // Odd FMAs
                      matf32.push_back(oddB);
                    }

                    // Load A-Matrix and do FMAs
                    for (int i = 0, k = 0; i < M; i++) {
                      SmallVector<OpFoldResult> offsets = {
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(i),
                          rewriter.getIndexAttr(0),
                      };
                      auto subview = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), lhsClone->getResult(0), offsets,
                          sizes_splat, strides_splat);
                      auto oddA =
                          rewriter.create<mlir::x86vector::BcstToPackedF32Op>(
                              kForOp.getLoc(), dstType, subview);

                      for (int j = 0; j < (N / sizeFactor); j++) {
                        auto fmaOdd = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), oddA, matf32[j],
                            iterArgsNewKForOp[k]);
                        k++;
                        evenFMAs.push_back(fmaOdd);
                      }
                    }
                  } else { // N->M
                    // Load the A Matrix
                    for (int i = 0; i < M; i++) {
                      SmallVector<OpFoldResult> offsets = {
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(i),
                          rewriter.getIndexAttr(0),
                      };
                      auto subview = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), lhsClone->getResult(0), offsets,
                          sizes_splat, strides_splat);
                      auto oddA =
                          rewriter.create<mlir::x86vector::BcstToPackedF32Op>(
                              kForOp.getLoc(), dstType, subview);

                      matf32.push_back(oddA);
                    }

                    // Load odd elements of B-Matrix, perform fma (odd), and
                    // store to a DS
                    for (int j = 0; j < N; j = j + (sizeFactor * 2)) {
                      SmallVector<OpFoldResult> offsets = {
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(0),
                          rewriter.getIndexAttr(j),
                      };

                      if ((N - j) <= 8) {
                        llvm::SmallVector<OpFoldResult> sizes = {
                            rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                            rewriter.getIndexAttr(8)};

                        auto subview = rewriter.create<memref::SubViewOp>(
                            kForOp.getLoc(), rhsClone->getResult(0), offsets,
                            sizes, strides_splat);

                        mlir::VectorType dstType = mlir::VectorType::get(
                            {sizeFactor / 2}, rewriter.getF32Type());

                        auto evenB = rewriter.create<
                            mlir::x86vector::CvtPackedEvenIndexedToF32Op>(
                            kForOp.getLoc(), dstType, subview);

                        auto oddB = rewriter.create<
                            mlir::x86vector::CvtPackedOddIndexedToF32Op>(
                            kForOp.getLoc(), dstType, subview);

                        auto shuffle = rewriter.create<vector::ShuffleOp>(
                            kForOp.getLoc(),
                            VectorType::get({sizeFactor},
                                            rewriter.getF32Type()),
                            evenB, oddB,
                            ArrayRef<int64_t>{0, 4, 1, 5, 2, 6, 3, 7});

                        for (int i = 0; i < M; i++) {
                          auto fmaOdd = rewriter.create<vector::FMAOp>(
                              kForOp.getLoc(), matf32[i], shuffle,
                              iterArgsNewKForOp[(j / sizeFactor) +
                                                (i * (N / sizeFactor))]);
                          oddFMAs.push_back(fmaOdd);
                        }

                        continue;
                      }

                      llvm::SmallVector<OpFoldResult> sizes = {
                          rewriter.getIndexAttr(1), rewriter.getIndexAttr(1),
                          rewriter.getIndexAttr(16)};
                      auto subview = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), rhsClone->getResult(0), offsets,
                          sizes, strides_splat);

                      auto evenB = rewriter.create<
                          mlir::x86vector::CvtPackedEvenIndexedToF32Op>(
                          kForOp.getLoc(), dstType, subview);

                      for (int i = 0; i < M; i++) {
                        auto fmaOdd = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), matf32[i], evenB,
                            iterArgsNewKForOp[(j / sizeFactor) +
                                              (i * (N / sizeFactor))]);
                        oddFMAs.push_back(fmaOdd);
                      }

                      auto subview1 = rewriter.create<memref::SubViewOp>(
                          kForOp.getLoc(), rhsClone->getResult(0), offsets,
                          sizes, strides_splat);

                      auto oddB = rewriter.create<
                          mlir::x86vector::CvtPackedOddIndexedToF32Op>(
                          kForOp.getLoc(), dstType, subview1);

                      // Odd FMAs
                      for (int i = 0; i < M; i++) {
                        auto fmaOdd = rewriter.create<vector::FMAOp>(
                            kForOp.getLoc(), matf32[i], oddB,
                            iterArgsNewKForOp[((j + sizeFactor) / sizeFactor) +
                                              (i * (N / sizeFactor))]);
                        oddFMAs.push_back(fmaOdd);
                      }
                    }

                    // Re-arrange the stored DPs in order of M -> N
                    for (int i = 0; i < M; i++) {
                      for (int j = 0; j < (N / sizeFactor); j++) {
                        evenFMAs.push_back(oddFMAs[i + (j * M)]);
                      }
                    }
                  }
                }

                if (isSplat && bf16dp) {
                  if (mDriven) { // M -> N
                    for (int j = 0; j < N; j = j + 32) {
                      Value indexOp_j = rewriter.create<arith::ConstantIndexOp>(
                          reductionForOp.getLoc(), j);

                      // B Matrix load.
                      // For case where `B` is one vector<32xbf16>, we do
                      // interleaving with two vector<16xbf16>
                      if ((N - j) <= 16) {
                        auto valueRow1 =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor, elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c0, indexOp_j});

                        auto valueRow2 =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor, elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c1, indexOp_j});

                        SmallVector<Value> shuffle =
                            performShuffle(kForOp.getLoc(), rewriter, valueRow1,
                                           valueRow2, 16, sizeFactor, vnni);

                        matf32.push_back(shuffle[0]);

                      } else { // For two vector<32xbf16>, we do two shuffle.
                        auto valueRow1 =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get({sizeFactor * 2}, elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c0, indexOp_j});

                        auto valueRow2 =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get({sizeFactor * 2}, elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c1, indexOp_j});

                        SmallVector<Value> shuffle =
                            performShuffle(kForOp.getLoc(), rewriter, valueRow1,
                                           valueRow2, 32, sizeFactor, vnni);

                        matf32.push_back(shuffle[0]);
                        matf32.push_back(shuffle[1]);
                      }
                    }

                    // Load elements of A matrix, do dp, and store in a DS
                    for (int i = 0, k = 0; i < M; i++) {
                      Value indexOp_i = rewriter.create<arith::ConstantIndexOp>(
                          reductionForOp.getLoc(), i);
                      auto valueRow = rewriterNewKForOp.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(2, elementType),
                          lhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_i, indexOp_c0});

                      auto valuef32 = performBroadcast(
                          kForOp.getLoc(), rewriter, valueRow, sizeFactor, vnni,
                          elementType, i32Type);

                      for (int j = 0; j < (N / sizeFactor); j++) {
                        auto dp = rewriter.create<mlir::x86vector::DotBF16Op>(
                            kForOp.getLoc(), dstType, iterArgsNewKForOp[k],
                            valuef32, matf32[j]);
                        k++;
                        evenFMAs.push_back(dp);
                      }
                    }

                  } else { // N -> M
                    // Load A matrix
                    for (int i = 0; i < M; i++) {
                      Value indexOp_i = rewriter.create<arith::ConstantIndexOp>(
                          reductionForOp.getLoc(), i);
                      auto valueRow = rewriterNewKForOp.create<vector::LoadOp>(
                          kForOp.getLoc(), VectorType::get(2, elementType),
                          lhsClone->getResult(0),
                          ValueRange{indexOp_c0, indexOp_i, indexOp_c0});
                      auto valuef32 = performBroadcast(
                          kForOp.getLoc(), rewriter, valueRow, sizeFactor, vnni,
                          elementType, i32Type);
                      matf32.push_back(valuef32);
                    }

                    // Load B Matrix
                    for (int j = 0; j < N; j = j + (sizeFactor * 2)) {
                      Value indexOp_j = rewriter.create<arith::ConstantIndexOp>(
                          reductionForOp.getLoc(), j);

                      if ((N - j) <= 16) {
                        auto valueRow1 =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor, elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c0, indexOp_j});

                        auto valueRow2 =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get(sizeFactor, elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c1, indexOp_j});

                        SmallVector<Value> shuffle =
                            performShuffle(kForOp.getLoc(), rewriter, valueRow1,
                                           valueRow2, 16, sizeFactor, vnni);

                        for (int i = 0; i < M; i++) {
                          auto dp = rewriter.create<mlir::x86vector::DotBF16Op>(
                              kForOp.getLoc(), dstType,
                              iterArgsNewKForOp[(j / sizeFactor) +
                                                (i * (N / sizeFactor))],
                              matf32[i], shuffle[0]);
                          oddFMAs.push_back(dp);
                        }

                      } else {
                        auto valueRow1 =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get({sizeFactor * 2}, elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c0, indexOp_j});

                        auto valueRow2 =
                            rewriterNewKForOp.create<vector::LoadOp>(
                                kForOp.getLoc(),
                                VectorType::get({sizeFactor * 2}, elementType),
                                rhsClone->getResult(0),
                                ValueRange{indexOp_c0, indexOp_c1, indexOp_j});

                        SmallVector<Value> shuffle =
                            performShuffle(kForOp.getLoc(), rewriter, valueRow1,
                                           valueRow2, 32, sizeFactor, vnni);

                        for (int i = 0; i < M; i++) {
                          auto dp = rewriter.create<mlir::x86vector::DotBF16Op>(
                              kForOp.getLoc(), dstType,
                              iterArgsNewKForOp[(j / sizeFactor) +
                                                (i * (N / sizeFactor))],
                              matf32[i], shuffle[0]);
                          oddFMAs.push_back(dp);
                        }

                        for (int i = 0; i < M; i++) {
                          auto dp = rewriter.create<mlir::x86vector::DotBF16Op>(
                              kForOp.getLoc(), dstType,
                              iterArgsNewKForOp[((j + sizeFactor) /
                                                 sizeFactor) +
                                                (i * (N / sizeFactor))],
                              matf32[i], shuffle[1]);
                          oddFMAs.push_back(dp);
                        }
                      }
                    }

                    // Re-arrange the stored DPs in order of M -> N
                    for (int i = 0; i < M; i++) {
                      for (int j = 0; j < (N / sizeFactor); j++) {
                        evenFMAs.push_back(oddFMAs[i + (j * M)]);
                      }
                    }
                  }
                }

                rewriterNewKForOp.create<scf::YieldOp>(locNewKForOp, evenFMAs);
              });

          rewriterNewReductionForOp.create<scf::YieldOp>(
              locNewReductionForOp, newKForOp.getResults());
        });

    SmallVector<Value> FMAs = newReductionForOp.getResults();

    if (isSplat && srf) {
      SmallVector<Value> splatFMAs;

      for (int i = 0, k = 0; i < M; i++) {
        for (int j = 0; j < N; j = j + (sizeFactor * 2)) {
          Value indexOp = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), i);
          Value indexOp_B1 = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j);
          Value indexOp_B2 = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j + sizeFactor);

          if ((N - j) <= 8) {
            Value valueCRow1 = rewriter.create<vector::LoadOp>(
                reductionForOp.getLoc(),
                VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
                ValueRange{indexOp, indexOp_B1});

            if (!outsElementType.isF32()) {
              valueCRow1 = performBitcast(reductionForOp.getLoc(), rewriter,
                                          valueCRow1, sizeFactor, vnni,
                                          elementType, i32Type, i16Type, cst16);
            }

            Value addOp1 = rewriter.create<arith::AddFOp>(
                reductionForOp.getLoc(), FMAs[k], valueCRow1);
            splatFMAs.push_back(addOp1);
            k++;
            continue;
          }

          auto shuffle1 = rewriter.create<vector::ShuffleOp>(
              kForOp.getLoc(),
              VectorType::get({sizeFactor}, rewriter.getF32Type()), FMAs[k],
              FMAs[k + 1], ArrayRef<int64_t>{0, 8, 1, 9, 2, 10, 3, 11});

          auto shuffle2 = rewriter.create<vector::ShuffleOp>(
              kForOp.getLoc(),
              VectorType::get({sizeFactor}, rewriter.getF32Type()), FMAs[k],
              FMAs[k + 1], ArrayRef<int64_t>{4, 12, 5, 13, 6, 14, 7, 15});

          Value valueCRow1 = rewriter.create<vector::LoadOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
              ValueRange{indexOp, indexOp_B1});

          if (!outsElementType.isF32()) {
            valueCRow1 = performBitcast(reductionForOp.getLoc(), rewriter,
                                        valueCRow1, sizeFactor, vnni,
                                        elementType, i32Type, i16Type, cst16);
          }

          Value addOp1 = rewriter.create<arith::AddFOp>(reductionForOp.getLoc(),
                                                        shuffle1, valueCRow1);
          splatFMAs.push_back(addOp1);

          Value valueCRow2 = rewriter.create<vector::LoadOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
              ValueRange{indexOp, indexOp_B2});

          if (!outsElementType.isF32()) {
            valueCRow2 = performBitcast(reductionForOp.getLoc(), rewriter,
                                        valueCRow2, sizeFactor, vnni,
                                        elementType, i32Type, i16Type, cst16);
          }

          Value addOp2 = rewriter.create<arith::AddFOp>(reductionForOp.getLoc(),
                                                        shuffle2, valueCRow2);
          splatFMAs.push_back(addOp2);

          k = k + 2;
        }
      }

      FMAs.clear();
      // Re-arrange the stored FMAs in order of M -> N.
      for (int j = 0; j < (N / sizeFactor); j++) {
        for (int i = 0; i < M; i++) {
          FMAs.push_back(splatFMAs[j + (i * (N / sizeFactor))]);
        }
      }
    }

    // On splat layout, we shuffle the acc and add the C
    // matriX value.
    if (isSplat && bf16dp) {
      SmallVector<Value> splatFMAs;

      for (int i = 0, k = 0; i < M; i++) {
        for (int j = 0; j < N; j = j + (sizeFactor * 2)) {
          Value indexOp = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), i);
          Value indexOp_B1 = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j);
          Value indexOp_B2 = rewriter.create<arith::ConstantIndexOp>(
              reductionForOp.getLoc(), j + sizeFactor);

          if ((N - j) <= 16) { // Case: one vector<32xbf16>
            Value valueCRow1 = rewriter.create<vector::LoadOp>(
                reductionForOp.getLoc(),
                VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
                ValueRange{indexOp, indexOp_B1});

            if (!outsElementType.isF32()) {
              valueCRow1 = performBitcast(reductionForOp.getLoc(), rewriter,
                                          valueCRow1, sizeFactor, vnni,
                                          elementType, i32Type, i16Type, cst16);
            }

            Value addOp = rewriter.create<arith::AddFOp>(
                reductionForOp.getLoc(), FMAs[k], valueCRow1);
            splatFMAs.push_back(addOp);
            k++;

          } else { // Case: two vector<32xbf16>
            auto shuffle1 = rewriter.create<vector::ShuffleOp>(
                kForOp.getLoc(),
                VectorType::get({sizeFactor}, rewriter.getF32Type()), FMAs[k],
                FMAs[k + 1],
                ArrayRef<int64_t>{0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20,
                                  21, 22, 23});

            auto shuffle2 = rewriter.create<vector::ShuffleOp>(
                kForOp.getLoc(),
                VectorType::get({sizeFactor}, rewriter.getF32Type()), FMAs[k],
                FMAs[k + 1],
                ArrayRef<int64_t>{8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15,
                                  28, 29, 30, 31});

            Value valueCRow1 = rewriter.create<vector::LoadOp>(
                reductionForOp.getLoc(),
                VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
                ValueRange{indexOp, indexOp_B1});

            if (!outsElementType.isF32()) {
              valueCRow1 = performBitcast(reductionForOp.getLoc(), rewriter,
                                          valueCRow1, sizeFactor, vnni,
                                          elementType, i32Type, i16Type, cst16);
            }

            Value addOp1 = rewriter.create<arith::AddFOp>(
                reductionForOp.getLoc(), shuffle1, valueCRow1);
            splatFMAs.push_back(addOp1);

            Value valueCRow2 = rewriter.create<vector::LoadOp>(
                reductionForOp.getLoc(),
                VectorType::get(sizeFactor, outsElementType), subviewOpAcc,
                ValueRange{indexOp, indexOp_B2});

            if (!outsElementType.isF32()) {
              valueCRow2 = performBitcast(reductionForOp.getLoc(), rewriter,
                                          valueCRow2, sizeFactor, vnni,
                                          elementType, i32Type, i16Type, cst16);
            }

            Value addOp2 = rewriter.create<arith::AddFOp>(
                reductionForOp.getLoc(), shuffle2, valueCRow2);
            splatFMAs.push_back(addOp2);

            k = k + 2;
          }
        }
      }

      FMAs.clear();
      // Re-arrange the stored FMAs in order of M -> N.
      for (int j = 0; j < (N / sizeFactor); j++) {
        for (int i = 0; i < M; i++) {
          FMAs.push_back(splatFMAs[j + (i * (N / sizeFactor))]);
        }
      }
    }

    // Check the mlp pattern
    arith::AddFOp addOp;
    arith::MaximumFOp maxOp;
    memref::SubViewOp subview_readOp;
    memref::GetGlobalOp global_readOp;

    auto zeroAttr = rewriter.getFloatAttr(rewriter.getF32Type(), 0.0);
    auto denseAttr = DenseElementsAttr::get(
        VectorType::get(sizeFactor, rewriter.getF32Type()), zeroAttr);
    auto cst_zero = rewriter.create<arith::ConstantOp>(
        reductionForOp.getLoc(),
        VectorType::get(sizeFactor, rewriter.getF32Type()), denseAttr);
    auto c1 = rewriter.create<arith::ConstantOp>(
        kForOp.getLoc(),
        DenseIntElementsAttr::get(VectorType::get(sizeFactor, i32Type), 1));
    auto c16 = rewriter.create<arith::ConstantOp>(
        kForOp.getLoc(),
        DenseIntElementsAttr::get(VectorType::get(sizeFactor, i32Type), 16));
    auto c7fff = rewriter.create<arith::ConstantOp>(
        kForOp.getLoc(), DenseIntElementsAttr::get(
                             VectorType::get(sizeFactor, i32Type), 0x7fff));

    // We first retrive the source of the C matrix
    auto subview_itr = subviewOpAcc.getSource();
    Value subview_tmp = subviewOpAcc;

    while (true) {
      if (auto subview_2 = subview_itr.getDefiningOp<memref::SubViewOp>()) {

        if (subview_2.getType().getRank() != 2) {
          break;
        }
        subview_tmp = subview_itr;
        subview_itr = subview_2.getSource();
        continue;
      }
      break;
    }

    Value source;
    if (subview_itr.getType().getRank() == 2) {
      source = subview_itr;
    } else {
      source = subview_tmp;
    }

    // Find the use of C matrix and
    // get the addOp and maxOp source for mlp
    for (auto users : source.getUsers()) {
      if (auto vectorRead = dyn_cast<vector::TransferReadOp>(users)) {
        for (auto read_users : vectorRead->getUsers()) {
          if (auto add_Op = dyn_cast<arith::AddFOp>(read_users)) {
            addOp = add_Op;
          }
          if (auto max_Op = dyn_cast<arith::MaximumFOp>(read_users)) {
            maxOp = max_Op;
          }
        }
      }
    }

    // get the 2nd input source for addOp via vector transfer read
    // ps: the 1st one is C matrix
    if (addOp && maxOp && !isF32 && !isI8) {
      vector::TransferReadOp readOp_add;
      if (auto vectBcst = addOp.getLhs().getDefiningOp<vector::BroadcastOp>()) {
        if (auto vectorRead =
                vectBcst.getSource().getDefiningOp<vector::TransferReadOp>()) {
          readOp_add = vectorRead;
        }
      }

      if (readOp_add) {
        if (auto subview =
                readOp_add.getBase().getDefiningOp<memref::SubViewOp>()) {
          subview_readOp = subview;
        }

        if (auto global_read =
                readOp_add.getBase().getDefiningOp<memref::GetGlobalOp>()) {
          global_readOp = global_read;
        }
      }
    }

    // B-matrix (N) induction variable
    auto nInductionVar = nForOp.getInductionVar();

    for (int j = 0, k = 0; j < N; j = j + sizeFactor) {
      for (int i = 0; i < M; i++) {
        Value indexOp =
            rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), i);
        Value indexOp_B =
            rewriter.create<arith::ConstantIndexOp>(reductionForOp.getLoc(), j);
        Type type;
        if (elementType.isBF16())
          type = rewriter.getBF16Type();
        if (elementType.isF16())
          type = rewriter.getF16Type();

        auto acc_value = FMAs[k];
        k++;

        if (addOp && maxOp && !isF32 && !isI8) {
          Value add_row;
          if (global_readOp) {
            auto index_mlp = rewriter.create<arith::AddIOp>(
                reductionForOp.getLoc(), rewriter.getIndexType(), nInductionVar,
                indexOp_B);
            add_row = rewriter.create<vector::LoadOp>(
                reductionForOp.getLoc(),
                VectorType::get(sizeFactor, elementType), global_readOp,
                ValueRange{index_mlp});
          }

          if (subview_readOp) {
            auto index_mlp = rewriter.create<arith::AddIOp>(
                reductionForOp.getLoc(), rewriter.getIndexType(), nInductionVar,
                indexOp_B);
            auto offsetsVec = subview_readOp.getMixedOffsets();
            llvm::ArrayRef<mlir::OpFoldResult> offsets = offsetsVec;
            auto val_offset = offsets[0].dyn_cast<mlir::Value>();
            add_row = rewriter.create<vector::LoadOp>(
                reductionForOp.getLoc(),
                VectorType::get(sizeFactor, elementType),
                subview_readOp.getSource(), ValueRange{val_offset, index_mlp});
          }

          // Fused mlp happens here
          if (add_row) {
            Value f32MLPVector;
            if (elementType.isBF16()) {
              auto bitcast_i16 = rewriter.create<vector::BitCastOp>(
                  reductionForOp.getLoc(), VectorType::get(sizeFactor, i16Type),
                  add_row);
              auto extend_i32 = rewriter.create<arith::ExtUIOp>(
                  reductionForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                  bitcast_i16);
              auto shiftOp = rewriter.create<arith::ShLIOp>(
                  reductionForOp.getLoc(), VectorType::get(sizeFactor, i32Type),
                  extend_i32,
                  rewriter.create<vector::BroadcastOp>(
                      reductionForOp.getLoc(),
                      VectorType::get(sizeFactor, i32Type), cst16));
              f32MLPVector = rewriter.create<vector::BitCastOp>(
                  reductionForOp.getLoc(),
                  VectorType::get(sizeFactor, rewriter.getF32Type()), shiftOp);
            }

            if (elementType.isF16()) {
              f32MLPVector = rewriter.create<arith::ExtFOp>(
                  reductionForOp.getLoc(),
                  VectorType::get(sizeFactor, rewriter.getF32Type()), add_row);
            }

            auto add = rewriter.create<arith::AddFOp>(
                reductionForOp.getLoc(),
                mlir::VectorType::get(sizeFactor, rewriter.getF32Type()),
                acc_value, f32MLPVector);
            auto max = rewriter.create<arith::MaximumFOp>(
                reductionForOp.getLoc(),
                mlir::VectorType::get(sizeFactor, rewriter.getF32Type()), add,
                cst_zero);
            acc_value = max;
          }
        }

        Value vec_final = acc_value;

        // We do f32 -> bf16 downconvert using rshift, truncate and rounding
        // the lsb for the fallback case.
        if (fallback && isBF16 && !outsElementType.isF32()) {
          auto vec = rewriter.create<vector::BitCastOp>(
              kForOp.getLoc(), VectorType::get(sizeFactor, i32Type), acc_value);
          auto rshift = rewriter.create<arith::ShRUIOp>(
              kForOp.getLoc(), VectorType::get(sizeFactor, i32Type), vec, c16);
          auto leastSB = rewriter.create<arith::AndIOp>(reductionForOp.getLoc(),
                                                        rshift, c1);
          auto roundBias = rewriter.create<arith::AddIOp>(
              reductionForOp.getLoc(), c7fff, leastSB);
          auto rounded_vec = rewriter.create<arith::AddIOp>(
              reductionForOp.getLoc(), vec, roundBias);
          auto shift = rewriter.create<arith::ShRUIOp>(reductionForOp.getLoc(),
                                                       rounded_vec, c16);
          auto truncate = rewriter.create<arith::TruncIOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, i16Type),
              shift);
          vec_final = rewriter.create<vector::BitCastOp>(
              reductionForOp.getLoc(),
              VectorType::get(sizeFactor, rewriter.getBF16Type()), truncate);
        }

        // We do arith.tuncf for f32 -> bf16 in SRF/ARL/SPR kind of machines
        if ((srf || bf16dp) && !outsElementType.isF32() && !isI8) {
          vec_final = rewriter.create<arith::TruncFOp>(
              reductionForOp.getLoc(), VectorType::get(sizeFactor, type),
              acc_value);
        }

        // Final store back the accumulate value into c matrix
        rewriter.create<vector::StoreOp>(reductionForOp.getLoc(), vec_final,
                                         subviewOpAcc,
                                         ValueRange{indexOp, indexOp_B});
      }
    }

    // Remove the transfer write operations
    if (addOp && maxOp && (subview_readOp || global_readOp)) {
      Operation *writeOp_add;
      for (auto val : addOp->getUsers()) {
        writeOp_add = dyn_cast<vector::TransferWriteOp>(val);
        if (writeOp_add) {
          rewriter.eraseOp(writeOp_add);
          break;
        }
      }

      Operation *writeOp_max;
      for (auto val : maxOp->getUsers()) {
        writeOp_max = dyn_cast<vector::TransferWriteOp>(val);
        if (writeOp_max) {
          rewriter.eraseOp(writeOp_max);
          break;
        }
      }
    }

    Value contractVal = reductionForOp.getResult(0);
    Operation *writeOp;
    for (auto val : contractVal.getUsers()) {
      writeOp = dyn_cast<vector::TransferWriteOp>(val);
      if (writeOp) {
        rewriter.eraseOp(writeOp);
        break;
      }
    }

    return success();
  }

private:
  MicroKernelsOptions options;
};

void populateMicroKernelsPatterns(RewritePatternSet &patterns,
                                  MicroKernelsOptions options) {
  patterns.add<MicroKernelsOp>(patterns.getContext(), options);
}

struct MicroKernels : public impl::MicroKernelsBase<MicroKernels> {
  using MicroKernelsBase::MicroKernelsBase;

  void runOnOperation() override {
    MicroKernelsOptions options;
    options.targetFeature = targetFeature;
    RewritePatternSet patterns(&getContext());
    populateMicroKernelsPatterns(patterns, options);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
