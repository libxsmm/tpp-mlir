//===- VectorContractToMicroKernelsAMX.cpp ------------------------*-C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction using AMX ops
// to micro kernels for flat layouts.
// Target types: bf16
// TODO: (a) VectorContractToAMX.cpp pass and this pass will be merged.
//       (b) Some functions are common between VectorContractToMicroKernels.cpp
//           and this pass. As part of upstreaming, those functions will be
//           moved to common utils.
//===----------------------------------------------------------------------===//
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_MICROKERNELSAMX
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

static Value performBitcast(Location loc, PatternRewriter &rewriter,
                            Value vector, int64_t sizeFactor, int64_t vnni,
                            Type elementType, Type i32Type, Type i16Type,
                            Value cst16) {

  auto bitcast_i16 = vector::BitCastOp::create(rewriter, 
      loc, VectorType::get(sizeFactor, i16Type), vector);
  auto extend_i32 = arith::ExtUIOp::create(rewriter, 
      loc, VectorType::get(sizeFactor, i32Type), bitcast_i16);
  auto vectType = VectorType::get(sizeFactor, i32Type);
  auto shiftOp = arith::ShLIOp::create(rewriter, 
      loc, vectType, extend_i32,
      vector::BroadcastOp::create(rewriter, loc, vectType, cst16));
  auto value = vector::BitCastOp::create(rewriter, 
      loc, VectorType::get(sizeFactor, rewriter.getF32Type()), shiftOp);

  return value;
}

// Function to pack Flat layout rows into VNNI packed using the 
// vpunpacklwd/hwd instructions.
static SmallVector<Value> performShuffle(Location loc,
                                         PatternRewriter &rewriter, Value vec1,
                                         Value vec2, int64_t elementSize,
                                         int64_t sizeFactor, int64_t vnni) {
  SmallVector<Value> vectors;
  if (elementSize == 16) {
    auto shuffle = vector::ShuffleOp::create(rewriter, 
        loc, VectorType::get({sizeFactor * vnni}, rewriter.getBF16Type()), vec1,
        vec2, ArrayRef<int64_t>{0,  16, 1,  17, 2,  18, 3,  19, 4,  20, 5,
                                21, 6,  22, 7,  23, 8,  24, 9,  25, 10, 26,
                                11, 27, 12, 28, 13, 29, 14, 30, 15, 31});

    vectors.push_back(shuffle);
  }

  if (elementSize == 32) {
    auto shuffle1 = vector::ShuffleOp::create(rewriter, 
        loc, VectorType::get({sizeFactor * vnni}, rewriter.getBF16Type()), vec1,
        vec2, ArrayRef<int64_t>{0,  32, 1,  33, 2,  34, 3,  35, 8,  40, 9,
                                41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50,
                                19, 51, 24, 56, 25, 57, 26, 58, 27, 59});
    vectors.push_back(shuffle1);
    auto shuffle2 = vector::ShuffleOp::create(rewriter, 
        loc, VectorType::get({sizeFactor * vnni}, rewriter.getBF16Type()), vec1,
        vec2, ArrayRef<int64_t>{4,  36, 5,  37, 6,  38, 7,  39, 12, 44, 13,
                                45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54,
                                23, 55, 28, 60, 29, 61, 30, 62, 31, 63});
    vectors.push_back(shuffle2);
  }

  return vectors;
}

// Function to load a 16x32 tiles.
static SmallVector<Value> loadTiles(Location loc, PatternRewriter &rewriter,
                                    Value subview, Value index, int64_t p1,
                                    int64_t p2) {

  SmallVector<Value> loads;
  auto tileType = amx::TileType::get({16, 32}, rewriter.getBF16Type());

  for (int j = 0; j < p2; j = j + 32) {
    for (int i = 0; i < p1; i = i + 16) {
      Value indexOp_i = arith::ConstantIndexOp::create(rewriter, loc, i);
      Value indexOp_j = arith::ConstantIndexOp::create(rewriter, loc, j);
      auto load = amx::TileLoadOp::create(rewriter, 
          loc, tileType, subview, ValueRange{index, indexOp_i, indexOp_j});
      loads.push_back(load);
    }
  }
  return loads;
}

// Function to compute tiled dot-product
static SmallVector<Value> computeTileMul(Location loc,
                                         PatternRewriter &rewriter,
                                         SmallVector<Value> tileA,
                                         SmallVector<Value> tileB,
                                         ValueRange acc, int64_t M, int64_t N) {

  SmallVector<Value> computedFMAs;
  auto resType = amx::TileType::get({16, 16}, rewriter.getF32Type());
  for (int i = 0, p = 0; i < M / 16; i++) {
    for (int j = 0; j < N / 16; j++) {

      auto fma = amx::TileMulFOp::create(rewriter, loc, resType, tileA[i],
                                                  tileB[j], acc[p]);
      computedFMAs.push_back(fma);
      p++;
    }
  }

  return computedFMAs;
}

// Function to load vector<32xbf16>, pack them into VNNI, and
// store them into a buffer.
// indx_r1 and indx_r2 used to load two vector<32xbf16> from 
// the subview. 
// indx_s1 to choose the 0th or 1st buffer. indx_s2 and indx_s3
// used to store the VNNI packed row into the buffer.
static void loadPackStore(Location loc, PatternRewriter &rewriter,
                          Value subview, Value bBuffer, Type elementType,
                          Value indx_r1, Value indx_r2, Value indx_s1,
                          Value indx_s2, Value indx_s3, int64_t N,
                          int64_t vnni) {
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  for (int j = 0; j < N; j = j + 32) {
    Value ind_j = arith::ConstantIndexOp::create(rewriter, loc, j);
    auto valueRow1 = vector::LoadOp::create(rewriter, 
        loc, VectorType::get(32, elementType), subview,
        ValueRange{c0, indx_r1, ind_j});

    auto valueRow2 = vector::LoadOp::create(rewriter, 
        loc, VectorType::get(32, elementType), subview,
        ValueRange{c0, indx_r2, ind_j});

    SmallVector<Value> shuffle =
        performShuffle(loc, rewriter, valueRow1, valueRow2, 32, 16, vnni);

    vector::StoreOp::create(rewriter, loc, shuffle[0], bBuffer,
                                     ValueRange{indx_s1, indx_s2, ind_j});
    vector::StoreOp::create(rewriter, loc, shuffle[1], bBuffer,
                                     ValueRange{indx_s1, indx_s3, ind_j});
  }
}

// Input IR:
// vector.contract with register blocked
//
// Output IR:
// s/w pipeline: load, pack to VNNI, and store the B sub matrix
// of the 0thbatch-reduce and K iteration.
// scf.for (0 to 31) {
// 	- load 0th and 1st  vector<32xbf16>, pack into VNNI, store the
// 	first shuffle in 0th and 2nd shuffle in 16th index of the
// 	buffer.
// }
// scf.for (0 to br-2) { batch-reduce loop
//   scf.for (0 to k-2) { K loop
// 	- load A matrix
//	- scf.loop for s/w pipeline: load, pack to VNNI, and store the B sub
// matrix 	for the next K loop iteration 	(c) load VNNI pack B matrix of K
// iteration from the buffer 	(d) compute the tiled dot-product
//   }
//   Last iteration of the the K Loop (k-1) {
//      - load A matrix
//      - scf.loop for s/w pipeline: load, pack to VNNI, and store the B sub
//      matrix for the next batch-reduce + K loop iteration (c) load VNNI pack B
//      matrix of K iteration from the buffer (d) compute the tiled dot-product
//   }
// }
// Last iteration of the batch-reduce loop (br-1) {
//   scf.for (0 to k-2) { K loop
//      - load A matrix
//      - scf.loop for s/w pipeline: load, pack to VNNI, and store the B sub
//      matrix for the next K loop iteration (c) load VNNI pack B matrix of K
//      iteration from the buffer (d) compute the tiled dot-product
//   }
//   Last iteration of the the K Loop (k-1) {
//      - load A matrix
//      - load VNNI pack B matrix of K iteration from the buffer
//      - compute the tiled dot-product
//   }
// }
//
// scf.for (0 to M)
//   scf.for (0 to N)
//     - Load the ith and i+1th acc
//     - Shuffle them as we packed using vpunpack
//     - Load C matrix and do arith.add with the shuffle
//     - Store back into C matrix
struct MicroKernelsAMXOp : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

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

    bool amx = vnni::utils::hasAMX();
    if (!amx)
      return rewriter.notifyMatchFailure(
          contractOp, "Lowering supported only for AMX flat layout.");

    bool isBF16 = elementType.isBF16();
    if (!isBF16)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only BF16 type supported.");

    // Check the operation type MatMul, B-MatMul, or BR-MatMul
    SmallVector<vector::IteratorType> contractIteratorTypes =
        contractOp.getIteratorTypesArray();
    int reductionCount =
        std::count(contractIteratorTypes.begin(), contractIteratorTypes.end(),
                   vector::IteratorType::reduction);
    auto lhsType = dyn_cast<ShapedType>(vectorReadOpLhs.getType());
    auto rhsType = dyn_cast<ShapedType>(vectorReadOpRhs.getType());

    bool isSplat = false;
    if (isBF16 && reductionCount == 2 && lhsType.getRank() == 3 &&
        rhsType.getRank() == 3)
      isSplat = true;

    if (!isSplat)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only AMX Flat layout supported.");

    if (reductionCount == 1)
      return rewriter.notifyMatchFailure(
          contractOp, "Batch matmul operation not supported yet");

    int64_t M = lhsType.getDimSize(lhsType.getRank() - 2);
    int64_t N = rhsType.getDimSize(lhsType.getRank() - 1);
    int64_t K = lhsType.getDimSize(lhsType.getRank() - 1);
    int64_t vnni = 2;

    if (K != 32)
      return rewriter.notifyMatchFailure(
          contractOp,
          "K tile size should be equal to 32 for Flat AMX lowering");

    // TODO: To support for the case where we have last set of two vectors as
    // vector<16xbf16>. For this case, we need to pack with the help of
    // vector.interleaving. This is not the prime focus now as optimal register
    // blocking (2x2) except N tile to be divisible by 32.
    if ((N % 32) != 0)
      return rewriter.notifyMatchFailure(
          contractOp, "Only, N tile size divisible by 32 is supported");

    if (isTransposedMatrix(contractOp, elementType, isSplat))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Matrices shoudn't be transposed.");

    if (!permutationCheck(contractOp, elementType, isSplat))
      return rewriter.notifyMatchFailure(
          contractOp, "Affine map permutation not supported.");

    rewriter.setInsertionPoint(mForOp);
    auto i32Type = rewriter.getIntegerType(32);
    auto i16Type = rewriter.getIntegerType(16);
    auto cst16 = arith::ConstantOp::create(rewriter, 
        reductionForOp.getLoc(), rewriter.getIntegerAttr(i32Type, 16));

    rewriter.setInsertionPoint(reductionForOp);
    llvm::SmallVector<Value> loopItrArgs;

    // Creating the Zero Tile for the iter args.
    if (isSplat && amx) {
      for (int i = 0; i < M; i = i + 16) {
        for (int j = 0; j < N; j = j + 16) {
          auto tileType = amx::TileType::get({16, 16}, rewriter.getF32Type());
          auto zeroTile = amx::TileZeroOp::create(rewriter, 
              reductionForOp.getLoc(), tileType);
          loopItrArgs.push_back(zeroTile);
        }
      }
    }

    SmallVector<Value> computedFMAs;
    SmallVector<Value> loadsA;
    SmallVector<Value> loadsB;

    // We need to capture the dim size of K. With whic, we reuse the
    // buffer effectively to store the VNNI packed B-Matrix.
    // Case 1: K dim > 32
    // Case 2: K dim / 32 = odd quotient.
    bool nDimK = false;
    bool oddDimK = false;

    int64_t ubVal = 32;
    mlir::Value ub = kForOp.getUpperBound();
    if (auto constOp = ub.getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto intAttr =
              llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
        ubVal = intAttr.getInt();
      }
    }

    nDimK = ubVal > 32;
    oddDimK = (((ubVal / 32) % 2) == 1) && nDimK;

    Value c0 =
        arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), 0);
    Value c1 =
        arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), 1);
    Value c2 =
        arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), 2);
    Value c16 =
        arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), 16);
    Value c32 =
        arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), 32);

    // Buffer to save the VNNI packed B-Matrix
    auto bufferType = MemRefType::get({2, K, N}, rewriter.getBF16Type());
    auto bBuffer =
        memref::AllocaOp::create(rewriter, kForOp.getLoc(), bufferType);

    // s/w pipeline: pack the first set of B-Matrix into VNNI before the
    // 0th batch-reduce iteration
    // We pack two vector<32xbf16> at i and i+1th index. Then store, the
    // two VNNI packed vector<32xbf16> at j and j+16th index of the buffer.
    scf::ForOp::create(rewriter, 
        reductionForOp.getLoc(), c0, c32, c2, ValueRange{},
        [&](OpBuilder &nestedBuilder, Location loc, Value iv,
            ValueRange iterArgs) {
          IRMapping rhsMapping;
          rhsMapping.map(
              vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1), c0);
          rhsMapping.map(
              vectorReadOpRhs.getBase().getDefiningOp()->getOperand(2), c0);
          auto rhsClone = rewriter.clone(
              *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

          // Add the iv by 1 to load the i+1th vector<32xbf16>
          Value i1_load =
              arith::AddIOp::create(rewriter, reductionForOp.getLoc(), c1, iv);

          // Divide the iv by 2 to get the jth position to store shuffled[0]
          // vector<32xbf16>
          Value j_pos =
              arith::DivUIOp::create(rewriter, reductionForOp.getLoc(), iv, c2);
          // Add 16 to the divided j_pos to get j+16th position to store
          // shuffled[1] vector<32xbf16>
          Value j16_pos = arith::AddIOp::create(rewriter, 
              reductionForOp.getLoc(), c16, j_pos);

          // Load two vector<32xbf16>, VNNI pack, and store into bBuffer
          loadPackStore(kForOp.getLoc(), rewriter, rhsClone->getResult(0),
                        bBuffer, elementType, iv, i1_load, c0, j_pos, j16_pos,
                        N, vnni);

          scf::YieldOp::create(nestedBuilder, reductionForOp.getLoc());
        });

    // Peel out the last iteration of batch-reduce and K loops
    Value subBRLoop = arith::SubIOp::create(rewriter, 
        reductionForOp.getLoc(), reductionForOp.getUpperBound(), c1);
    Value subKloop = arith::SubIOp::create(rewriter, 
        kForOp.getLoc(), kForOp.getUpperBound(), c32);

    // Code to re-create the batch-reduce loop (0 to n-2) and K loop (0 to n-2)
    // with iter args
    auto newReductionForOp = scf::ForOp::create(rewriter, 
        reductionForOp.getLoc(), reductionForOp.getLowerBound(), subBRLoop,
        reductionForOp.getStep(), loopItrArgs,
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp,
            Value ivNewReductionForOp, ValueRange iterArgsNewReductionForOp) {
          auto newKForOp = scf::ForOp::create(rewriter, 
              kForOp.getLoc(), kForOp.getLowerBound(), subKloop,
              kForOp.getStep(), iterArgsNewReductionForOp,
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                // A-Matrix subview
                IRMapping mapping;
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(3),
                    ivNewKForOp);
                auto lhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

                // We extract the B-Matrix subview to load + VNNI pack the next
                // set of sub-matrix
                Value ivK_add32 = arith::AddIOp::create(rewriter, 
                    reductionForOp.getLoc(), c32, ivNewKForOp);

                // B-Matrix subview
                IRMapping rhsMapping;
                rhsMapping.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                rhsMapping.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(2),
                    ivK_add32);
                auto rhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpRhs.getBase().getDefiningOp(), rhsMapping);

                // Load A-Matrix
                loadsA = loadTiles(kForOp.getLoc(), rewriter,
                                   lhsClone->getResult(0), c0, M, K);

                // s/w pipeline: pack the current iteration + 1th  of B
                // sub-Matrix into VNNI int the current iteration. To store the
                // packed B sub-matrix into 0th or 1st buffer is decied based on
                // the remainder of BR or K loop induction variables.
                scf::ForOp::create(rewriter, 
                    reductionForOp.getLoc(), c0, c32, c2, ValueRange{},
                    [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                        ValueRange iterArgs) {
                      Value i1_load = arith::AddIOp::create(rewriter, 
                          reductionForOp.getLoc(), c1, iv);
                      Value j_pos = arith::DivUIOp::create(rewriter, 
                          reductionForOp.getLoc(), iv, c2);
                      Value j16_pos = arith::AddIOp::create(rewriter, 
                          reductionForOp.getLoc(), c16, j_pos);

                      // Code to decide whether to store the packed sub-matrix
                      // either 0th or 1st buffer
                      Value quotient_K = arith::DivUIOp::create(rewriter, 
                          reductionForOp.getLoc(), ivK_add32, c32);
                      Value rem_K = arith::RemUIOp::create(rewriter, 
                          reductionForOp.getLoc(), rewriter.getIndexType(),
                          quotient_K, c2);

                      // if K quotient is odd. Then, BR loop iv is taken
                      // into consideration
                      if (oddDimK) {
                        auto rem_BR = arith::RemUIOp::create(rewriter, 
                            reductionForOp.getLoc(), rewriter.getIndexType(),
                            ivNewReductionForOp, c2);
                        auto remAdd = arith::AddIOp::create(rewriter, 
                            reductionForOp.getLoc(), rewriter.getIndexType(),
                            rem_K, rem_BR);
                        rem_K = arith::RemUIOp::create(rewriter, 
                            reductionForOp.getLoc(), rewriter.getIndexType(),
                            remAdd, c2);
                      }

                      // Load two vector<32xbf16>, VNNI pack, and store into
                      // bBuffer
                      loadPackStore(kForOp.getLoc(), rewriter,
                                    rhsClone->getResult(0), bBuffer,
                                    elementType, iv, i1_load, rem_K, j_pos,
                                    j16_pos, N, vnni);

                      scf::YieldOp::create(nestedBuilder, 
                          reductionForOp.getLoc());
                    });

                // Load B sub-matrix from the buffer.
                // Same way like storing, we load from buffer 0th or 1st
                // based on the
                // remainder of BR or K loop induction variables.
                Value quotient_K = arith::DivUIOp::create(rewriter, 
                    reductionForOp.getLoc(), ivNewKForOp, c32);
                auto rem_K = arith::RemUIOp::create(rewriter, 
                    reductionForOp.getLoc(), rewriter.getIndexType(),
                    quotient_K, c2);

                if (oddDimK) {
                  auto rem_BR = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      ivNewReductionForOp, c2);
                  auto remAdd = arith::AddIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(), rem_K,
                      rem_BR);
                  rem_K = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(), remAdd,
                      c2);
                }

                loadsB =
                    loadTiles(kForOp.getLoc(), rewriter, bBuffer, rem_K, K, N);

                // Compute the tiled dot-product with the A and B tile loads.
                computedFMAs = computeTileMul(kForOp.getLoc(), rewriter, loadsA,
                                              loadsB, iterArgsNewKForOp, M, N);

                scf::YieldOp::create(rewriterNewKForOp, locNewKForOp,
                                                       computedFMAs);
              });

          // Create the last iteration of K-Loop within the {0 to n-2} reduction
          // loop
          auto newKForOp_last = scf::ForOp::create(rewriter, 
              kForOp.getLoc(), subKloop, kForOp.getUpperBound(),
              kForOp.getStep(), newKForOp.getResults(),
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                // A subview
                IRMapping mapping;
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(3),
                    ivNewKForOp);
                auto lhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

                // B subview
                Value ivK_add32 = arith::AddIOp::create(rewriter, 
                    reductionForOp.getLoc(), c1, ivNewReductionForOp);

                IRMapping mapping1;
                mapping1.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1),
                    ivK_add32);
                mapping1.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(2),
                    c0);
                auto rhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpRhs.getBase().getDefiningOp(), mapping1);

                // Load A-Matrix
                loadsA = loadTiles(kForOp.getLoc(), rewriter,
                                   lhsClone->getResult(0), c0, M, K);

                // s/w pipeline: load, pack, and store into buffer
                scf::ForOp::create(rewriter, 
                    reductionForOp.getLoc(), c0, c32, c2, ValueRange{},
                    [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                        ValueRange iterArgs) {
                      Value i1_load = arith::AddIOp::create(rewriter, 
                          reductionForOp.getLoc(), c1, iv);
                      Value j_pos = arith::DivUIOp::create(rewriter, 
                          reductionForOp.getLoc(), iv, c2);
                      Value j16_pos = arith::AddIOp::create(rewriter, 
                          reductionForOp.getLoc(), c16, j_pos);

                      Value cIndex = c0;
                      if (!nDimK || oddDimK) {
                        cIndex = arith::RemUIOp::create(rewriter, 
                            reductionForOp.getLoc(), rewriter.getIndexType(),
                            ivK_add32, c2);
                      }

                      // Load two vector<32xbf16>, VNNI pack, and store into
                      // bBuffer
                      loadPackStore(kForOp.getLoc(), rewriter,
                                    rhsClone->getResult(0), bBuffer,
                                    elementType, iv, i1_load, cIndex, j_pos,
                                    j16_pos, N, vnni);

                      scf::YieldOp::create(nestedBuilder, 
                          reductionForOp.getLoc());
                    });

                // Load B-Matrix
                Value quotient_K = arith::DivUIOp::create(rewriter, 
                    reductionForOp.getLoc(), ivNewKForOp, c32);
                Value rem_K = arith::RemUIOp::create(rewriter, 
                    reductionForOp.getLoc(), rewriter.getIndexType(),
                    quotient_K, c2);

                if (!nDimK) {
                  rem_K = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      ivNewReductionForOp, c2);
                }

                if (oddDimK) {
                  auto rem_BR = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      ivNewReductionForOp, c2);
                  auto remAdd = arith::AddIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(), rem_K,
                      rem_BR);
                  rem_K = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(), remAdd,
                      c2);
                }

                loadsB =
                    loadTiles(kForOp.getLoc(), rewriter, bBuffer, rem_K, K, N);

                // Compute the tiled dot-product with the A and B tile loads.
                computedFMAs = computeTileMul(kForOp.getLoc(), rewriter, loadsA,
                                              loadsB, iterArgsNewKForOp, M, N);

                scf::YieldOp::create(rewriterNewKForOp, locNewKForOp,
                                                       computedFMAs);
              });

          scf::YieldOp::create(rewriterNewReductionForOp, 
              locNewReductionForOp, newKForOp_last.getResults());
        });

    // Create the last iteration of the batch-reduce loop (br-1). Inside create
    // both the {0 to n-2} and {n-1} K loop.
    auto newReductionForOp_last = scf::ForOp::create(rewriter, 
        reductionForOp.getLoc(), subBRLoop, reductionForOp.getUpperBound(),
        reductionForOp.getStep(), newReductionForOp.getResults(),
        [&](OpBuilder &rewriterNewReductionForOp, Location locNewReductionForOp,
            Value ivNewReductionForOp, ValueRange iterArgsNewReductionForOp) {
          auto newKForOp_last = scf::ForOp::create(rewriter, 
              kForOp.getLoc(), kForOp.getLowerBound(), subKloop,
              kForOp.getStep(), iterArgsNewReductionForOp,
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                // A subview
                IRMapping mapping;
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(3),
                    ivNewKForOp);
                auto lhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

                // B subview
                Value ivK_add32 = arith::AddIOp::create(rewriter, 
                    reductionForOp.getLoc(), c32, ivNewKForOp);
                IRMapping mapping1;
                mapping1.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                mapping1.map(
                    vectorReadOpRhs.getBase().getDefiningOp()->getOperand(2),
                    ivK_add32);
                auto rhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpRhs.getBase().getDefiningOp(), mapping1);

                //  Load A Matrix
                loadsA = loadTiles(kForOp.getLoc(), rewriter,
                                   lhsClone->getResult(0), c0, M, K);

                // s/w pipeline: load, pack, and store into buffer
                scf::ForOp::create(rewriter, 
                    reductionForOp.getLoc(), c0, c32, c2, ValueRange{},
                    [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                        ValueRange iterArgs) {
                      Value i1_load = arith::AddIOp::create(rewriter, 
                          reductionForOp.getLoc(), c1, iv);
                      Value j_pos = arith::DivUIOp::create(rewriter, 
                          reductionForOp.getLoc(), iv, c2);
                      Value j16_pos = arith::AddIOp::create(rewriter, 
                          reductionForOp.getLoc(), c16, j_pos);

                      Value quotient_K = arith::DivUIOp::create(rewriter, 
                          reductionForOp.getLoc(), ivK_add32, c32);
                      auto rem_K = arith::RemUIOp::create(rewriter, 
                          reductionForOp.getLoc(), rewriter.getIndexType(),
                          quotient_K, c2);

                      if (oddDimK) {
                        auto rem_BR = arith::RemUIOp::create(rewriter, 
                            reductionForOp.getLoc(), rewriter.getIndexType(),
                            ivNewReductionForOp, c2);
                        auto remAdd = arith::AddIOp::create(rewriter, 
                            reductionForOp.getLoc(), rewriter.getIndexType(),
                            rem_K, rem_BR);
                        rem_K = arith::RemUIOp::create(rewriter, 
                            reductionForOp.getLoc(), rewriter.getIndexType(),
                            remAdd, c2);
                      }

                      // Load two vector<32xbf16>, VNNI pack, and store into
                      // bBuffer
                      loadPackStore(kForOp.getLoc(), rewriter,
                                    rhsClone->getResult(0), bBuffer,
                                    elementType, iv, i1_load, rem_K, j_pos,
                                    j16_pos, N, vnni);

                      scf::YieldOp::create(nestedBuilder, 
                          reductionForOp.getLoc());
                    });

                // Load B-Matrix
                Value quotient_K = arith::DivUIOp::create(rewriter, 
                    reductionForOp.getLoc(), ivNewKForOp, c32);
                auto rem_K = arith::RemUIOp::create(rewriter, 
                    reductionForOp.getLoc(), rewriter.getIndexType(),
                    quotient_K, c2);

                if (oddDimK) {
                  auto rem_BR = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      ivNewReductionForOp, c2);
                  auto remAdd = arith::AddIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(), rem_K,
                      rem_BR);
                  rem_K = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(), remAdd,
                      c2);
                }
                loadsB =
                    loadTiles(kForOp.getLoc(), rewriter, bBuffer, rem_K, K, N);

                // Compute the tiled dot-product with the A and B tile loads.
                computedFMAs = computeTileMul(kForOp.getLoc(), rewriter, loadsA,
                                              loadsB, iterArgsNewKForOp, M, N);

                scf::YieldOp::create(rewriterNewKForOp, locNewKForOp,
                                                       computedFMAs);
              });

          // Create the last iteration of K loop
          auto newKForOp_last_br = scf::ForOp::create(rewriter, 
              kForOp.getLoc(), subKloop, kForOp.getUpperBound(),
              kForOp.getStep(), newKForOp_last.getResults(),
              [&](OpBuilder &rewriterNewKForOp, Location locNewKForOp,
                  Value ivNewKForOp, ValueRange iterArgsNewKForOp) {
                // A subview
                IRMapping mapping;
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(1),
                    ivNewReductionForOp);
                mapping.map(
                    vectorReadOpLhs.getBase().getDefiningOp()->getOperand(3),
                    ivNewKForOp);
                auto lhsClone = rewriterNewKForOp.clone(
                    *vectorReadOpLhs.getBase().getDefiningOp(), mapping);

                // Load A matrix
                loadsA = loadTiles(kForOp.getLoc(), rewriter,
                                   lhsClone->getResult(0), c0, M, K);

                // Load B-Matrix
                Value quotient_K = arith::DivUIOp::create(rewriter, 
                    reductionForOp.getLoc(), ivNewKForOp, c32);
                Value rem_K = arith::RemUIOp::create(rewriter, 
                    reductionForOp.getLoc(), rewriter.getIndexType(),
                    quotient_K, c2);

                if (!nDimK) {
                  rem_K = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      ivNewReductionForOp, c2);
                }

                if (oddDimK) {
                  auto rem_BR = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      ivNewReductionForOp, c2);
                  auto remAdd = arith::AddIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(), rem_K,
                      rem_BR);
                  rem_K = arith::RemUIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(), remAdd,
                      c2);
                }
                loadsB =
                    loadTiles(kForOp.getLoc(), rewriter, bBuffer, rem_K, K, N);

                // Compute the tiled dot-product with the A and B tile loads.
                computedFMAs = computeTileMul(kForOp.getLoc(), rewriter, loadsA,
                                              loadsB, iterArgsNewKForOp, M, N);

                scf::YieldOp::create(rewriterNewKForOp, locNewKForOp,
                                                       computedFMAs);
              });

          scf::YieldOp::create(rewriterNewReductionForOp,
              locNewReductionForOp, newKForOp_last_br.getResults());
        });

    SmallVector<Value> FMAs = newReductionForOp_last.getResults();

    //  Check the mlp pattern
    arith::AddFOp addOp;
    arith::MaximumFOp maxOp;
    memref::SubViewOp subview_readOp;
    memref::GetGlobalOp global_readOp;

    auto zeroAttr = rewriter.getFloatAttr(rewriter.getF32Type(), 0.0);
    auto denseAttr = DenseElementsAttr::get(
        VectorType::get(16, rewriter.getF32Type()), zeroAttr);
    auto cst_zero = arith::ConstantOp::create(rewriter, 
        reductionForOp.getLoc(), VectorType::get(16, rewriter.getF32Type()),
        denseAttr);

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
    if (addOp && maxOp) {
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

    if (amx) {
      auto bufferType = MemRefType::get({M, N}, rewriter.getF32Type());
      auto bBuffer =
          memref::AllocaOp::create(rewriter, kForOp.getLoc(), bufferType);

      for (int i = 0, k = 0; i < M; i = i + 16) {
        for (int j = 0; j < N; j = j + 16) {
          Value indexOp_i = arith::ConstantIndexOp::create(rewriter, 
              reductionForOp.getLoc(), i);
          Value indexOp_j = arith::ConstantIndexOp::create(rewriter, 
              reductionForOp.getLoc(), j);
          amx::TileStoreOp::create(rewriter, reductionForOp.getLoc(), bBuffer,
                                            ValueRange{indexOp_i, indexOp_j},
                                            FMAs[k]);
          k++;
        }
      }

      auto c0 =
          arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), 0);
      auto one =
          arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), 1);
      auto mBound =
          arith::ConstantIndexOp::create(rewriter, reductionForOp.getLoc(), M);

      scf::ForOp::create(rewriter, 
          reductionForOp.getLoc(), c0, mBound, one, ValueRange{},
          [&](OpBuilder &nestedBuilder, Location loc, Value iv,
              ValueRange iterArgs) {
            for (int j = 0; j < N; j = j + 32) {
              Value ind_1 = arith::ConstantIndexOp::create(rewriter, 
                  reductionForOp.getLoc(), j);
              Value ind_2 = arith::ConstantIndexOp::create(rewriter, 
                  reductionForOp.getLoc(), j + 16);
              auto row = vector::LoadOp::create(rewriter, 
                  reductionForOp.getLoc(),
                  VectorType::get(16, rewriter.getF32Type()), bBuffer,
                  ValueRange{iv, ind_1});

              auto row2 = vector::LoadOp::create(rewriter, 
                  reductionForOp.getLoc(),
                  VectorType::get(16, rewriter.getF32Type()), bBuffer,
                  ValueRange{iv, ind_2});

              auto shuffle1 = vector::ShuffleOp::create(rewriter, 
                  kForOp.getLoc(), VectorType::get(16, rewriter.getF32Type()),
                  row, row2,
                  ArrayRef<int64_t>{0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20,
                                    21, 22, 23});

              auto shuffle2 = vector::ShuffleOp::create(rewriter, 
                  kForOp.getLoc(), VectorType::get(16, rewriter.getF32Type()),
                  row, row2,
                  ArrayRef<int64_t>{8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14,
                                    15, 28, 29, 30, 31});

              // Load the C-Matrix
              Value valueCRow1 = vector::LoadOp::create(rewriter, 
                  reductionForOp.getLoc(), VectorType::get(16, outsElementType),
                  subviewOpAcc, ValueRange{iv, ind_1});

              Value valueCRow2 = vector::LoadOp::create(rewriter, 
                  reductionForOp.getLoc(), VectorType::get(16, outsElementType),
                  subviewOpAcc, ValueRange{iv, ind_2});

              if (!outsElementType.isF32()) {
                valueCRow1 = performBitcast(reductionForOp.getLoc(), rewriter,
                                            valueCRow1, 16, vnni, elementType,
                                            i32Type, i16Type, cst16);

                valueCRow2 = performBitcast(reductionForOp.getLoc(), rewriter,
                                            valueCRow2, 16, vnni, elementType,
                                            i32Type, i16Type, cst16);
              }

              Value addOp = arith::AddFOp::create(rewriter, 
                  reductionForOp.getLoc(), shuffle1, valueCRow1);

              Value addOp2 = arith::AddFOp::create(rewriter, 
                  reductionForOp.getLoc(), shuffle2, valueCRow2);

              if (addOp && maxOp) {
                Value add_row;
                Value add_row1;
                if (global_readOp) {
                  auto index_mlp = arith::AddIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      nInductionVar, ind_1);
                  auto index_mlp1 = arith::AddIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      nInductionVar, ind_2);
                  add_row = vector::LoadOp::create(rewriter, 
                      reductionForOp.getLoc(), VectorType::get(16, elementType),
                      global_readOp, ValueRange{index_mlp});
                  add_row1 = vector::LoadOp::create(rewriter, 
                      reductionForOp.getLoc(), VectorType::get(16, elementType),
                      global_readOp, ValueRange{index_mlp1});
                }

                if (subview_readOp) {
                  auto index_mlp = arith::AddIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      nInductionVar, ind_1);
                  auto index_mlp1 = arith::AddIOp::create(rewriter, 
                      reductionForOp.getLoc(), rewriter.getIndexType(),
                      nInductionVar, ind_2);

                  auto offsetsVec = subview_readOp.getMixedOffsets();
                  llvm::ArrayRef<mlir::OpFoldResult> offsets = offsetsVec;
                  auto val_offset = offsets[0].dyn_cast<mlir::Value>();

                  add_row = vector::LoadOp::create(rewriter, 
                      reductionForOp.getLoc(), VectorType::get(16, elementType),
                      subview_readOp.getSource(),
                      ValueRange{val_offset, index_mlp});
                  add_row1 = vector::LoadOp::create(rewriter, 
                      reductionForOp.getLoc(), VectorType::get(16, elementType),
                      subview_readOp.getSource(),
                      ValueRange{val_offset, index_mlp1});
                }

                // Fused mlp happens here
                if (add_row) {
                  Value f32MLPVector;
                  Value f32MLPVector1;

                  if (elementType.isBF16()) {
                    auto bitcast_i16 = vector::BitCastOp::create(rewriter, 
                        reductionForOp.getLoc(), VectorType::get(16, i16Type),
                        add_row);
                    auto extend_i32 = arith::ExtUIOp::create(rewriter, 
                        reductionForOp.getLoc(), VectorType::get(16, i32Type),
                        bitcast_i16);
                    auto shiftOp = arith::ShLIOp::create(rewriter, 
                        reductionForOp.getLoc(), VectorType::get(16, i32Type),
                        extend_i32,
                        vector::BroadcastOp::create(rewriter, 
                            reductionForOp.getLoc(),
                            VectorType::get(16, i32Type), cst16));
                    f32MLPVector = vector::BitCastOp::create(rewriter, 
                        reductionForOp.getLoc(),
                        VectorType::get(16, rewriter.getF32Type()), shiftOp);

                    auto bitcast_i16_1 = vector::BitCastOp::create(rewriter, 
                        reductionForOp.getLoc(), VectorType::get(16, i16Type),
                        add_row1);
                    auto extend_i32_1 = arith::ExtUIOp::create(rewriter, 
                        reductionForOp.getLoc(), VectorType::get(16, i32Type),
                        bitcast_i16_1);
                    auto shiftOp_1 = arith::ShLIOp::create(rewriter, 
                        reductionForOp.getLoc(), VectorType::get(16, i32Type),
                        extend_i32_1,
                        vector::BroadcastOp::create(rewriter, 
                            reductionForOp.getLoc(),
                            VectorType::get(16, i32Type), cst16));
                    f32MLPVector1 = vector::BitCastOp::create(rewriter, 
                        reductionForOp.getLoc(),
                        VectorType::get(16, rewriter.getF32Type()), shiftOp_1);
                  }

                  auto add = arith::AddFOp::create(rewriter, 
                      reductionForOp.getLoc(),
                      mlir::VectorType::get(16, rewriter.getF32Type()), addOp,
                      f32MLPVector);
                  auto max = arith::MaximumFOp::create(rewriter, 
                      reductionForOp.getLoc(),
                      mlir::VectorType::get(16, rewriter.getF32Type()), add,
                      cst_zero);
                  addOp = max;

                  auto add_1 = arith::AddFOp::create(rewriter, 
                      reductionForOp.getLoc(),
                      mlir::VectorType::get(16, rewriter.getF32Type()), addOp2,
                      f32MLPVector1);
                  auto max_1 = arith::MaximumFOp::create(rewriter, 
                      reductionForOp.getLoc(),
                      mlir::VectorType::get(16, rewriter.getF32Type()), add_1,
                      cst_zero);
                  addOp2 = max_1;
                }
              }

              auto cvtF32ToBf16 = arith::TruncFOp::create(rewriter, 
                  reductionForOp.getLoc(), VectorType::get({16}, elementType),
                  addOp);

              auto cvtF32ToBf16_2 = arith::TruncFOp::create(rewriter, 
                  reductionForOp.getLoc(), VectorType::get({16}, elementType),
                  addOp2);

              vector::StoreOp::create(rewriter, reductionForOp.getLoc(),
                                               cvtF32ToBf16, subviewOpAcc,
                                               ValueRange{iv, ind_1});

              vector::StoreOp::create(rewriter, reductionForOp.getLoc(),
                                               cvtF32ToBf16_2, subviewOpAcc,
                                               ValueRange{iv, ind_2});
            }
            // Yield results from inner loop to outer loop
            scf::YieldOp::create(nestedBuilder, reductionForOp.getLoc());
          });
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
};

void populateMicroKernelsAMXPatterns(RewritePatternSet &patterns) {
  patterns.add<MicroKernelsAMXOp>(patterns.getContext());
}

struct MicroKernelsAMX : public impl::MicroKernelsAMXBase<MicroKernelsAMX> {
  using MicroKernelsAMXBase::MicroKernelsAMXBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateMicroKernelsAMXPatterns(patterns);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};
} // namespace tpp
} // namespace mlir
