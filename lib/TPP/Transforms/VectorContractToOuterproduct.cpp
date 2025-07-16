//===--------------- VectorContractToOuterproduct.cpp ------------*- C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction to vector outerproduct.
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <cmath>
#include <cstdint>

#define DEBUG_TYPE "vector-contract-to-outerproduct"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORCONTRACTTOOUTERPRODUCT
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace {
/// Returns true if the \p map is transposed.
static bool isTransposed(AffineMap map) {
  auto results = map.getResults();
  // Assert if the map does not have 3 inputs (m, n, k).
  assert(map.getNumInputs() > 2 && "Al least 3 input dim expected");
  // Assert if the result is not 2D.
  assert(map.getNumResults() == 2 && "Only 2 output dim expected");

  // Check the last two dimensions for transposition.
  auto dimExpr0 = dyn_cast<AffineDimExpr>(results[0]);
  auto dimExpr1 = dyn_cast<AffineDimExpr>(results[1]);
  assert((dimExpr0 && dimExpr1) && "Unexpected dim expression");

  // Exclude output map result.
  bool isOutputResultMap =
      dimExpr0 == mlir::getAffineDimExpr(0, map.getContext()) &&
      dimExpr1 == mlir::getAffineDimExpr(1, map.getContext());
  assert(!isOutputResultMap && "Output result map not expected");

  // It's transposed if result found as (k, m) or (n, k), else not transposed.
  if ((dimExpr0 == mlir::getAffineDimExpr(2, map.getContext()) &&
       dimExpr1 == mlir::getAffineDimExpr(0, map.getContext())) ||
      (dimExpr0 == mlir::getAffineDimExpr(1, map.getContext()) &&
       dimExpr1 == mlir::getAffineDimExpr(2, map.getContext())))
    return true;
  return false;
}
} // namespace

namespace mlir {
namespace tpp {
// Enum to represent the type of matmul operation
enum class MatMulType { Standard, Batch, BatchReduce };

struct VectorContractToOuterproductPattern
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(
          contractOp,
          "Unsupported combining kind, only supports ADD at the moment)");

    SmallVector<AffineMap, 3> maps = contractOp.getIndexingMapsArray();
    if (llvm::any_of(
            maps, [](AffineMap map) { return !map.isProjectedPermutation(); }))
      return rewriter.notifyMatchFailure(contractOp, "Unexpected map");

    // Check for the variant of matrix multiply.
    auto iteratorTypes = contractOp.getIteratorTypesArray();
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
      return rewriter.notifyMatchFailure(contractOp, "Not a gemm");
    }

    if (matmulType == MatMulType::Batch)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Batch matmul not supported");
    if (iteratorTypes[outerDimIndex] != vector::IteratorType::parallel ||
        iteratorTypes[outerDimIndex + 1] != vector::IteratorType::parallel ||
        iteratorTypes[outerDimIndex + 2] != vector::IteratorType::reduction)
      return rewriter.notifyMatchFailure(contractOp, "Not a gemm");

    Value acc = contractOp.getAcc();
    // Find the original tensor operands
    auto lhsDefiningOp =
        contractOp.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto rhsDefiningOp =
        contractOp.getRhs().getDefiningOp<vector::TransferReadOp>();
    auto accDefiningOp = acc.getDefiningOp<vector::TransferReadOp>();
    if (!lhsDefiningOp || !rhsDefiningOp || !accDefiningOp)
      return failure();

    // Make sure the inputs being read are whole tensor or subview.
    if (!llvm::all_of(lhsDefiningOp.getIndices(), isZeroInteger) ||
        !llvm::all_of(rhsDefiningOp.getIndices(), isZeroInteger)) {
      return failure();
    }

    auto lhsType = cast<ShapedType>(lhsDefiningOp.getType());
    auto rhsType = cast<ShapedType>(rhsDefiningOp.getType());
    auto accType = cast<ShapedType>(accDefiningOp.getType());

    if (matmulType == MatMulType::BatchReduce &&
        (lhsType.getRank() != 3 || rhsType.getRank() != 3))
      return failure();

    if (matmulType == MatMulType::Standard &&
        (lhsType.getRank() != 2 || rhsType.getRank() != 2))
      return failure();

    // Only 2-D output expected.
    if (accType.getRank() != 2)
      return failure();

    // Handle 3D subviews
    auto mapLHS = maps[0];
    auto mapRHS = maps[1];
    if (matmulType == MatMulType::BatchReduce) {
      mapLHS = mapLHS.dropResult(outerDimIndex);
      mapRHS = mapRHS.dropResult(outerDimIndex);
    }

    int64_t M = accType.getDimSize(0);
    int64_t N = accType.getDimSize(1);
    int64_t K = !isTransposed(mapLHS)
                    ? lhsType.getDimSize(lhsType.getRank() - 1)
                    : lhsType.getDimSize(lhsType.getRank() - 2);

    // Create constants
    Location loc = contractOp.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value cK = rewriter.create<arith::ConstantIndexOp>(loc, K);

    auto elementType = lhsType.getElementType();
    FloatType floatType = cast<FloatType>(elementType);
    Value f0 = rewriter.create<arith::ConstantFloatOp>(
        loc, floatType, APFloat::getZero(floatType.getFloatSemantics()));

    // Create the outer scf.for loop
    auto forOp = rewriter.create<scf::ForOp>(
        loc, c0, cK, c1, ValueRange{acc},
        [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
            ValueRange iterArgs) {
          // Prepare indices and map to iterate over rows/colums and read
          // slices of lhs/rhs input operands.
          SmallVector<Value, 3> lhsIndices, rhsIndices;
          AffineMap lhsMap, rhsMap;
          for (int i = 0; i < lhsType.getRank() - 2; ++i)
            lhsIndices.push_back(c0);
          // LHS operand
          if (!isTransposed(mapLHS)) {
            // If not transposed, iterate over colums and read each column
            // using map.
            lhsIndices.push_back(c0);
            lhsIndices.push_back(iv);
            lhsMap = AffineMap::get(lhsType.getRank(), 0,
                                    {nestedBuilder.getAffineDimExpr(0)},
                                    nestedBuilder.getContext());
          } else {
            // If transposed, iterate over rows and read each row with default
            // map.
            lhsIndices.push_back(iv);
            lhsIndices.push_back(c0);
            lhsMap = AffineMap::get(lhsType.getRank(), 0,
                                    {nestedBuilder.getAffineDimExpr(1)},
                                    nestedBuilder.getContext());
          }

          for (int i = 0; i < rhsType.getRank() - 2; ++i)
            rhsIndices.push_back(c0);
          // RHS operand
          if (!isTransposed(mapRHS)) {
            // If not transposed, iterate over rows and read each row using
            // default map.
            rhsIndices.push_back(iv);
            rhsIndices.push_back(c0);
            rhsMap = AffineMap::get(rhsType.getRank(), 0,
                                    {nestedBuilder.getAffineDimExpr(1)},
                                    nestedBuilder.getContext());
          } else {
            // If transposed, iterate over columns and read each column with
            // default map.
            rhsIndices.push_back(c0);
            rhsIndices.push_back(iv);
            rhsMap = AffineMap::get(rhsType.getRank(), 0,
                                    {nestedBuilder.getAffineDimExpr(0)},
                                    nestedBuilder.getContext());
          }

          Value lhsTensor = lhsDefiningOp.getBase();
          Value rhsTensor = rhsDefiningOp.getBase();
          // Read vector slices using TransferReadOp
          auto lhsSlice = nestedBuilder.create<vector::TransferReadOp>(
              nestedLoc, VectorType::get({M}, lhsType.getElementType()),
              lhsTensor, lhsIndices, AffineMapAttr::get(lhsMap), f0, Value(),
              rewriter.getBoolArrayAttr({true}));

          auto rhsSlice = nestedBuilder.create<vector::TransferReadOp>(
              nestedLoc, VectorType::get({N}, rhsType.getElementType()),
              rhsTensor, rhsIndices, rhsMap, f0, Value(),
              rewriter.getBoolArrayAttr({true}));

          // Perform outer product
          auto outerProduct = nestedBuilder.create<vector::OuterProductOp>(
              nestedLoc, accType, lhsSlice, rhsSlice, iterArgs[0],
              vector::CombiningKind::ADD);

          // Yield the result
          nestedBuilder.create<scf::YieldOp>(nestedLoc,
                                             ValueRange{outerProduct});
        });

    // Replace the original contraction with the result of the loop
    rewriter.replaceOp(contractOp, forOp.getResults());

    return success();
  }
};

struct VectorContractToOuterproduct
    : public tpp::impl::VectorContractToOuterproductBase<
          VectorContractToOuterproduct> {

  using VectorContractToOuterproductBase::VectorContractToOuterproductBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<VectorContractToOuterproductPattern>(context);

    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace tpp
} // namespace mlir
