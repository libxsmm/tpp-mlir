//===- VectorContractToAMX.cpp ----------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of vector contraction to amx.
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "vector-contract-to-amx"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_VECTORCONTRACTTOAMX
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace {
// Helper function to down convert and copy the result back to the output
// matrix.
static void downConvertAndCopyResult(OpBuilder &rewriter, Location loc,
                                     Value src, Value dst,
                                     MemRefType bufferType, ShapedType accType,
                                     int64_t m, int64_t n) {
  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto sixteen = arith::ConstantIndexOp::create(rewriter, loc, 16);
  auto mBound = arith::ConstantIndexOp::create(rewriter, loc, m);
  auto nBound = arith::ConstantIndexOp::create(rewriter, loc, n);
  scf::ForOp::create(rewriter, 
      loc, c0, mBound, one, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location loc, Value iv,
          ValueRange iterArgs) {
        scf::ForOp::create(nestedBuilder, 
            loc, c0, nBound, sixteen, iterArgs,
            [&](OpBuilder &innerBuilder, Location loc, Value innerIv,
                ValueRange innerIterArgs) {
              auto elementType = bufferType.getElementType();
              FloatType floatType = cast<FloatType>(elementType);
              Value f0 = arith::ConstantFloatOp::create(rewriter, 
                  loc, floatType,
                  APFloat::getZero(floatType.getFloatSemantics()));
              // Read
              auto readC = vector::TransferReadOp::create(rewriter, 
                  loc, VectorType::get({16}, bufferType.getElementType()), src,
                  ValueRange{iv, innerIv}, f0, ArrayRef{true});
              // Convert
              auto cvtF32ToBf16 = arith::TruncFOp::create(rewriter, 
                  loc, VectorType::get({16}, accType.getElementType()), readC);
              // Write
              vector::TransferWriteOp::create(rewriter, loc, cvtF32ToBf16, dst,
                                                   ValueRange{iv, innerIv},
                                                   ArrayRef{true}).getResult();

              scf::YieldOp::create(innerBuilder, loc);
            });
        // Yield results from inner loop to outer loop
        scf::YieldOp::create(nestedBuilder, loc);
      });
}

// Helper function to up-convert and copy the accumulator.
static void upConvertAndCopyAccumulator(OpBuilder &rewriter, Location loc,
                                        Value src, Value dst,
                                        Type inputElementType,
                                        Type outputElementType, int64_t m,
                                        int64_t n) {
  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto sixteen = arith::ConstantIndexOp::create(rewriter, loc, 16);
  auto mBound = arith::ConstantIndexOp::create(rewriter, loc, m);
  auto nBound = arith::ConstantIndexOp::create(rewriter, loc, n);
  scf::ForOp::create(rewriter, 
      loc, c0, mBound, one, ValueRange{},
      [&](OpBuilder &nestedBuilder, Location loc, Value iv,
          ValueRange iterArgs) {
        scf::ForOp::create(nestedBuilder, 
            loc, c0, nBound, sixteen, iterArgs,
            [&](OpBuilder &innerBuilder, Location loc, Value innerIv,
                ValueRange innerIterArgs) {
              // Read
              auto readC = vector::TransferReadOp::create(rewriter, 
                  loc, VectorType::get({16}, inputElementType), src,
                  ValueRange{iv, innerIv}, std::nullopt, ArrayRef{true});
              auto bitcastLoad = vector::BitCastOp::create(rewriter, 
                  loc, VectorType::get({16}, rewriter.getI16Type()), readC);
              // Convert
              auto cvtSIToUI32 = arith::ExtSIOp::create(rewriter, 
                  loc, VectorType::get({16}, rewriter.getI32Type()),
                  bitcastLoad);
              int8_t bitsToShiftLeft = 16;
              auto shiftLeft16bit = arith::ShLIOp::create(rewriter, 
                  loc, cvtSIToUI32,
                  arith::ConstantOp::create(rewriter, 
                      loc, DenseElementsAttr::get(
                               VectorType::get({16}, rewriter.getI32Type()),
                               rewriter.getI32IntegerAttr(bitsToShiftLeft))));
              auto bitcast = arith::BitcastOp::create(rewriter, 
                  loc, VectorType::get({16}, rewriter.getF32Type()),
                  shiftLeft16bit);
              // Write
              vector::TransferWriteOp::create(rewriter, 
                  loc, bitcast, dst, ValueRange{iv, innerIv}, ArrayRef{true});
              scf::YieldOp::create(innerBuilder, loc);
            });

        // Yield results from inner loop to outer loop
        scf::YieldOp::create(nestedBuilder, loc);
      });
}

// Helper function to initialize accumulators using amx.tile_load.
static SmallVector<Value> initializeAccumulators(OpBuilder &rewriter,
                                                 Location loc, Value buffer,
                                                 Type accElementType, int64_t m,
                                                 int64_t n) {
  SmallVector<Value> initAccs;
  auto amxTile16x16xF32Ty = mlir::amx::TileType::get({16, 16}, accElementType);
  for (auto mIndices = 0; mIndices < m; mIndices += 16) {
    for (auto nIndices = 0; nIndices < n; nIndices += 16) {
      auto acc = amx::TileLoadOp::create(rewriter, 
          loc, amxTile16x16xF32Ty, buffer,
          ValueRange{arith::ConstantIndexOp::create(rewriter, loc, mIndices),
                     arith::ConstantIndexOp::create(rewriter, loc, nIndices)});
      initAccs.push_back(acc);
    }
  }
  return initAccs;
}

/// Returns true if the \p map is transposed.
static bool isTransposed(AffineMap map) {
  auto numInputDims = map.getNumInputs();
  // Assert if the map does not have 4 or 5 inputs ([] m, n, k).
  assert((numInputDims == 4 || numInputDims == 5) &&
         "4 or 5 input dim expected");
  // Assert if the result is not 2D.
  assert(map.getNumResults() == 2 && "Only 2 output dim expected");

  // Check the last two dimensions for transposition.
  auto results = map.getResults();
  auto dimExpr0 = dyn_cast<AffineDimExpr>(results[0]);
  auto dimExpr1 = dyn_cast<AffineDimExpr>(results[1]);
  assert((dimExpr0 && dimExpr1) && "Unexpected dim expression");

  MLIRContext *context = map.getContext();
  auto mDim = mlir::getAffineDimExpr(numInputDims - 3, context);
  auto nDim = mlir::getAffineDimExpr(numInputDims - 2, context);
  auto kDim = mlir::getAffineDimExpr(numInputDims - 1, context);
  // Exclude output map result.
  if ((dimExpr0 != mDim) && (dimExpr1 != nDim))
    return false;

  // It's transposed if result found as (k, m) or (n, k), else not transposed.
  return (dimExpr0 == kDim && dimExpr1 == mDim) ||
         (dimExpr0 == nDim && dimExpr1 == kDim);
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

/// This pass lowers vector.contract (linalg.batch_reduce_matmul) for bf16
/// type into sequence of amx.tile_load, amx.tile_mulf, amx.tile_store along
/// with the required up-convert and down-convert.
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
///
///   // allocate local buffer for result accumulation in <32x32xf32>
///   memref.alloca
///
///   // Up-convert C matrix and copy to local buffer
///   vector.transfer_read
///   vector.bitcast + arith.extsi + arith.shli + vector.bitcast
///   vector.transfer_write
///
///   // load tiles of <16x16xf32> from local buffer
///   amx.tile_load // 4 loads, pass them as iterargs
///
///   scf.for (iterargs = loaded tiles) // batch-reduce
///    scf.for (iterargs = batch-reduce iterArgs) // k-tile
///     amx.load // 2 loads from A matrix
///     amx.load // 2 loads from B matrix
///     amx.tile_mulf // 4 multiply and accumulate in <32x32xf32>
///     scf.yield
///   scf.yield
///   amx.tile_store // store back into local buffer
///
///   // Down-convert local buffer and store back to C matrix
///   vector.transfer_read
///   x86vector.avx512.intr.cvtneps2bf16.512
///   vector.transfer_write
///  .............
///  ............
struct VectorContractToAMX
    : public tpp::impl::VectorContractToAMXBase<VectorContractToAMX> {

  using VectorContractToAMXBase::VectorContractToAMXBase;

  void runOnOperation() override;

private:
  TransformationContext ctx;
};

namespace {
// Returns true if the loop has an iterarg.
static bool hasIterArg(scf::ForOp loop) {
  return loop.getNumRegionIterArgs() > 0;
}

// Returns true if the argument val is an iterarg of the loop.
static bool containsIterArg(Value val, scf::ForOp loop) {
  return llvm::any_of(loop.getRegionIterArgs(),
                      [&](BlockArgument arg) { return val == arg; });
}

// Returns the outermost loop with argument acc as iterarg accumulator.
static scf::ForOp getOutermostLoopWithIterargAccumulator(scf::ForOp loop,
                                                         Value &acc) {
  scf::ForOp outermostLoop = loop;
  auto parentOp = loop->getParentOfType<scf::ForOp>();
  while (parentOp && hasIterArg(parentOp)) {
    outermostLoop = parentOp;
    parentOp = parentOp->getParentOfType<scf::ForOp>();
  }
  return outermostLoop;
}

static bool hasUserWriteOp(Value matResult) {
  for (auto user : matResult.getUsers()) {
    if (isa<vector::TransferWriteOp>(user))
      return true;
  }
  return false;
}

// Verifies that the accumulator is coming through a chain of iterargs of nested
// loop and it is define by 'TransferReadOp'.
static LogicalResult verifyAccumulator(PatternRewriter &rewriter,
                                       vector::ContractionOp op,
                                       mlir::tpp::TransformationContext &ctx,
                                       Value &acc,
                                       vector::TransferReadOp &accDefiningOp) {

  ctx.innerForOp = op->getParentOfType<scf::ForOp>();
  if (!ctx.innerForOp)
    return rewriter.notifyMatchFailure(op, "Inner loop not found");

  // Verify original inner loop has only one iterarg.
  auto origIterArgs = ctx.innerForOp.getRegionIterArgs();
  if (origIterArgs.size() != 1)
    return rewriter.notifyMatchFailure(op, "Exactly one iterarg expected");

  // Verify chain, accumulator must be inner loop's iterarg.
  auto bbArg = dyn_cast<BlockArgument>(acc);

  // This block arg must be init arg, not induction variable.
  if (bbArg && ((bbArg.getOwner() != ctx.innerForOp.getBody()) ||
                (bbArg.getArgNumber() == 0)))
    return rewriter.notifyMatchFailure(op, "Accumulator is not an iterarg");

  // This iterarg must be intialized by outer loop's iterarg.
  auto innerInitValue = ctx.innerForOp.getInitArgs()[bbArg.getArgNumber() - 1];
  auto outerBBArg = dyn_cast<BlockArgument>(innerInitValue);
  ctx.outerForOp = ctx.innerForOp->getParentOfType<scf::ForOp>();

  Value matResult = ctx.outerForOp && hasIterArg(ctx.outerForOp)
                        ? ctx.outerForOp.getResult(0)
                        : ctx.innerForOp.getResult(0);
  if (!hasUserWriteOp(matResult))
    return rewriter.notifyMatchFailure(
        op, "Store of accumulated result is not found");

  acc = outerBBArg && ctx.outerForOp && hasIterArg(ctx.outerForOp) &&
                containsIterArg(innerInitValue, ctx.outerForOp)
            ? ctx.outerForOp.getInitArgs()[outerBBArg.getArgNumber() - 1]
            : innerInitValue;

  //  This must be defined by vector.transfer_read
  accDefiningOp = acc.getDefiningOp<vector::TransferReadOp>();
  if (!accDefiningOp)
    return rewriter.notifyMatchFailure(op,
                                       "Accumulator intializer did not match");

  // Only 2-D output expected.
  auto accType = cast<ShapedType>(accDefiningOp.getType());
  if (accType.getRank() != 2)
    return rewriter.notifyMatchFailure(op, "Only 2-D output is expected");

  return success();
}

// Helper to create a collapse_shape op to collapse the inner dimensions of
// a memref. The firstDimToCollapse is the first dimension that will be
// collapsed with the rest of the dimensions.
static Value collapseInnerDims(OpBuilder &builder, mlir::Location loc,
                               Value input, int64_t firstDimToCollapse) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  if (inputType.getRank() == 1)
    return input;
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < firstDimToCollapse; ++i)
    reassociation.push_back(ReassociationIndices{i});
  ReassociationIndices collapsedIndices;
  for (int64_t i = firstDimToCollapse; i < inputType.getRank(); ++i)
    collapsedIndices.push_back(i);
  reassociation.push_back(collapsedIndices);
  return memref::CollapseShapeOp::create(builder, loc, input, reassociation);
}

// Helper to create collapse_shape and tile_load ops for the input tiles.
static SmallVector<Value, 4> createTileLoads(OpBuilder &builder, Location loc,
                                             amx::TileType resType,
                                             Value subview, int dimSize,
                                             Value c0, bool isLHS = true) {
  SmallVector<Value, 4> loadTiles;
  auto subviewType = cast<ShapedType>(subview.getType());
  // Choose step for the considered amx tile type <16x32xbf16> for A and B
  // matrix.
  unsigned dimStep = isLHS                                   ? 16
                     : subviewType.getElementType().isBF16() ? 32
                                                             : 64;
  for (int i = 0; i < dimSize; i += dimStep) {
    auto mIndex = isLHS ? arith::ConstantIndexOp::create(builder, loc, i) : c0;
    auto nIndex = isLHS ? c0 : arith::ConstantIndexOp::create(builder, loc, i);
    auto subviewRank = subviewType.getRank();
    auto collapsedOpnd =
        collapseInnerDims(builder, loc, subview, subviewRank - 2);
    auto elem = amx::TileLoadOp::create(builder, loc, resType, collapsedOpnd,
                                                ValueRange{c0, mIndex, nIndex});
    loadTiles.push_back(elem);
  }
  return loadTiles;
}

// Helper to create tile mul ops for the input tiles.
static SmallVector<Value> createTileMuls(OpBuilder &builder, Location loc,
                                         amx::TileType resType,
                                         SmallVector<Value, 4> aLoadTiles,
                                         SmallVector<Value, 4> bLoadTiles,
                                         ValueRange iterArgs) {
  SmallVector<Value> results;
  int numIterArgs = 0;
  for (unsigned i = 0; i < aLoadTiles.size(); i++) {
    for (unsigned j = 0; j < bLoadTiles.size(); j++) {
      auto amx =
          resType.getElementType().isFloat()
              ? amx::TileMulFOp::create(builder, loc, resType, aLoadTiles[i],
                                                bLoadTiles[j],
                                                iterArgs[numIterArgs++])
              : amx::TileMulIOp::create(builder, loc, resType, aLoadTiles[i],
                                                bLoadTiles[j],
                                                iterArgs[numIterArgs++]);
      results.push_back(amx->getResult(0));
    }
  }
  return results;
}
} // namespace

struct VectorContractToAMXPattern
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;
  VectorContractToAMXPattern(MLIRContext *context, TransformationContext &ctx)
      : OpRewritePattern<vector::ContractionOp>(context), ctx(ctx) {}

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(
          op, "Unsupported combining kind, only supports ADD at the moment)");

    auto maskableOp = cast<vector::MaskableOpInterface>(op.getOperation());
    if (maskableOp.isMasked()) {
      return rewriter.notifyMatchFailure(op, "Masked contractOp not supported");
    }

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

    if (matmulType == MatMulType::Batch || matmulType == MatMulType::Standard)
      return rewriter.notifyMatchFailure(op,
                                         "Standard/Batch matmul not supported");

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
      return rewriter.notifyMatchFailure(
          op, "LHS or RHS not defined by TransferReadOp");

    if (accDefiningOp)
      return rewriter.notifyMatchFailure(
          op, "Accumulator defined by TransferReadOp");

    if (!llvm::all_of(lhsDefiningOp.getIndices(), isZeroInteger) ||
        !llvm::all_of(rhsDefiningOp.getIndices(), isZeroInteger))
      return rewriter.notifyMatchFailure(
          op, "Inputs are not whole tensor or subview");

    auto lhsType = cast<ShapedType>(lhsDefiningOp.getType());
    auto rhsType = cast<ShapedType>(rhsDefiningOp.getType());
    auto expectedRank = matmulType == MatMulType::BatchReduce ? 4 : 3;

    if (!vnni::utils::isInVnniLayout(expectedRank, lhsType) ||
        !vnni::utils::isInVnniLayout(expectedRank, rhsType))
      return rewriter.notifyMatchFailure(op, "Expects VNNI layout");

    auto vnniFactor = vnni::utils::getVnniBlockingFactor(rhsType);

    if (vnniFactor != 2 && vnniFactor != 4)
      return rewriter.notifyMatchFailure(op, "Unexpected VNNI factor");

    if (matmulType == MatMulType::BatchReduce &&
        (lhsType.getRank() != 4 || rhsType.getRank() != 4))
      return rewriter.notifyMatchFailure(
          op, "BatchReduce matmul type in VNNI with incorrect rank\n");

    if (matmulType == MatMulType::Standard &&
        (lhsType.getRank() != 3 || rhsType.getRank() != 3))
      return rewriter.notifyMatchFailure(
          op, "Standard matmul type in VNNI with incorrect rank\n");

    // Check for non-transposed matrices.
    auto mapLHS = maps[0];
    auto mapRHS = maps[1];
    if (matmulType == MatMulType::BatchReduce) {
      mapLHS = mapLHS.dropResult(0);
      mapLHS = mapLHS.dropResult(1);
      mapRHS = mapRHS.dropResult(0);
      mapRHS = mapRHS.dropResult(1);
    }

    if (isTransposed(mapLHS) || isTransposed(mapRHS))
      return rewriter.notifyMatchFailure(
          op, "Transposed matrices are not expected");

    if (failed(verifyAccumulator(rewriter, op, ctx, acc, accDefiningOp)))
      return rewriter.notifyMatchFailure(
          op, "Failed to verify accumulator and loop structure\n");

    auto accType = cast<ShapedType>(accDefiningOp.getType());
    int64_t M = accType.getDimSize(0);
    int64_t N = accType.getDimSize(1);
    // M and N must be divisible by 16.
    if (M % 16 != 0 || N % 16 != 0)
      return rewriter.notifyMatchFailure(
          op, "Output matrix dimensions must be divisible by 16");

    auto accSubview = accDefiningOp.getBase();
    Location loc = op.getLoc();
    scf::ForOp insertAt =
        getOutermostLoopWithIterargAccumulator(ctx.innerForOp, acc);
    rewriter.setInsertionPoint(insertAt->getBlock(),
                               std::next(insertAt->getIterator(), 1));

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Type accElementType;
    Type outputElementType = accType.getElementType();
    outputElementType.isFloat() ? accElementType = rewriter.getF32Type()
                                : accElementType = rewriter.getI32Type();

    Value accBuffer;
    MemRefType bufferType;
    SmallVector<Value> initAccs;
    if (outputElementType.isBF16()) {
      // Create a new buffer to hold the accumulator at higher precision.
      bufferType = MemRefType::get({M, N}, accElementType);
      accBuffer = memref::AllocaOp::create(rewriter, loc, bufferType);

      // Up Convert and copy the original accumulator to the buffer.
      upConvertAndCopyAccumulator(rewriter, loc, accSubview, accBuffer,
                                  accType.getElementType(), accElementType, M,
                                  N);
    }

    // Intialize each accumulator with a tileType of size 16x16
    initAccs = initializeAccumulators(
        rewriter, loc, outputElementType.isBF16() ? accBuffer : accSubview,
        accElementType, M, N);

    SmallVector<Value, 4> results;
    auto amxTile16x16xF32Ty =
        mlir::amx::TileType::get({16, 16}, accElementType);
    auto inputTileElemType = lhsType.getElementType();
    auto amxInputTilesOf16xColxTy =
        mlir::amx::TileType::get({16, 16 * vnniFactor}, inputTileElemType);

    // Lamda to create inner loop body.
    auto createLoopBody = [&](OpBuilder &innerBuilder, Location loc, Value iv,
                              Value innerIv, ValueRange innerIterArgs) {
      IRMapping mapping;
      // Update index of LHS matrix subview for batch dimension if corresponding
      // loop is needed.
      if (iv)
        mapping.map(lhsDefiningOp.getBase().getDefiningOp()->getOperand(1),
                    iv);
      // Update index of LHS matrix subview for K dimension.
      mapping.map(
          lhsDefiningOp.getBase().getDefiningOp()->getOperand(iv ? 3 : 1),
          innerIv);
      auto lhsClone = innerBuilder.clone(
          *lhsDefiningOp.getBase().getDefiningOp(), mapping);
      // Load matrix A tile
      SmallVector<Value, 4> aLoadTiles =
          createTileLoads(innerBuilder, loc, amxInputTilesOf16xColxTy,
                          lhsClone->getResult(0), M, c0, true);

      IRMapping rhsMapping;
      // Update index of LHS matrix subview for batch dimension if corresponding
      // loop is needed.
      if (iv)
        rhsMapping.map(rhsDefiningOp.getBase().getDefiningOp()->getOperand(1),
                       iv);
      // Update index of LHS matrix subview for K dimension.
      rhsMapping.map(
          rhsDefiningOp.getBase().getDefiningOp()->getOperand(iv ? 2 : 1),
          innerIv);
      auto rhsClone = innerBuilder.clone(
          *rhsDefiningOp.getBase().getDefiningOp(), rhsMapping);
      // Load matrix B tile, vnni factor and N tile size will be collapsed as
      // effective tilse size.
      SmallVector<Value, 4> bLoadTiles =
          createTileLoads(innerBuilder, loc, amxInputTilesOf16xColxTy,
                          rhsClone->getResult(0), N * vnniFactor, c0, false);

      // Create MxN/16x16 different AMXs TileMulFOp.
      results = createTileMuls(innerBuilder, loc, amxTile16x16xF32Ty,
                               aLoadTiles, bLoadTiles, innerIterArgs);

      // Yield all results
      scf::YieldOp::create(innerBuilder, loc, results);
    };

    // Lamda to create inner loop with loop body.
    auto createInnerLoop = [&](OpBuilder &nestedBuilder, Location loc, Value iv,
                               ValueRange iterArgs) {
      return scf::ForOp::create(nestedBuilder, 
          loc, ctx.innerForOp.getLowerBound(), ctx.innerForOp.getUpperBound(),
          ctx.innerForOp.getStep(), iterArgs,
          [&](OpBuilder &innerBuilder, Location loc, Value innerIv,
              ValueRange innerIterArgs) {
            createLoopBody(innerBuilder, loc, iv, innerIv, innerIterArgs);
          });
    };

    scf::ForOp newOuterForOp;
    // Single iteration outer loop may have been optimized out due to the
    // register blocking factor. If the outer loop is not optimized out, we need
    // to create a new outer.
    if (ctx.outerForOp && hasIterArg(ctx.outerForOp)) {
      // Create new outer loop with M/blocking-factor different accumulators.
      newOuterForOp = scf::ForOp::create(rewriter, 
          loc, ctx.outerForOp.getLowerBound(), ctx.outerForOp.getUpperBound(),
          ctx.outerForOp.getStep(), initAccs,
          [&](OpBuilder &nestedBuilder, Location loc, Value iv,
              ValueRange iterArgs) {
            auto newInnerForOp =
                createInnerLoop(nestedBuilder, loc, iv, iterArgs);
            scf::YieldOp::create(nestedBuilder, loc, newInnerForOp.getResults());
          });
    } else {
      // Create single loop with M/blocking-factor different accumulators.
      newOuterForOp = scf::ForOp::create(rewriter, 
          loc, ctx.innerForOp.getLowerBound(), ctx.innerForOp.getUpperBound(),
          ctx.innerForOp.getStep(), initAccs,
          [&](OpBuilder &innerBuilder, Location loc, Value innerIv,
              ValueRange innerIterArgs) {
            createLoopBody(innerBuilder, loc, nullptr, innerIv, innerIterArgs);
          });
    }

    Value matResult = ctx.outerForOp && hasIterArg(ctx.outerForOp)
                          ? ctx.outerForOp.getResult(0)
                          : ctx.innerForOp.getResult(0);
    Operation *writeOp;
    for (auto user : matResult.getUsers()) {
      writeOp = dyn_cast<vector::TransferWriteOp>(user);
      if (writeOp)
        break;
    }

    // Store final results back to original locations.
    if (writeOp) {
      int numIterArgs = 0;
      for (auto mIndices = 0; mIndices < M; mIndices += 16) {
        for (auto nIndices = 0; nIndices < N; nIndices += 16) {
          amx::TileStoreOp::create(rewriter, 
              loc, outputElementType.isBF16() ? accBuffer : accSubview,
              ValueRange{
                  arith::ConstantIndexOp::create(rewriter, loc, mIndices),
                  arith::ConstantIndexOp::create(rewriter, loc, nIndices)},
              newOuterForOp.getResult(numIterArgs++));
        }
      }
    }

    // Down convert and copy the result back to the result matrix.
    if (outputElementType.isBF16()) {
      downConvertAndCopyResult(rewriter, loc, accBuffer, accSubview, bufferType,
                               accType, M, N);
    }

    // Erase original write.
    if (writeOp)
      rewriter.eraseOp(writeOp);
    return success();
  }

private:
  TransformationContext &ctx;
};

void VectorContractToAMX::runOnOperation() {
  auto funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.add<VectorContractToAMXPattern>(context, ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace tpp
} // namespace mlir

std::unique_ptr<Pass> createVectorContractToAMX() {
  return std::make_unique<VectorContractToAMX>();
}