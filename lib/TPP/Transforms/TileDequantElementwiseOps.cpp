//===- TileDequantElementwiseOps.cpp - Scalar Broadcast Vectorization -===//
//
// This pass transforms element-wise linalg.generic operations to use scalar
// broadcast vectorization for better register efficiency on AVX-512.
//
// Strategy:
// - Outer loop iterates serially over one dimension
// - Load scalar from that dimension
// - Broadcast scalar to vector using vector.splat (hoisted outside inner loop)
// - Inner loop processes vectorized chunks of the other dimension
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_TILEDEQUANTELEMENTWISEOPS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace mlir {
namespace tpp {

/// Check if a linalg.generic op contains dequantization op sequence and is
/// suitable for tiling and vectorization. For example, for a pattern like this:
/// ```
///   %0 = linalg.generic
///     ... = arith.mulf
///     ... = arith.sitofp
///     ... = arith.mulf
/// ```
/// The first arith.mulf is the multiplication of two scale values, arith.sitofp
/// is the casting of i32 gemm result to float, and the second arith.mulf is the
/// multiplication of the casted value with the scale.
bool hasValidOpPattern(linalg::GenericOp op) {
  // Must operate on memrefs
  if (!op.hasPureBufferSemantics())
    return false;

  Block &body = op.getRegion().front();
  // Exit if no non-terminating operations in the body (e.g., just a yield)
  if (body.without_terminator().empty())
    return false;

  if (op.getNumDpsInits() != 1)
    return false;

  // Only handle 2D operations for now
  if (op.getNumLoops() != 2)
    return false;

  if (!llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType type) {
        return type == utils::IteratorType::parallel;
      }))
    return false;

  return llvm::any_of(body.without_terminator(), [](Operation &bodyOp) {
    return isa<arith::MulFOp, arith::MulIOp>(&bodyOp);
  });
}

/// Keep info of broadcast operands in linalg.generic body. We will use this
/// info to check if the broadcast operands are used in the same multiplication
/// operation and to identify which operand is broadcasted along which
/// dimension.
struct BroadcastOperandInfo {
  Value operand;
  unsigned operandIndex;
};

SmallVector<BroadcastOperandInfo> findBroadcastOperands(linalg::GenericOp op) {
  SmallVector<BroadcastOperandInfo> broadcastOps;
  unsigned numLoops = op.getNumLoops();
  auto indexingMaps = op.getIndexingMapsArray();

  for (auto [idx, operand] : llvm::enumerate(op.getDpsInputs())) {
    AffineMap map = indexingMaps[idx];

    // Check if this operand has fewer dimensions than the loop space.
    if (map.getNumResults() < numLoops) {
      BroadcastOperandInfo info;
      info.operand = operand;
      info.operandIndex = idx;
      broadcastOps.push_back(info);
    }
  }
  return broadcastOps;
}

/// Pattern to tile and vectorize element-wise linalg.generic with broadcast
/// operands.
class TileDequantElementwiseOpsPattern
    : public OpRewritePattern<linalg::GenericOp> {
public:
  TileDequantElementwiseOpsPattern(MLIRContext *context, unsigned vectorSize)
      : OpRewritePattern<linalg::GenericOp>(context), vectorSize(vectorSize) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!hasValidOpPattern(op))
      return rewriter.notifyMatchFailure(
          op, "Does not match expected pattern of element-wise dequantization");

    // Find broadcast operands
    auto broadcastOps = findBroadcastOperands(op);
    if (broadcastOps.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "Expected exactly two broadcast operands");

    auto lhsBcastOp = broadcastOps[0];
    auto rhsBcastOp = broadcastOps[1];
    BlockArgument bcastBlockArg1 = op.getMatchingBlockArgument(
        op.getDpsInputOperand(lhsBcastOp.operandIndex));
    BlockArgument bcastBlockArg2 = op.getMatchingBlockArgument(
        op.getDpsInputOperand(rhsBcastOp.operandIndex));
    // Check broadcast block arguments(bcastBlockArg1 & bcastBlockArg2)
    // has single user.
    SmallVector<Value, 2> oneDimArgs = {bcastBlockArg1, bcastBlockArg2};
    auto getSingleUser = [](Value arg) -> Operation * {
      if (!arg.hasOneUse())
        return nullptr;
      return *arg.getUsers().begin();
    };
    for (auto arg : oneDimArgs) {
      auto *user = getSingleUser(arg);
      // If there are multiple users or no users, return failure.
      if (!user)
        return rewriter.notifyMatchFailure(
            op, "Broadcast operand does not have single user");
      // Unary op user must be arith::ExtFOp.
      if (user->getNumOperands() == 1 && !isa<arith::ExtFOp>(user))
        return rewriter.notifyMatchFailure(op, "Expected unary arith::ExtFOp");
    }

    Location loc = op.getLoc();
    auto outputType = cast<MemRefType>(op.getDpsInits()[0].getType());
    int64_t serialDimSize = outputType.getShape()[0];
    int64_t vectorDimSize = outputType.getShape()[1];

    // Check if dimensions are static
    if (ShapedType::isDynamic(serialDimSize) ||
        ShapedType::isDynamic(vectorDimSize))
      return rewriter.notifyMatchFailure(
          op, "Expected static dimensions for output memref");

    // Create constants
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value serialBound = arith::ConstantIndexOp::create(rewriter, loc, serialDimSize);
    Value vectorBound = arith::ConstantIndexOp::create(rewriter, loc, vectorDimSize);
    Value stepValue =
        arith::ConstantIndexOp::create(rewriter, loc, this->vectorSize);

    // Create outer serial loop
    auto outerLoop = scf::ForOp::create(rewriter, loc, c0, serialBound, c1);
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    Value outerIV = outerLoop.getInductionVar();

    // Load and broadcast the LHS broadcast operand (hoisted outside inner loop)
    Value lhsScalar =
        memref::LoadOp::create(rewriter, loc, lhsBcastOp.operand, outerIV)
            ->getResult(0);
    Type scalarType = lhsScalar.getType();
    VectorType vecType = VectorType::get({this->vectorSize}, scalarType);
    Value lhsBroadcastVec =
        vector::BroadcastOp::create(rewriter, loc, vecType, lhsScalar);

    // Create inner vectorized loop
    auto innerLoop = scf::ForOp::create(rewriter, loc, c0, vectorBound, stepValue);
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    Value innerIV = innerLoop.getInductionVar();

    // Build vector operations based on the original linalg.generic body
    Block &linalgBody = op.getRegion().front();
    IRMapping mapping;
    // Map the LHS broadcast operand to the hoisted broadcast vector
    mapping.map(linalgBody.getArgument(lhsBcastOp.operandIndex),
                lhsBroadcastVec);

    // Load other operand as vector.
    auto indexingMaps = op.getIndexingMapsArray();
    unsigned argIdx = 0;
    for (auto operand : op.getDpsInputs()) {
      // Skip if this is already mapped (the LHS broadcast operand)
      if (mapping.contains(linalgBody.getArgument(argIdx))) {
        argIdx++;
        continue;
      }

      // Determine indices based on indexing map
      auto map = indexingMaps[argIdx];
      SmallVector<Value> indices;
      if (map.getNumResults() == 2) {
        // 2D operand
        indices = {outerIV, innerIV};
      } else if (map.getNumResults() == 1) {
        // 1D operand
        auto expr = dyn_cast<AffineDimExpr>(map.getResult(0));
        if (expr)
          indices = {innerIV};
      }

      auto memrefType = cast<MemRefType>(operand.getType());
      Type elemType = memrefType.getElementType();
      VectorType loadVecType = VectorType::get({this->vectorSize}, elemType);

      Value padding = arith::ConstantOp::create(rewriter, loc, elemType,
                                                rewriter.getZeroAttr(elemType));
      Value vectorLoad = vector::TransferReadOp::create(
          rewriter, loc, loadVecType, operand, indices, padding);
      mapping.map(linalgBody.getArgument(argIdx), vectorLoad);
      argIdx++;
    }

    // Clone the computation from linalg.generic body.
    Value result;
    for (Operation &bodyOp : linalgBody.without_terminator()) {
      Operation *cloned = rewriter.clone(bodyOp, mapping);

      // Check if this cloned operation has a constant operand which can be
      // vectorized.
      for (auto operand : cloned->getOperands()) {
        if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
          // Vectorize the constant by creating a vector constant with the same value
          auto constValue = constOp.getValue();
          if (auto intAttr = dyn_cast<IntegerAttr>(constValue)) {
            auto elemType = intAttr.getType();
            VectorType constVecType = VectorType::get({vectorSize}, elemType);
            Value vecConst = arith::ConstantOp::create(rewriter, loc, constVecType, rewriter.getZeroAttr(constVecType));
            tensor::SplatOp::create(rewriter, loc, constVecType, operand);
            // Update the operand in the cloned operation to use the vectorized
            // constant.
            for (unsigned i = 0; i < cloned->getNumOperands(); ++i) {
              if (cloned->getOperand(i) == operand) {
                cloned->setOperand(i, vecConst);
                mapping.map(operand, vecConst);
                break;
              }
            }
          }
        }
      }

      // Set the result type of cloned op to appropriate vector type.
      auto origResultElemType = bodyOp.getResult(0).getType();
      VectorType clonedVecType = VectorType::get({vectorSize}, origResultElemType);
      cloned->getResult(0).setType(clonedVecType);

      // Map results for next operations.
      for (auto [original, cloned_val] :
           llvm::zip(bodyOp.getResults(), cloned->getResults())) {
        mapping.map(original, cloned_val);
      }

      // Track the final result
      result = cloned->getResult(0);
    }

    // Store result vector
    SmallVector<Value> storeIndices{outerIV, innerIV};
    vector::TransferWriteOp::create(rewriter, loc, result, op.getDpsInits()[0],
                                    storeIndices, true);

    // Remove the original linalg.generic
    rewriter.eraseOp(op);
    return success();
  }

private:
  unsigned vectorSize;
};

struct TileDequantElementwiseOps
    : public tpp::impl::TileDequantElementwiseOpsBase<
          TileDequantElementwiseOps> {

  // Add constructor that accepts option.
  using TileDequantElementwiseOpsBase::TileDequantElementwiseOpsBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    TileDequantElementwiseOpsOptions options;
    options.vectorSize = vectorSize;
    patterns.add<TileDequantElementwiseOpsPattern>(context, options.vectorSize);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace tpp
} // namespace mlir
