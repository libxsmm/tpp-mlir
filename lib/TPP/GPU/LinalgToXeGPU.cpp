//===- LinalgToXeGPU.cpp -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/Dialect/XeGPU/IR/XeGPUOps.h"
#include "TPP/IR/MatcherUtils.h"
#include "TPP/IR/StructuredOpMatcher.h"
#include "TPP/Transforms/Utils/ValueUtils.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/TypeSwitch.h"

#include <numeric>
#include <optional>

using namespace mlir;
using namespace mlir::tpp;
using namespace imex;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LINALGTOXEGPU
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

struct VnniConfig {
  int vnniFactor;
  int vnniAxis;
};

struct TilesArray {
  TilesArray() = delete;
  TilesArray(int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
      tileMatrix.push_back(SmallVector<Value>{});
      for (int j = 0; j < numCols; j++)
        tileMatrix[i].push_back(Value{});
    }
  }
  ~TilesArray() = default;

  Value getTile(int row, int col) { return tileMatrix[row][col]; }

  void setTile(int row, int col, Value val) { tileMatrix[row][col] = val; }

  SmallVector<SmallVector<Value, 8>, 8> tileMatrix;
};

// Return DPAS tile sizes if the gemm-like operation fits DPAS hardware.
static std::optional<SmallVector<int64_t>>
getDPASConfig(linalg::LinalgOp linalgOp, int kTile) {
  if (!(isa<linalg::MatmulOp>(linalgOp) ||
        isa<linalg::BatchReduceMatmulOp>(linalgOp) ||
        isa<linalg::GenericOp>(linalgOp))) {
    return std::nullopt;
  }

  // Only static shapes are supported.
  if (linalgOp.hasDynamicShape())
    return std::nullopt;

  auto aType = linalgOp.getDpsInputs()[0].getType().cast<ShapedType>();
  auto bType = linalgOp.getDpsInputs()[1].getType().cast<ShapedType>();
  auto cType = linalgOp.getDpsInits()[0].getType().cast<ShapedType>();

  auto elemTypeA = aType.getElementType();
  auto elemTypeB = bType.getElementType();
  auto elemTypeC = cType.getElementType();

  // TODO: Add more DPAS combinations.
  bool isSupportedPrecision =
      (elemTypeA.isF16() && elemTypeB.isF16() && elemTypeC.isF16()) ||
      (elemTypeA.isF16() && elemTypeB.isF16() && elemTypeC.isF32());
  if (!isSupportedPrecision)
    return std::nullopt;

  auto mDim = cType.getShape()[0];
  auto nDim = cType.getShape()[1];
  auto kDim = aType.getShape().back();

  // DPAS hardware sizes in MxNxK format.
  // TODO: In case more hardware configurations are available,
  //       add some automatic selection for optimal sizes.
  SmallVector<int64_t> dpasTile{8, 16, 16};

  // Validate workload sizes.
  // The computation dimensions must fit into the tiles.
  // Reduction dimension tile size has to be compatible
  // with the warp tile.
  int dpasTileM = dpasTile[0];
  int dpasTileN = dpasTile[1];
  int dpasTileK = dpasTile[2];
  if ((mDim % dpasTileM != 0) || (nDim % dpasTileN != 0) ||
      (kDim % dpasTileK != 0) || (kTile % dpasTileK != 0)) {
    return std::nullopt;
  }

  return dpasTile;
}

// Verify if linalg operands fulfill XeGPU constraints.
LogicalResult isValidMemrefOperand(linalg::LinalgOp linalgOp, Value operand,
                                   PatternRewriter &rewriter,
                                   unsigned maxDims = 2) {
  auto type = dyn_cast<MemRefType>(operand.getType());
  if (!type) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect memref operand for XeGPU lowering");
  }

  if (type.getShape().size() > maxDims) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Too high dimensionality for XeGPU operations");
  }

  auto strides = utils::getStaticStrides(operand);

  if (failed(strides)) {
    return rewriter.notifyMatchFailure(
        linalgOp, "Expect static strides for XeGPU lowering");
  }
  if (strides->back() != 1) {
    return rewriter.notifyMatchFailure(linalgOp,
                                       "Expect unit stride in the innermost "
                                       "dimension for XeGPU operations");
  }

  return success();
}

// Match and, if possible, lower a generic operation to an XeGPU compatible op.
// Returns the result of the lowered op or nullopt, otherwise.
static std::optional<Value> lowerGenericOp(linalg::GenericOp genericOp,
                                           ArrayRef<Value> operands,
                                           VectorType resType,
                                           PatternRewriter &rewriter) {
  Location loc = genericOp.getLoc();

  // Expect operands to be already loaded vectors.
  for (auto operand : operands) {
    if (!isa<VectorType>(operand.getType()))
      return std::nullopt;
  }

  if (structured_match::utils::isTwoDReluOp(genericOp, /*operands=*/nullptr)) {
    assert(operands.size() == 1 &&
           "Invalid number of operands for generic 2D ReLU");

    auto eltType = resType.getElementType();
    Value zeroConst;

    if (isa<FloatType>(eltType)) {
      auto floatType = cast<FloatType>(eltType);
      zeroConst = rewriter.create<arith::ConstantFloatOp>(
          loc, APFloat::getZero(floatType.getFloatSemantics()), floatType);
    } else if (isa<IntegerType>(eltType)) {
      zeroConst = rewriter.create<arith::ConstantIntOp>(loc, 0, eltType);
    } else {
      // Unhandled type. Bail out.
      return std::nullopt;
    }

    auto zeroVec =
        rewriter.create<vector::BroadcastOp>(loc, resType, zeroConst);

    return rewriter
        .create<arith::MaximumFOp>(loc, resType, operands[0], zeroVec)
        .getResult();
  }

  if (structured_match::utils::isTwoDAddOp(genericOp, /*operands=*/nullptr)) {
    assert(operands.size() == 2 &&
           "Invalid number of operands for generic 2D add");
    return rewriter
        .create<arith::AddFOp>(loc, resType, operands[0], operands[1])
        .getResult();
  }

  return std::nullopt;
}

// Lower an elementwise operation to an XeGPU compatible op.
// Returns the result of the lowered op or nullopt, otherwise.
static std::optional<Value> lowerEltwiseOp(linalg::LinalgOp linalgOp,
                                           ArrayRef<Value> operands,
                                           VectorType resType,
                                           PatternRewriter &rewriter) {
  Location loc = linalgOp.getLoc();

  // Expect operands to be already loaded vectors.
  for (auto operand : operands) {
    if (!isa<VectorType>(operand.getType()))
      return std::nullopt;
  }

  auto eltType = resType.getElementType();

  return llvm::TypeSwitch<Operation *, std::optional<Value>>(linalgOp)
      .Case([&](linalg::AbsOp absOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for abs");
        if (isa<FloatType>(eltType)) {
          return rewriter.create<math::AbsFOp>(loc, resType, operands[0])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter.create<math::AbsIOp>(loc, resType, operands[0])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::AddOp addOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for add");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::AddFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::AddIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::CeilOp ceilOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for ceil");
        return rewriter.create<math::CeilOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::DivOp divOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for div");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::DivFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::DivSIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::DivUnsignedOp divUnsignedOp) -> std::optional<Value> {
        assert(operands.size() == 2 &&
               "Invalid number of operands for unsigned div");
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::DivUIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::ExpOp expOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for exp");
        return rewriter.create<math::ExpOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::FloorOp floorOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for floor");
        return rewriter.create<math::FloorOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::MaxOp maxOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for max");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::MaximumFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          if (eltType.isUnsignedInteger()) {
            return rewriter
                .create<arith::MaxUIOp>(loc, resType, operands[0], operands[1])
                .getResult();
          } else {
            return rewriter
                .create<arith::MaxSIOp>(loc, resType, operands[0], operands[1])
                .getResult();
          }
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::MulOp mulOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for mul");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::MulFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::MulIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::NegfOp negfOp) -> std::optional<Value> {
        assert(operands.size() == 1 && "Invalid number of operands for negf");
        return rewriter.create<arith::NegFOp>(loc, resType, operands[0])
            .getResult();
      })
      .Case([&](linalg::SubOp subOp) -> std::optional<Value> {
        assert(operands.size() == 2 && "Invalid number of operands for sub");
        if (isa<FloatType>(eltType)) {
          return rewriter
              .create<arith::SubFOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        if (isa<IntegerType>(eltType)) {
          return rewriter
              .create<arith::SubIOp>(loc, resType, operands[0], operands[1])
              .getResult();
        }
        // Unhandled type. Bail out.
        return std::nullopt;
      })
      .Case([&](linalg::GenericOp genericOp) -> std::optional<Value> {
        return lowerGenericOp(genericOp, operands, resType, rewriter);
      })
      .Default(
          [&](Operation *op) -> std::optional<Value> { return std::nullopt; });
}

// Fuse an elementwise consumer operation.
// Returns updated store ops or nullopt if the fusion fails.
static std::optional<SmallVector<xegpu::StoreNDOp>>
eltwiseFusion(linalg::LinalgOp rootOp, linalg::LinalgOp consumerOp,
              SmallVector<xegpu::StoreNDOp> rootStoreOps,
              PatternRewriter &rewriter) {
  assert(rootStoreOps.size() > 0 && "Requires at least one store op");

  Location loc = rootOp.getLoc();
  auto ctx = rootOp.getContext();

  auto rootOutput = rootOp.getDpsInits()[0];

  // Gather additional operands of the fused consumer.
  // This excludes the root's output which values are already loaded into
  // registers and accessible through the store ops.
  SmallVector<Value> extraOperands;
  for (auto operand : consumerOp.getDpsInputOperands()) {
    if (operand->get() != rootOutput)
      extraOperands.push_back(operand->get());
  }

  // Insert fused eltwise ops before the stores and later replace the stores
  // with a new results.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(rootStoreOps[0]);

  // Collect new results after fusion.
  SmallVector<Value> fusedRes;
  auto readCacheHint =
      xegpu::CacheReadHintAttr::get(ctx, xegpu::CacheReadHint::CACHED);

  // For each store op, take a corresponding slice from the consumer operands
  // and load them into registers.
  for (auto storeOp : rootStoreOps) {
    auto storedVal = storeOp.getValue();
    auto storeDesc = storeOp.getTensorDesc();
    auto descOp = cast<xegpu::CreateNdDescOp>(storeDesc.getDefiningOp());

    // Create descriptors for the extra operands.
    SmallVector<Value> tensorDescs;
    for (auto operand : extraOperands) {
      auto tensorDesc = rewriter
                            .create<xegpu::CreateNdDescOp>(
                                loc, storeDesc.getType(), operand,
                                descOp.getOffsets(), descOp.getShape(),
                                descOp.getStrides(), descOp.getStaticOffsets(),
                                descOp.getBoundaryCheck(), descOp.getMode())
                            .getResult();
      tensorDescs.push_back(tensorDesc);
    }

    // Operands for the consumer op.
    // This always includes the previous result held by the store op.
    // Load values of the extra operands into registers.
    SmallVector<Value> operands{storedVal};
    for (auto tensorDesc : tensorDescs) {
      auto loadedVec = rewriter.create<xegpu::LoadNDOp>(
          loc, storedVal.getType(), tensorDesc, /*vnni_axis=*/nullptr,
          /*transpose=*/nullptr,
          /*l1_hint=*/readCacheHint,
          /*l2_hint=*/readCacheHint, /*l3_hint=*/readCacheHint,
          storeOp.getMode());
      operands.push_back(loadedVec);
    }

    // Lower to a vectorized eltwise op.
    auto newRes = lowerEltwiseOp(
        consumerOp, operands, cast<VectorType>(storedVal.getType()), rewriter);
    if (!newRes)
      return std::nullopt;

    fusedRes.push_back(*newRes);
  }

  // Fusion must have failed, if number of new results is different.
  // Bail out.
  if (fusedRes.size() != rootStoreOps.size())
    return std::nullopt;

  // Store the new result.
  auto writeCacheHint =
      xegpu::CacheWriteHintAttr::get(ctx, xegpu::CacheWriteHint::WRITE_BACK);
  SmallVector<xegpu::StoreNDOp> newStores;

  for (size_t i = 0; i < rootStoreOps.size(); i++) {
    auto storeDesc = rootStoreOps[i].getTensorDesc();

    auto newStore = rewriter.create<xegpu::StoreNDOp>(
        loc, storeDesc, fusedRes[i],
        /*l1_hint=*/writeCacheHint,
        /*l2_hint=*/writeCacheHint,
        /*l3_hint=*/writeCacheHint, rootStoreOps[i].getMode());
    newStores.push_back(newStore);
  }

  // Replace store ops and cleanup standalone consumer.
  for (size_t i = 0; i < rootStoreOps.size(); i++)
    rewriter.replaceOp(rootStoreOps[i], newStores[i]);

  rewriter.eraseOp(consumerOp);

  return newStores;
}

// Find operations fusable with the given root op.
//
// A simple fusion strategy that looks at the other operations after the root
// linalg op and tries to fuse them.
static SmallVector<linalg::LinalgOp>
getFusableConsumers(linalg::LinalgOp rootOp) {
  auto *parentOp = rootOp->getParentOp();
  auto rootOutput = rootOp.getDpsInits()[0];

  // Traverse other ops within the same region and collect consumers.
  SmallVector<linalg::LinalgOp> consumers;
  Operation *nextOp = rootOp;
  while ((nextOp = nextOp->getNextNode())) {
    // Potential consumers must be within the same region.
    if (nextOp->getParentOp() != parentOp)
      break;

    // Only other linalg ops are expected as consumers.
    // TODO: might need to be relaxed to skip over ops without side effects
    auto consumer = dyn_cast<linalg::LinalgOp>(nextOp);
    if (!consumer || !linalg::isElementwise(consumer))
      break;
    // Require the same iteration space.
    if (consumer.getNumParallelLoops() != rootOp.getNumParallelLoops())
      break;

    auto outBuf = consumer.getDpsInitOperand(0)->get();
    // Check that the op reuses the same output buffer as the root op.
    // Otherwise, it is assumed that the op cannot be fused.
    // TODO: Consider adding support for eltwise with broadcast.
    if (outBuf != rootOutput)
      break;

    consumers.push_back(consumer);
  }

  return consumers;
}

// Fuse elementwise consumers within a GPU kernel.
//
// Fusion bails on the first mismatch.
// Returns updated store ops.
static SmallVector<xegpu::StoreNDOp>
fuseEltwiseConsumers(linalg::LinalgOp rootOp,
                     SmallVector<xegpu::StoreNDOp> rootStoreOps,
                     PatternRewriter &rewriter) {
  auto consumers = getFusableConsumers(rootOp);

  for (auto consumer : consumers) {
    std::optional<SmallVector<xegpu::StoreNDOp>> updatedStoreOps = std::nullopt;

    updatedStoreOps = eltwiseFusion(rootOp, consumer, rootStoreOps, rewriter);

    // Failed to fuse operation. Bail out.
    if (!updatedStoreOps)
      break;

    rootStoreOps = *updatedStoreOps;
  }

  return rootStoreOps;
}

// Get static GPU block sizes represented by a surrounding operation
// like a kernel launch or parallel loop.
// Returns known block sizes if they are all static or failure, otherwise.
static FailureOr<SmallVector<int64_t>> getStaticBlockSizes(Operation *op) {
  if (!op)
    return failure();

  auto getConstVal = [&](Value val) -> std::optional<int64_t> {
    if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>()) {
      return constOp.value();
    }
    return std::nullopt;
  };

  if (auto launchOp = dyn_cast<gpu::LaunchOp>(op)) {
    auto sizeX = getConstVal(launchOp.getBlockSizeX());
    auto sizeY = getConstVal(launchOp.getBlockSizeY());
    auto sizeZ = getConstVal(launchOp.getBlockSizeZ());
    if (!sizeX || !sizeY || !sizeZ)
      return failure();

    return SmallVector<int64_t>{*sizeX, *sizeY, *sizeZ};
  }

  // TODO: Remove when the lowering only occurs within a gpu.launch op.
  //       Manually computing this is brittle and duplicated parallel
  //       loops to gpu conversion.
  if (auto blockLoop = dyn_cast<scf::ParallelOp>(op)) {
    auto gridLoop = blockLoop->getParentOfType<scf::ParallelOp>();

    // Blocks or number of threads are represented by the first parallel loop
    // nested within another parallel loop.
    //
    // Fail if there is no outer parallel loop or current loop is nested more
    // than once.
    if (!gridLoop || (gridLoop->getParentOfType<scf::ParallelOp>())) {
      return failure();
    }

    SmallVector<int64_t> blockSizes;
    for (auto [lb, ub, step] :
         llvm::zip_equal(blockLoop.getLowerBound(), blockLoop.getUpperBound(),
                         blockLoop.getStep())) {
      auto lbVal = getConstVal(lb);
      auto ubVal = getConstVal(ub);
      auto stepVal = getConstVal(step);
      if (!lbVal || !ubVal || !stepVal)
        return failure();

      int64_t blockSize = (*ubVal - *lbVal) / *stepVal;

      // Assume that at least one thread/workitem is created for the given
      // dimension. Otherwise, outlining will fail anyway.
      blockSizes.push_back(blockSize < 0 ? 1 : blockSize);
    }

    // Too many dimensions, something went wrong. Bail out.
    if (blockSizes.size() > 3)
      return failure();

    return blockSizes;
  }

  return failure();
}

static Value getGpuLinearThreadId(PatternRewriter &rewriter, Location loc) {
  SmallVector<Value, 3> threadIds;
  SmallVector<Value, 3> blockDims;

  for (auto dim : {gpu::Dimension::x, gpu::Dimension::y, gpu::Dimension::z}) {
    threadIds.push_back(rewriter.create<gpu::ThreadIdOp>(loc, dim));
    blockDims.push_back(rewriter.create<gpu::BlockDimOp>(loc, dim));
  }

  // The default GPU indexing is modeled after CUDA:
  // linear index = (z * sizeY + y) * sizeX + x
  Value threadId =
      rewriter.create<arith::MulIOp>(loc, threadIds[2], blockDims[1]);
  threadId = rewriter.create<arith::AddIOp>(loc, threadId, threadIds[1]);
  threadId = rewriter.create<arith::MulIOp>(loc, threadId, blockDims[0]);
  threadId = rewriter.create<arith::AddIOp>(loc, threadId, threadIds[0]);

  return threadId;
}

// Create a GEMM input tile to be loaded by each thread/workitem in
// cooperative fashion.
// Optionally accepts batch IV for batched GEMM input loading.
// Returns failure if it is unable to split block/workgroup for
// prefetching.
static FailureOr<xegpu::CreateNdDescOp>
createGemmCoopPrefetchTile(PatternRewriter &rewriter, linalg::LinalgOp linalgOp,
                           unsigned inputPos, int64_t numThreads,
                           ArrayRef<int> blockTile, ArrayRef<int> threadTile,
                           int tileStep, xegpu::Mode mode) {
  assert(inputPos <= 1 && "Can handle only GEMM inputs: mat A or mat B");
  Location loc = linalgOp.getLoc();

  Value src = linalgOp.getDpsInputs()[inputPos];

  // Get a top level view into the whole matrix not only the thread slice.
  if (auto subview = dyn_cast_or_null<memref::SubViewOp>(src.getDefiningOp())) {
    src = subview.getSource();
  }

  const int tileRows = inputPos == 0 ? blockTile[0] : tileStep;
  const int tileCols = inputPos == 0 ? tileStep : blockTile[1];

  const int numElements = tileRows * tileCols;
  const int elementsPerThread = numElements / numThreads;

  // Limit the maximum prefetching row length to avoid very wide tiles.
  //
  // Currently, the max row size is capped by the hardware max load width.
  //
  // TODO: Expose as a tunable parameter or add some heuristics.
  const int maxRowLength = 32;

  // Prioritize first loading contiguous elements (row lenght/number of
  // columns) only then gather any remaining elements to be loaded from
  // further rows.
  // Also, ensure that the prefetch tile stays within the tile bounds.
  //
  // Ideally, prefetch tile sizes should be derived from total number of
  // elements to be loaded, number of threads/workitems, and hardware load
  // size limits. Large prefetch tiles might need to be split into sub-tiles.
  const int numCols =
      std::min(std::min(elementsPerThread, tileCols), maxRowLength);
  const int numRows = elementsPerThread / numCols;

  // Bail on invalid prefetching tiles config.
  if (numRows == 0 ||
      ((numRows * numCols * numThreads) > (tileRows * tileCols)))
    return failure();

  auto srcType = src.getType().cast<ShapedType>();

  auto prefetchType =
      xegpu::TensorDescType::get({numRows, numCols}, srcType.getElementType());

  Value threadId = getGpuLinearThreadId(rewriter, loc);

  // TODO: Simplify block offsets.
  //       Prefetching tile should be derived from the matmul op operands and
  //       exposed as a subview.
  //
  // Add offset if there are multiple blocks in the current tile's non-reduction
  // dimension.
  Value blockOffset = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  if (blockTile[inputPos] / threadTile[inputPos] > 1) {
    Value blockSize =
        rewriter.create<arith::ConstantIndexOp>(loc, blockTile[inputPos]);

    // For matrix B, pick correct block dimension.
    // Block min X has to be used if there is no thread tiling in the rows
    // (dim X) but only in columns (dim Y).
    gpu::Dimension gpuDim = gpu::Dimension::x;
    if ((inputPos == 1) && (blockTile[0] / threadTile[0] > 1)) {
      gpuDim = gpu::Dimension::y;
    }
    Value blockId = rewriter.create<gpu::BlockIdOp>(loc, gpuDim);

    blockOffset = rewriter.create<arith::MulIOp>(loc, blockId, blockSize);
  }

  Value numColTiles =
      rewriter.create<arith::ConstantIndexOp>(loc, tileStep / numCols);
  if (inputPos == 1) {
    numColTiles =
        rewriter.create<arith::ConstantIndexOp>(loc, blockTile[1] / numCols);
  }
  Value tileRowOffset =
      rewriter.create<arith::DivUIOp>(loc, threadId, numColTiles);
  Value tileColOffset =
      rewriter.create<arith::RemUIOp>(loc, threadId, numColTiles);

  Value tileRowSize = rewriter.create<arith::ConstantIndexOp>(loc, numRows);
  Value tileColSize = rewriter.create<arith::ConstantIndexOp>(loc, numCols);
  Value eltRowOffset =
      rewriter.create<arith::MulIOp>(loc, tileRowOffset, tileRowSize);
  Value eltColOffset =
      rewriter.create<arith::MulIOp>(loc, tileColOffset, tileColSize);

  if (inputPos == 0) {
    eltRowOffset =
        rewriter.create<arith::AddIOp>(loc, eltRowOffset, blockOffset);
  } else {
    eltColOffset =
        rewriter.create<arith::AddIOp>(loc, eltColOffset, blockOffset);
  }

  SmallVector<mlir::OpFoldResult> prefetchOffsets{eltRowOffset, eltColOffset};

  return rewriter.create<xegpu::CreateNdDescOp>(loc, prefetchType, src,
                                                prefetchOffsets,
                                                /*boundary_check=*/true, mode);
}

static void prefetchTiles(PatternRewriter &rewriter, Location loc,
                          ValueRange prefetchTiles,
                          xegpu::CacheReadHintAttr readCacheHint,
                          xegpu::Mode mode) {
  // Prefetch the next set of input tiles.
  for (auto tile : prefetchTiles) {
    rewriter.create<xegpu::PrefetchNDOp>(loc, tile,
                                         /*l1_hint=*/readCacheHint,
                                         /*l2_hint=*/readCacheHint,
                                         /*l3_hint=*/readCacheHint, mode);
  }
}

static SmallVector<Value> updateTilesOffsets(PatternRewriter &rewriter,
                                             Location loc, ValueRange tiles,
                                             ValueRange offsets,
                                             xegpu::Mode mode) {
  SmallVector<Value> updatedTiles;
  for (auto tile : tiles) {
    auto updatedTile = rewriter
                           .create<xegpu::UpdateNDOffsetOp>(loc, tile.getType(),
                                                            tile, offsets, mode)
                           .getResult();
    updatedTiles.push_back(updatedTile);
  }

  return updatedTiles;
}

// Split a source into a series of descriptor tiles.
// The descriptors collectively load a 2D shape at the specified offsets from
// the given source.
// The offsets and the load shape must stay within the source boundaries.
//
// The descriptor sub-tiles are ordered in row-major fashion with respect to the
// whole load tile.
static SmallVector<Value>
createDescriptorTiles(PatternRewriter &rewriter, Location loc, Value src,
                      ArrayRef<int64_t> loadShape,
                      ArrayRef<int64_t> loadOffsets, ArrayRef<int64_t> descTile,
                      xegpu::Mode mode, int arrayLength = 1) {
  assert(arrayLength == 1 && "Array descriptors are not supported");

  auto type = src.getType().cast<ShapedType>();
  auto descType = xegpu::TensorDescType::get(descTile, type.getElementType());

  // Create the root descriptor.
  //
  // It is more efficient to create remainig descriptors by only updating its
  // offsets compared to creating separate descriptors.
  // The original tile is split into contiguous sub-tiles so, the first tile
  // can be used as an anchor.
  Value rootOffsetRow =
      rewriter.create<arith::ConstantIndexOp>(loc, loadOffsets[0]);
  Value rootOffsetCol =
      rewriter.create<arith::ConstantIndexOp>(loc, loadOffsets[1]);

  mlir::SmallVector<mlir::OpFoldResult> offsets{rootOffsetRow, rootOffsetCol};
  auto rootTile =
      rewriter
          .create<xegpu::CreateNdDescOp>(loc, descType, src, offsets,
                                         /*boundary_check=*/true, mode)
          .getResult();

  SmallVector<Value> tiles;
  for (int i = 0; i < loadShape[0]; i += descTile[0]) {
    for (int j = 0; j < loadShape[1]; j += descTile[1] * arrayLength) {
      Value rowOffset = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value colOffset = rewriter.create<arith::ConstantIndexOp>(loc, j);

      auto tile = rewriter
                      .create<xegpu::UpdateNDOffsetOp>(
                          loc, descType, rootTile,
                          ValueRange{rowOffset, colOffset}, mode)
                      .getResult();
      tiles.push_back(tile);
    }
  }

  return tiles;
}

// Create a GEMM sub-tiles to be loaded by the current thread/workitem.
//
// The shape to be loaded is split into largest 2D loads supported
// by the hardware.
//
// The load tiles are ordered in row-major fashion with respect to the
// thread/workitem tile.
static SmallVector<Value> createCoarseLoadTiles(PatternRewriter &rewriter,
                                                Location loc, Value src,
                                                ArrayRef<int64_t> loadShape,
                                                xegpu::Mode mode, bool isVnni) {
  // TODO: Fetch actual list of supported load configs
  //       as they also depend on data type width.
  //       Here only 16 bit wide elements are assumed.
  const int64_t maxHeight = 32;
  const int64_t maxWidth = isVnni ? 16 : 32;
  const int64_t maxArrayLength = 4;
  int64_t loadRows = std::min(loadShape[0], maxHeight);
  int64_t loadCols = std::min(loadShape[1], maxWidth);
  int64_t arrayLength = std::min(maxWidth / loadCols, maxArrayLength);
  // In case of partial fit, load only single tile.
  if (maxWidth % loadCols != 0 || arrayLength != 4 || arrayLength != 2)
    arrayLength = 1;

  // TODO: Bump IMEX version to bring array_length attr support.
  arrayLength = 1;

  return createDescriptorTiles(rewriter, loc, src, loadShape, {0, 0},
                               {loadRows, loadCols}, mode, arrayLength);
}

static VectorType getVnniVector(ArrayRef<int64_t> shape, Type elementType,
                                VnniConfig vnniConf) {
  SmallVector<int64_t> vecShape{shape};
  vecShape[vnniConf.vnniAxis] /= vnniConf.vnniFactor;
  vecShape.push_back(vnniConf.vnniFactor);
  return VectorType::get(vecShape, elementType);
}

static SmallVector<Value>
loadGemmTiles(PatternRewriter &rewriter, Location loc, ValueRange loadTiles,
              xegpu::Mode mode, xegpu::CacheReadHintAttr hint,
              std::optional<VnniConfig> vnniConf = std::nullopt,
              DenseI64ArrayAttr transpose = nullptr) {
  // Assume all tiles have the same shape.
  xegpu::TensorDescType tileType =
      loadTiles[0].getType().cast<xegpu::TensorDescType>();
  assert(llvm::all_of(loadTiles,
                      [&](Value tile) { return tile.getType() == tileType; }) &&
         "All load tiles must have the same type.");

  VectorType vecLoadType =
      VectorType::get(tileType.getShape(), tileType.getElementType());
  IntegerAttr vnniAxisAttr = nullptr;
  if (vnniConf) {
    vnniAxisAttr = IntegerAttr::get(rewriter.getI32Type(), vnniConf->vnniAxis);
    vecLoadType = getVnniVector(tileType.getShape(), tileType.getElementType(),
                                *vnniConf);
  }

  SmallVector<Value> loadVec;
  for (auto tile : loadTiles) {
    auto loadOp = rewriter.create<xegpu::LoadNDOp>(
        loc, vecLoadType, tile, vnniAxisAttr, transpose,
        /*l1_hint=*/hint,
        /*l2_hint=*/hint, /*l3_hint=*/hint, mode);
    loadVec.push_back(loadOp);
  }
  // TODO: Add split over the array_length > 1.
  //       The split must preserve row-major ordering of the load tiles.

  return loadVec;
}

static TilesArray
extractVecDpasTiles(PatternRewriter &rewriter, Location loc,
                    ValueRange loadVecTiles, ArrayRef<int64_t> loadTile,
                    ArrayRef<int64_t> loadSubTile, ArrayRef<int64_t> dpasTile,
                    std::optional<VnniConfig> vnniConf = std::nullopt) {
  auto vecLoadType = loadVecTiles[0].getType().cast<VectorType>();
  assert(llvm::all_of(loadVecTiles,
                      [&](Value tile) {
                        return tile.getType().cast<VectorType>() == vecLoadType;
                      }) &&
         "All loaded vectors must have the same type.");

  // Accumulate all dimensions as the vector might have extra VNNI dimensions.
  int loadVecSize = std::accumulate(vecLoadType.getShape().begin(),
                                    vecLoadType.getShape().end(), 1,
                                    std::multiplies<int64_t>());
  auto loadVecFlat = VectorType::get(loadVecSize, vecLoadType.getElementType());

  VectorType vecDpasType =
      VectorType::get(dpasTile, vecLoadType.getElementType());
  if (vnniConf) {
    vecDpasType =
        getVnniVector(dpasTile, vecLoadType.getElementType(), *vnniConf);
  }

  const int numLoadTilesRows = loadTile[0] / loadSubTile[0];
  const int numLoadTilesCol = loadTile[1] / loadSubTile[1];

  const int numDpasPerLoadRow = loadSubTile[0] / dpasTile[0];
  const int numDpasPerLoadCol = loadSubTile[1] / dpasTile[1];

  const int numDpasTileRows = loadTile[0] / dpasTile[0];
  const int numDpasTileCols = loadTile[1] / dpasTile[1];
  TilesArray dpasTiles(numDpasTileRows, numDpasTileCols);

  // Iterate over load tile.
  // Each load tile contains one or more DPAS tiles.
  for (int m = 0; m < numLoadTilesRows; m++) {
    for (int k = 0; k < numLoadTilesCol; k++) {
      // Load tiles are ordered in row-major fashion.
      int loadIdx = m * numLoadTilesCol + k;
      auto loadTile = loadVecTiles[loadIdx];
      auto castFlat =
          rewriter.create<vector::ShapeCastOp>(loc, loadVecFlat, loadTile);

      // Iterate over DPAS tiles.
      for (int i = 0; i < numDpasPerLoadRow; i++) {
        for (int j = 0; j < numDpasPerLoadCol; j++) {
          const int dpasTileSize = dpasTile[0] * dpasTile[1];
          int dpasIdx = i * numDpasPerLoadCol + j;
          int offset = dpasIdx * dpasTileSize;

          auto slice = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, castFlat, /*offsets=*/ArrayRef<int64_t>{offset},
              /*sizes=*/ArrayRef<int64_t>{dpasTileSize},
              /*strides=*/ArrayRef<int64_t>{1});
          auto castTile =
              rewriter.create<vector::ShapeCastOp>(loc, vecDpasType, slice);

          // Insert the sub-tiles in their position relative to the whole
          // thread/workitem tile.
          int rowIdx = m * numDpasPerLoadRow + i;
          int colIdx = k * numDpasPerLoadCol + j;
          dpasTiles.setTile(rowIdx, colIdx, castTile);
        }
      }
    }
  }

  return dpasTiles;
}

// Create XeGPU DPAS kernel out of GEMM-like operation.
static LogicalResult createDPASKernel(linalg::LinalgOp linalgOp,
                                      ArrayRef<int64_t> dpasTile, int kTile,
                                      int prefetchStages,
                                      PatternRewriter &rewriter) {
  assert((isa<linalg::MatmulOp>(linalgOp) ||
          isa<linalg::BatchReduceMatmulOp>(linalgOp) ||
          isa<linalg::GenericOp>(linalgOp)) &&
         "Requires a GEMM-like op for DPAS lowering");

  Location loc = linalgOp.getLoc();
  auto ctx = linalgOp.getContext();

  auto matA = linalgOp.getDpsInputs()[0];
  auto matB = linalgOp.getDpsInputs()[1];
  auto matC = linalgOp.getDpsInits()[0];

  auto typeA = matA.getType().cast<ShapedType>();
  auto typeC = matC.getType().cast<ShapedType>();

  int64_t dpasTileM = dpasTile[0];
  int64_t dpasTileN = dpasTile[1];
  int64_t dpasTileK = dpasTile[2];

  // Instruction mode - use workgroup intristic directly.
  auto xegpuMode = xegpu::Mode::VC;

  // Cache hints for loads and stores.
  auto readCacheHint =
      xegpu::CacheReadHintAttr::get(ctx, xegpu::CacheReadHint::CACHED);
  auto writeCacheHint =
      xegpu::CacheWriteHintAttr::get(ctx, xegpu::CacheWriteHint::WRITE_BACK);

  bool isBrgemm = isa<linalg::BatchReduceMatmulOp>(linalgOp);

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  int dimM = typeC.getShape()[0];
  int dimN = typeC.getShape()[1];
  int dimK = typeA.getShape().back();

  // Create C sub-tiles.
  auto dpasTypeC = xegpu::TensorDescType::get({dpasTileM, dpasTileN},
                                              typeC.getElementType());
  SmallVector<Value> tilesC =
      createDescriptorTiles(rewriter, loc, matC, typeC.getShape(), {0, 0},
                            dpasTypeC.getShape(), xegpuMode);

  // Load C sub-tiles.
  // Fetch the inital values of the output accumulator.
  SmallVector<Value> loadVecC =
      loadGemmTiles(rewriter, loc, tilesC, xegpuMode, readCacheHint);
  rewriter.create<xegpu::CompileHintOp>(loc);

  // DPAS only works with F32 accumulators.
  auto dpasResType =
      VectorType::get(dpasTypeC.getShape(), FloatType::getF32(ctx));

  // Extend the accumulation values if needed.
  auto isOutF16 = typeC.getElementType().isF16();
  if (isOutF16) {
    for (size_t i = 0; i < loadVecC.size(); i++) {
      auto extOp =
          rewriter.create<arith::ExtFOp>(loc, dpasResType, loadVecC[i]);
      loadVecC[i] = extOp.getOut();
    }
  }

  // Create a loop and step into it.
  auto startLoop = [&](int lb, int ub, int step,
                       ValueRange iterArgs) -> scf::ForOp {
    Value lbCst = rewriter.create<arith::ConstantIndexOp>(loc, lb);
    Value ubCst = rewriter.create<arith::ConstantIndexOp>(loc, ub);
    Value stepCst = rewriter.create<arith::ConstantIndexOp>(loc, step);
    scf::ForOp loopOp =
        rewriter.create<scf::ForOp>(loc, lbCst, ubCst, stepCst, iterArgs);
    rewriter.setInsertionPointToStart(loopOp.getBody());
    return loopOp;
  };
  auto getLoopIterValues = [&](scf::ForOp loopOp) -> SmallVector<Value> {
    SmallVector<Value> loopIterVals;
    for (auto iterArg : loopOp.getRegionIterArgs())
      loopIterVals.push_back(iterArg);
    return loopIterVals;
  };

  OpBuilder::InsertionGuard guard(rewriter);

  // Construct and move into batch reduction loop.
  // Propagate output values as iter args.
  scf::ForOp batchLoop;
  Value batchIv;
  if (isBrgemm) {
    batchLoop = startLoop(0, typeA.getShape()[0], 1, loadVecC);
    batchIv = batchLoop.getInductionVar();
    loadVecC = getLoopIterValues(batchLoop);
    // TODO: Replace input matrices A and B with subviews on the current
    //       batchIV as loads can only be performed on 2D memrefs.
  }

  // Create A sub-tiles.
  SmallVector<Value> tilesA = createCoarseLoadTiles(
      rewriter, loc, matA, {dimM, kTile}, xegpuMode, /*isVnni=*/true);

  // Create B sub-tiles.
  SmallVector<Value> tilesB = createCoarseLoadTiles(
      rewriter, loc, matB, {kTile, dimN}, xegpuMode, /*isVnni=*/true);

  // Create input prefetch tiles.
  int64_t numThreads = 1;
  auto blockDims =
      getStaticBlockSizes(linalgOp->getParentOfType<scf::ParallelOp>());
  if (succeeded(blockDims)) {
    numThreads = std::accumulate(blockDims->begin(), blockDims->end(), 1,
                                 std::multiplies<int64_t>());
  }
  // Disable prefetching when there is no block/workgroup parallelism.
  bool isCoopPrefetch = numThreads > 1;

  Value prefetchA;
  Value prefetchB;
  xegpu::TensorDescType prefetchTypeA;
  xegpu::TensorDescType prefetchTypeB;
  Value kTileOffset = rewriter.create<arith::ConstantIndexOp>(loc, kTile);
  if (isCoopPrefetch) {
    // Return dimension size on which the whole block/workgroup operates.
    auto getBlockLevelSize = [&](Value val, int dim) -> int {
      if (auto subview =
              dyn_cast_or_null<memref::SubViewOp>(val.getDefiningOp())) {
        val = subview.getSource();
      }

      return cast<ShapedType>(val.getType()).getShape()[dim];
    };

    int blockRows = getBlockLevelSize(matC, 0);
    int blockCols = getBlockLevelSize(matC, 1);

    auto prefetchDescA = createGemmCoopPrefetchTile(
        rewriter, linalgOp, /*inputPos=*/0, numThreads, {blockRows, blockCols},
        {dimM, dimN}, kTile, xegpuMode);
    auto prefetchDescB = createGemmCoopPrefetchTile(
        rewriter, linalgOp, /*inputPos=*/1, numThreads, {blockRows, blockCols},
        {dimM, dimN}, kTile, xegpuMode);

    if (succeeded(prefetchDescA) && succeeded(prefetchDescB)) {
      prefetchA = prefetchDescA->getResult();
      prefetchTypeA = prefetchDescA->getType();
      prefetchB = prefetchDescB->getResult();
      prefetchTypeB = prefetchDescB->getType();

      // Start data prefetching by multistage data load.
      for (int i = 0; i < prefetchStages; i++) {
        prefetchTiles(rewriter, loc, ValueRange{prefetchA}, readCacheHint,
                      xegpuMode);
        prefetchTiles(rewriter, loc, ValueRange{prefetchB}, readCacheHint,
                      xegpuMode);
        prefetchA =
            updateTilesOffsets(rewriter, loc, ValueRange{prefetchA},
                               ValueRange{zero, kTileOffset}, xegpuMode)[0];
        prefetchB =
            updateTilesOffsets(rewriter, loc, ValueRange{prefetchB},
                               ValueRange{kTileOffset, zero}, xegpuMode)[0];
      }

      // Ensure that block/workgroup is synchronized after prefetching.
      rewriter.create<xegpu::CompileHintOp>(loc);
    } else {
      // Disable coop prefetching on failure.
      isCoopPrefetch = false;
    }
  }

  // Construct and move into GEMM reduction dimension tiling loop.
  // Propagate output values as iter args.
  SmallVector<Value> iterArgs;
  iterArgs.append(loadVecC);
  iterArgs.append(tilesA);
  iterArgs.append(tilesB);
  if (isCoopPrefetch) {
    iterArgs.push_back(prefetchA);
    iterArgs.push_back(prefetchB);
  }
  scf::ForOp kDimLoop = startLoop(0, dimK, kTile, iterArgs);
  auto iterValues = getLoopIterValues(kDimLoop);

  loadVecC = SmallVector<Value>{iterValues.begin(),
                                iterValues.begin() + loadVecC.size()};
  tilesA =
      SmallVector<Value>{iterValues.begin() + loadVecC.size(),
                         iterValues.begin() + loadVecC.size() + tilesA.size()};
  tilesB = SmallVector<Value>{iterValues.begin() + loadVecC.size() + tilesA.size(),
                              iterValues.begin() + loadVecC.size() + tilesA.size() + tilesB.size()};
  if (isCoopPrefetch) {
    prefetchA = *(iterValues.end() - 2);
    prefetchB = *(iterValues.end() - 1);
  }

  // Periodically synchronize the block/workgroup to minimize impact on cache
  // due to replacement of sub-tiles before all threads/workitems consumed
  // inputs for reduction dimension step.
  //
  // TODO: Synchronization frequency should derived from tile and cache size.
  int syncFreq = 4;
  int maxSyncStep = 1024;
  int syncStep = std::min(std::max(dimK / syncFreq, maxSyncStep), maxSyncStep);
  auto syncStepConst = rewriter.create<arith::ConstantIndexOp>(loc, syncStep);
  auto loopStepMod = rewriter.create<arith::RemUIOp>(
      loc, kDimLoop.getInductionVar(), syncStepConst);
  auto syncBlockCond = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, loopStepMod, zero);
  rewriter.create<scf::IfOp>(
      loc, syncBlockCond,
      /*thenBuilder=*/
      [](OpBuilder &b, Location loc) {
        b.create<gpu::BarrierOp>(loc);
        b.create<scf::YieldOp>(loc);
      },
      /*elseBuilder=*/nullptr);

  // TODO: Make the VNNI factor a flexible parameter.
  const int vnniFactor = 2;
  VnniConfig vnniConfA{.vnniFactor = vnniFactor, .vnniAxis = 1};
  VnniConfig vnniConfB{.vnniFactor = vnniFactor, .vnniAxis = 0};

  // Load A sub-tiles.
  SmallVector<Value> loadVecA =
      loadGemmTiles(rewriter, loc, tilesA, xegpuMode, readCacheHint, vnniConfA);
  xegpu::TensorDescType tileTypeA =
      tilesA[0].getType().cast<xegpu::TensorDescType>();

  // Load B sub-tiles.
  SmallVector<Value> loadVecB =
      loadGemmTiles(rewriter, loc, tilesB, xegpuMode, readCacheHint, vnniConfB);
  xegpu::TensorDescType tileTypeB =
      tilesB[0].getType().cast<xegpu::TensorDescType>();

  // Ensure that all load are scheduled together.
  rewriter.create<xegpu::CompileHintOp>(loc);

  // Update offsets of the input tiles.
  // Shift along the reduction dimension.
  tilesA = updateTilesOffsets(rewriter, loc, tilesA,
                              ValueRange{zero, kTileOffset}, xegpuMode);
  tilesB = updateTilesOffsets(rewriter, loc, tilesB,
                              ValueRange{kTileOffset, zero}, xegpuMode);

  // Prefetch the next set of input tiles.
  if (isCoopPrefetch) {
    // Prefetch all block/workgroup tiles cooperatively.
    prefetchTiles(rewriter, loc, ValueRange{prefetchA}, readCacheHint,
                  xegpuMode);
    prefetchTiles(rewriter, loc, ValueRange{prefetchB}, readCacheHint,
                  xegpuMode);
    prefetchA = updateTilesOffsets(rewriter, loc, ValueRange{prefetchA},
                                   ValueRange{zero, kTileOffset}, xegpuMode)[0];
    prefetchB = updateTilesOffsets(rewriter, loc, ValueRange{prefetchB},
                                   ValueRange{kTileOffset, zero}, xegpuMode)[0];
  } else {
    // Apply naive prefetching for each thread/workitem.
    prefetchTiles(rewriter, loc, tilesA, readCacheHint, xegpuMode);
    prefetchTiles(rewriter, loc, tilesB, readCacheHint, xegpuMode);
  }

  // Ensure that prefetches are scheduled before computation starts.
  rewriter.create<xegpu::CompileHintOp>(loc);

  // Extract DPAS tiles from loaded sub-tiles.
  TilesArray dpasVecA = extractVecDpasTiles(rewriter, loc, loadVecA,
                                            {dimM, kTile}, tileTypeA.getShape(),
                                            {dpasTileM, dpasTileK}, vnniConfA);
  TilesArray dpasVecB = extractVecDpasTiles(rewriter, loc, loadVecB,
                                            {kTile, dimN}, tileTypeB.getShape(),
                                            {dpasTileK, dpasTileN}, vnniConfB);

  // Finalize vector reshaping before DPAS.
  rewriter.create<xegpu::CompileHintOp>(loc);

  const int numTilesM = dimM / dpasTileM;
  const int numTilesN = dimN / dpasTileN;
  const int numTilesK = kTile / dpasTileK;

  // Compute sub-tiles of the C tile.
  //
  // Iterate over the reduction dimension sub-tiles as the outermost
  // loop to minimize read after write conflicts between partial
  // computations of the same C sub-tile.
  SmallVector<Value> dpasResults = loadVecC;

  for (int k = 0; k < numTilesK; k++) {
    for (int m = 0; m < numTilesM; m++) {
      for (int n = 0; n < numTilesN; n++) {
        int cIdx = m * numTilesN + n;

        Value result =
            rewriter
                .create<xegpu::DpasOp>(loc, dpasResType, dpasVecA.getTile(m, k),
                                       dpasVecB.getTile(k, n),
                                       dpasResults[cIdx], xegpuMode)
                .getResult();

        // Update sub-tile partial result.
        dpasResults[cIdx] = result;
      }
    }
  }

  // Ensure that DPAS computation is finished before the input tiles are
  // replaced with new values.
  rewriter.create<xegpu::CompileHintOp>(loc);

  // Create loop terminator and exit the loop.
  auto terminateLoop = [&](scf::ForOp loopOp, SmallVector<Value> resultValues) {
    rewriter.setInsertionPointToEnd(loopOp.getBody());
    rewriter.create<scf::YieldOp>(loc, resultValues);
    rewriter.setInsertionPointAfter(loopOp);
  };

  SmallVector<Value> yieldVals;
  yieldVals.append(dpasResults);
  yieldVals.append(tilesA);
  yieldVals.append(tilesB);
  if (isCoopPrefetch) {
    yieldVals.push_back(prefetchA);
    yieldVals.push_back(prefetchB);
  }

  // Terminate and exit reduction dim loop.
  terminateLoop(kDimLoop, yieldVals);
  yieldVals = kDimLoop.getResults();

  SmallVector<Value> results{yieldVals.begin(),
                             yieldVals.begin() + dpasResults.size()};

  // Terminate and exit batch reduce loop.
  if (isBrgemm) {
    terminateLoop(batchLoop, results);
    results = batchLoop.getResults();
  }

  // Truncate the result values if needed.
  if (isOutF16) {
    auto truncType =
        VectorType::get(dpasTypeC.getShape(), FloatType::getF16(ctx));
    for (size_t i = 0; i < results.size(); i++) {
      auto truncOp =
          rewriter.create<arith::TruncFOp>(loc, truncType, results[i]);
      results[i] = truncOp.getOut();
    }
  }

  // Write back the final C sub-tiles results to the output buffer.
  SmallVector<xegpu::StoreNDOp> storeOps;
  for (size_t i = 0; i < tilesC.size(); i++) {
    auto storeOp = rewriter.create<xegpu::StoreNDOp>(loc, tilesC[i], results[i],
                                                     /*l1_hint=*/writeCacheHint,
                                                     /*l2_hint=*/writeCacheHint,
                                                     /*l3_hint=*/writeCacheHint,
                                                     xegpuMode);
    storeOps.push_back(storeOp);
  }

  (void)fuseEltwiseConsumers(linalgOp, storeOps, rewriter);

  rewriter.eraseOp(linalgOp);

  return success();
}

// Create XeGPU kernel out of elementwise operation.
LogicalResult createEltwiseKernel(linalg::LinalgOp linalgOp,
                                  ArrayRef<int64_t> tileSizes,
                                  PatternRewriter &rewriter) {
  assert(tileSizes.size() == 2 && "Require 2D tile size for eltwise lowering");

  Location loc = linalgOp.getLoc();
  auto ctx = linalgOp.getContext();

  auto output = linalgOp.getDpsInits()[0];
  auto outputType = output.getType().cast<ShapedType>();
  auto outputShape = outputType.getShape();

  bool isOutput2D = outputShape.size() == 2;

  auto dimM = outputShape[0];
  auto dimN = isOutput2D ? outputShape[1] : 0;

  SmallVector<int64_t> eltwiseTileShape{tileSizes[0]};
  if (isOutput2D)
    eltwiseTileShape.push_back(tileSizes[1]);

  // Linalg named elementwise operations guarantee that all operands
  // have the same shape and type. Thus, the same tensor descriptor
  // type can be used for all operands.
  auto tensorDesc = xegpu::TensorDescType::get(
      eltwiseTileShape, output.getType().cast<ShapedType>().getElementType());
  auto xegpuMode = xegpu::Mode::VC;

  // Create tiled tensor descriptors.
  auto createDescs = [&](Value source) -> SmallVector<Value> {
    SmallVector<Value> tiles;
    // TODO: Use larger tile size in case of 1D inputs.
    for (int m = 0; m < dimM; m += tileSizes[0]) {
      Value rowIdx = rewriter.create<arith::ConstantIndexOp>(loc, m);

      // Flexibly create 1D or 2D descriptors.
      int n = 0;
      do {
        mlir::SmallVector<mlir::OpFoldResult> loadOffsets{rowIdx};

        if (dimN > 0) {
          Value colIdx = rewriter.create<arith::ConstantIndexOp>(loc, n);
          loadOffsets.push_back(colIdx);
        }

        Value tile = rewriter
                         .create<xegpu::CreateNdDescOp>(
                             loc, tensorDesc, source, loadOffsets,
                             /*boundary_check=*/true, xegpuMode)
                         .getResult();
        tiles.push_back(tile);

        n += tileSizes[1];
      } while (n < dimN);
    }

    return tiles;
  };

  // Output descriptors for later stores.
  SmallVector<Value> outputTiles = createDescs(output);

  // Create descriptors and load values for all inputs.
  auto vecType =
      VectorType::get(tensorDesc.getShape(), tensorDesc.getElementType());

  SmallVector<SmallVector<Value>> loadedInputs;
  for (auto input : linalgOp.getDpsInputs()) {
    SmallVector<Value> inputTiles = createDescs(input);
    SmallVector<Value> loadedVals;
    for (auto tile : inputTiles) {
      auto loadedVec = rewriter.create<xegpu::LoadNDOp>(
          loc, vecType, tile, /*vnni_axis=*/nullptr,
          /*transpose=*/nullptr,
          /*l1_hint=*/nullptr,
          /*l2_hint=*/nullptr, /*l3_hint=*/nullptr, xegpuMode);
      loadedVals.push_back(loadedVec);
    }
    loadedInputs.push_back(loadedVals);
  }

  // Perform vectorized computations for each output tile.
  SmallVector<Value> results;
  for (size_t i = 0; i < outputTiles.size(); i++) {
    SmallVector<Value> operands;
    for (auto inputs : loadedInputs) {
      operands.push_back(inputs[i]);
    }
    auto res = lowerEltwiseOp(linalgOp, operands, vecType, rewriter);
    if (!res)
      return failure();

    results.push_back(*res);
  }

  // Store results.
  auto writeCacheHint =
      xegpu::CacheWriteHintAttr::get(ctx, xegpu::CacheWriteHint::WRITE_BACK);
  for (size_t i = 0; i < outputTiles.size(); i++) {
    rewriter.create<xegpu::StoreNDOp>(loc, outputTiles[i], results[i],
                                      /*l1_hint=*/writeCacheHint,
                                      /*l2_hint=*/writeCacheHint,
                                      /*l3_hint=*/writeCacheHint, xegpuMode);
  }

  rewriter.eraseOp(linalgOp);

  return success();
}

// Convert a GEMM-like operation to an XeGPU kernel.
template <typename LinalgOpTy>
struct ConvertGemmLikeToXeGPU : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;
  // Constrain conversion to the supported GEMM-like ops.
  static_assert(
      llvm::is_one_of<LinalgOpTy, linalg::MatmulOp, linalg::BatchReduceMatmulOp,
                      linalg::GenericOp>::value);

  ConvertGemmLikeToXeGPU(MLIRContext *ctx, LinalgToXeGPUOptions options)
      : OpRewritePattern<LinalgOpTy>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOpTy gemmLikeOp,
                                PatternRewriter &rewriter) const override {
    if (!gemmLikeOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Linalg GEMM-like to GPU expects memref type");
    }
    if (gemmLikeOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Expect static shape when mapping to GPU");
    }

    using namespace structured_match;
    auto matmulMatcher =
        StructuredOpMatcher::make<linalg::GenericOp>()
            .operation(NumDpsInits(EqualsTo(1)))
            .operation(NumDpsInputs(EqualsTo(2)))
            .operation(NumRegions(EqualsTo(1)))
            .operation(NumOfLoops(EqualsTo(3)))
            .input(MatchAll(), HasStaticShape())
            .output(MatchAll(), HasStaticShape())
            .region(MatchOne(0), WithOpChain<arith::MulFOp, arith::AddFOp>());
    if (isa<linalg::GenericOp>(gemmLikeOp) &&
        !matmulMatcher.match(gemmLikeOp)) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "Generic does not represent a GEMM-like operation");
    }

    for (auto input : gemmLikeOp.getDpsInputs()) {
      // 3D inputs are also acceptable in case of brgemm.
      auto isInputValid =
          isValidMemrefOperand(gemmLikeOp, input, rewriter, /*maxDims=*/3);
      if (failed(isInputValid))
        return isInputValid;
    }
    auto isOutputValid =
        isValidMemrefOperand(gemmLikeOp, gemmLikeOp.getDpsInits()[0], rewriter);
    if (failed(isOutputValid))
      return isOutputValid;

    // Ensure that reduction dimension tiling also works for smaller
    // workloads.
    auto aType =
        gemmLikeOp.getDpsInputs()[0].getType().template cast<ShapedType>();
    auto kDim = aType.getShape().back();
    auto kTile = kDim < options.kTile ? kDim : options.kTile;

    auto dpasConfig = getDPASConfig(gemmLikeOp, kTile);
    if (!dpasConfig) {
      return rewriter.notifyMatchFailure(
          gemmLikeOp, "GEMM-like compute does not fit in DPAS tiles");
    }

    return createDPASKernel(gemmLikeOp, *dpasConfig, kTile, options.stages,
                            rewriter);
  }

private:
  LinalgToXeGPUOptions options;
};

// Convert a named elementwise operation to an XeGPU kernel.
template <typename LinalgOpTy>
struct ConvertNamedEltwiseToXeGPU : public OpRewritePattern<LinalgOpTy> {
  using OpRewritePattern<LinalgOpTy>::OpRewritePattern;

  ConvertNamedEltwiseToXeGPU(MLIRContext *ctx, LinalgToXeGPUOptions options)
      : OpRewritePattern<LinalgOpTy>(ctx), options(options) {}

  LogicalResult matchAndRewrite(LinalgOpTy eltwiseOp,
                                PatternRewriter &rewriter) const override {
    if (!eltwiseOp.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(
          eltwiseOp, "Linalg eltwise to GPU expects memref type");
    }
    if (eltwiseOp.hasDynamicShape()) {
      return rewriter.notifyMatchFailure(
          eltwiseOp, "Expect static shape when mapping to GPU");
    }

    for (auto input : eltwiseOp.getDpsInputs()) {
      auto isInputValid = isValidMemrefOperand(eltwiseOp, input, rewriter);
      if (failed(isInputValid))
        return isInputValid;
    }
    auto isOutputValid =
        isValidMemrefOperand(eltwiseOp, eltwiseOp.getDpsInits()[0], rewriter);
    if (failed(isOutputValid))
      return isOutputValid;

    // TODO: Tile sizes for vectorized eltwise operations should be chosen
    //       dynamically based on the workload and target hardware.
    SmallVector<int64_t> tileSizes{8, 16};

    return createEltwiseKernel(eltwiseOp, tileSizes, rewriter);
  }

private:
  LinalgToXeGPUOptions options;
};

// TODO: Finalize BRGEMM support and register the pattern.
void populateLinalgGemmToXeGPUPatterns(RewritePatternSet &patterns,
                                       LinalgToXeGPUOptions options) {
  patterns.add<ConvertGemmLikeToXeGPU<linalg::MatmulOp>,
               ConvertGemmLikeToXeGPU<linalg::GenericOp>>(patterns.getContext(),
                                                          options);
}

void populateLinalgEltwiseToXeGPUPatterns(RewritePatternSet &patterns,
                                          LinalgToXeGPUOptions options) {
  patterns.add<ConvertNamedEltwiseToXeGPU<linalg::AbsOp>,
               ConvertNamedEltwiseToXeGPU<linalg::AddOp>,
               ConvertNamedEltwiseToXeGPU<linalg::CeilOp>,
               ConvertNamedEltwiseToXeGPU<linalg::DivOp>,
               ConvertNamedEltwiseToXeGPU<linalg::DivUnsignedOp>,
               ConvertNamedEltwiseToXeGPU<linalg::ExpOp>,
               ConvertNamedEltwiseToXeGPU<linalg::FloorOp>,
               ConvertNamedEltwiseToXeGPU<linalg::MaxOp>,
               ConvertNamedEltwiseToXeGPU<linalg::MulOp>,
               ConvertNamedEltwiseToXeGPU<linalg::NegfOp>,
               ConvertNamedEltwiseToXeGPU<linalg::SubOp>>(patterns.getContext(),
                                                          options);
}

struct LinalgToXeGPU : public tpp::impl::LinalgToXeGPUBase<LinalgToXeGPU> {
  using LinalgToXeGPUBase::LinalgToXeGPUBase;

  void runOnOperation() override {
    LinalgToXeGPUOptions options{kTile, stages};

    // Run GEMM pattern first to allow fusion with its consumers.
    RewritePatternSet gemmPatterns(&getContext());
    populateLinalgGemmToXeGPUPatterns(gemmPatterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(gemmPatterns));

    // Convert other remaining ops.
    RewritePatternSet patterns(&getContext());
    populateLinalgEltwiseToXeGPUPatterns(patterns, options);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
