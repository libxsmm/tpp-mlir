//===- TileConsumerAndFuseProducers.cpp --------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/DLTIUtils.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <optional>
#include <queue>

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_TILECONSUMERANDFUSEPRODUCERS
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_ELEMENTWISEFUSION
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

#define DEBUG_TYPE "tile-consumer-and-fuse-producers"

namespace {

// Replace the iter operand of the outermost loop with the region iter argument
// of the innermost loop in the region of the innermost loop. This fix-up
// destination passing style in tile-consumer-and-fuse-producers:
// https://github.com/llvm/llvm-project/issues/61386
struct ReplaceIterArgs : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  void replaceIterArgs(PatternRewriter &rewriter, scf::ForOp outerFor,
                       scf::ForOp innerFor) const {
    assert(outerFor.getInitArgs().size() == innerFor.getInitArgs().size() &&
           "expect same number of iter args");
    Block *block = &(*innerFor.getRegion().begin());
    for (auto it : llvm::zip_equal(outerFor.getInitArgsMutable(),
                                   innerFor.getRegionIterArgs())) {
      auto &source = std::get<0>(it);
      auto target = std::get<1>(it);
      rewriter.replaceUsesWithIf(source.get(), target, [&](OpOperand &use) {
        return use.getOwner()->getBlock() == block;
      });
    }
  }

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto metadata =
        forOp->getAttrOfType<StringAttr>(linalgx::utils::kLoopParallel);
    if (!metadata || metadata.getValue() != linalgx::utils::kLoopRoot)
      return failure();
    if (forOp.getNumRegionIterArgs() != 1)
      return failure();
    SmallVector<scf::ForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, forOp);
    if (nestedLoops.size() == 0)
      return failure();
    replaceIterArgs(rewriter, forOp, nestedLoops[nestedLoops.size() - 1]);
    return success();
  }
};

static bool isConvolutionLike(Operation *op) {
  if (isa_and_nonnull<linalg::GenericOp>(op))
    return linalgx::utils::isBlockedConvolution(op);
  return false;
}

// Return true if `op` can be tiled using `tileSizes`. Require to statically
// know the range and the tile factor. The tile must be full.
static bool canBeTiledWithCurrentSpec(Operation *op,
                                      ArrayRef<OpFoldResult> tileSizes,
                                      int64_t minTileFactor) {
  assert(isa<TilingInterface>(op) &&
         "expect an op implementing the tiling interface");
  assert(!tileSizes.empty() && "expect tile sizes to be non-empty");
  SmallVector<utils::IteratorType> loopIteratorTypes =
      cast<TilingInterface>(op).getLoopIteratorTypes();
  if (tileSizes.size() > loopIteratorTypes.size())
    return false;

  // Validate tiles:
  // - All zeros, nothing to do.
  // - Each tile must be statically known and perfectly divides the dimension.
  // - Require tiling on parallel dimensions only.
  if (llvm::all_of(tileSizes, [](OpFoldResult tile) {
        return isConstantIntValue(tile, 0);
      })) {
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "Running tile validations ----\n");
  if (!linalgx::utils::validateFullTilesOnDims(
          cast<TilingInterface>(op), tileSizes, /*dim=*/{}, minTileFactor)) {
    LLVM_DEBUG(llvm::dbgs() << "FAILED\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "OK\n");

  for (auto tileIdx : llvm::seq<size_t>(0, tileSizes.size())) {
    if (isConstantIntValue(tileSizes[tileIdx], 0))
      continue;
    if (!linalg::isParallelIterator(loopIteratorTypes[tileIdx]))
      return false;
  }

  // Candidate op is good to go.
  return true;
}

// Returns a bit vector of size number of loops of the `interfaceOp` with
// the bits corresponding to outer parallel loops set to `true`.
// Adapted from IREE: `FormDispatchRegions.cpp`.
static llvm::SmallBitVector getOuterParallelLoops(Operation *op) {
  auto interfaceOp = dyn_cast<TilingInterface>(op);
  if (!interfaceOp)
    return llvm::SmallBitVector{};

  SmallVector<utils::IteratorType> loopIteratorTypes =
      interfaceOp.getLoopIteratorTypes();
  llvm::SmallBitVector parallelLoops(loopIteratorTypes.size());
  for (auto iteratorType : llvm::enumerate(loopIteratorTypes)) {
    if (iteratorType.value() != utils::IteratorType::parallel)
      continue;
    parallelLoops.set(iteratorType.index());
  }
  return parallelLoops;
}

// Return the tile sizes as bit vector.
static llvm::SmallBitVector
getTileBitVectorConfig(ArrayRef<OpFoldResult> tileSizes,
                       size_t rootConsumerParallelLoops) {
  assert(tileSizes.size() <= rootConsumerParallelLoops);
  llvm::SmallBitVector tileConfig(rootConsumerParallelLoops, false);
  for (size_t idx : llvm::seq<size_t>(0, tileSizes.size()))
    if (!isConstantIntValue(tileSizes[idx], 0))
      tileConfig.set(idx);
  return tileConfig;
}

// Returns true if `map` is an identity map with zeros, i.e. if you
// drop the result exprs that are constant zeros, the `map` will become an
// identity.
// Taken from IREE
static bool isIdentityMapWithZeros(AffineMap map) {
  if (map.getNumSymbols() != 0)
    return false;
  unsigned dimsSeen = 0;
  for (auto result : map.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(result)) {
      if (dimExpr.getPosition() != dimsSeen)
        return false;
      dimsSeen++;
      continue;
    }
    if (auto constExpr = dyn_cast<AffineConstantExpr>(result)) {
      if (constExpr.getValue() == 0)
        continue;
    }
    return false;
  }
  return dimsSeen == map.getNumDims();
}

// Helper fuction to print a bit vector.
[[maybe_unused]] static void printBitVector(std::string banner,
                           const llvm::SmallBitVector &bitVector,
                           llvm::raw_ostream &os) {
  os << banner << "  ";
  for (size_t idx : llvm::seq<size_t>(0, bitVector.size())) {
    os << bitVector.test(idx);
    if (idx != bitVector.size() - 1)
      os << ", ";
  }
  os << "\n";
}

static FailureOr<AffineMap> getProducerOutputMap(Operation *producer) {
  assert(producer->getNumResults() == 1);
  if (auto linalgProducer = dyn_cast_or_null<linalg::LinalgOp>(producer)) {
    return linalgProducer.getIndexingMapMatchingResult(
        linalgProducer.getTiedOpResult(
            &linalgProducer.getDpsInitsMutable()[0]));
  }
  return failure();
}

static FailureOr<AffineMap> getConsumerOperandMap(Operation *consumer,
                                                  OpOperand &operand) {
  if (auto linalgConsumer = dyn_cast_or_null<linalg::LinalgOp>(consumer))
    return linalgConsumer.getMatchingIndexingMap(&operand);
  return failure();
}

static FailureOr<llvm::SmallBitVector> getTileConfigProducer(
    OpOperand &operand, Operation *producer, size_t numLoopsProd,
    llvm::SmallBitVector tileSpecConsumer,
    llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> &tileSizes) {

  // Happy path, we already have a tile configuration.
  if (tileSizes.count(producer)) {
    return getTileBitVectorConfig(tileSizes.at(producer), numLoopsProd);
  }
  Operation *consumer = operand.getOwner();

  auto producerOutputMap = getProducerOutputMap(producer);
  auto consumerOperandMap = getConsumerOperandMap(consumer, operand);
  if (failed(producerOutputMap) || failed(consumerOperandMap))
    return failure();
  if (!producerOutputMap->isProjectedPermutation() ||
      !consumerOperandMap->isProjectedPermutation()) {
    return failure();
  }

  assert(consumerOperandMap->getNumResults() == numLoopsProd);
  llvm::SmallBitVector tileConfigProducer(numLoopsProd);
  SmallVector<OpFoldResult> tileProducer(numLoopsProd);
  for (auto expr : llvm::enumerate(consumerOperandMap->getResults())) {
    auto dim = cast<AffineDimExpr>(expr.value());
    tileConfigProducer[expr.index()] = tileSpecConsumer[dim.getPosition()];
    tileProducer[expr.index()] = tileSizes.at(consumer)[dim.getPosition()];
  }
  // Insert the tile configuration in the map.
  tileSizes[producer] = tileProducer;
  return tileConfigProducer;
}

// Check if the the tile specification for the consumer (operand's owner)
// is compatible with the producer. If the producer is not yet visited (i.e., it
// does not have a tile specification attempt to infer it using
// `getTileConfigProducer`).
static bool hasCompatibleParallelLoops(
    OpOperand &operand, Operation *producer,
    llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> &tileSizes) {
  assert(operand.get().getDefiningOp() == producer);
  Operation *consumer = operand.getOwner();

  if (tileSizes.find(consumer) == tileSizes.end())
    return false;

  llvm::SmallBitVector producerParallelLoops =
      getOuterParallelLoops(cast<TilingInterface>(producer));
  llvm::SmallBitVector consumerParallelLoops =
      getOuterParallelLoops(cast<TilingInterface>(consumer));

  llvm::SmallBitVector tileSpecConsumer = getTileBitVectorConfig(
      tileSizes.at(consumer), consumerParallelLoops.size());
  auto tileSpecProducer =
      getTileConfigProducer(operand, producer, producerParallelLoops.size(),
                            tileSpecConsumer, tileSizes);
  if (failed(tileSpecProducer)) {
    return false;
  }

  auto producerOutputMap = getProducerOutputMap(producer);
  auto consumerOperandMap = getConsumerOperandMap(consumer, operand);

  producerParallelLoops &= *tileSpecProducer;
  producerParallelLoops.flip();
  auto producerProjectedMap =
      getProjectedMap(*producerOutputMap, producerParallelLoops);

  consumerParallelLoops &= tileSpecConsumer;
  consumerParallelLoops.flip();
  auto consumerProjectedMap =
      getProjectedMap(*consumerOperandMap, consumerParallelLoops);

  LLVM_DEBUG(
      llvm::dbgs() << "Producer: " << *producer << "\n";
      llvm::dbgs() << "Consumer: " << *consumer << "\n"; printBitVector(
          "PRODUCER LOOPS      ", producerParallelLoops, llvm::dbgs());
      printBitVector("CONSUMER LOOPS      ", consumerParallelLoops,
                     llvm::dbgs());
      printBitVector("TILE CONFIG CONSUMER", tileSpecConsumer, llvm::dbgs());
      printBitVector("TILE CONFIG PRODUCER", *tileSpecProducer, llvm::dbgs());

      llvm::dbgs() << "Producer output  map: " << *producerOutputMap << "\n";
      llvm::dbgs() << "Consumer operand map: " << *consumerOperandMap << "\n";
      llvm::dbgs() << "Producer projected map: " << producerProjectedMap
                   << "\n";
      llvm::dbgs() << "Consumer projected map: " << consumerProjectedMap
                   << "\n";);

  return isIdentityMapWithZeros(producerProjectedMap) &&
         isIdentityMapWithZeros(consumerProjectedMap);
}

// Return true if the op has all users in the worklist. This is an important
// condition as we want to fuse together 'isolated islands' of op. Basically
// each op in the worklist must have users in the worklist. If this is not the
// case we can introduce recomputation.
bool hasAllUsersInWorklist(Operation *op,
                           const llvm::SmallDenseSet<Operation *> &worklist) {
  assert(op->getNumResults() == 1 && "expect single result op");
  Value result = op->getResult(0);
  for (Operation *user : result.getUsers())
    if (worklist.count(user) == 0)
      return false;
  return true;
}

void incDepthAndSwap(std::queue<Operation *> &frontier,
                     std::queue<Operation *> &nextFrontier, int64_t &depth) {
  if (frontier.empty() && !nextFrontier.empty()) {
    std::swap(frontier, nextFrontier);
    depth++;
  }
}

// Return a list of producers op that can be fused together based on what has
// already been fused and the current tile specification.
static llvm::SmallDenseSet<Operation *> collectFusableProducers(
    TilingInterface rootConsumer,
    llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> &tileSizes,
    const llvm::SmallDenseSet<Operation *> &alreadyFusedOps, int64_t maxDepth) {
  if (alreadyFusedOps.count(rootConsumer.getOperation()))
    return {};

  llvm::SmallDenseSet<Operation *> worklist;
  std::queue<Operation *> processingQueue;
  std::queue<Operation *> nextProcessingQueue;
  processingQueue.push(rootConsumer);
  worklist.insert(rootConsumer);
  LLVM_DEBUG(llvm::dbgs() << "WORKLIST INSERT CONSUMER: "
                          << rootConsumer.getOperation() << "\n");
  LLVM_DEBUG(
      llvm::dbgs()
          << "---------------------------------------------------------\n";
      llvm::dbgs() << *rootConsumer.getOperation();
      llvm::dbgs()
      << "\n---------------------------------------------------------\n";);
  int64_t depth = 0;
  while (!processingQueue.empty()) {
    if (depth >= maxDepth)
      break;
    Operation *currentOp = processingQueue.front();
    processingQueue.pop();
    for (OpOperand &operand : currentOp->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (producer && isa<linalg::FillOp>(producer)) {
        nextProcessingQueue.push(producer);
        incDepthAndSwap(processingQueue, nextProcessingQueue, depth);
        worklist.insert(producer);
        continue;
      }
      if (producer && isa<TilingInterface>(producer) &&
          !worklist.count(producer) && producer->getNumResults() == 1 &&
          !alreadyFusedOps.count(producer) &&
          hasCompatibleParallelLoops(operand, producer, tileSizes) &&
          hasAllUsersInWorklist(producer, worklist)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "WORKLIST INSERT PRODUCER: " << producer << "\n");
        LLVM_DEBUG(llvm::dbgs() << "=============================\n";
                   llvm::dbgs() << *producer << "\n";
                   llvm::dbgs() << "=============================\n";);
        nextProcessingQueue.push(producer);
        incDepthAndSwap(processingQueue, nextProcessingQueue, depth);
        worklist.insert(producer);
      }
    }
  }
  return worklist;
}

// Entry point for fusion with element-wise operations.
static FailureOr<scf::SCFTileAndFuseResult> fuseWithEltwise(
    RewriterBase &rewriter, TilingInterface consumer,
    llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> &tileSizes,
    llvm::SmallDenseSet<Operation *> &alreadyFusedOps, int64_t maxDepth,
    int64_t minTileFactor) {
  // Step 0. Early exit if tileSizes are empty.
  if (tileSizes.empty() || !tileSizes.count(consumer)) {
    LLVM_DEBUG(llvm::dbgs() << "EMPTY TILE SIZES\n");
    return failure();
  }

  // Step 1. If the consumer is already tiled and fused, bail out.
  if (alreadyFusedOps.count(consumer)) {
    LLVM_DEBUG(llvm::dbgs()
               << "CONSUMER: " << consumer << "\nALREADY TILED AND FUSED\n");
    return failure();
  }

  // Step 2. Check if the tile configuration fits the consumer.
  if (!canBeTiledWithCurrentSpec(consumer, tileSizes.at(consumer),
                                 minTileFactor)) {
    LLVM_DEBUG(llvm::dbgs() << "CONSUMER: " << consumer
                            << "\nCANNOT BE TILED WITH CURRENT CONFIG\n");
    return failure();
  }

  // Step 3. Collect the operations that can be tiled and fused.
  llvm::SmallDenseSet<Operation *> worklist =
      collectFusableProducers(consumer, tileSizes, alreadyFusedOps, maxDepth);
  LLVM_DEBUG(llvm::dbgs() << "#WORKLIST: " << worklist.size() << "\n");
  if (worklist.size() < 1)
    return failure();

  // Step 4. Tile the consumer and move the producers
  // in the fusion domain.
  scf::SCFTilingOptions options;
  options.setTileSizes(tileSizes.at(consumer));
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.setTilingOptions(options);
  scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
      [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
          bool isDestinationOperand)
      -> std::optional<scf::SCFTileAndFuseOptions::ControlFnResult> {
    Operation *candidateOp = originalProducer.getOwner();
    if (!candidateOp || worklist.count(candidateOp) == 0 ||
        (alreadyFusedOps.count(candidateOp) &&
         !isa<linalg::FillOp>(candidateOp))) {
      return std::nullopt;
    }
    scf::SCFTileAndFuseOptions::ControlFnResult res;
    res.yieldProducerReplacement = false;
    return res;
  };
  tileAndFuseOptions.setFusionControlFn(controlFn);
  FailureOr<scf::SCFTileAndFuseResult> tileAndFuseResult =
      scf::tileConsumerAndFuseProducersUsingSCF(rewriter, consumer,
                                                tileAndFuseOptions);
  if (failed(tileAndFuseResult)) {
    return rewriter.notifyMatchFailure(
        consumer, "failed to tile and fuse with op as root");
  }
  if (!tileAndFuseResult->loops.empty()) {
    tileAndFuseResult->loops[0]->setAttr(
        linalgx::utils::kLoopParallel,
        rewriter.getStringAttr(linalgx::utils::kLoopRoot));
  }
  return *tileAndFuseResult;
}

// Trivial tile selection. If the dimension is statically known, it perfectly
// divides the tile, and we have enough iterations return a default of 32.
static int64_t getTileForDim(linalg::LinalgOp linalgOp, unsigned dim) {
  int64_t tile = 32;

  // Check if a tile size hint is associated to the IR via DLTI.
  auto tileValue = dlti::utils::query(linalgOp, {"CPU", "tile_size"});
  if (succeeded(tileValue))
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(*tileValue))
      tile = intAttr.getInt();

  SmallVector<int64_t, 4> loopsRange = linalgOp.getStaticLoopRanges();
  if (loopsRange[dim] == ShapedType::kDynamic)
    return tile;
  if (loopsRange[dim] < tile || loopsRange[dim] % tile != 0)
    return 0;
  return tile;
}

// Return tile sizes for `linalgOp`.
// - For linalg.matmul -> {32, 32}
// - For all other matmul-like contractions: tile fully all the parallel loops
// that are not involved in a GEMM computation.
static SmallVector<int64_t>
getDefaultTileSizesForMatmulLikeOp(linalg::LinalgOp linalgOp) {
  SmallVector<int64_t> tiles(linalgOp.getNumLoops(), 0);
  if (isa<linalg::MatmulOp>(linalgOp)) {
    tiles[0] = getTileForDim(linalgOp, 0); // i loop
    tiles[1] = getTileForDim(linalgOp, 1); // j loop
    return tiles;
  }

  auto contractionDims = linalgx::utils::isContraction(linalgOp);
  if (failed(contractionDims))
    return tiles;

  SmallVector<unsigned, 2> mDims = contractionDims->m;
  SmallVector<unsigned, 2> nDims = contractionDims->n;
  SmallVector<unsigned, 2> kDims = contractionDims->k;
  SmallVector<unsigned, 2> batchDims = contractionDims->batch;
  // Trivial GEMM-like contractions.
  if (tiles.size() == 3 && batchDims.size() == 0 && mDims.size() == 1 &&
      nDims.size() == 1 && kDims.size() == 1) {
    tiles[mDims[0]] = getTileForDim(linalgOp, mDims[0]); // i loop
    tiles[nDims[0]] = getTileForDim(linalgOp, nDims[0]); // j loop
  } else {
    // Non-trivial contraction: Drop the minor dimensions on m and n. These
    // dimensions are part of the GEMM computation and should not be tiled.
    mDims.pop_back();
    nDims.pop_back();

    int64_t constexpr tileFactor = 1;
    for (auto dim : mDims)
      tiles[dim] = tileFactor;
    for (auto dim : nDims)
      tiles[dim] = tileFactor;
    for (auto dim : batchDims)
      tiles[dim] = tileFactor;
  }
  return tiles;
}

static FailureOr<SmallVector<int64_t>>
getDefaultTileSizes(linalg::LinalgOp linalgOp,
                    ArrayRef<int64_t> userProvidedTiles) {
  // The user-provided tiles are considered from the outer
  // most loop. If not enough tiles are provided we pad with
  // zeros.
  if (!userProvidedTiles.empty()) {
    size_t numParallelLoops = linalgOp.getNumParallelLoops();
    size_t nonZeros = 0;
    for (auto tile : userProvidedTiles)
      if (tile != 0)
        nonZeros++;
    if (nonZeros > numParallelLoops ||
        userProvidedTiles.size() > linalgOp.getNumLoops()) {
      return failure();
    }

    SmallVector<int64_t> userTiles(linalgOp.getNumLoops(), 0);
    for (auto tile : llvm::enumerate(userProvidedTiles))
      userTiles[tile.index()] = tile.value();
    return userTiles;
  }
  // Blocked convolutions are tiled and fused along the three outermost parallel
  // loops to expose a BRGEMM.
  // TODO: this should merge with `getDefaultTileSizesForMatmulLikeOp`.
  if (linalgx::utils::isBlockedConvolution(linalgOp))
    return SmallVector<int64_t>{1, 1, 1, 0, 0, 0, 0, 0, 0};
  return getDefaultTileSizesForMatmulLikeOp(linalgOp);
}

// Propagate the tile specification from producer to consumer. Example,
// Tile spec producer:  (1,  0, 0,  0, 1,  0)
// Output map producer: (i, ii, k, kk, j, jj) -> (i, ii, j, jj)
// Assuming an eltwise consumer, with map:
// (i, ii, j, jj) -> (i, ii, j, jj) the tiling specification will be:
// (1, 0, 1, 0).
static SmallVector<OpFoldResult>
getTileForEltWiseConsumer(Operation *consumer, Operation *producer,
                          SmallVector<OpFoldResult> tilesProducer) {
  assert(consumer && producer);
  if (consumer == producer)
    return tilesProducer;

  assert(isa<linalg::LinalgOp>(consumer) && isa<linalg::LinalgOp>(producer) &&
         linalg::isElementwise(cast<linalg::LinalgOp>(consumer)));

  // Case 1. producer and consumer are eltwise.
  if (linalg::isElementwise(cast<linalg::LinalgOp>(producer)))
    return tilesProducer;

  // Case 2. producer is not an eltwise.
  linalg::LinalgOp producerOp = cast<linalg::LinalgOp>(producer);
  assert(producerOp.getNumDpsInits() == 1);
  AffineMap outputMap =
      producerOp.getMatchingIndexingMap(&producerOp.getDpsInitsMutable()[0]);
  assert(outputMap.isProjectedPermutation());
  assert(outputMap.getNumDims() == tilesProducer.size());
  SmallVector<OpFoldResult> eltWiseTiles;
  for (auto expr : outputMap.getResults()) {
    eltWiseTiles.push_back(
        tilesProducer[cast<AffineDimExpr>(expr).getPosition()]);
  }
  return eltWiseTiles;
}

static bool hasOneUser(Value op) {
  if (op.hasOneUse())
    return true;
  auto users = op.getUsers();
  Operation *firstUser = *users.begin();
  if (firstUser)
    return llvm::all_of(users,
                        [&](Operation *user) { return user == firstUser; });

  return false;
}

static Operation *getLastFusableEltWiseConsumer(
    linalg::LinalgOp linalgOp,
    llvm::SmallDenseSet<Operation *> &visitedConsumers,
    llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> &tiles) {
  assert(linalgOp->getNumResults() == 1 && "Expect single result operation");
  Value linalgOpRes = linalgOp->getResult(0);
  // If we allow use, we may end up doing recomputation. Unclear if it is
  // profitablem thus disallow for now.
  if (!hasOneUser(linalgOpRes))
    return linalgOp;

  // Start checking consumers.
  Operation *nextConsumer = *(linalgOpRes.getUsers().begin());
  Operation *currentConsumer = linalgOp;
  visitedConsumers.insert(currentConsumer);

  auto isValidEltWiseConsumer = [&](Operation *op) {
    // Make sure we visit each consumer only once.
    if (visitedConsumers.count(op))
      return false;

    if (!isa<linalg::LinalgOp>(op) ||
        !linalg::isElementwise(cast<linalg::LinalgOp>(op))) {
      return false;
    }
    // Require same iteration space.
    if (cast<linalg::LinalgOp>(op).getNumParallelLoops() !=
        cast<linalg::LinalgOp>(currentConsumer).getNumParallelLoops())
      return false;

    return op->getNumResults() == 1;
  };

  while (isValidEltWiseConsumer(nextConsumer)) {
    Value resNextConsumer = nextConsumer->getResult(0);
    currentConsumer = nextConsumer;
    tiles[currentConsumer] =
        getTileForEltWiseConsumer(currentConsumer, linalgOp, tiles[linalgOp]);
    visitedConsumers.insert(currentConsumer);
    // Require each eltwise to have a single user.
    if (!hasOneUser(resNextConsumer)) {
      break;
    }
    nextConsumer = *(resNextConsumer.getUsers().begin());
  }
  LLVM_DEBUG(llvm::dbgs() << "LAST FUSABLE CONSUMER: " << currentConsumer
                          << "\n");
  return currentConsumer;
}

// Run `fuseWithEltwise` on contraction-like operations.
static void doFusion(RewriterBase &rewriter, func::FuncOp func,
                     ArrayRef<int64_t> tileSizes, int64_t maxDepth,
                     int64_t minTileFactor) {
  // Set to keep track of fused ops.
  llvm::SmallDenseSet<Operation *> fusedOps;

  SmallVector<linalg::LinalgOp> linalgContractionOperations;
  // Walk postorder to increase fusion boundaries.
  func->walk<WalkOrder::PostOrder>([&](linalg::LinalgOp linalgOp) {
    if ((isConvolutionLike(linalgOp) ||
         succeeded(linalgx::utils::isContraction(linalgOp))) &&
        linalgOp.hasPureTensorSemantics())
      linalgContractionOperations.push_back(linalgOp);
  });

  if (linalgContractionOperations.empty())
    return;

  llvm::SmallDenseSet<Operation *> visitedConsumers;
  llvm::SmallDenseSet<Operation *> fusionRoots;

  // Compute the tile sizes for each contraction operations or
  // use the default one.
  llvm::DenseMap<Operation *, SmallVector<OpFoldResult>> defaultTiles;
  for (auto contractionOp : linalgContractionOperations) {
    auto tiles = getDefaultTileSizes(contractionOp, tileSizes);
    if (failed(tiles)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to compute default tile sizes for: "
                              << contractionOp << "\n");
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "Tiles to use for op:\n");
    LLVM_DEBUG(llvm::dbgs() << contractionOp << "\n");
    LLVM_DEBUG(llvm::interleaveComma(*tiles, llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "\n\n");
    defaultTiles[contractionOp] =
        getAsOpFoldResult(rewriter.getI64ArrayAttr(*tiles));
  }

  for (linalg::LinalgOp contractionOp : linalgContractionOperations) {
    Operation *consumerOp = getLastFusableEltWiseConsumer(
        contractionOp, visitedConsumers, defaultTiles);
    fusionRoots.insert(consumerOp);
  }
  LLVM_DEBUG(llvm::dbgs() << "#fusionRoots: " << fusionRoots.size() << "\n");

  SmallVector<Operation *> allLinalgOps;
  func->walk(
      [&](linalg::LinalgOp linalgOp) { allLinalgOps.push_back(linalgOp); });

  // We want to walk operations in reverse order as fusion will likely find
  // larger cluster of operations to fuse if we go bottom-up.
  std::reverse(allLinalgOps.begin(), allLinalgOps.end());

  for (Operation *linalgOp : allLinalgOps) {
    if (fusionRoots.count(linalgOp)) {
      LLVM_DEBUG(llvm::dbgs() << "\n\n");
      FailureOr<scf::SCFTileAndFuseResult> fuseAndTileResult =
          fuseWithEltwise(rewriter, cast<TilingInterface>(linalgOp),
                          defaultTiles, fusedOps, maxDepth, minTileFactor);
      LLVM_DEBUG(llvm::dbgs() << "\n\n");
      if (succeeded(fuseAndTileResult)) {
        rewriter.replaceOp(
            linalgOp,
            (*fuseAndTileResult).replacements[linalgOp->getResults()[0]]);
      }
    }
  }
}

struct TileConsumerAndFuseProducers
    : tpp::impl::TileConsumerAndFuseProducersBase<
          TileConsumerAndFuseProducers> {
  using TileConsumerAndFuseProducersBase::TileConsumerAndFuseProducersBase;

  void runOnOperation() override {
    auto &ctx = getContext();

    {
      // Attempt to recover named ops.
      RewritePatternSet patterns(&ctx);
      linalg::populateLinalgDeGeneralizationPatterns(patterns);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    int64_t numIters = this->numIters;
    do {
      func::FuncOp func = getOperation();
      IRRewriter rewriter(&getContext());
      doFusion(rewriter, func, this->tileSizes, this->maxDepth,
               this->minTileFactor);

      {
        RewritePatternSet patterns(&ctx);
        // Fold unit-extent dims for linalg on tensors. Since
        // `populateFoldUnitExtentDimsViaSlicesPatterns` works only with
        // linalg.generic we need to generalize first using
        // `populateLinalgNamedOpsGeneralizationPatterns`.
        linalg::ControlDropUnitDims options;
        options.rankReductionStrategy = linalg::ControlDropUnitDims::
            RankReductionStrategy::ExtractInsertSlice;
        linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
        tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);

        // TODO: Remove the generalization of named ops after resolving the
        // above dependency with "populateFoldUnitExtentDimsViaSlicesPatterns".
        linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
        (void)applyPatternsGreedily(getOperation(), std::move(patterns));
      }
    } while (--numIters);

    {
      // Patterns for scf.for.
      RewritePatternSet patterns(&ctx);
      patterns.add<ReplaceIterArgs>(&ctx);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    {
      // Patterns for scf.forall.
      RewritePatternSet patterns(&ctx);
      if (this->useForAll)
        linalgx::utils::populateScfForToForAllRewritePattern(patterns);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }

    {
      // Attempt to recover named ops.
      RewritePatternSet patterns(&ctx);
      linalg::populateLinalgDeGeneralizationPatterns(patterns);
      scf::ForallOp::getCanonicalizationPatterns(patterns, &ctx);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
  }
};

static bool areFusableOps(Operation *producerOp, Operation *consumerOp) {
  if ((!isa<linalg::LinalgOp>(producerOp)) ||
      (!isa<linalg::LinalgOp>(consumerOp)))
    return false;
  if (!linalg::isElementwise(cast<linalg::LinalgOp>(producerOp)) ||
      !linalg::isElementwise(cast<linalg::LinalgOp>(consumerOp))) {
    return false;
  }
  return producerOp->hasOneUse();
}

struct ElementWiseFusion : tpp::impl::ElementWiseFusionBase<ElementWiseFusion> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    linalg::ControlFusionFn fuseElementwiseOpsControlFn =
        [&](OpOperand *fusedOperand) {
          Operation *producer = fusedOperand->get().getDefiningOp();
          if (!producer)
            return false;
          Operation *consumer = fusedOperand->getOwner();
          return areFusableOps(producer, consumer);
        };

    linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                 fuseElementwiseOpsControlFn);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace
