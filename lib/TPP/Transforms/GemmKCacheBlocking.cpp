//===- GemmKCacheBlocking.cpp ------------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the K-dimension cache-blocking half of the two-level
// GEMM cache-blocking scheme (the M/N cache panels come from the prior
// TileConsumerAndFuseProducers step). It tiles the batch-reduce (outer K)
// dimension of a batch-reduce GEMM by a cache-block factor so the reused A/B
// panels stay L2-resident instead of being evicted and refetched from L3 on
// every reuse. It runs after tile-and-fuse, so the K-block loop nests inside
// the spatial (M/N) cache panel and threads that panel's small high-precision
// (f32/i32) accumulator across K-blocks. When the panel spans multiple
// register tiles it also splits them into inner register-tile loops (hoisting
// the K-block A/B slices so each is loaded once and reused across the panel).
// The accumulator is written to C exactly once, by the (unchanged)
// truncf/requant epilogue that already follows the tiled GEMM.
//
//===----------------------------------------------------------------------===//
#include "TPP/Passes.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <functional>

#define DEBUG_TYPE "gemm-k-cache-blocking"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GEMMKCACHEBLOCKING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace {

// Tile the batch-reduce (outer K) dimension of a batch-reduce GEMM by the
// cache-block factor. Runs after tile-and-fuse: the GEMM is the small per-tile
// op inside the spatial scf.forall, so threading its own DPS accumulator (a
// high-precision f32/i32 tile) through the resulting sequential K-block loop
// keeps a single, full-precision accumulator that the existing epilogue writes
// to C once.
template <typename BrgemmOp>
struct KCacheBlockingTiling : OpRewritePattern<BrgemmOp> {
  KCacheBlockingTiling(MLIRContext *ctx, unsigned kCacheBlocking)
      : OpRewritePattern<BrgemmOp>(ctx), kCacheBlocking(kCacheBlocking) {}

  LogicalResult matchAndRewrite(BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    if (kCacheBlocking == 0)
      return rewriter.notifyMatchFailure(brgemmOp, "K cache blocking disabled");

    auto linalgOp = cast<linalg::LinalgOp>(brgemmOp.getOperation());

    // Only batch-reduce GEMMs (fp32: BR+K reductions, bf16 VNNI: BR+K+vnni)
    // carry an outer batch-reduce dimension to cache-block.
    SmallVector<utils::IteratorType> iteratorTypes =
        linalgOp.getIteratorTypesArray();
    int reductionCount = std::count(iteratorTypes.begin(), iteratorTypes.end(),
                                    utils::IteratorType::reduction);
    int parallelCount = std::count(iteratorTypes.begin(), iteratorTypes.end(),
                                   utils::IteratorType::parallel);
    // Match the tiled GEMM inside the spatial scf.forall: the M/N tile dims are
    // parallel iterators and the batch-reduce (outer K) dim is still a full
    // reduction, so tiling it here yields a K-block loop nested inside the
    // spatial tile.
    if (parallelCount < 2 || reductionCount < 2 || reductionCount > 3)
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "Expected batch-reduce GEMM");

    auto vnniOpt = vnni::utils::isInVnniLayout(brgemmOp);
    if (reductionCount == 3 && !vnniOpt)
      return rewriter.notifyMatchFailure(
          brgemmOp, "Expected VNNI layout for 3 reduction dims");

    // Locate the batch-reduce dimension from operand A's indexing map.
    // A is (BR, M, K, [vnni]); BR is the outermost result.
    auto rankA =
        cast<ShapedType>(brgemmOp->getOperand(0).getType()).getRank();
    AffineMap mapA =
        linalgOp.getMatchingIndexingMap(&brgemmOp->getOpOperand(0));
    unsigned brResultIdx = vnniOpt ? (rankA - 4) : (rankA - 3);
    auto dimExpr = dyn_cast<AffineDimExpr>(mapA.getResult(brResultIdx));
    if (!dimExpr)
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "Non-trivial batch-reduce map");
    unsigned dimBR = dimExpr.getPosition();

    if (iteratorTypes[dimBR] != utils::IteratorType::reduction)
      return rewriter.notifyMatchFailure(
          brgemmOp, "Batch-reduce dimension is not a reduction");

    // Require a static, cleanly divisible extent so no partial tail is emitted
    // and the pattern does not re-match the tiled inner op.
    SmallVector<int64_t> loopRanges = linalgOp.getStaticLoopRanges();
    int64_t brExtent = loopRanges[dimBR];
    if (ShapedType::isDynamic(brExtent) || brExtent <= 0)
      return rewriter.notifyMatchFailure(brgemmOp,
                                         "Dynamic batch-reduce extent");
    // When k-cache-blocking covers all (or more) K-blocks the K-loop degenerates
    // to a single iteration; clamp to the extent so any M/N cache panel is still
    // split into register tiles below (bailing here would leave a panelized GEMM
    // the AMX/nano path cannot lower). A partial tail (non-divisor) is rejected.
    unsigned effKBlock = kCacheBlocking;
    if (static_cast<int64_t>(effKBlock) > brExtent)
      effKBlock = static_cast<unsigned>(brExtent);
    if (brExtent % static_cast<int64_t>(effKBlock) != 0)
      return rewriter.notifyMatchFailure(
          brgemmOp, "K cache block must divide batch-reduce extent");

    // GEMM-centric K-cache-blocking: tile the batch-reduce (outer K) reduction
    // dimension and thread the running result through the resulting sequential
    // loop so the partial products sum correctly across blocks. Element-type
    // agnostic.
    Value initOperand = linalgOp.getDpsInitOperand(0)->get();

    // Locate the batch-reduce tensor dim on the A and B operands.
    AffineMap mapB = linalgOp.getMatchingIndexingMap(&brgemmOp->getOpOperand(1));
    auto findDim = [&](AffineMap m) -> unsigned {
      for (unsigned i = 0, e = m.getNumResults(); i < e; ++i)
        if (auto d = dyn_cast<AffineDimExpr>(m.getResult(i)))
          if (d.getPosition() == dimBR)
            return i;
      return 0;
    };
    unsigned aDim = findDim(mapA);
    unsigned bDim = findDim(mapB);

    Location loc = brgemmOp.getLoc();
    rewriter.setInsertionPoint(brgemmOp);

    Value A = brgemmOp->getOperand(0);
    Value Bmat = brgemmOp->getOperand(1);
    auto aType = cast<RankedTensorType>(A.getType());
    auto bTy = cast<RankedTensorType>(Bmat.getType());

    // Slice A and B along the batch-reduce dim to a K-block starting at `iv`.
    auto sliceBR = [&](OpBuilder &b, Location l, Value iv, Value v,
                       RankedTensorType t, unsigned d) -> Value {
      SmallVector<OpFoldResult> off(t.getRank(), b.getIndexAttr(0));
      SmallVector<OpFoldResult> sz;
      for (int64_t i = 0; i < t.getRank(); ++i)
        sz.push_back(b.getIndexAttr(t.getDimSize(i)));
      SmallVector<OpFoldResult> str(t.getRank(), b.getIndexAttr(1));
      off[d] = iv;
      sz[d] = b.getIndexAttr(effKBlock);
      return tensor::ExtractSliceOp::create(b, l, v, off, sz, str);
    };

    // Detect the multi-register-tile panels: after two-level tile-and-fuse the
    // spatial forall owns a panel of register tiles in M and/or N, so the GEMM
    // carries extra leading parallel (panel) dimensions and its accumulator is
    // rank (numPanels + 2) instead of rank-2 (M, N). Each such panel dim must be
    // split into an inner register-tile loop so the innermost op is a single
    // 32x32 tile GEMM the AMX/nano path can lower. An M panel lives on A+C (B is
    // reused across it); an N panel lives on B+C (A is reused across it). The
    // K-block slices are hoisted above the inner loops so each is loaded once
    // and reused across all register tiles in the panel for the L2 win.
    MLIRContext *ctx = rewriter.getContext();
    auto outTy = cast<RankedTensorType>(initOperand.getType());
    AffineMap mapC =
        linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));
    auto findResultForDim = [](AffineMap m,
                               unsigned dimPos) -> std::optional<unsigned> {
      for (unsigned i = 0, e = m.getNumResults(); i < e; ++i)
        if (auto d = dyn_cast<AffineDimExpr>(m.getResult(i)))
          if (d.getPosition() == dimPos)
            return i;
      return std::nullopt;
    };

    // The trailing two C result dims are the 32x32 register tile; every leading
    // C result dim is a panel to split.
    unsigned numPanels =
        isa<linalg::GenericOp>(brgemmOp.getOperation()) && outTy.getRank() > 2
            ? outTy.getRank() - 2
            : 0;
    SmallVector<unsigned> panelIterPos(numPanels);
    SmallVector<int64_t> panelExtent(numPanels);
    bool hasPanel = numPanels > 0;
    for (unsigned p = 0; p < numPanels; ++p) {
      panelIterPos[p] = cast<AffineDimExpr>(mapC.getResult(p)).getPosition();
      panelExtent[p] = outTy.getDimSize(p);
      // Each panel must live on exactly one of A/B (and, implicitly, C); the
      // other operand is then reused across the panel. Otherwise fall back.
      bool onA = findResultForDim(mapA, panelIterPos[p]).has_value();
      bool onB = findResultForDim(mapB, panelIterPos[p]).has_value();
      if (onA == onB)
        hasPanel = false;
    }

    // Nothing to tile when the K-loop is a single block and there is no panel
    // to split: leave the canonical single-tile GEMM for the AMX/nano path and
    // avoid re-matching the tiled inner op.
    if (effKBlock == static_cast<unsigned>(brExtent) && !hasPanel)
      return rewriter.notifyMatchFailure(
          brgemmOp, "single K-block and no panel: nothing to tile");

    auto isPanelPos = [&](unsigned pos) {
      return llvm::is_contained(panelIterPos, pos);
    };
    // Rebuild an indexing map with all panel iterators dropped and the
    // remaining dims renumbered to be contiguous.
    auto rebuildMap = [&](AffineMap m) -> AffineMap {
      SmallVector<AffineExpr> res;
      for (AffineExpr e : m.getResults()) {
        unsigned p = cast<AffineDimExpr>(e).getPosition();
        if (isPanelPos(p))
          continue;
        unsigned shift = llvm::count_if(
            panelIterPos, [&](unsigned q) { return q < p; });
        res.push_back(getAffineDimExpr(p - shift, ctx));
      }
      return AffineMap::get(m.getNumDims() - numPanels, 0, res, ctx);
    };
    // Extract the single register tile of an operand at the panel induction
    // vars `ivs`, rank-reducing away every panel dim the operand carries.
    auto extractTile = [&](OpBuilder &b, Location l, Value full, AffineMap m,
                           ArrayRef<Value> ivs) -> Value {
      auto t = cast<RankedTensorType>(full.getType());
      SmallVector<OpFoldResult> off(t.getRank(), b.getIndexAttr(0));
      SmallVector<OpFoldResult> sz;
      for (int64_t i = 0; i < t.getRank(); ++i)
        sz.push_back(b.getIndexAttr(t.getDimSize(i)));
      SmallVector<OpFoldResult> str(t.getRank(), b.getIndexAttr(1));
      SmallVector<bool> drop(t.getRank(), false);
      for (unsigned p = 0; p < numPanels; ++p)
        if (auto idx = findResultForDim(m, panelIterPos[p])) {
          off[*idx] = ivs[p];
          sz[*idx] = b.getIndexAttr(1);
          drop[*idx] = true;
        }
      SmallVector<int64_t> shp;
      for (int64_t i = 0; i < t.getRank(); ++i)
        if (!drop[i])
          shp.push_back(t.getDimSize(i));
      auto resTy = RankedTensorType::get(shp, t.getElementType());
      return tensor::ExtractSliceOp::create(b, l, resTy, full, off, sz, str);
    };

    AffineMap newMapA = rebuildMap(mapA);
    AffineMap newMapB = rebuildMap(mapB);
    AffineMap newMapC = rebuildMap(mapC);
    SmallVector<utils::IteratorType> newIters;
    for (unsigned i = 0, e = iteratorTypes.size(); i < e; ++i)
      if (!isPanelPos(i))
        newIters.push_back(iteratorTypes[i]);
    Block &origBody = brgemmOp->getRegion(0).front();

    // Tile the panel-group accumulator zero-init along the same panel dims as
    // the truncf epilogue below. Left whole the fill vectorizes into one
    // `dense<0.0> : vector<numPanels x 32 x 32 x f32>` splat constant; once the
    // panel is large (e.g. an 8x8 M/N panel = 65536 elements) that overflows the
    // target's build_vector operand limit (X86 SelectionDAG SDNode) and aborts
    // instruction selection. Tiled, each fill initializes a single 32x32 tile.
    Value loopInit = initOperand;
    if (hasPanel) {
      if (auto fillOp = initOperand.getDefiningOp<linalg::FillOp>()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(fillOp);
        Location fl = fillOp.getLoc();
        Value fillVal = fillOp.getInputs()[0];
        Value fillDst = fillOp.getDpsInitOperand(0)->get();

        auto fillTileOffsets = [&](OpBuilder &b, RankedTensorType t,
                                   ArrayRef<Value> ivs,
                                   SmallVector<OpFoldResult> &off,
                                   SmallVector<OpFoldResult> &sz,
                                   SmallVector<OpFoldResult> &str) {
          off.assign(t.getRank(), b.getIndexAttr(0));
          sz.clear();
          for (int64_t i = 0; i < t.getRank(); ++i)
            sz.push_back(b.getIndexAttr(t.getDimSize(i)));
          str.assign(t.getRank(), b.getIndexAttr(1));
          for (unsigned p = 0; p < numPanels; ++p) {
            off[p] = ivs[p];
            sz[p] = b.getIndexAttr(1);
          }
        };

        std::function<Value(OpBuilder &, Location, Value, SmallVector<Value> &)>
            buildFill = [&](OpBuilder &nb, Location nl, Value dstCur,
                            SmallVector<Value> &ivs) -> Value {
          if (ivs.size() == numPanels) {
            auto t = cast<RankedTensorType>(dstCur.getType());
            SmallVector<OpFoldResult> off, sz, str;
            fillTileOffsets(nb, t, ivs, off, sz, str);
            SmallVector<int64_t> shp(t.getShape().drop_front(numPanels));
            auto rTy = RankedTensorType::get(shp, t.getElementType());
            Value tile =
                tensor::ExtractSliceOp::create(nb, nl, rTy, dstCur, off, sz, str);
            Value filled =
                linalg::FillOp::create(nb, nl, ValueRange{fillVal},
                                       ValueRange{tile})
                    .getResult(0);
            return tensor::InsertSliceOp::create(nb, nl, filled, dstCur, off, sz,
                                                 str);
          }
          unsigned d = ivs.size();
          Value pLb = arith::ConstantIndexOp::create(nb, nl, 0);
          Value pUb = arith::ConstantIndexOp::create(nb, nl, panelExtent[d]);
          Value pStep = arith::ConstantIndexOp::create(nb, nl, 1);
          auto loop = scf::ForOp::create(
              nb, nl, pLb, pUb, pStep, ValueRange{dstCur},
              [&](OpBuilder &lb2, Location ll, Value pv, ValueRange lArgs) {
                SmallVector<Value> ivs2(ivs);
                ivs2.push_back(pv);
                Value res = buildFill(lb2, ll, lArgs[0], ivs2);
                scf::YieldOp::create(lb2, ll, res);
              });
          return loop.getResult(0);
        };

        SmallVector<Value> fivs;
        loopInit = buildFill(rewriter, fl, fillDst, fivs);
        rewriter.replaceOp(fillOp, loopInit);
      }
    }

    Value lb = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value ub = arith::ConstantIndexOp::create(rewriter, loc, brExtent);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, effKBlock);

    // Thread the GEMM's own accumulator (its DPS init) through the loop; each
    // block accumulates onto it (beta=1, since the GEMM body computes
    // `out + A*B`). Because this runs after spatial tiling, that accumulator is
    // the small per-tile f32/i32 tensor, kept at full precision across all
    // K-blocks. The consumer epilogue runs once, unchanged, after the loop.
    auto forOp = scf::ForOp::create(
        rewriter, loc, lb, ub, step, ValueRange{loopInit},
        [&](OpBuilder &b, Location l, Value iv, ValueRange args) {
          Value acc = args[0];
          Value slicedA = sliceBR(b, l, iv, A, aType, aDim);
          Value slicedB = sliceBR(b, l, iv, Bmat, bTy, bDim);

          if (!hasPanel) {
            IRMapping map;
            map.map(A, slicedA);
            map.map(Bmat, slicedB);
            map.map(initOperand, acc);
            Operation *blockMM = b.clone(*brgemmOp.getOperation(), map);
            scf::YieldOp::create(b, l, blockMM->getResult(0));
            return;
          }

          // Build the nested register-tile loops, one per panel dim. The K-block
          // slices (slicedA/slicedB) are hoisted above this nest, so an operand
          // that lacks a given panel dim is reused across that loop. The
          // innermost body is a single 32x32 tile GEMM.
          std::function<Value(OpBuilder &, Location, Value, SmallVector<Value> &)>
              buildNest = [&](OpBuilder &nb, Location nl, Value accCur,
                              SmallVector<Value> &ivs) -> Value {
            if (ivs.size() == numPanels) {
              Value aMi = extractTile(nb, nl, slicedA, mapA, ivs);
              Value bMi = extractTile(nb, nl, slicedB, mapB, ivs);
              // Accumulator tile: offsets/sizes needed for the write-back.
              SmallVector<OpFoldResult> cOff(outTy.getRank(),
                                             nb.getIndexAttr(0));
              SmallVector<OpFoldResult> cSz;
              for (int64_t i = 0; i < outTy.getRank(); ++i)
                cSz.push_back(nb.getIndexAttr(outTy.getDimSize(i)));
              SmallVector<OpFoldResult> cStr(outTy.getRank(),
                                             nb.getIndexAttr(1));
              SmallVector<int64_t> cShp;
              for (unsigned p = 0; p < numPanels; ++p) {
                unsigned idx = *findResultForDim(mapC, panelIterPos[p]);
                cOff[idx] = ivs[p];
                cSz[idx] = nb.getIndexAttr(1);
              }
              for (int64_t i = 0; i < outTy.getRank(); ++i)
                if (i >= static_cast<int64_t>(numPanels))
                  cShp.push_back(outTy.getDimSize(i));
              auto accMiType =
                  RankedTensorType::get(cShp, outTy.getElementType());
              Value accMi = tensor::ExtractSliceOp::create(
                  nb, nl, accMiType, accCur, cOff, cSz, cStr);
              auto g = linalg::GenericOp::create(
                  nb, nl, TypeRange{accMiType}, ValueRange{aMi, bMi},
                  ValueRange{accMi},
                  ArrayRef<AffineMap>{newMapA, newMapB, newMapC}, newIters,
                  [&](OpBuilder &gb, Location gl, ValueRange bargs) {
                    IRMapping bmap;
                    for (auto [oldArg, newArg] :
                         llvm::zip(origBody.getArguments(), bargs))
                      bmap.map(oldArg, newArg);
                    for (Operation &op : origBody.without_terminator())
                      gb.clone(op, bmap);
                    auto yield =
                        cast<linalg::YieldOp>(origBody.getTerminator());
                    SmallVector<Value> yielded;
                    for (Value v : yield.getOperands())
                      yielded.push_back(bmap.lookupOrDefault(v));
                    linalg::YieldOp::create(gb, gl, yielded);
                  });
              return tensor::InsertSliceOp::create(nb, nl, g.getResult(0),
                                                   accCur, cOff, cSz, cStr);
            }
            unsigned d = ivs.size();
            Value pLb = arith::ConstantIndexOp::create(nb, nl, 0);
            Value pUb = arith::ConstantIndexOp::create(nb, nl, panelExtent[d]);
            Value pStep = arith::ConstantIndexOp::create(nb, nl, 1);
            auto loop = scf::ForOp::create(
                nb, nl, pLb, pUb, pStep, ValueRange{accCur},
                [&](OpBuilder &lb2, Location ll, Value pv, ValueRange lArgs) {
                  SmallVector<Value> ivs2(ivs);
                  ivs2.push_back(pv);
                  Value res = buildNest(lb2, ll, lArgs[0], ivs2);
                  scf::YieldOp::create(lb2, ll, res);
                });
            return loop.getResult(0);
          };

          SmallVector<Value> ivs;
          scf::YieldOp::create(b, l, buildNest(b, l, acc, ivs));
        });

    rewriter.replaceOp(brgemmOp, forOp.getResult(0));

    // Tile the fused f32->bf16 truncf epilogue along the same panel dims. Left
    // whole it vectorizes into one `vector<numPanels x 32 x 32>` op that spills
    // catastrophically; tiled, each iteration truncs a single 32x32 C tile
    // (which downstream lowering further tiles to rows, like the non-panel
    // path). The panel dims are the leading dims of the accumulator/C tensors.
    if (hasPanel) {
      Value accFull = forOp.getResult(0);
      linalg::GenericOp consumer;
      for (Operation *user : accFull.getUsers()) {
        auto g = dyn_cast<linalg::GenericOp>(user);
        if (g && g.getNumParallelLoops() == g.getNumLoops() &&
            g.getNumDpsInputs() == 1 && g.getNumDpsInits() == 1 &&
            g.getDpsInputOperand(0)->get() == accFull) {
          consumer = g;
          break;
        }
      }
      if (consumer) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(consumer);
        Location cl = consumer.getLoc();
        Value cInit = consumer.getDpsInitOperand(0)->get();
        Block &cBody = consumer->getRegion(0).front();

        // Slice off the leading `numPanels` dims of `v` at `ivs` (size 1,
        // rank-reduced), keeping the trailing 32x32 tile.
        auto tileOffsets = [&](OpBuilder &b, RankedTensorType t,
                               ArrayRef<Value> ivs,
                               SmallVector<OpFoldResult> &off,
                               SmallVector<OpFoldResult> &sz,
                               SmallVector<OpFoldResult> &str) {
          off.assign(t.getRank(), b.getIndexAttr(0));
          sz.clear();
          for (int64_t i = 0; i < t.getRank(); ++i)
            sz.push_back(b.getIndexAttr(t.getDimSize(i)));
          str.assign(t.getRank(), b.getIndexAttr(1));
          for (unsigned p = 0; p < numPanels; ++p) {
            off[p] = ivs[p];
            sz[p] = b.getIndexAttr(1);
          }
        };
        auto sliceTile = [&](OpBuilder &b, Location l, Value v,
                             ArrayRef<Value> ivs) -> Value {
          auto t = cast<RankedTensorType>(v.getType());
          SmallVector<OpFoldResult> off, sz, str;
          tileOffsets(b, t, ivs, off, sz, str);
          SmallVector<int64_t> shp(t.getShape().drop_front(numPanels));
          auto rTy = RankedTensorType::get(shp, t.getElementType());
          return tensor::ExtractSliceOp::create(b, l, rTy, v, off, sz, str);
        };
        AffineMap idMap =
            rewriter.getMultiDimIdentityMap(2 /*32x32 tile*/);
        SmallVector<utils::IteratorType> tileIters(
            2, utils::IteratorType::parallel);

        std::function<Value(OpBuilder &, Location, Value, SmallVector<Value> &)>
            buildTrunc = [&](OpBuilder &nb, Location nl, Value cCur,
                             SmallVector<Value> &ivs) -> Value {
          if (ivs.size() == numPanels) {
            Value accTile = sliceTile(nb, nl, accFull, ivs);
            Value cTile = sliceTile(nb, nl, cCur, ivs);
            auto tileTy = cast<RankedTensorType>(cTile.getType());
            auto g = linalg::GenericOp::create(
                nb, nl, TypeRange{tileTy}, ValueRange{accTile},
                ValueRange{cTile}, ArrayRef<AffineMap>{idMap, idMap},
                tileIters,
                [&](OpBuilder &gb, Location gl, ValueRange bargs) {
                  IRMapping bmap;
                  for (auto [o, n] : llvm::zip(cBody.getArguments(), bargs))
                    bmap.map(o, n);
                  for (Operation &op : cBody.without_terminator())
                    gb.clone(op, bmap);
                  auto y = cast<linalg::YieldOp>(cBody.getTerminator());
                  SmallVector<Value> yv;
                  for (Value v : y.getOperands())
                    yv.push_back(bmap.lookupOrDefault(v));
                  linalg::YieldOp::create(gb, gl, yv);
                });
            auto cTy = cast<RankedTensorType>(cCur.getType());
            SmallVector<OpFoldResult> off, sz, str;
            tileOffsets(nb, cTy, ivs, off, sz, str);
            return tensor::InsertSliceOp::create(nb, nl, g.getResult(0), cCur,
                                                 off, sz, str);
          }
          unsigned d = ivs.size();
          Value pLb = arith::ConstantIndexOp::create(nb, nl, 0);
          Value pUb = arith::ConstantIndexOp::create(nb, nl, panelExtent[d]);
          Value pStep = arith::ConstantIndexOp::create(nb, nl, 1);
          auto loop = scf::ForOp::create(
              nb, nl, pLb, pUb, pStep, ValueRange{cCur},
              [&](OpBuilder &lb2, Location ll, Value pv, ValueRange lArgs) {
                SmallVector<Value> ivs2(ivs);
                ivs2.push_back(pv);
                Value res = buildTrunc(lb2, ll, lArgs[0], ivs2);
                scf::YieldOp::create(lb2, ll, res);
              });
          return loop.getResult(0);
        };

        SmallVector<Value> ivs;
        Value tiled = buildTrunc(rewriter, cl, cInit, ivs);
        rewriter.replaceOp(consumer, tiled);
      }
    }
    return success();
  }

private:
  unsigned kCacheBlocking;
};

struct GemmKCacheBlocking
    : public tpp::impl::GemmKCacheBlockingBase<GemmKCacheBlocking> {

  using GemmKCacheBlockingBase::GemmKCacheBlockingBase;

  void runOnOperation() override {
    if (kCacheBlocking == 0)
      return;

    RewritePatternSet patterns(&getContext());
    patterns.add<KCacheBlockingTiling<linalg::GenericOp>,
                 KCacheBlockingTiling<linalg::BatchReduceMatmulOp>>(
        &getContext(), kCacheBlocking);
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);

    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};

} // namespace
