//===- NanoGemmKCacheBlocking.cpp --------------------------------*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Cache-block the reduction (K) dimension of a nano-kernel (AMX) GEMM AFTER the
// vector.contract has already been lowered to x86.amx.* ops. The batch-reduce
// reduction loop that carries the AMX accumulator tiles is split by a
// cache-block factor and the resulting outer K-block loop is hoisted OUTSIDE
// the spatial scf.forall, so the A/B panel of a K-block stays L2-resident while
// the whole tile grid is swept. The output tile C becomes the cross-block
// accumulator (beta=1): the first K-block seeds a zero bias, every later block
// adds the running accumulator into the partial before writing C back. For the
// bf16 path C itself carries the running sum (extf to f32); for the quantized
// i8->i8 path C cannot hold the wide partial and requant is non-linear, so a
// dedicated full-grid i32 carrier threads the running sum across K-blocks and
// the last block requantizes the full reduction. The non-quant i8->i32 path
// already accumulates into its i32 C, so only the reduction is narrowed.
//
//===----------------------------------------------------------------------===//
#include "TPP/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "nano-gemm-k-cache-blocking"

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_NANOGEMMKCACHEBLOCKING
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace mlir::tpp;

namespace {

// True when `op` is an x86 AMX op with the given mnemonic (checked by name so
// the pass does not need to depend on the vendored X86 dialect headers).
static bool isAmxOp(Operation *op, StringRef mnemonic) {
  return op->getName().getStringRef() == mnemonic;
}

// The nano GEMM tile lowered inside a single scf.forall iteration. For the
// float (bf16) path all fields must be found; the integer (i8->i32) path only
// needs the reduction loop because its epilogue already accumulates into C.
struct NanoGemmTile {
  scf::ForOp brFor;         // batch-reduce reduction loop (carries AMX tiles)
  scf::ForOp biasFor;       // per-row loop adding the bias into the f32 partial
  memref::SubViewOp cTile;  // the 32x32 bf16 output tile subview
  Value partialBuf;         // f32 scratch the AMX tiles are stored into
  SmallVector<vector::LoadOp> biasLoads; // loads of the (zero) bias to rewrite
  int64_t brExtent = 0;     // number of batch-reduce blocks (brFor upper bound)
  bool isInteger = false;   // true for the i8 (tile_muli) path
  bool isQuant = false;     // true for the quantized i8->i8 requant path
};

// Locate the blockable nano GEMM tile inside `forall`, or return failure.
static FailureOr<NanoGemmTile> matchNanoGemmTile(scf::ForallOp forall) {
  NanoGemmTile tile;

  // The reduction loop is the scf.for whose body issues tile_mulf (bf16) or
  // tile_muli (i8->i32).
  forall.walk([&](scf::ForOp f) {
    if (tile.brFor)
      return;
    for (Operation &op : f.getBody()->without_terminator()) {
      if (isAmxOp(&op, "x86.amx.tile_mulf")) {
        tile.brFor = f;
        tile.isInteger = false;
        return;
      }
      if (isAmxOp(&op, "x86.amx.tile_muli")) {
        tile.brFor = f;
        tile.isInteger = true;
        return;
      }
    }
  });
  if (!tile.brFor)
    return failure();

  // A constant, zero-based, unit-step reduction loop is required so the split
  // into an outer K-block loop is exact.
  auto lb = getConstantIntValue(tile.brFor.getLowerBound());
  auto ub = getConstantIntValue(tile.brFor.getUpperBound());
  auto step = getConstantIntValue(tile.brFor.getStep());
  if (!lb || !ub || !step || *lb != 0 || *step != 1)
    return failure();
  tile.brExtent = *ub;

  // Distinguish the two integer pathss normal or quant.
  if (tile.isInteger) {
    bool hasRequant = false;
    forall.walk([&](arith::FPToSIOp) { hasRequant = true; });
    if (!hasRequant)
      return tile;
    tile.isQuant = true;
  }

  // The AMX tiles are stored into a scratch buffer (tile_store operand 0): f32
  // for bf16, i32 for quantized i8.
  forall.walk([&](Operation *op) {
    if (isAmxOp(op, "x86.amx.tile_store"))
      tile.partialBuf = op->getOperand(0);
  });
  if (!tile.partialBuf)
    return failure();

  // The bias-add loop reads the partial buffer plus a bias buffer, adds them
  // and writes the sum back to the partial buffer. The bias loads are the
  // ones re-sourced from the accumulator for beta=1.
  forall.walk([&](scf::ForOp f) {
    if (f == tile.brFor)
      return;
    bool loadsPartial = false;
    SmallVector<vector::LoadOp> others;
    for (auto ld : f.getBody()->getOps<vector::LoadOp>()) {
      if (ld.getBase() == tile.partialBuf)
        loadsPartial = true;
      else
        others.push_back(ld);
    }
    bool hasAdd = tile.isQuant
                      ? !f.getBody()->getOps<arith::AddIOp>().empty()
                      : !f.getBody()->getOps<arith::AddFOp>().empty();
    if (loadsPartial && hasAdd && !others.empty()) {
      tile.biasFor = f;
      tile.biasLoads = others;
    }
  });
  if (!tile.biasFor)
    return failure();

  // The output tile is the 2-D subview at the top level of the forall body
  // bf16 for the float path, i8 for the quantized path.
  for (auto sv : forall.getBody()->getOps<memref::SubViewOp>()) {
    auto ty = dyn_cast<MemRefType>(sv.getType());
    if (!ty || ty.getRank() != 2)
      continue;
    if (tile.isQuant ? ty.getElementType().isInteger(8)
                     : ty.getElementType().isBF16())
      tile.cTile = sv;
  }
  if (!tile.cTile)
    return failure();

  return tile;
}

// Rewrite one nano GEMM tile into a K-cache-blocked form.
static void blockNanoGemmTile(scf::ForallOp forall, NanoGemmTile &tile,
                              unsigned kCacheTile) {
  Location loc = forall.getLoc();
  OpBuilder b(forall);

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
  Value cUB = arith::ConstantIndexOp::create(b, loc, tile.brExtent);
  Value cKB = arith::ConstantIndexOp::create(b, loc, kCacheTile);

  // The quantized path needs a full-grid i32 carrier that survives across
  // K-blocks.
  Value accBuf;
  if (tile.isQuant) {
    auto srcTy = cast<MemRefType>(tile.cTile.getSource().getType());
    auto accTy = MemRefType::get(srcTy.getShape(), b.getI32Type());
    accBuf = memref::AllocOp::create(b, loc, accTy);
  }

  // Outer K-block loop that encloses the whole spatial tile sweep.
  auto kLoop = scf::ForOp::create(b, loc, c0, cUB, cKB);
  Value kb = kLoop.getInductionVar();
  forall->moveBefore(kLoop.getBody()->getTerminator());

  if (accBuf) {
    OpBuilder db(kLoop);
    db.setInsertionPointAfter(kLoop);
    memref::DeallocOp::create(db, loc, accBuf);
  }

  // Restrict the reduction loop to the current K-block: [kb, kb + kCacheTile).
  {
    OpBuilder rb(tile.brFor);
    Value kbEnd = arith::AddIOp::create(rb, loc, kb, cKB);
    tile.brFor.getLowerBoundMutable().assign(kb);
    tile.brFor.getUpperBoundMutable().assign(kbEnd);
  }

  // The non-quant i8->i32 epilogue already accumulates into the i32 output C,
  // hoisting the K-loop and narrowing the reduction is sufficient.
  if (tile.isInteger && !tile.isQuant)
    return;

  OpBuilder eb(tile.biasFor);
  Value isFirst =
      arith::CmpIOp::create(eb, loc, arith::CmpIPredicate::eq, kb, c0);

  if (tile.isQuant) {
    // beta=1 against the dedicated i32 carrier: seed zero on the first K-block,
    // otherwise re-load the running accumulator, and mirror the summed partial
    // back into the carrier. The requant epilogue reads the running sum, so the
    // last K-block requantizes the full K reduction.
    auto cI8Ty = cast<MemRefType>(tile.cTile.getType());
    auto accSvTy = MemRefType::get(cI8Ty.getShape(), eb.getI32Type(),
                                   cI8Ty.getLayout(), cI8Ty.getMemorySpace());
    auto accSv = memref::SubViewOp::create(
        eb, loc, accSvTy, accBuf, tile.cTile.getMixedOffsets(),
        tile.cTile.getMixedSizes(), tile.cTile.getMixedStrides());

    for (vector::LoadOp bl : tile.biasLoads) {
      OpBuilder lb(bl);
      auto vecTy = cast<VectorType>(bl.getType());
      Value accLoad = vector::LoadOp::create(lb, loc, vecTy, accSv.getResult(),
                                             bl.getIndices());
      Value zero = arith::ConstantOp::create(
          lb, loc, vecTy, DenseElementsAttr::get(vecTy, APInt(32, 0)));
      Value bias = arith::SelectOp::create(lb, loc, isFirst, zero, accLoad);
      bl.getResult().replaceAllUsesWith(bias);
      bl.erase();
    }

    SmallVector<vector::StoreOp> partialStores;
    for (auto st : tile.biasFor.getBody()->getOps<vector::StoreOp>())
      if (st.getBase() == tile.partialBuf)
        partialStores.push_back(st);
    for (vector::StoreOp st : partialStores) {
      OpBuilder sb(st);
      sb.setInsertionPointAfter(st);
      vector::StoreOp::create(sb, loc, st.getValueToStore(), accSv.getResult(),
                              st.getIndices());
    }
    return;
  }

  // bf16 beta=1: replace each zero-bias load with the running C tile (extf),
  // except on the first K-block where the bias is zero.
  Operation *cTileClone = eb.clone(*tile.cTile.getOperation());
  Value cTileVal = cTileClone->getResult(0);

  for (vector::LoadOp bl : tile.biasLoads) {
    OpBuilder lb(bl);
    auto f32VecTy = cast<VectorType>(bl.getType());
    auto bf16VecTy = VectorType::get(f32VecTy.getShape(), lb.getBF16Type());
    Value cLoad = vector::LoadOp::create(lb, loc, bf16VecTy, cTileVal,
                                         bl.getIndices());
    Value cExt = arith::ExtFOp::create(lb, loc, f32VecTy, cLoad);
    Value zero = arith::ConstantOp::create(
        lb, loc, f32VecTy,
        DenseElementsAttr::get(f32VecTy, APFloat(0.0f)));
    Value bias = arith::SelectOp::create(lb, loc, isFirst, zero, cExt);
    bl.getResult().replaceAllUsesWith(bias);
    bl.erase();
  }
}

struct NanoGemmKCacheBlocking
    : public tpp::impl::NanoGemmKCacheBlockingBase<NanoGemmKCacheBlocking> {
  using NanoGemmKCacheBlockingBase::NanoGemmKCacheBlockingBase;

  void runOnOperation() override {
    if (kCacheTile == 0)
      return;

    SmallVector<std::pair<scf::ForallOp, NanoGemmTile>> targets;
    getOperation()->walk([&](scf::ForallOp forall) {
      auto tile = matchNanoGemmTile(forall);
      if (failed(tile))
        return;
      // A block that spans (or exceeds) the whole K extent is a no-op; leave
      // such tiles as the unblocked baseline. Only exact divisors are handled.
      if (kCacheTile >= (unsigned)tile->brExtent ||
          tile->brExtent % (int64_t)kCacheTile != 0)
        return;
      targets.emplace_back(forall, *tile);
    });

    for (auto &[forall, tile] : targets)
      blockNanoGemmTile(forall, tile, kCacheTile);
  }
};

} // namespace
