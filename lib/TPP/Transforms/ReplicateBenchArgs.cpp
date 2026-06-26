//===- ReplicateBenchArgs.cpp -----------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replicate benchmark kernel arguments for cold-cache timing.
//
// Runs after bufferization on the benchmark wrapper produced by tpp-run. The
// single kernel call inside every `perf.bench` region is wrapped in an
// `scf.for` loop over a new "replica" dimension. For each kernel argument a
// flat `i8` global of `factor * sizeof(arg)` bytes is allocated once; every
// iteration feeds the kernel a distinct `memref.view` into that buffer (pure
// pointer arithmetic, no allocation or copy). `memref.view` is used instead of
// `memref.subview` on purpose: it always yields an identity-layout, offset-0
// result that exactly matches the original (contiguous) argument type, so any
// layout-sensitive ops inside the kernel (e.g. the `memref.expand_shape`
// introduced by bf16 VNNI packing) keep verifying. A strided subview with a
// dynamic offset would instead change the argument layout and break them.
//
// This mirrors the "n_layers" replication used by libxsmm's cold-cache GEMM
// benchmark: the same problem is run on different memory so caches stay cold.
//
// Each float buffer is filled once, before any timed region. By default it is
// filled with the constant 1.0; with random initialization enabled (the
// `random-init` option or the `tpp.bench_replication_random_init` module
// attribute) it is instead filled with PRNG-generated values in [1, 2).
// All-zero inputs let the FMA units run at an unrealistically high clock, so
// neither default value is zero.
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "TPP/Dialect/Perf/PerfOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_REPLICATEBENCHARGS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

constexpr StringLiteral kReplicationFactorAttr = "tpp.bench_replication_factor";
constexpr StringLiteral kReplicationRandomInitAttr =
    "tpp.bench_replication_random_init";

// Emit a cheap counter-based PRNG that maps a linear element index to a
// floating-point value in [1, 2). The [1, 2) range guarantees normal (non-zero,
// non-denormal, finite) values, so the FMA units are exercised realistically
// without the risk of NaN/Inf slowdowns. f32 and bf16 are handled explicitly;
// for any other float type it falls back to a constant 1.0.
static Value emitRandomFloat(OpBuilder &b, Location loc, Value linearIdx,
                             Type elemTy) {
  Type i32Ty = b.getI32Type();
  auto c32 = [&](uint32_t v) -> Value {
    return arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(i32Ty, static_cast<int32_t>(v)));
  };

  // MurmurHash3-style finalizer using modulo-2^32 arithmetic.
  Value h = arith::IndexCastOp::create(b, loc, i32Ty, linearIdx);
  h = arith::MulIOp::create(b, loc, h, c32(0x9E3779B1u));
  h = arith::XOrIOp::create(b, loc, h,
                            arith::ShRUIOp::create(b, loc, h, c32(16)));
  h = arith::MulIOp::create(b, loc, h, c32(0x85EBCA77u));
  h = arith::XOrIOp::create(b, loc, h,
                            arith::ShRUIOp::create(b, loc, h, c32(13)));

  if (elemTy.isF32()) {
    // mantissa = h & 0x7FFFFF; bits = 0x3F800000 | mantissa -> [1, 2).
    Value mant = arith::AndIOp::create(b, loc, h, c32(0x7FFFFFu));
    Value bits = arith::OrIOp::create(b, loc, mant, c32(0x3F800000u));
    return arith::BitcastOp::create(b, loc, elemTy, bits);
  }
  if (elemTy.isBF16()) {
    // bf16 is the top 16 bits of an f32; build the [1, 2) pattern directly.
    Type i16Ty = b.getIntegerType(16);
    Value h16 = arith::TruncIOp::create(b, loc, i16Ty, h);
    auto c16 = [&](uint16_t v) -> Value {
      return arith::ConstantOp::create(
          b, loc, b.getIntegerAttr(i16Ty, static_cast<int16_t>(v)));
    };
    Value mant = arith::AndIOp::create(b, loc, h16, c16(0x7Fu));
    Value bits = arith::OrIOp::create(b, loc, mant, c16(0x3F80u));
    return arith::BitcastOp::create(b, loc, elemTy, bits);
  }
  return arith::ConstantOp::create(b, loc, b.getFloatAttr(elemTy, 1.0));
}

struct ReplicateBenchArgs
    : public tpp::impl::ReplicateBenchArgsBase<ReplicateBenchArgs> {
  using ReplicateBenchArgsBase::ReplicateBenchArgsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Resolve the replication factor: command-line option wins, otherwise read
    // the attribute stamped by the benchmark producer.
    int64_t factor = replicationFactor;
    if (factor <= 0) {
      if (auto attr =
              module->getAttrOfType<IntegerAttr>(kReplicationFactorAttr))
        factor = attr.getInt();
    }
    module->removeAttr(kReplicationFactorAttr);

    // Resolve random initialization: command-line option OR module attribute.
    bool doRandomInit = randomInit || module->hasAttr(kReplicationRandomInitAttr);
    module->removeAttr(kReplicationRandomInitAttr);

    if (factor <= 1)
      return;

    // Collect the benchmark timing regions.
    SmallVector<perf::BenchOp> benches;
    module.walk([&](perf::BenchOp bench) { benches.push_back(bench); });
    if (benches.empty())
      return;

    // Identify the benchmarked kernel from the first call inside any bench.
    func::FuncOp kernel;
    for (auto bench : benches) {
      bench.getBodyRegion().walk([&](func::CallOp call) {
        if (!kernel)
          kernel = module.lookupSymbol<func::FuncOp>(call.getCalleeAttr());
      });
      if (kernel)
        break;
    }
    if (!kernel) {
      module.emitError("replicate-bench-args: no kernel call found in perf.bench");
      return signalPassFailure();
    }

    MLIRContext *ctx = module.getContext();
    Location loc = kernel.getLoc();
    auto origInputs = kernel.getFunctionType().getInputs();

    // For each kernel argument, allocate a flat i8 global large enough to hold
    // `factor` contiguous copies of the argument. The buffer is zero
    // initialized: the all-zero byte pattern reinterprets to +0.0 for floats
    // and 0 for integers, both of which are normal values that avoid the
    // denormal/NaN penalties that uninitialized memory could introduce. The
    // float buffers are then overwritten at runtime (see below) with 1.0 by
    // default, or random values when `doRandomInit` is set; the zero global
    // init still serves as a safe default for any element type the runtime
    // fill does not cover (e.g. integers).
    OpBuilder globalBuilder(ctx);
    globalBuilder.setInsertionPointToStart(module.getBody());
    Type i8Ty = globalBuilder.getI8Type();
    SmallVector<StringRef> globalNames(origInputs.size());
    SmallVector<int64_t> replicaByteSizes(origInputs.size());
    SmallVector<int64_t> totalElemCounts(origInputs.size());
    auto alignment = globalBuilder.getI64IntegerAttr(128);
    for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
      auto memrefTy = dyn_cast<MemRefType>(inTy);
      if (!memrefTy || !memrefTy.hasStaticShape()) {
        module.emitError("replicate-bench-args: kernel arguments must be "
                         "statically shaped memrefs");
        return signalPassFailure();
      }
      if (!memrefTy.getLayout().isIdentity()) {
        module.emitError("replicate-bench-args: kernel arguments must have "
                         "identity layout");
        return signalPassFailure();
      }
      Type elemTy = memrefTy.getElementType();
      if (!elemTy.isIntOrFloat()) {
        module.emitError("replicate-bench-args: unsupported element type");
        return signalPassFailure();
      }

      int64_t numElements = 1;
      for (int64_t d : memrefTy.getShape())
        numElements *= d;
      int64_t eltBytes = (elemTy.getIntOrFloatBitWidth() + 7) / 8;
      int64_t replicaBytes = numElements * eltBytes;
      replicaByteSizes[idx] = replicaBytes;
      totalElemCounts[idx] = numElements * factor;
      int64_t totalBytes = replicaBytes * factor;

      auto flatTy = MemRefType::get({totalBytes}, i8Ty);
      auto tensorTy = RankedTensorType::get({totalBytes}, i8Ty);
      auto initAttr = DenseElementsAttr::get(
          tensorTy, globalBuilder.getIntegerAttr(i8Ty, 0));

      std::string name = "__bench_replica_" + std::to_string(idx);
      auto global = memref::GlobalOp::create(
          globalBuilder, loc, name, globalBuilder.getStringAttr("private"),
          flatTy, initAttr, /*constant=*/false, alignment);
      globalNames[idx] = global.getName();
    }

    // Fill each replicated float buffer once, before any timed region. All-zero
    // inputs let the FMA units run at an unrealistically high clock, so by
    // default every float buffer is filled with the constant 1.0; with random
    // initialization enabled it is instead filled with random values in
    // [1, 2), which exercises the units more realistically. The fill loop sits
    // before the first perf.bench so it is never timed. Integer/other buffers
    // keep the safe zero initialization.
    {
      OpBuilder initBuilder(benches.front());
      Value c0 = arith::ConstantIndexOp::create(initBuilder, loc, 0);
      Value c1 = arith::ConstantIndexOp::create(initBuilder, loc, 1);
      for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
        auto memrefTy = cast<MemRefType>(inTy);
        Type elemTy = memrefTy.getElementType();
        // Only float buffers are filled; integer/other buffers keep the safe
        // zero initialization.
        if (!isa<FloatType>(elemTy))
          continue;

        auto flatTy = cast<MemRefType>(
            cast<memref::GlobalOp>(module.lookupSymbol(globalNames[idx]))
                .getType());
        Value flat = memref::GetGlobalOp::create(initBuilder, loc, flatTy,
                                                 globalNames[idx]);
        // Typed view over the whole buffer: memref<totalElems x elemTy>.
        auto viewTy = MemRefType::get({totalElemCounts[idx]}, elemTy);
        Value typedView = memref::ViewOp::create(initBuilder, loc, viewTy, flat,
                                                 c0, /*sizes=*/ValueRange{});
        Value ub = arith::ConstantIndexOp::create(initBuilder, loc,
                                                  totalElemCounts[idx]);
        auto fill = scf::ForOp::create(initBuilder, loc, c0, ub, c1);
        OpBuilder fb(fill.getBody(), fill.getBody()->begin());
        Value iv = fill.getInductionVar();
        Value v = doRandomInit
                      ? emitRandomFloat(fb, loc, iv, elemTy)
                      : arith::ConstantOp::create(
                            fb, loc, fb.getFloatAttr(elemTy, 1.0));
        memref::StoreOp::create(fb, loc, v, typedView, ValueRange{iv});
      }
    }

    // Wrap every kernel call inside a perf.bench in a replication loop.
    for (auto bench : benches) {
      SmallVector<func::CallOp> calls;
      bench.getBodyRegion().walk([&](func::CallOp call) {
        if (call.getCallee() == kernel.getSymName())
          calls.push_back(call);
      });

      for (func::CallOp call : calls) {
        OpBuilder builder(call);

        // Hoist the flat global handles out of the loop; they are invariant.
        SmallVector<Value> globals(origInputs.size());
        for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
          auto flatTy = cast<MemRefType>(
              cast<memref::GlobalOp>(module.lookupSymbol(globalNames[idx]))
                  .getType());
          globals[idx] = memref::GetGlobalOp::create(builder, loc, flatTy,
                                                      globalNames[idx]);
        }

        Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
        Value one = arith::ConstantIndexOp::create(builder, loc, 1);
        Value ub = arith::ConstantIndexOp::create(builder, loc, factor);
        auto loop = scf::ForOp::create(builder, loc, zero, ub, one);

        OpBuilder bodyBuilder(loop.getBody(), loop.getBody()->begin());
        Value iv = loop.getInductionVar();

        SmallVector<Value> viewArgs(origInputs.size());
        for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
          auto memrefTy = cast<MemRefType>(inTy);
          // Byte offset of replica `iv`: iv * sizeof(arg).
          Value replicaBytes = arith::ConstantIndexOp::create(
              bodyBuilder, loc, replicaByteSizes[idx]);
          Value byteShift =
              arith::MulIOp::create(bodyBuilder, loc, iv, replicaBytes);
          // memref.view yields an identity-layout, offset-0 memref that matches
          // the original argument type exactly.
          viewArgs[idx] = memref::ViewOp::create(
              bodyBuilder, loc, memrefTy, globals[idx], byteShift,
              /*sizes=*/ValueRange{});
        }

        func::CallOp::create(bodyBuilder, loc, kernel, viewArgs);
        call.erase();
      }
    }
  }
};

} // namespace
