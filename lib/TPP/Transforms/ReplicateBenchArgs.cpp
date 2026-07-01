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
// single kernel call inside non-warmup `perf.bench` region is wrapped in an
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
// Before any timed region, each replica slot is seeded by copying the real
// benchmark data from the corresponding `__wrapper_*` global that tpp-run built
// for the kernel argument.
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

    // The random-init attribute is obsolete: replicas are now initialized by
    // copying the real benchmark data from the __wrapper_* globals. Drop it so
    // it does not leak into later passes.
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

    Location loc = kernel.getLoc();
    auto origInputs = kernel.getFunctionType().getInputs();

    // Trace the original kernel call's operands back to their source globals.
    // tpp-run's benchmark wrapper feeds each kernel argument from a
    // `memref.get_global @__wrapper_*`; capturing those names lets us seed every
    // replica with the same real benchmark data (crucial for the i8 inputs,
    // which would otherwise stay all-zero). Output/scratch operands produced by
    // an alloc rather than a global leave an empty entry and are skipped.
    SmallVector<StringRef> srcGlobalNames(origInputs.size());
    {
      func::CallOp origCall;
      for (auto bench : benches) {
        bench.getBodyRegion().walk([&](func::CallOp call) {
          if (!origCall && call.getCallee() == kernel.getSymName())
            origCall = call;
        });
        if (origCall)
          break;
      }
      if (origCall) {
        for (auto [idx, operand] : llvm::enumerate(origCall.getOperands())) {
          if (idx >= srcGlobalNames.size())
            break;
          if (auto gg = operand.getDefiningOp<memref::GetGlobalOp>())
            srcGlobalNames[idx] = gg.getName();
        }
      }
    }

    // For each kernel argument, allocate at runtime a flat i8 buffer large
    // enough to hold `factor` contiguous copies of the argument. A runtime
    // `memref.alloc` (heap allocation) is used instead of a static
    // `memref.global`. Each replica is then overwritten with a copy of the
    // corresponding __wrapper_* global so every replica carries the real
    // benchmark data. The allocs are placed before the first perf.bench so they
    // dominate both the seeding loop and the timed region.
    OpBuilder allocBuilder(benches.front());
    Type i8Ty = allocBuilder.getI8Type();
    SmallVector<Value> replicaBufs(origInputs.size());
    SmallVector<int64_t> replicaStrides(origInputs.size());
    auto alignmentAttr = allocBuilder.getI64IntegerAttr(128);
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
      // Replicas are laid out back-to-back; the stride from one replica to the
      // next is exactly its own byte size.
      replicaStrides[idx] = replicaBytes;
      int64_t totalBytes = replicaBytes * factor;

      auto flatTy = MemRefType::get({totalBytes}, i8Ty);
      replicaBufs[idx] =
          memref::AllocOp::create(allocBuilder, loc, flatTy, alignmentAttr);
    }

    // Seed every replica with the real benchmark data by copying the
    // corresponding __wrapper_* global into each replica slot, before any timed
    // region. This mirrors sfc_ca_gemm, which fills each of its cold-cache
    // buffers with the same initialized data. The copy loop sits before the
    // first perf.bench so it is never timed. Any argument without a source
    // global (e.g. an alloc'd output) is left untouched.
    {
      OpBuilder initBuilder(benches.front());
      Value c0 = arith::ConstantIndexOp::create(initBuilder, loc, 0);
      Value c1 = arith::ConstantIndexOp::create(initBuilder, loc, 1);
      for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
        // Skip arguments whose data does not come from a global.
        if (srcGlobalNames[idx].empty())
          continue;
        auto memrefTy = cast<MemRefType>(inTy);

        Value flat = replicaBufs[idx];
        Value src = memref::GetGlobalOp::create(initBuilder, loc, memrefTy,
                                                srcGlobalNames[idx]);
        Value strideVal = arith::ConstantIndexOp::create(
            initBuilder, loc, replicaStrides[idx]);
        Value factorVal =
            arith::ConstantIndexOp::create(initBuilder, loc, factor);

        // Outer loop over replicas: copy the source global into each slot.
        auto repLoop = scf::ForOp::create(initBuilder, loc, c0, factorVal, c1);
        OpBuilder rb(repLoop.getBody(), repLoop.getBody()->begin());
        Value repIdx = repLoop.getInductionVar();
        Value byteOffset = arith::MulIOp::create(rb, loc, repIdx, strideVal);

        // Typed view over this replica, matching the source's type exactly.
        Value replicaView = memref::ViewOp::create(rb, loc, memrefTy, flat,
                                                   byteOffset,
                                                   /*sizes=*/ValueRange{});
        memref::CopyOp::create(rb, loc, src, replicaView);
      }
    }

    // Wrap the kernel call in a replication loop, but only inside the timed
    // benchmark region which is the last perf.bench.
    perf::BenchOp timedBench = benches.back();
    {
      auto bench = timedBench;
      SmallVector<func::CallOp> calls;
      bench.getBodyRegion().walk([&](func::CallOp call) {
        if (call.getCallee() == kernel.getSymName())
          calls.push_back(call);
      });

      for (func::CallOp call : calls) {
        OpBuilder builder(call);

        Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
        Value one = arith::ConstantIndexOp::create(builder, loc, 1);
        Value ub = arith::ConstantIndexOp::create(builder, loc, factor);
        auto loop = scf::ForOp::create(builder, loc, zero, ub, one);

        OpBuilder bodyBuilder(loop.getBody(), loop.getBody()->begin());
        Value iv = loop.getInductionVar();

        SmallVector<Value> viewArgs(origInputs.size());
        for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
          auto memrefTy = cast<MemRefType>(inTy);
          // Byte offset of replica `iv`: iv * replicaStride (contiguous).
          Value replicaStride = arith::ConstantIndexOp::create(
              bodyBuilder, loc, replicaStrides[idx]);
          Value byteShift =
              arith::MulIOp::create(bodyBuilder, loc, iv, replicaStride);
          // memref.view yields an identity-layout, offset-0 memref that matches
          // the original argument type exactly.
          viewArgs[idx] = memref::ViewOp::create(
              bodyBuilder, loc, memrefTy, replicaBufs[idx], byteShift,
              /*sizes=*/ValueRange{});
        }

        func::CallOp::create(bodyBuilder, loc, kernel, viewArgs);
        call.erase();
      }
    }

    // Free the replica buffers after the timed region so the heap allocation is
    // balanced. Deallocs are placed right after the last perf.bench.
    {
      OpBuilder deallocBuilder(module.getContext());
      deallocBuilder.setInsertionPointAfter(timedBench);
      for (Value buf : replicaBufs)
        memref::DeallocOp::create(deallocBuilder, loc, buf);
    }
  }
};

} // namespace
