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
// `scf.for` loop over a new outer "replica" dimension. Each kernel argument is
// backed by a freshly allocated, value-initialized global memref whose leading
// dimension is the replication factor; every iteration feeds the kernel a
// distinct `memref.subview` (pure pointer arithmetic, no allocation or copy).
// This mirrors the "n_layers" replication used by libxsmm's cold-cache GEMM
// benchmark: the same problem is run on different memory so caches stay cold.
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

// Build a splat initializer (value 1) for the replicated global. Using a
// constant non-zero value keeps the buffers free of denormals/garbage that
// would otherwise distort floating-point timing.
static TypedAttr getOneScalar(OpBuilder &builder, Type elemTy) {
  if (isa<FloatType>(elemTy))
    return builder.getFloatAttr(elemTy, 1.0);
  if (auto intTy = dyn_cast<IntegerType>(elemTy))
    return builder.getIntegerAttr(elemTy, 1);
  return nullptr;
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

    // For each kernel argument, create a replicated, initialized global memref
    // and pre-compute the rank-reduced strided type of a single replica slice.
    OpBuilder globalBuilder(ctx);
    globalBuilder.setInsertionPointToStart(module.getBody());
    SmallVector<StringRef> globalNames(origInputs.size());
    SmallVector<MemRefType> sliceTypes(origInputs.size());
    auto alignment = globalBuilder.getI64IntegerAttr(128);
    for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
      auto memrefTy = dyn_cast<MemRefType>(inTy);
      if (!memrefTy || !memrefTy.hasStaticShape()) {
        module.emitError("replicate-bench-args: kernel arguments must be "
                         "statically shaped memrefs");
        return signalPassFailure();
      }

      // Replicated global shape: [factor, origShape...].
      SmallVector<int64_t> replShape;
      replShape.push_back(factor);
      replShape.append(memrefTy.getShape().begin(), memrefTy.getShape().end());
      auto replTy = MemRefType::get(replShape, memrefTy.getElementType());

      TypedAttr one = getOneScalar(globalBuilder, memrefTy.getElementType());
      if (!one) {
        module.emitError("replicate-bench-args: unsupported element type");
        return signalPassFailure();
      }
      auto tensorTy =
          RankedTensorType::get(replShape, memrefTy.getElementType());
      auto initAttr = DenseElementsAttr::get(tensorTy, one);

      std::string name = "__bench_replica_" + std::to_string(idx);
      auto global = memref::GlobalOp::create(
          globalBuilder, loc, name, globalBuilder.getStringAttr("private"),
          replTy, initAttr, /*constant=*/false, alignment);
      globalNames[idx] = global.getName();

      // Type of a single replica slice: drop the leading replica dimension,
      // keeping a (statically strided, dynamically offset) view into the
      // contiguous replicated buffer.
      SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
      staticOffsets.push_back(ShapedType::kDynamic);
      staticSizes.push_back(1);
      staticStrides.push_back(1);
      for (int64_t d : memrefTy.getShape()) {
        staticOffsets.push_back(0);
        staticSizes.push_back(d);
        staticStrides.push_back(1);
      }
      sliceTypes[idx] = memref::SubViewOp::inferRankReducedResultType(
          memrefTy.getShape(), replTy, staticOffsets, staticSizes,
          staticStrides);
    }

    // Relax the kernel signature so it accepts the strided replica slices.
    SmallVector<Type> newInputs(sliceTypes.begin(), sliceTypes.end());
    kernel.setType(FunctionType::get(ctx, newInputs,
                                     kernel.getFunctionType().getResults()));
    for (auto [blockArg, sliceTy] :
         llvm::zip_equal(kernel.getBody().getArguments(), sliceTypes))
      blockArg.setType(sliceTy);

    // Wrap every kernel call inside a perf.bench in a replication loop.
    for (auto bench : benches) {
      SmallVector<func::CallOp> calls;
      bench.getBodyRegion().walk([&](func::CallOp call) {
        if (call.getCallee() == kernel.getSymName())
          calls.push_back(call);
      });

      for (func::CallOp call : calls) {
        OpBuilder builder(call);

        // Hoist the global handles out of the loop; they are loop invariant.
        SmallVector<Value> globals(origInputs.size());
        for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
          auto replTy = cast<MemRefType>(
              cast<memref::GlobalOp>(module.lookupSymbol(globalNames[idx]))
                  .getType());
          globals[idx] = memref::GetGlobalOp::create(builder, loc, replTy,
                                                      globalNames[idx]);
        }

        Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
        Value one = arith::ConstantIndexOp::create(builder, loc, 1);
        Value ub = arith::ConstantIndexOp::create(builder, loc, factor);
        auto loop = scf::ForOp::create(builder, loc, zero, ub, one);

        OpBuilder bodyBuilder(loop.getBody(), loop.getBody()->begin());
        Value iv = loop.getInductionVar();

        SmallVector<Value> sliceArgs(origInputs.size());
        for (auto [idx, inTy] : llvm::enumerate(origInputs)) {
          auto memrefTy = cast<MemRefType>(inTy);
          SmallVector<OpFoldResult> offsets, sizes, strides;
          offsets.push_back(iv);
          sizes.push_back(bodyBuilder.getIndexAttr(1));
          strides.push_back(bodyBuilder.getIndexAttr(1));
          for (int64_t d : memrefTy.getShape()) {
            offsets.push_back(bodyBuilder.getIndexAttr(0));
            sizes.push_back(bodyBuilder.getIndexAttr(d));
            strides.push_back(bodyBuilder.getIndexAttr(1));
          }
          sliceArgs[idx] = memref::SubViewOp::create(
              bodyBuilder, loc, sliceTypes[idx], globals[idx], offsets, sizes,
              strides);
        }

        func::CallOp::create(bodyBuilder, loc, kernel, sliceArgs);
        call.erase();
      }
    }
  }
};

} // namespace
