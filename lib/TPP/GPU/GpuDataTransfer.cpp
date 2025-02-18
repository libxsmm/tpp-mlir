//===- GpuDataTransfer.cpp ---------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_GPUDATATRANSFER
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

namespace {

// Returns the source operation of the given value (data) e.g., host allocation,
// if it should be moved to the device.
// Nullopt is returned (nothing to do), if the source is unknown and/or
// the buffer is believed to already live on the device.
static std::optional<Operation *> getDeviceTransferableBuffer(Value val) {
  // Not a buffer - nothing to do.
  if (!isa<MemRefType>(val.getType()))
    return std::nullopt;

  // Assume it is a valid buffer, if the operation that defines the value
  // is not available e.g., buffer coming from a function argument.
  // TODO: This assumption about function arguments should be optional
  //       and controlled by a flag.
  Operation *op = val.getDefiningOp();
  if (!op)
    return std::nullopt;

  // Host-space allocation or a global variable can be easily transfered to
  // a device.
  if (isa<memref::AllocOp, memref::GetGlobalOp>(op))
    return op;

  // Follow through memref alias to get to the real source.
  if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(op))
    return getDeviceTransferableBuffer(viewOp.getViewSource());

  // Nothing to do for all the other cases.
  // Assume it is a valid buffer, if the value comes from a device allocation,
  // a function call, a function arguments, device allocation etc.
  return std::nullopt;
}

// Matches the kernel operand and device buffer types such that data can be
// transfered correctly from the original host buffer.
// Returns failure if the two types cannot be connected.
static FailureOr<Value> matchTransferBuffers(RewriterBase &rewriter,
                                             Value kernelOperand,
                                             Value gpuBuffer) {
  auto loc = gpuBuffer.getLoc();

  auto operandType = cast<MemRefType>(kernelOperand.getType());
  auto gpuAllocType = cast<MemRefType>(gpuBuffer.getType());

  // Kernel operands type already matches the device buffer.
  // The host and device buffers can be used directly.
  if (operandType == gpuAllocType)
    return gpuBuffer;

  // Examine view source type. More type conversions might be needed.
  if (auto viewOp =
          dyn_cast<mlir::ViewLikeOpInterface>(kernelOperand.getDefiningOp())) {
    auto newBuffer =
        matchTransferBuffers(rewriter, viewOp.getViewSource(), gpuBuffer);
    if (failed(newBuffer))
      return failure();
    gpuBuffer = *newBuffer;
  }

  // Cast device allocation to match the operand if possible.
  if (memref::CastOp::areCastCompatible(gpuAllocType, operandType)) {
    return rewriter.create<memref::CastOp>(loc, operandType, gpuBuffer)
        .getDest();
  }

  // Take an equal subview of the device buffer if the operand is a subview.
  // This ensures correct data access and eliminates need to change kernel
  // argument types.
  if (auto subview =
          dyn_cast<memref::SubViewOp>(kernelOperand.getDefiningOp())) {
    return rewriter
        .create<memref::SubViewOp>(
            loc, operandType, gpuBuffer, subview.getOffsets(),
            subview.getSizes(), subview.getStrides(),
            subview.getStaticOffsets(), subview.getStaticSizes(),
            subview.getStaticStrides())
        .getResult();
  }

  // No way to connect the device buffer to the kernel operand.
  return failure();
}

// Transfer host allocated data between the host and a device.
// Optionally, copies data back to the host after device computations.
// Returns device-allocated buffer with the moved data.
static FailureOr<Value> transferMemref(RewriterBase &rewriter,
                                       gpu::LaunchFuncOp launchFuncOp,
                                       Value operand, Value hostBuffer,
                                       bool copyDataBack = true) {
  // A memref buffer is expected.
  assert(isa<MemRefType>(hostBuffer.getType()) &&
         "Transfer memref expects memref buffer");
  if (!isa<MemRefType>(hostBuffer.getType()))
    return failure();

  auto loc = launchFuncOp.getLoc();
  auto &block = launchFuncOp->getParentOfType<func::FuncOp>().getBody().front();

  OpBuilder::InsertionGuard guard(rewriter);

  // Allocate device buffer.
  rewriter.setInsertionPointToStart(&block);
  auto gpuAlloc = rewriter.create<gpu::AllocOp>(
      loc, TypeRange(hostBuffer), ValueRange{}, ValueRange{}, ValueRange{});
  Value gpuBuffer = gpuAlloc.getMemref();

  // Copy data to the device.
  rewriter.setInsertionPoint(launchFuncOp);
  if (operand.getType() != gpuBuffer.getType()) {
    // Use the kernel operand directly as the host buffer to ensure the type
    // matches correctly for data copy.
    // Device buffer type has to be adapted.
    hostBuffer = operand;
    auto newBuffer = matchTransferBuffers(rewriter, operand, gpuBuffer);
    if (failed(newBuffer))
      return failure();
    gpuBuffer = *newBuffer;
  }
  rewriter.create<gpu::MemcpyOp>(loc, std::nullopt, ValueRange{}, gpuBuffer,
                                 hostBuffer);

  // If requested, copy data back to the host.
  if (copyDataBack) {
    rewriter.setInsertionPointAfter(launchFuncOp);
    rewriter.create<gpu::MemcpyOp>(loc, std::nullopt, ValueRange{}, hostBuffer,
                                   gpuBuffer);
  }

  // Cleanup device buffer.
  rewriter.setInsertionPoint(block.getTerminator());
  rewriter.create<gpu::DeallocOp>(loc, std::nullopt, gpuAlloc.getMemref());

  return gpuBuffer;
}

// Move host data used by GPU kernel calls to the device.
struct TransferDataToGpu : public OpRewritePattern<gpu::LaunchFuncOp> {
  using OpRewritePattern<gpu::LaunchFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::LaunchFuncOp launchFuncOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> newOperands;

    // Track if there are any changes to the root.
    bool updatedOperands = false;

    for (auto operand : launchFuncOp.getKernelOperands()) {
      auto src = getDeviceTransferableBuffer(operand);
      if (!src) {
        // Not a transferable operand. Keep it as is.
        newOperands.push_back(operand);
        continue;
      }

      // Operand can be moved to the device.
      // The kernel launch will be updated.
      updatedOperands = true;

      FailureOr<Value> newOperand = failure();

      if (auto allocOp = dyn_cast<memref::AllocOp>(*src)) {
        // Copy data back to the host as it might have been updated on
        // the device.
        newOperand =
            transferMemref(rewriter, launchFuncOp, operand, allocOp.getMemref(),
                           /*copyDataBack=*/true);
      }
      if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(*src)) {
        // Data does not need to be updated if the global data is constant
        // (read-only).
        bool isGlobalConst = false;
        if (auto *symbolTableOp =
                getGlobalOp->getParentWithTrait<OpTrait::SymbolTable>()) {
          if (auto globalOp = dyn_cast_or_null<memref::GlobalOp>(
                  SymbolTable::lookupSymbolIn(symbolTableOp,
                                              getGlobalOp.getNameAttr()))) {
            isGlobalConst = globalOp.getConstant();
          }
        }
        newOperand = transferMemref(rewriter, launchFuncOp, operand,
                                    getGlobalOp.getResult(),
                                    /*copyDataBack=*/!isGlobalConst);
      }

      if (failed(newOperand))
        return failure();
      newOperands.push_back(*newOperand);
    }

    // If there are any new operands, update the kernel launch.
    if (updatedOperands) {
      rewriter.modifyOpInPlace(launchFuncOp, [&]() {
        launchFuncOp.getKernelOperandsMutable().assign(newOperands);
      });
    }

    return success();
  }
};

// Transfer data from host to a GPU device.
class GpuDataTransfer : public tpp::impl::GpuDataTransferBase<GpuDataTransfer> {
public:
  GpuDataTransfer() = default;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    // TODO: Add cleanup patterns to minimize data copies.
    patterns.add<TransferDataToGpu>(ctx);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
