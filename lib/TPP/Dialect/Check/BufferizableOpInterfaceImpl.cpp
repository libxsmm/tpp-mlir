//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Check/CheckOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/Operation.h"
using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::check;

namespace mlir {
namespace check {
namespace {

// TODO: bufferization interface for check ops
struct ExpectTrueLayoutInterface
    : public BufferizableOpInterface::ExternalModel<ExpectTrueLayoutInterface,
                                                    check::ExpectTrueOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return opOperand.getOperandNumber() == 0;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    check::ExpectTrueOp expectTrueOp = cast<check::ExpectTrueOp>(op);

    FailureOr<Value> maybeSrcBuffer =
        getBuffer(rewriter, expectTrueOp.getOperand(), options, state);
    if (failed(maybeSrcBuffer))
      return failure();
    Value srcBuffer = *maybeSrcBuffer;

    check::ExpectTrueOp::create(rewriter, op->getLoc(), srcBuffer);
    return success();
  }
};

struct ExpectAlmostEqLayoutInterface
    : public BufferizableOpInterface::ExternalModel<
          ExpectAlmostEqLayoutInterface, check::ExpectAlmostEqOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    check::ExpectAlmostEqOp almostEqOp = cast<check::ExpectAlmostEqOp>(op);
    FailureOr<Value> maybeFirstBuffer =
        getBuffer(rewriter, almostEqOp.getLhs(), options, state);
    if (failed(maybeFirstBuffer))
      return failure();
    Value firstBuffer = *maybeFirstBuffer;

    FailureOr<Value> maybeSecondBuffer =
        getBuffer(rewriter, almostEqOp.getRhs(), options, state);
    if (failed(maybeSecondBuffer))
      return failure();
    Value secondBuffer = *maybeSecondBuffer;

    auto newExpectOp = check::ExpectAlmostEqOp::create(rewriter,
        op->getLoc(), firstBuffer, secondBuffer, almostEqOp.getThreshold());
    op->replaceAllUsesWith(newExpectOp);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExpectSaneLayoutInterface
    : public BufferizableOpInterface::ExternalModel<ExpectSaneLayoutInterface,
                                                    check::ExpectSaneOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    return BufferRelation::Equivalent;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    check::ExpectSaneOp saneOp = cast<check::ExpectSaneOp>(op);
    FailureOr<Value> maybeBuffer =
        getBuffer(rewriter, saneOp.getOperand(), options, state);
    if (failed(maybeBuffer)) {
      return failure();
    }
    Value buffer = *maybeBuffer;

    auto newExpectOp =
        check::ExpectSaneOp::create(rewriter, op->getLoc(), buffer);
    op->replaceAllUsesWith(newExpectOp);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace
} // namespace check
} // namespace mlir

namespace mlir {
namespace check {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, check::CheckDialect *dialect) {
    ExpectTrueOp::attachInterface<check::ExpectTrueLayoutInterface>(*ctx);
    ExpectAlmostEqOp::attachInterface<check::ExpectAlmostEqLayoutInterface>(
        *ctx);
    ExpectSaneOp::attachInterface<check::ExpectSaneLayoutInterface>(*ctx);
  });
}
} // namespace check
} // namespace mlir
