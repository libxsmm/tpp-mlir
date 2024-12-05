//===- XsmmUtils.h - --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_XSMM_XSMMUTILS_H
#define TPP_DIALECT_XSMM_XSMMUTILS_H

#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include <variant>

namespace mlir {
class Type;
class RewriterBase;
class Value;
class ArrayAttr;
class Operation;
class PatternRewriter;
class VectorType;
class MemRefType;

namespace func {
class CallOp;
}

namespace xsmm {
class UnaryKindAttr;

struct UnaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldi;
  int64_t ldo;
};

struct BinaryInfo {
  unsigned m;
  unsigned n;

  int64_t ldiLhs;
  int64_t ldiRhs;
  int64_t ldo;
};

/// Represents a chain of XSMM ops that can be fused. All broadcast ops
/// should have already been converted to flags. All stray allocations
/// should have already been converted to in-place reuse.
struct FusedMatch {
  // This is the (optional) zero op that precedes the GEMM op
  UnaryOp zeroOp;
  // This is the BRGEMM op
  BrgemmOp brgemmOp;
  // This is the (optional) binary op that follows the GEMM
  BinaryOp binaryOp;
  BinaryKind binaryKind;
  // This is the (optional) unary op that follows the GEMM/Binary
  UnaryOp unaryOp;
  UnaryKind unaryKind;
};

namespace utils {

DataTypeAttr getDataType(RewriterBase &rewriter, Type type);

FailureOr<UnaryInfo>
getVectorUnaryInfo(MemRefType shapeType, MemRefType inputType,
                   MemRefType outputType, VectorType inputVectorType,
                   VectorType outputVectorType, UnaryFlags inputFlag);

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  UnaryFlags inputFlag);

FailureOr<BinaryInfo> getBinaryInfo(Value lhs, BinaryFlags lhsFlag, Value rhs,
                                    BinaryFlags rhsFlag, Value output);

void replaceOpWithUnary(RewriterBase &rewriter, Operation *operation,
                        ArrayRef<Value> operands, UnaryInfo unaryInfo,
                        ArrayAttr flags, UnaryKindAttr kind);

// Compute the broadcasting flags for 'inputType' based 'outputType'.
// Rules for broadcasting follows Numpy-style, and are only allowed in
// 'inputType'. see: https://numpy.org/doc/stable/user/basics.broadcasting.html
FailureOr<UnaryFlags> getUnaryFlags(Type inputType, Type outputType);

// Compute the broadcasting flags for 'operandType' based on 'outputType'.
enum class OperandPos { LHS = 0, RHS = 1 };
FailureOr<BinaryFlags> getBinaryFlags(Type operandType, Type outputType,
                                      OperandPos operandNumber);

FailureOr<FusedMatch> getFusedBrgemmSequenceFromProducer(Operation *op);

ArrayAttr getUnaryDispatchFlags(UnaryOp op);

ArrayAttr getBinaryDispatchFlags(BinaryOp op);

template <typename DispatchOpTy>
FailureOr<SmallVector<Attribute>> getBrgemmFlags(PatternRewriter &rewriter,
                                                 DispatchOpTy dispatchOpTy,
                                                 bool returnNone);

SmallVector<Type> extractOperandTypes(OpBuilder &builder,
                                      ArrayRef<Value> operands);

enum class XsmmCallType { DISPATCH = 0, INVOKE = 1 };

typedef struct {
  XsmmCallType CallType;
  Value CallResult;
} XsmmCall;

// This is Value type and not MemRefType only in case we want to support some
// other object types in the future
typedef std::variant<int64_t, Value, XsmmCall> XsmmOperand;

func::CallOp buildXsmmCall(RewriterBase &rewriter, XsmmCallType callType,
                           Location loc, DataTypeAttr dtype,
                           SmallVector<XsmmOperand> operands, TypeRange results,
                           FlatSymbolRefAttr fnName, Operation *parentOp,
                           Operation *insertBefore);
} // namespace utils
} // namespace xsmm
} // namespace mlir

#endif // TPP_DIALECT_XSMM_XSMMUTILS_H
