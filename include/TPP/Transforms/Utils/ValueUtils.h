//===- ValueUtils.h - -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_VALUEUTILS_H
#define TPP_TRANSFORMS_UTILS_VALUEUTILS_H

namespace mlir {
class Value;
class OpBuilder;
class Operation;
class MemRefType;

namespace utils {

// Returns true if the value is a constant float or integer.
bool isValConstZero(Value val);

// Returns true if the op defining `val` represents a zero filled tensor.
bool isZeroTensor(Value val);

// Returns true if the operation represents a zero filled tensor.
bool isZeroOp(Operation *);

// Returns the strides of `val`. The method returns something usefull
// only if the `val` type is a strided memref and the strides are statically
// known.
FailureOr<SmallVector<int64_t>> getStaticStrides(Value val);

FailureOr<SmallVector<int64_t>> getStaticStrides(MemRefType valueType);

// Return the offset and ptr for `val`. Assert if `val`
// is not a memref.
std::pair<Value, Value> getPtrAndOffset(OpBuilder &builder, Value val,
                                        Location loc);

} // namespace utils
} // namespace mlir

#endif // TPP_TRANSFORMS_UTILS_VALUEUTILS_H
