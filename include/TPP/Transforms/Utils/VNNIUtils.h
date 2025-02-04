//===- VNNIUtils.h -----------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TRANSFORMS_UTILS_VNNIUTILS_H
#define TPP_TRANSFORMS_UTILS_VNNIUTILS_H

#include "mlir/Support/LogicalResult.h"
#include <cstdint>
#include <optional>

namespace mlir {
class Type;
class ShapedType;
class OpOperand;
class AffineDimExpr;
class AffineMap;
class VectorType;
class Operation;

namespace linalg {
class LinalgOp;
} // namespace linalg

namespace vnni {
namespace utils {
enum class VnniOperandRank {
  TRANSPOSE = 3,
  GEMM = 3,
  BRGEMM_INS = 4,
  BRGEMM_OUTS = 3
};

// Returns True if the current architecture supports AMX instructions.
bool hasAMX();

// Return the VNNI blocking factor if it can be determined for the given type or
// zero, otherwise.
// Optionally, an operation can be provided to give access to DLTI.
unsigned getVnniBlockingFactor(Type type, Operation *op = nullptr);

// Return true if the shaped type is in VNNI layout with rank `expectedRank`.
// Optionally, the check can be constrained to a specific VNNI blocking factor.
bool isInVnniLayout(VnniOperandRank expectedRank, ShapedType shape,
                    std::optional<unsigned> blockingFactor = std::nullopt);

// Return true if the shaped type is in VNNI layout with rank `expectedRank`.
// Optionally, the check can be constrained to a specific VNNI blocking factor.
bool isInVnniLayout(int64_t expectedRank, ShapedType shape,
                    std::optional<unsigned> blockingFactor = std::nullopt);

// Return true if the linalg operation is in VNNI layout.
// Optionally, the check can be constrained to a specific VNNI blocking factor.
bool isInVnniLayout(linalg::LinalgOp linalgOp,
                    std::optional<unsigned> blockingFactor = std::nullopt);

} // namespace utils
} // namespace vnni
} // namespace mlir

#endif
