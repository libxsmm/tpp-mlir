//===- TuneDialect.h - Tune dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_DIALECT_TUNE_TUNEDIALECT_H
#define TPP_DIALECT_TUNE_TUNEDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace tune {

class TuneDialect : public Dialect {
public:
  explicit TuneDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tune"; }
};

} // namespace tune
} // namespace mlir

#endif // TPP_DIALECT_TUNE_TUNEDIALECT_H
