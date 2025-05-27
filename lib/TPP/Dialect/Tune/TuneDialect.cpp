// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TPP/Dialect/Tune/TuneDialect.h"

#include "mlir/Parser/Parser.h"

namespace mlir {
namespace tune {

TuneDialect::TuneDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TuneDialect>()) {
//#define GET_OP_LIST
//  addOperations<
//#include "TPP/Dialect/Tune/TuneOps.cpp.inc"
//      >();
}

} // namespace tune
} // namespace mlir
