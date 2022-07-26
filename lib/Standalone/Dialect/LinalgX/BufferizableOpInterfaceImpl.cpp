//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Standalone/Dialect/LinalgX/BufferizableOpInterfaceImpl.h"
#include "Standalone/Dialect/LinalgX/LinalgXDialect.h"
#include "Standalone/Dialect/LinalgX/LinalgXOps.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalgx;

namespace mlir {
namespace linalx {
namespace {

/* write something here */

} // namespace
} // namespace linalx
} // namespace mlir

void mlir::linalgx::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, linalgx::LinalgXDialect *dialect) {});
}
