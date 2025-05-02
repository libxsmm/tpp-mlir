//===- LoadTppDialects.cpp -----------------------------------------*- C++-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass is a no-op as it is only used for the side-effect of loading dialects.
//
//===----------------------------------------------------------------------===//
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"


namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_LOADTPPDIALECTS
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

using namespace mlir;
using namespace std;

namespace mlir {
namespace tpp {
struct LoadTppDialects
    : public impl::LoadTppDialectsBase<LoadTppDialects> {
  void runOnOperation() override {}
};
} // namespace tpp
} // namespace mlir

