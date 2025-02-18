//===- TestForToForAllRewrite.cpp - ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Transforms/Utils/TransformUtils.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {
// This is a test pass to check the scf.for to scf.forall rewrite.
struct TestForToForAllRewrite
    : public PassWrapper<TestForToForAllRewrite,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestForToForAllRewrite)
  void runOnOperation() override;
  StringRef getArgument() const final { return "test-scf-for-rewrite"; }
  StringRef getDescription() const final {
    return "Test scf.for to scf.forall rewrite.";
  }
};

} // namespace

void TestForToForAllRewrite::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  linalgx::utils::populateScfForToForAllRewritePattern(patterns);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

namespace mlir {
namespace tpp {
void registerTestForToForAllRewrite() {
  PassRegistration<TestForToForAllRewrite>();
}
} // namespace tpp
} // namespace mlir
