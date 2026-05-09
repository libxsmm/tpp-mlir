//===- ConvertXsmmToFunc.cpp -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Xsmm/XsmmEnum.h"
#include "TPP/Dialect/Xsmm/XsmmOps.h"
#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/ValueUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::xsmm;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_CONVERTXSMMTOFUNC
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

// NOTE: The ordering of operands to XSMM function calls as it is defined here
// is strictly followed by XsmmRunnerUtils for XSMM calls. Please change
// the ordering of the fields in XsmmRunnerUtils for any such change in this
// file.

namespace {

static SmallVector<Type> extractInvokeOperandTypes(OpBuilder &builder,
                                                   OperandRange operands) {
  SmallVector<Type> results;
  // One extra operand for datatype
  IntegerType integer64 = IntegerType::get(builder.getContext(), 64);
  results.push_back(integer64);
  for (Value operand : operands) {
    Type operandType = operand.getType();
    if (auto memrefType = dyn_cast<MemRefType>(operandType)) {
      // TODO: non-POD will require an LLVMTypeConverter.
      Type basePtrType = LLVM::LLVMPointerType::get(builder.getContext());
      results.push_back(basePtrType);
      results.push_back(builder.getIndexType()); // offset
    } else {
      results.push_back(operand.getType());
    }
  }
  return results;
}

// Extract the operands to be used in the function call. For each memref operand
// extract the aligned pointer and the offset.
static SmallVector<Value> getOperands(OpBuilder &builder, Location loc,
                                      ValueRange operands,
                                      IntegerAttr dataTypeAttr) {
  SmallVector<Value> res;
  IntegerType integer64 = IntegerType::get(builder.getContext(), 64);
  res.push_back(
      arith::ConstantOp::create(builder, loc, integer64, dataTypeAttr));

  for (Value operand : operands) {
    auto memrefType = dyn_cast<MemRefType>(operand.getType());
    if (!memrefType) {
      res.push_back(operand);
      continue;
    }
    auto [ptr, offset] = utils::getPtrAndOffset(builder, operand, loc);
    res.push_back(ptr);
    res.push_back(offset);
  }
  return res;
}

static void buildInvokeCall(OpBuilder &builder, Location loc,
                            const std::string &funcName, Operation *op,
                            IntegerAttr dataTypeAttr) {
  FlatSymbolRefAttr fnName = SymbolRefAttr::get(op->getContext(), funcName);
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto libFnType = builder.getFunctionType(
      extractInvokeOperandTypes(builder, op->getOperands()), {});

  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(builder);
    // Insert before module terminator.
    builder.setInsertionPoint(module.getBody(),
                              std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        func::FuncOp::create(builder, loc, fnName.getValue(), libFnType);
    funcOp.setPrivate();
  }

  func::CallOp::create(builder, 
      loc, fnName.getValue(), TypeRange(),
      getOperands(builder, loc, op->getOperands(), dataTypeAttr));
}

struct ConvertGemmXsmmOp : public OpRewritePattern<GemmOp> {
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmOp gemmOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName = "xsmm_gemm_invoke";
    buildInvokeCall(rewriter, gemmOp.getLoc(), funcName, gemmOp,
                    gemmOp.getDataTypeAttr());
    rewriter.eraseOp(gemmOp);
    return success();
  }
};

struct ConvertBrgemmXsmmOp : public OpRewritePattern<BrgemmOp> {
  using OpRewritePattern<BrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrgemmOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName = "xsmm_brgemm_invoke";
    buildInvokeCall(rewriter, brgemmOp.getLoc(), funcName, brgemmOp,
                    brgemmOp.getDataTypeAttr());
    rewriter.eraseOp(brgemmOp);
    return success();
  }
};

struct ConvertUnaryXsmmOp : public OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOp unaryOp,
                                PatternRewriter &rewriter) const override {
    // Handle the scalar case. There is no operator overloading
    // in MLIR (thus we need to change the function name from
    // "unary" to "unary_scalar"). We also don't want to convert
    // the scalar to a memref by using an alloc/alloca.
    std::string funcName = "xsmm_unary_invoke";
    if (unaryOp.hasScalarInput())
      funcName = "xsmm_unary_scalar_invoke";
    buildInvokeCall(rewriter, unaryOp.getLoc(), funcName, unaryOp,
                    unaryOp.getDataTypeAttr());
    rewriter.eraseOp(unaryOp);
    return success();
  }
};

struct ConvertBinaryXsmmOp : public OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp binaryOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName = "xsmm_binary_invoke";
    buildInvokeCall(rewriter, binaryOp.getLoc(), funcName, binaryOp,
                    binaryOp.getDataTypeAttr());
    rewriter.eraseOp(binaryOp);
    return success();
  }
};

struct ConvertFusedBrgemmXsmmOp : public OpRewritePattern<FusedBrgemmOp> {
  using OpRewritePattern<FusedBrgemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FusedBrgemmOp fusedBrgemmOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName = "xsmm_fused_brgemm_invoke";
    buildInvokeCall(rewriter, fusedBrgemmOp.getLoc(), funcName, fusedBrgemmOp,
                    fusedBrgemmOp.getDataTypeAttr());
    rewriter.eraseOp(fusedBrgemmOp);
    return success();
  }
};

struct ConvertIntelAMXTileConfigXsmmOp
    : public OpRewritePattern<IntelAMXTileConfigOp> {
  using OpRewritePattern<IntelAMXTileConfigOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IntelAMXTileConfigOp tileConfigOp,
                                PatternRewriter &rewriter) const override {
    std::string funcName = "xsmm_intel_amx_tile_config_invoke";
    buildInvokeCall(
        rewriter, tileConfigOp.getLoc(), funcName, tileConfigOp,
        xsmm::DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16));
    rewriter.eraseOp(tileConfigOp);
    return success();
  }
};

static func::CallOp buildDispatchCall(RewriterBase &rewriter, Location loc,
                                      ArrayRef<Value> dispatchOperands,
                                      ArrayRef<Type> dispatchOperandTypes,
                                      ModuleOp module,
                                      FlatSymbolRefAttr fnName) {
  auto libFnType = rewriter.getFunctionType(
      dispatchOperandTypes, IntegerType::get(rewriter.getContext(), 64));

  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        func::FuncOp::create(rewriter, loc, fnName.getValue(), libFnType);
    funcOp.setPrivate();
  }

  func::CallOp call = func::CallOp::create(rewriter, 
      loc, fnName.getValue(), IntegerType::get(rewriter.getContext(), 64),
      dispatchOperands);
  return call;
}

template <typename OpTy, typename = std::enable_if_t<
                             std::is_same<OpTy, xsmm::UnaryDispatchOp>::value ||
                             std::is_same<OpTy, xsmm::BinaryDispatchOp>::value>>
void addKindOperand(RewriterBase &rewriter, OpTy dispatchOp,
                    SmallVectorImpl<Value> &dispatchOperands,
                    SmallVectorImpl<Type> &dispatchOperandTypes) {
  Location loc = dispatchOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  dispatchOperands.push_back(arith::ConstantOp::create(rewriter, 
      loc, integer64, cast<TypedAttr>(dispatchOp.getKindAttr())));
  dispatchOperandTypes.push_back(integer64);
}

void addKindOperand(RewriterBase &rewriter, GemmDispatchOp dispatchOp,
                    SmallVectorImpl<Value> &dispatchOperands,
                    SmallVectorImpl<Type> &dispatchOperandTypes) {
  /* do nothing */
}

void addKindOperand(RewriterBase &rewriter, BrgemmDispatchOp dispatchOp,
                    SmallVectorImpl<Value> &dispatchOperands,
                    SmallVectorImpl<Type> &dispatchOperandTypes) {
  /* do nothing */
}

void addKindOperand(RewriterBase &rewriter, FusedBrgemmDispatchOp dispatchOp,
                    SmallVectorImpl<Value> &dispatchOperands,
                    SmallVectorImpl<Type> &dispatchOperandTypes) {
  /* do nothing */
}

void addKindOperand(RewriterBase &rewriter,
                    IntelAMXTileConfigDispatchOp dispatchOp,
                    SmallVectorImpl<Value> &dispatchOperands,
                    SmallVectorImpl<Type> &dispatchOperandTypes) {
  /* do nothing */
}

static int64_t getOredFlags(ArrayAttr flags) {
  int64_t oredFlag = 0;
  for (auto flag : flags) {
    int64_t intAttr = dyn_cast<IntegerAttr>(flag).getInt();
    // LIBXSMM is col-major, swap A and B flags.
    if (auto gemmFlag = dyn_cast_or_null<xsmm::GemmFlagsAttr>(flag)) {
      if (gemmFlag.getValue() == GemmFlags::VNNI_A)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_B);
      if (gemmFlag.getValue() == GemmFlags::VNNI_B)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_A);
    }
    oredFlag |= intAttr;
  }
  return oredFlag;
}

// Fused brgemm requires additional flags:
// 1. Unary flags.
// 2. Type of the unary operation (i.e., relu).
// 3. Binary flags.
// 4. Type of the binary operation (i.e., add).
void addUnaryAndBinaryFlags(RewriterBase &rewriter,
                            FusedBrgemmDispatchOp dispatchOp,
                            SmallVectorImpl<Value> &dispatchOperands,
                            SmallVectorImpl<Type> &dispatchOperandTypes) {
  Location loc = dispatchOp.getLoc();
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

  int64_t oredFlag = getOredFlags(dispatchOp.getUnaryFlags());
  dispatchOperands.push_back(arith::ConstantOp::create(rewriter, 
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);

  dispatchOperands.push_back(arith::ConstantOp::create(rewriter, 
      loc, integer64, cast<TypedAttr>(dispatchOp.getUnaryKindAttr())));
  dispatchOperandTypes.push_back(integer64);

  oredFlag = getOredFlags(dispatchOp.getBinaryFlags());
  dispatchOperands.push_back(arith::ConstantOp::create(rewriter, 
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);

  dispatchOperands.push_back(arith::ConstantOp::create(rewriter, 
      loc, integer64, cast<TypedAttr>(dispatchOp.getBinaryKindAttr())));
  dispatchOperandTypes.push_back(integer64);
}

template <typename OpTy>
static LogicalResult buildDispatchOp(RewriterBase &rewriter, OpTy dispatchOp,
                                     std::string funcName) {
  Location loc = dispatchOp.getLoc();
  FlatSymbolRefAttr fnName =
      SymbolRefAttr::get(rewriter.getContext(), funcName);

  ModuleOp module = dispatchOp->template getParentOfType<ModuleOp>();
  SmallVector<Value, 10> dispatchOperands;
  SmallVector<Type, 10> dispatchOperandTypes;
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

  // If `OpTy` is unary or binary we need to dispatch and extra
  // integer for the kind of operation to invoke.
  if (std::is_same<OpTy, xsmm::UnaryDispatchOp>::value ||
      std::is_same<OpTy, xsmm::BinaryDispatchOp>::value) {
    addKindOperand(rewriter, dispatchOp, dispatchOperands,
                   dispatchOperandTypes);
  }

  // Dispatch the data type.
  dispatchOperands.push_back(arith::ConstantOp::create(rewriter, 
      loc, integer64, cast<TypedAttr>(dispatchOp.getDataTypeAttr())));
  dispatchOperandTypes.push_back(integer64);

  // Dispatch the inputs.
  ArrayRef<int64_t> integers = dispatchOp.getInputsAttr().asArrayRef();
  size_t arrayAttrSize = integers.size();
  for (size_t idx = 0; idx < arrayAttrSize; idx++) {
    IntegerAttr attr = IntegerAttr::get(rewriter.getI64Type(), integers[idx]);
    dispatchOperands.push_back(
        arith::ConstantOp::create(rewriter, loc, integer64, attr));
    dispatchOperandTypes.push_back(integer64);
  }

  // Dispatch the flags. Pass to the library the already ored-flag to
  // avoid changing the interface every time we add a new flag. Flags
  // are assumed to be verified before (i.e., op verifier).
  int64_t oredFlag = getOredFlags(dispatchOp.getFlagsAttr());

  dispatchOperands.push_back(arith::ConstantOp::create(rewriter, 
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);

  if (auto dispatchBrgemmOp = dyn_cast_or_null<xsmm::FusedBrgemmDispatchOp>(
          dispatchOp.getOperation())) {
    addUnaryAndBinaryFlags(rewriter, dispatchBrgemmOp, dispatchOperands,
                           dispatchOperandTypes);
  }

  func::CallOp call = buildDispatchCall(rewriter, loc, dispatchOperands,
                                        dispatchOperandTypes, module, fnName);
  rewriter.replaceOp(dispatchOp, call.getResult(0));
  return success();
}

struct ConvertGemmDispatchOp : public OpRewritePattern<GemmDispatchOp> {
  using OpRewritePattern<GemmDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<GemmDispatchOp>(rewriter, dispatchOp,
                                           "xsmm_gemm_dispatch");
  }
};

struct ConvertBrgemmDispatchOp : public OpRewritePattern<BrgemmDispatchOp> {
  using OpRewritePattern<BrgemmDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BrgemmDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<BrgemmDispatchOp>(rewriter, dispatchOp,
                                             "xsmm_brgemm_dispatch");
  }
};

struct ConvertBinaryDispatchOp : public OpRewritePattern<BinaryDispatchOp> {
  using OpRewritePattern<BinaryDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<BinaryDispatchOp>(rewriter, dispatchOp,
                                             "xsmm_binary_dispatch");
  }
};

struct ConvertUnaryDispatchOp : public OpRewritePattern<UnaryDispatchOp> {
  using OpRewritePattern<UnaryDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<UnaryDispatchOp>(rewriter, dispatchOp,
                                            "xsmm_unary_dispatch");
  }
};

struct ConvertIntelAMXTileConfigDispatchOp
    : public OpRewritePattern<IntelAMXTileConfigDispatchOp> {
  using OpRewritePattern<IntelAMXTileConfigDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IntelAMXTileConfigDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    return buildDispatchOp<IntelAMXTileConfigDispatchOp>(
        rewriter, dispatchOp, "xsmm_intel_amx_tile_config_dispatch");
  }
};

struct ConvertFusedBrgemmOp : public OpRewritePattern<FusedBrgemmDispatchOp> {
  using OpRewritePattern<FusedBrgemmDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FusedBrgemmDispatchOp dispatchOp,
                                PatternRewriter &rewriter) const override {
    // Currently LIBXSMM support only BCAST_COL_IN_0 as binary flag with bias
    // addition.
    auto isFusedAdd = dispatchOp.getBinaryKind() == xsmm::BinaryKind::ADD;
    auto binaryFlags = dispatchOp.getBinaryFlags();
    if (isFusedAdd && (binaryFlags.size() != 1 ||
                       cast<BinaryFlagsAttr>(binaryFlags[0]).getValue() !=
                           BinaryFlags::BCAST_COL_IN_0)) {
      return failure();
    }
    return buildDispatchOp<FusedBrgemmDispatchOp>(rewriter, dispatchOp,
                                                  "xsmm_fused_brgemm_dispatch");
  }
};

struct ConvertXsmmToFunc
    : public tpp::impl::ConvertXsmmToFuncBase<ConvertXsmmToFunc> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertBinaryXsmmOp, ConvertUnaryXsmmOp, ConvertGemmXsmmOp,
                 ConvertBrgemmXsmmOp, ConvertFusedBrgemmXsmmOp,
                 ConvertIntelAMXTileConfigXsmmOp>(patterns.getContext());
    patterns.add<ConvertBinaryDispatchOp, ConvertUnaryDispatchOp,
                 ConvertGemmDispatchOp, ConvertBrgemmDispatchOp,
                 ConvertFusedBrgemmOp, ConvertIntelAMXTileConfigDispatchOp>(
        patterns.getContext());
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
