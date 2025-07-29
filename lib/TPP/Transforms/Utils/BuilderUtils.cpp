//===- BuilderUtils.cpp ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"

#include "TPP/Transforms/Utils/BuilderUtils.h"

#include <optional>

namespace mlir {

namespace {
template <class ValueT>
arith::ConstantOp getConstant(OpBuilder &builder, Type type, ValueT value) {
  auto unkLoc = builder.getUnknownLoc();
  TypedAttr attr;
  if constexpr (std::numeric_limits<ValueT>::is_integer) {
    attr = builder.getIntegerAttr(type, value);
  } else if constexpr (llvm::is_one_of<ValueT, float, double>()) {
    attr = builder.getFloatAttr(type, value);
  }
  assert(attr && "Unsupported ConstantOp type");
  return builder.create<arith::ConstantOp>(unkLoc, type, attr);
}
} // anonymous namespace

func::FuncOp createFunction(OpBuilder &builder, ModuleOp module, StringRef name,
                            TypeRange args, TypeRange ret, bool createBody) {
  auto oper = module.lookupSymbol(name);
  if (oper)
    return dyn_cast<func::FuncOp>(oper);
  else {
    auto unkLoc = builder.getUnknownLoc();
    auto funcType = FunctionType::get(builder.getContext(), args, ret);
    auto func = func::FuncOp::create(unkLoc, name, funcType);
    func.setVisibility(SymbolTable::Visibility::Private);
    if (createBody) {
      func.setVisibility(SymbolTable::Visibility::Public);
      auto *entryBlock = func.addEntryBlock();
      builder.setInsertionPointToEnd(entryBlock);
    }
    module.push_back(func);
    return func;
  }
}

Value getConstInt(OpBuilder &builder, int value, int width) {
  switch (width) {
  case 32:
    return getConstant(builder, builder.getI32Type(), value);
  case 64:
    return getConstant(builder, builder.getI64Type(), value);
  default:
    assert(false && "Invalid constant integer size");
  }
  abort();
}

Value getConstFloat(OpBuilder &builder, float value, FloatType type) {
  return getConstant(builder, type, value);
}

Value getConstIndex(OpBuilder &builder, int value) {
  return getConstant(builder, builder.getIndexType(), value);
}

Value createDenseTensor(OpBuilder &builder, TensorInitType initType,
                        TensorType type, int seed) {
  auto unkLoc = builder.getUnknownLoc();
  auto init = getTensorInit(initType, type.getElementType(), seed);
  auto floatInit = init->get(type);
  assert(!failed(floatInit) && "Invalid dense tensor initializer");
  return builder.create<arith::ConstantOp>(unkLoc, type, floatInit.value());
}

Value createDenseMemref(OpBuilder &builder, ModuleOp module,
                        TensorInitType initType, MemRefType type, int seed) {
  auto unkLoc = builder.getUnknownLoc();
  StringRef globalName;
  // First create the global
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&module->getRegions().front().front());

    // Simple auto increment
    static unsigned order = 0;

    // Create global dense memrefs (Module insertion point)
    auto privAttr = builder.getStringAttr("private");

    // Auto incremental naming system
    std::string name = "__wrapper_" + std::to_string(order++);

    auto alignment = builder.getIntegerAttr(builder.getI64Type(), 128);
    auto init = getTensorInit(initType, type.getElementType(), seed);
    auto floatInit = init->get(type);
    assert(!failed(floatInit) && "Invalid dense tensor initializer");

    // Create the global object in the Module's region
    auto global = builder.create<memref::GlobalOp>(
        unkLoc, StringRef(name), privAttr, type, floatInit.value(),
        /*constant=*/false, alignment);
    globalName = global.getName();
  }
  // Get the created global value and use it
  // as an input to the kernel
  auto nameAttr = builder.getStringAttr(globalName);
  return builder.create<memref::GetGlobalOp>(unkLoc, type, nameAttr);
}

TypedAttr getTypedAttr(OpBuilder &builder, Type type, double value) {
  if (isa<FloatType>(type))
    return builder.getFloatAttr(type, value);
  if (isa<IndexType>(type))
    return builder.getIndexAttr(value);
  if (auto intTp = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(type, APInt(intTp.getWidth(), value));
  if (isa<RankedTensorType, VectorType>(type)) {
    auto shapedType = cast<ShapedType>(type);
    if (auto one = getTypedAttr(builder, shapedType.getElementType(), value))
      return DenseElementsAttr::get(shapedType, one);
  }
  llvm_unreachable("Unsupported attribute type");
}

} // namespace mlir
