//===- PerfOps.cpp - Perf dialect ops ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Dialect/Perf/PerfOps.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Perf/PerfOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "TPP/Dialect/Perf/PerfOpsTypes.cpp.inc"

using namespace mlir;
using namespace mlir::perf;

//===----------------------------------------------------------------------===//
// StopTimerOp
//===----------------------------------------------------------------------===//

LogicalResult StopTimerOp::verify() {
  auto timerSrc = getTimer().getDefiningOp();
  if (!timerSrc || !isa<StartTimerOp>(timerSrc))
    return emitOpError("invalid timer input");

  // Any timer can only be stopped once. It is unusable afterwards.
  int numStopTimers = 0;
  for (auto user : timerSrc->getUsers()) {
    if (isa<StopTimerOp>(*user))
      ++numStopTimers;
  }
  if (numStopTimers != 1)
    return emitOpError("timer stopped multiple times");

  return success();
}

//===----------------------------------------------------------------------===//
// BenchOp
//===----------------------------------------------------------------------===//

void BenchOp::build(OpBuilder &builder, OperationState &result, Value numIters,
                    Value deltas, ValueRange iterArgs) {
  result.addOperands({numIters, deltas});
  result.addOperands(iterArgs);

  // Results have to match the input arguments
  for (Value v : iterArgs)
    result.addTypes(v.getType());

  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();

  // Create the default terminator if the arguments are not provided.
  // Otherwise, leave this to the caller because we don't know which values to
  // return from the body.
  if (iterArgs.empty()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    builder.create<perf::YieldOp>(result.location);
  }
}

YieldOp BenchOp::getYieldOp() {
  return cast<perf::YieldOp>(getRegion().front().getTerminator());
}

void BenchOp::print(OpAsmPrinter &printer) {
  printer << "(";
  printer << getNumIters();
  printer << ", ";
  printer << getDeltas();
  printer << " : ";
  printer << getNumIters().getType();
  printer << ", ";
  {
    auto type = getDeltas().getType();
    if (auto validType = type.dyn_cast<::mlir::BaseMemRefType>())
      printer.printStrippedAttrOrType(validType);
    else
      printer << type;
  }
  printer << ")";
  if (!getIterArgs().empty()) {
    printer << " iter_args";
    printer << "(";
    printer << getIterArgs();
    printer << " : ";
    printer << getIterArgs().getTypes();
    printer << ")";
  }
  printer << ' ';

  {
    bool printTerminator = true;
    if (auto *term = getRegion().empty()
                         ? nullptr
                         : getRegion().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    printer.printRegion(getRegion(), /*printEntryBlockArgs=*/true,
                        /*printBlockTerminators=*/printTerminator);
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  if (!getBodyResults().empty()) {
    printer << " -> ";
    printer << getBodyResults().getTypes();
  }
}

ParseResult BenchOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> types;
  SmallVector<llvm::SMLoc> locs;

  // Types
  // Avoid using parseTypeList because it crashes on GCC
  auto parseType = [&]() -> ParseResult {
    if (parser.parseType(types.emplace_back()))
      return failure();
    return success();
  };

  // Parse num_it + delta
  if (parser.parseLParen())
    return failure();

  locs.push_back(parser.getCurrentLocation());
  if (parser.parseOperand(operands.emplace_back()))
    return failure();
  if (parser.parseComma())
    return failure();

  locs.push_back(parser.getCurrentLocation());
  if (parser.parseOperand(operands.emplace_back()))
    return failure();
  if (parser.parseColon())
    return failure();

  if (parser.parseCommaSeparatedList(parseType)) {
    return failure();
  }
  if (parser.parseRParen())
    return failure();

  // Validate arguments
  if (operands.size() != 2)
    return parser.emitError(locs[0], "expect two arguments");
  if (types.size() != 2)
    return parser.emitError(locs[0], "expect two types for arguments");
  if (parser.resolveOperand(operands[0], types[0], result.operands) ||
      !types[0].isa<IntegerType>())
    return parser.emitError(locs[0], "expect integer number of iterations");
  if (parser.resolveOperand(operands[1], types[1], result.operands) ||
      !types[1].isa<MemRefType>())
    return parser.emitError(locs[1], "expect memref for results");
  operands.clear();
  types.clear();
  locs.clear();

  // Parse iter_args, if any
  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    if (parser.parseLParen())
      return failure();

    locs.push_back(parser.getCurrentLocation());
    if (parser.parseOperandList(operands))
      return failure();
    if (parser.parseColon())
      return failure();

    // Avoid using parseTypeList because it crashes on GCC
    if (parser.parseCommaSeparatedList(parseType))
      return failure();
    if (parser.parseRParen())
      return failure();

    // Validate operands
    if (!operands.size())
      return parser.emitError(parser.getCurrentLocation(), "expect arguments");
    if (operands.size() != types.size())
      return parser.emitError(locs[0], "expect same number of types as args");
    if (parser.resolveOperands(operands, types, locs[0], result.operands))
      return failure();
  }

  // Parse region
  auto region = std::make_unique<Region>();
  if (parser.parseRegion(*region))
    return failure();
  ensureTerminator(*region, parser.getBuilder(), result.location);
  result.addRegion(std::move(region));

  // Attributes, if any
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Return type
  SmallVector<Type> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  result.addTypes(resultTypes);

  return success();
}

LogicalResult BenchOp::verify() {
  Operation *terminator = getRegion().front().getTerminator();
  if (!dyn_cast_or_null<perf::YieldOp>(terminator)) {
    auto diag = emitOpError("expects region to terminate with 'perf.yield'");
    if (terminator)
      diag.attachNote(terminator->getLoc()) << "terminator here";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SinkOp
//===----------------------------------------------------------------------===//

// Apply custom function name mangling for various data types.
// It is assumed that all relevant perf operations accept only unranked memory
// types. This allows for simpler name mangling and leaner perf runtime.
std::string SinkOp::applyTypeMangling(std::string name, Type type) {
  llvm::raw_string_ostream mangledName(name);

  TypeSwitch<Type>(type)
      .Case<MemRefType>([&](Type t) {
        mangledName << "_memref"
                    << "_" << cast<MemRefType>(t).getElementType();
      })
      .Case<TensorType>([&](Type t) {
        mangledName << "_tensor"
                    << "_" << cast<TensorType>(t).getElementType();
      })
      .Default([&](Type t) { mangledName << "_" << t; });

  return mangledName.str();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

void YieldOp::print(OpAsmPrinter &printer) {
  if (!getOperands().empty()) {
    printer << ' ';
    printer << getOperands();
    printer << " : ";
    printer << getOperands().getTypes();
  }
}

ParseResult YieldOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SmallVector<Type> types;

  // Attributes, if any
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Operands
  if (parser.parseOperandList(operands))
    return failure();

  if (!operands.empty()) {
    if (parser.parseColon())
      return failure();

    // Types
    // Avoid using parseTypeList because it crashes on GCC
    auto parseType = [&]() -> ParseResult {
      if (parser.parseType(types.emplace_back()))
        return failure();
      return success();
    };
    if (parser.parseCommaSeparatedList(parseType))
      return failure();
  }

  if (parser.resolveOperands(operands, types, parser.getCurrentLocation(),
                             result.operands))
    return failure();
  return success();
}
