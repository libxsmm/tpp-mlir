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
  auto *timerSrc = getTimer().getDefiningOp();
  if (!timerSrc || !isa<StartTimerOp>(timerSrc))
    return emitOpError("invalid timer input");

  // Any timer can only be stopped once. It is unusable afterwards.
  int numStopTimers = 0;
  for (auto *user : timerSrc->getUsers()) {
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

OperandRange BenchOp::getEntrySuccessorOperands(RegionSuccessor point) {
  return getIterArgs();
}

void BenchOp::getSuccessorRegions(RegionBranchPoint parentReg,
                                  SmallVectorImpl<RegionSuccessor> &regions) {
  // The `body` region branch back to the parent operation. Drop the first
  // argument which is the timer value.
  if (!parentReg.isParent()) {
    regions.push_back(RegionSuccessor::parent());
    return;
  }

  // Otherwise the successor is the body region.
  regions.push_back(RegionSuccessor(&getBodyRegion()));
}

ValueRange BenchOp::getSuccessorInputs(RegionSuccessor successor) {
  if (successor.isParent())
    return getBodyResults().drop_front();
  return getIterArgs();
}

void BenchOp::build(OpBuilder &builder, OperationState &result, Value numIters,
                    ValueRange iterArgs) {
  result.addOperands({numIters});
  result.addOperands(iterArgs);

  // First result is always the deltas
  result.addTypes(builder.getF64Type());

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
    perf::YieldOp::create(builder, result.location);
  }
}

YieldOp BenchOp::getYieldOp() {
  return cast<perf::YieldOp>(getRegion().front().getTerminator());
}

void BenchOp::print(OpAsmPrinter &printer) {
  // Print base args
  printer << "(";
  printer << getNumIters();
  printer << " : ";
  printer << getNumIters().getType();
  printer << ")";

  // Print iter_args
  if (!getIterArgs().empty()) {
    auto blockArgs = getRegion().getArguments();
    auto initArgs = ValueRange{getIterArgs()};
    assert(blockArgs.size() == initArgs.size() &&
           "expected same length of arguments and initializers");
    printer << " iter_args(";
    llvm::interleaveComma(
        llvm::zip(blockArgs, initArgs), printer, [&](auto it) {
          printer << std::get<0>(it) << " = " << std::get<1>(it);
        });
    printer << ")";
  }
  printer << " -> " << getResultTypes() << " ";

  // Print region body
  {
    bool printTerminator = true;
    if (auto *term = getRegion().empty()
                         ? nullptr
                         : getRegion().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    printer.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/printTerminator);
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
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
  if (parser.parseColon())
    return failure();
  if (parser.parseCommaSeparatedList(parseType))
    return failure();
  if (parser.parseRParen())
    return failure();

  // Validate arguments
  if (operands.size() != 1)
    return parser.emitError(locs[0], "expect one argument");
  if (types.size() != 1)
    return parser.emitError(locs[0], "expect one types for argument");
  if (parser.resolveOperand(operands[0], types[0], result.operands) ||
      !isa<IntegerType>(types[0]))
    return parser.emitError(locs[0], "expect integer number of iterations");
  operands.clear();
  types.clear();
  locs.clear();

  // Parse iter_args, if any
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  bool hasIterArgs = succeeded(parser.parseOptionalKeyword("iter_args"));
  if (hasIterArgs && parser.parseAssignmentList(regionArgs, operands))
    return failure();

  // Must have at least one return type (stats), more if iter_args
  if (parser.parseArrowTypeList(result.types))
    return failure();

  // First return type is %stat, rest is iter_args
  if (regionArgs.size() + 1 != result.types.size()) {
    return parser.emitError(
        parser.getNameLoc(),
        "mismatch in number of loop-carried values and defined values");
  }
  SmallVector<Type> opTypes(result.types.begin() + 1, result.types.end());

  // Resolve input operands.
  if (hasIterArgs) {
    for (auto argOperandType :
         llvm::zip_equal(regionArgs, operands, opTypes)) {
      Type type = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = type;
      if (parser.resolveOperand(std::get<1>(argOperandType), type,
                                result.operands)) {
        return failure();
      }
    }
  }

  // Parse region
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs))
    return failure();
  ensureTerminator(*body, parser.getBuilder(), result.location);

  // Attributes, if any
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

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
