// GCC emits a false-positive -Wmaybe-uninitialized in LLVM's Hashing.h from the
// generated computePropertiesHash(); LLVM builds with -Werror. The pragma must
// precede the includes so it is active while Hashing.h is parsed. Guarded
// because clang rejects the unknown -Wmaybe-uninitialized group under -Werror.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "TPP/Dialect/Check/CheckOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Check/CheckOps.cpp.inc"
