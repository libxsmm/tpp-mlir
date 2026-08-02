#include "TPP/Dialect/Check/CheckOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OpImplementation.h"

// TableGen's computePropertiesHash() for ops with empty properties calls the
// empty-pack llvm::hash_combine<>(), whose uninitialized staging buffer trips a
// GCC -Wmaybe-uninitialized false positive under -Werror. Providing a
// behavior-identical specialization keeps that body from being instantiated
// here without a diagnostic pragma or an LLVM-header change.
#include "llvm/ADT/Hashing.h"
namespace llvm {
template <> inline hash_code hash_combine<>() {
  std::array<char, 1> buf{};
  return hashing::detail::combine_bytes(buf.data(), 0);
}
} // namespace llvm

#define GET_OP_CLASSES
#include "TPP/Dialect/Check/CheckOps.cpp.inc"
