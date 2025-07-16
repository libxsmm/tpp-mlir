#ifndef TPP_DIALECT_FFI_FFITRANSFORMOPS_H
#define TPP_DIALECT_FFI_FFITRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Transform/FfiExtension/TransformOps.h.inc"

namespace mlir {
namespace transform {
namespace ffi {

using Handler = std::function<SmallVector<SmallVector<transform::MappedValue>>(
    StringRef, SmallVector<SmallVector<transform::MappedValue>>)>;

extern Handler handler;

void registerDialectExtension(DialectRegistry &registry);
} // namespace ffi
} // namespace transform
} // namespace mlir

#endif // MLIR_FFI_TRANSFORM_OPS_H
