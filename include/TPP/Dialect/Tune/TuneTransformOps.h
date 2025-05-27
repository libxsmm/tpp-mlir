#ifndef TPP_DIALECT_TUNE_TUNETRANSFORMOPS_H
#define TPP_DIALECT_TUNE_TUNETRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TPP/Dialect/Tune/TuneTransformOps.h.inc"

namespace mlir {
namespace tune {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace tune
} // namespace mlir

#endif // MLIR_TUNE_TRANSFORM_OPS_H
