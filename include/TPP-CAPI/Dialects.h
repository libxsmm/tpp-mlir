#ifndef TPP_CAPI_DIALECTS_H
#define TPP_CAPI_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Check, check);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Xsmm, xsmm);
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Perf, perf);

#ifdef __cplusplus
}
#endif

#endif // TPP_CAPI_DIALECTS_H
