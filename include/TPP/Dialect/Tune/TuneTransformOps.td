#ifndef TUNE_TRANSFORM_OPS
#define TUNE_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def TuneSelectOp : Op<Transform_Dialect, "tune.select", [
  DeclareOpInterfaceMethods<TransformOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Non-deterministically select a value from a set of values";
  let description = [{
    TODO
  }];

  let arguments = (ins SymbolRefAttr:$name,
                       ArrayAttr:$options);
  let results = (outs TransformParamTypeInterface:$selected);
  let assemblyFormat =
      "$name `from` $options attr-dict `:` type(results)";
}

#endif // TUNE_TRANSFORM_OPS
