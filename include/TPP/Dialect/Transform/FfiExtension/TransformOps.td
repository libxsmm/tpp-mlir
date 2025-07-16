#ifndef FFI_TRANSFORM_OPS
#define FFI_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def CallbackOp : Op<Transform_Dialect, "ffi.callback", [
  DeclareOpInterfaceMethods<TransformOpInterface>,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let cppNamespace = [{ mlir::transform::ffi }];

  let summary = "Callback to an external handler";
  let description = [{
    A transform op which will invoke an external handler routine - not further
    specified - which is supposed to delegate to the right callback based on
    the provided `name`.

    Takes any number of arguments of any handle type and produces any number of
    results of any handle type. For example,

    ```mlir
    %res:3 = transform.ffi.callback @py_transform(%op, %param, %value) :
        (!transform.any_op, !transform.any_param, !transform.any_value) ->
            (!transform.any_param, !transform.any_value, !transform.any_op)
    ```
  }];

  let arguments = (ins SymbolRefAttr:$name,
                       Variadic<Transform_AnyHandleOrParamType>:$payloads);
  let results = (outs Variadic<Transform_AnyHandleOrParamType>:$results);
  let assemblyFormat =
      "$name `(` $payloads `)` attr-dict `:` functional-type(operands, results)";
}

#endif // FFI_TRANSFORM_OPS
