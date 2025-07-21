from typing import Callable, Union, Dict, Optional

import atexit

from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import ffi, structured


# Wrapper to addresss verbosity.
def apply_registered_pass(*args, **kwargs):
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


# Wrapper to addresss verbosity.
def match(*args, **kwargs):
    return structured.MatchOp(transform.AnyOpType.get(), *args, **kwargs)


# Global mapping callback names to Python-function callback functions.
HANDLER_MAPPING: Dict[str, Callable] = {}


# The python function that actually gets called from C++ to deal with
# transform.ffi.callback callbacks.
def callback_handler(name, *args):
    if (handler := HANDLER_MAPPING.get(name)) is None:
        raise RuntimeError(f"callback '{name}' requested but was not registered")
    return handler(*args)


ffi.register_callback_handler(callback_handler)
atexit.register(ffi.register_callback_handler, None)


# Decorator to register named Python callback functions. Return types need to be
# provided as part of the signature.
def callback(function: Callable, context: Optional[ir.Context] = None):
    setattr(ir.Context.current, "_callback_handler", callback_handler)

    if function.__name__ in HANDLER_MAPPING:
        raise RuntimeError("tried to register a callback with the same name twice")
    HANDLER_MAPPING[function.__name__] = function
    results_type = function.__annotations__.get("return", ())

    def wrapper(
        *args: Union[
            transform.AnyOpType, transform.AnyValueType, transform.AnyParamType
        ]
    ):
        return ffi.callback(results_type, function.__name__, *args)

    return wrapper


# Decorator to register named Python callback function and immediately call it.
# Return types need to be provided as part of the signature.
def call_with(
    *args: Union[transform.AnyOpType, transform.AnyValueType, transform.AnyParamType]
):
    def decorator(function: Callable):
        if function.__name__ in HANDLER_MAPPING:
            raise RuntimeError("tried to register a callback with the same name twice")
        HANDLER_MAPPING[function.__name__] = function
        results_type = function.__annotations__.get("return", ())
        return ffi.callback(results_type, function.__name__, *args)

    return decorator
