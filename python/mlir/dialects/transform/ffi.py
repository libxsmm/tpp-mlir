from ..._mlir_libs import get_dialect_registry
from ..._mlir_libs._tppDialects.transform.ffi import (
    register_dialect_extension,
    register_callback_handler,
)

register_dialect_extension(get_dialect_registry())

from ...ir import ArrayAttr, SymbolRefAttr, Attribute, Type
from ...dialects import transform
from .._transform_ffi_ops_gen import *

from collections.abc import Sequence
from typing import Union


def callback(
    results: Type,
    name: Union[str, Attribute],
    *payloads: Union[
        transform.AnyOpType, transform.AnyParamType, transform.AnyValueType
    ],
    loc=None,
    ip=None
):
    if isinstance(name, str):
        name = SymbolRefAttr.get([name])

    return CallbackOp(
        results_=results,
        name=name,
        payloads=payloads,
        loc=loc,
        ip=ip,
    )
