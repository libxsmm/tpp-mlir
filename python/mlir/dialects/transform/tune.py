from ..._mlir_libs import get_dialect_registry
from ..._mlir_libs._tppDialects.transform.tune import (
    register_dialect_extension,
    register_callback_handler,
)
from ..._mlir_libs._tppDialects.transform import tune

tune._callback = None
from ..._mlir_libs import _tppDialects

_tppDialects._callback = lambda: print("callbacked")

register_dialect_extension(get_dialect_registry())

from ...ir import ArrayAttr, SymbolRefAttr, Attribute, Type, Operation, Value
from ...dialects import transform
from .._tune_transform_ops_gen import *

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

    return TuneCallbackOp(
        results_=results,
        name=name,
        payloads=payloads,
        loc=loc,
        ip=ip,
    )


def select(
    selected: Type,  # transform.any_param or transform.param<...>
    name: Union[str, Attribute],
    options: Union[ArrayAttr, Sequence[Attribute]],
    loc=None,
    ip=None,
) -> TuneSelectOp:
    if isinstance(name, str):
        name = SymbolRefAttr.get([name])

    return TuneSelectOp(
        selected=selected,
        name=name,
        options=options,
        loc=loc,
        ip=ip,
    )
