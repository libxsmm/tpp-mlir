from ..._mlir_libs import get_dialect_registry
from ..._mlir_libs._tppDialects.transform.tune import register_dialect_extension

register_dialect_extension(get_dialect_registry())

from ...ir import ArrayAttr, SymbolRefAttr, Attribute, Type
from .._tune_transform_ops_gen import TuneSelectOp

from collections.abc import Sequence
from typing import Union


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
