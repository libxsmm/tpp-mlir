from ..._mlir_libs import get_dialect_registry
from ..._mlir_libs._tppDialects.transform.tune import register_dialect_extension

register_dialect_extension(get_dialect_registry())

from ...ir import (
    ArrayAttr,
    SymbolRefAttr,
    Attribute,
    Type,
    StringAttr,
    IntegerAttr,
    IntegerType,
    BoolAttr,
)
from .._tune_transform_ops_gen import TuneSelectOp

from collections.abc import Sequence
from typing import Union


def select(
    selected: Type,  # transform.any_param or transform.param<...>
    name: Union[str, Attribute],
    options: Union[ArrayAttr, Sequence[Union[Attribute, str, int, bool]]],
    loc=None,
    ip=None,
) -> TuneSelectOp:
    if isinstance(name, str):
        name = SymbolRefAttr.get([name])

    if not isinstance(options, ArrayAttr):
        option_attrs = []
        for option in options:
            if isinstance(option, str):
                option_attrs.append(StringAttr.get(option))
            elif isinstance(option, int):
                int_type = IntegerType.get_signless(64)
                option_attrs.append(IntegerAttr.get(int_type, option))
            elif isinstance(option, bool):
                option_attrs.append(BoolAttr.get(option))
            elif isinstance(option, Attribute):
                option_attrs.append(option)
        options = ArrayAttr.get(option_attrs)

    return TuneSelectOp(
        selected=selected,
        name=name,
        options=options,
        loc=loc,
        ip=ip,
    )
