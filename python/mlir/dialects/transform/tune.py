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
from .._tune_transform_ops_gen import *

from collections.abc import Sequence
from typing import Union, Optional


def select(
    result: Type,  # transform.any_param or transform.param<...>
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
        result=result,
        name=name,
        options=options,
        loc=loc,
        ip=ip,
    )


def pick(
    result: Type,  # transform.any_param or transform.param<...>
    name: Union[str, Attribute],
    options: Union[ArrayAttr, Sequence[Union[Attribute, str, int, bool]]],
    *,
    selected: Optional[Union[Attribute, str, int, bool]] = None,
    loc=None,
    ip=None,
) -> TunePickOp:
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
            else:
                assert False
        options = ArrayAttr.get(option_attrs)


    if selected is None:
        pass
    elif isinstance(selected, str):
        selected = StringAttr.get(selected)
    elif isinstance(selected, int):
        int_type = IntegerType.get_signless(64)
        selected = IntegerAttr.get(int_type, selected)
    elif isinstance(selected, bool):
        selected = BoolAttr.get(selected)
    elif not isinstance(selected, Attribute):
        assert False

    return TunePickOp(
        result=result,
        name=name,
        options=options,
        selected=selected,
        loc=loc,
        ip=ip,
    )
