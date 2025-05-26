from ...ir import *
from .._tune_transform_ops_gen import *

from collections.abc import Sequence
from typing import List, Optional, Union

def select(
    selected: Type, # transform.param or transform.param<...>
    name: Union[str, Attribute],
    options: Union[ArrayAttr, Sequence[Attribute]],
    loc=None,
    ip=None,
) -> TuneSelectOp:
    if isinstance(name, str):
        name = SymbolRefAttr.get([name])
    if isinstance(options, Sequence):
        options = ArrayAttr.get(options)

    return TuneSelectOp(
        selected=selected,
        name=name,
        options=options,
        loc=loc,
        ip=ip,
    )
