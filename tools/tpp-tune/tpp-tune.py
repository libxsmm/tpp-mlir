#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Union, Sequence, Dict
import random

# Enable automagically finding TPP-MLIR's python modules (which include
# and extend MLIR's Python bindings).
python_packages_path = Path(__file__).parent.parent / "python_packages"
if python_packages_path.exists():
    sys.path = [str(python_packages_path)] + sys.path


from mlir import ir
from mlir.dialects import transform
from mlir.dialects.transform import tune as transform_tune


def walker(f):
    def wrapper(op: Union[ir.OpView, ir.Operation]):
        f(op)
        for region in op.regions:
            for block in region.blocks:
                for child_op in block:
                    wrapper(child_op)

    return wrapper


def autotune(choices: Dict[str, Sequence[ir.Attribute]]) -> Dict[str, ir.Attribute]:
    # Aint tuning easy!!
    return {key: random.choice(values) for key, values in choices.items()}


file = sys.stdin
if len(sys.argv) > 1 and sys.argv[1] != "-":
    file = open(sys.argv[1])


with ir.Context(), ir.Location.unknown():
    schedule = ir.Module.parse(file.read())

    choices = {}

    @walker
    def choices_finder(op):
        if isinstance(op, transform_tune.TuneSelectOp):
            if op.name in choices:
                raise RuntimeError(f"options name collision: {op.name} used twice")
            choices[op.name] = tuple(op.options)

    choices_finder(schedule.operation)

    selected = autotune(choices)

    @walker
    def selected_rewriter(op: Union[ir.OpView, ir.Operation]):
        if isinstance(op, transform_tune.TuneSelectOp):
            with ir.InsertionPoint(op):
                param = transform.param_constant(
                    transform.AnyParamType.get(), selected[op.name]
                )
                for use in op.result.uses:
                    use.owner.operands[use.operand_number] = param

    selected_rewriter(schedule.operation)

    print(schedule)
