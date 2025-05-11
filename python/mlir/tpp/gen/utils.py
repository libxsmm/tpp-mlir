import struct
import random
from enum import Enum, auto
from collections import abc
from functools import reduce
from operator import mul
from typing import Union

import numpy as np

from mlir import ir
from mlir.dialects import arith, linalg, tensor

class ConstantInitKind(Enum):
    ones = auto()
    distinct = auto()
    random = auto()

CONSTANT_INIT_KIND = ConstantInitKind.ones
GAUSSIAN_SAMPLING = True

splat_value = 0.3


def affine_map(dim_count, exprs, *, symb_count=0):
    return ir.AffineMap.get(dim_count, symb_count, exprs)


parallel = linalg.IteratorType.parallel
reduction = linalg.IteratorType.reduction


def floats(shape: abc.Sequence[int], elementType: ir.Type) -> np.ndarray:
    def gen_bf16_uniform():
        value = random.random()
        element = struct.pack("f", value)[:2]
        return np.frombuffer(element, np.uint16)[0]

    def gen_bf16_gaussian():
        value = random.gauss(0.0, 0.2)
        clamped = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
        element = struct.pack("f", clamped)[:2]
        return np.frombuffer(element, np.uint16)[0]

    def gen_f32_uniform():
        value = random.random()
        element = struct.pack("f", value)
        return np.frombuffer(element, np.float32)

    def gen_f32_gaussian():
        value = random.gauss(0.0, 0.2)
        clamped = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
        element = struct.pack("f", clamped)
        return np.frombuffer(element, np.float32)

    if isinstance(elementType, ir.BF16Type):
        gen_elt = gen_bf16_gaussian if GAUSSIAN_SAMPLING else gen_bf16_uniform
    elif isinstance(elementType, ir.F32Type):
        gen_elt = gen_f32_gaussian if GAUSSIAN_SAMPLING else gen_f32_uniform
    else:
        assert False

    size_iteration_space = reduce(mul, (shape))
    return np.array([gen_elt() for _ in range(size_iteration_space)]).reshape(*shape)


def gen_tensor_cst(tensor_type: ir.RankedTensorType) -> ir.Value:
    if CONSTANT_INIT_KIND == CONSTANT_INIT_KIND.ones:
        splat_attr = ir.FloatAttr.get(tensor_type.element_type, 1.0)
        value = ir.DenseElementsAttr.get_splat(tensor_type, splat_attr)
    elif CONSTANT_INIT_KIND == CONSTANT_INIT_KIND.distinct:
        global splat_value
        splat_attr = ir.FloatAttr.get(tensor_type.element_type, splat_value)
        splat_value += 0.01
        value = ir.DenseElementsAttr.get_splat(tensor_type, splat_attr)
    elif CONSTANT_INIT_KIND == CONSTANT_INIT_KIND.random:
        value = ir.DenseElementsAttr.get(
            floats(tensor_type.shape, tensor_type.element_type), type=tensor_type
        )
    else:
        assert False, "unreachable"
    return arith.constant(tensor_type, value)


def get_outputs(outputs_or_outputs_type: Union[ir.Value, ir.Type]) -> ir.Value:
    if isinstance(outputs_or_outputs_type, ir.Value):
        return outputs_or_outputs_type
    else:
        assert isinstance(outputs_or_outputs_type, ir.RankedTensorType)
        shape, elem_type = (
            outputs_or_outputs_type.shape,
            outputs_or_outputs_type.element_type,
        )
        out_uninit = tensor.EmptyOp(shape, elem_type)
        zero = arith.constant(elem_type, 0.0)
        return linalg.fill(zero, outs=out_uninit)


def get_weights(weights_or_weights_type: Union[ir.Value, ir.Type]) -> ir.Value:
    if isinstance(weights_or_weights_type, ir.Value):
        return weights_or_weights_type
    else:
        assert isinstance(weights_or_weights_type, ir.RankedTensorType)
        return gen_tensor_cst(weights_or_weights_type)


get_bias = get_weights  # NB: implementation is exactly the same
