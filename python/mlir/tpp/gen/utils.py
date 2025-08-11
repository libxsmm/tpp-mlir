import struct
from enum import Enum, auto
from collections import abc
from typing import Union

import numpy as np

from mlir import ir
from mlir.dialects import arith, linalg, tensor


class ConstantInitKind(Enum):
    ones = auto()
    distinct = auto()
    random = auto()
    identity = auto()


CONSTANT_INIT_KIND = ConstantInitKind.ones
GAUSSIAN_SAMPLING = True
RNG = None

splat_value = 0.3


def affine_map(dim_count, exprs, *, symb_count=0):
    return ir.AffineMap.get(dim_count, symb_count, exprs)


parallel = linalg.IteratorType.parallel
reduction = linalg.IteratorType.reduction


def floats(shape: abc.Sequence[int], elementType: ir.Type) -> np.ndarray:
    if GAUSSIAN_SAMPLING:
        random_tensor = RNG.normal(0.0, 0.2, shape)
    else:
        if isinstance(elementType, ir.F32Type):
            random_tensor = RNG.random(shape, dtype=np.float32)
            return random_tensor
        random_tensor = RNG.random(shape, dtype=np.float32)

    if isinstance(elementType, ir.BF16Type):

        def to_bf16(value):
            clamped = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
            element = struct.pack("f", clamped)[:2]
            return np.frombuffer(element, np.uint16)[0]

        return np.vectorize(to_bf16)(random_tensor).reshape(shape)
    elif isinstance(elementType, ir.F32Type):

        def to_f32(value):
            clamped = 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)
            element = struct.pack("f", clamped)
            return np.frombuffer(element, np.float32)

        return np.vectorize(to_f32)(random_tensor).reshape(shape)
    else:
        assert False


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
        random_tensor = floats(tensor_type.shape, tensor_type.element_type)
        value = ir.DenseElementsAttr.get(random_tensor, type=tensor_type)
    elif CONSTANT_INIT_KIND == CONSTANT_INIT_KIND.identity:
        if tensor_type.rank == 1:
            # FIXME(rolf): do not detect bias case through tensor shape.
            splat_attr = ir.FloatAttr.get(tensor_type.element_type, 1.0)
            value = ir.DenseElementsAttr.get_splat(tensor_type, splat_attr)
        else:
            shape = tensor_type.shape
            assert len(shape) == 2 and shape[0] == shape[1]
            dtype = {
                ir.F32Type.get(): np.float32,
                ir.BF16Type.get(): np.uint16,
            }[tensor_type.element_type]
            ident_matrix = np.identity(tensor_type.shape[0], dtype)
            value = ir.DenseElementsAttr.get(ident_matrix, type=tensor_type)
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
