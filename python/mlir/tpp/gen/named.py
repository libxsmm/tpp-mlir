from typing import Union

from mlir import ir
from mlir.dialects import linalg, tensor, arith

from .utils import get_outputs, get_weights, get_bias


def times_weights(
    inputs: ir.Value,
    weights_or_weights_type: Union[ir.Value, ir.RankedTensorType],
    outputs_or_outputs_type: Union[ir.Value, ir.RankedTensorType],
) -> ir.Value:
    weights: ir.Value = get_weights(weights_or_weights_type)
    outputs: ir.Value = get_outputs(outputs_or_outputs_type)

    assert inputs.type.element_type == outputs.type.element_type

    if inputs.type.rank == 2:
        return linalg.matmul(inputs, weights, outs=(outputs,))
    elif inputs.type.rank == 4:
        tp_shape = weights.type.shape
        tp_shape[2], tp_shape[3] = tp_shape[3], tp_shape[2]
        tranpose_uninit = tensor.EmptyOp(tp_shape, weights.type.element_type)

        transposed_weights = linalg.transpose(
            weights, outs=(tranpose_uninit,), permutation=(0, 1, 3, 2)
        )
        return linalg.mmt4d(inputs, transposed_weights, outs=(outputs,))
    assert False


def add_bias(inputs: ir.Value, bias_or_bias_type: Union[ir.Value, ir.Type]) -> ir.Value:
    bias: ir.Value = get_bias(bias_or_bias_type)

    out_uninit = tensor.EmptyOp(inputs.type.shape, inputs.type.element_type)

    dimensions = {2: [0], 4: [0, 2]}[inputs.type.rank]
    bias_bcast = linalg.broadcast(bias, outs=(out_uninit,), dimensions=dimensions)
    return linalg.add(bias_bcast, inputs, outs=(out_uninit,))


def relu(inputs: ir.Value) -> ir.Value:
    zero = arith.constant(inputs.type.element_type, 0.0)
    out_uninit = tensor.EmptyOp(inputs.type.shape, inputs.type.element_type)
    out = linalg.fill(zero, outs=out_uninit)
    return linalg.max(inputs, out, outs=(out_uninit,))


def softmax(
    inputs: ir.Value, softmax_buf_or_softmax_buf_type: Union[ir.Value, ir.Type]
) -> ir.Value:
    softmax_buf = get_outputs(softmax_buf_or_softmax_buf_type)
    return linalg.softmax((softmax_buf.type,), inputs, softmax_buf, dimension=1)
