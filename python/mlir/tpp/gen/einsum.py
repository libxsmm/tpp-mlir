from typing import Union

from mlir import ir
from mlir.dialects import linalg, tensor

from . import named, generic
from .utils import get_outputs, get_weights


def times_weights(
    inputs: ir.Value,
    weights_or_weights_type: Union[ir.Value, ir.RankedTensorType],
    outputs_or_outputs_type: Union[ir.Value, ir.RankedTensorType],
) -> ir.Value:
    weights: ir.Value = get_weights(weights_or_weights_type)
    outputs: ir.Value = get_outputs(outputs_or_outputs_type)

    affine_maps, _ = generic.affine_maps_and_iter_types(weights.type.rank)

    if weights.type.rank == 5:  # tiled weights with vnni blocking
        vnni_block = weights.type.get_dim_size(4)
        assert inputs.type.shape[-1] % vnni_block == 0

        expanded_shape = (
            inputs.type.shape[:-1]
            + [inputs.type.shape[-1] // vnni_block]
            + [vnni_block]
        )
        expanded_type = ir.RankedTensorType.get(
            expanded_shape, inputs.type.element_type
        )
        inputs = tensor.expand_shape(
            expanded_type,
            inputs,
            reassociation=[[0], [1], [2], [3, 4]],
            output_shape=[],
            static_output_shape=expanded_shape,
        )

    return linalg.contract(inputs, weights, outs=[outputs], indexing_maps=affine_maps)


# TODO: enable python-bindings for elementwise ops and use add with affine_maps
add_bias = named.add_bias


# TODO: enable python-bindings for elementwise ops and use max with affine_maps
relu = named.relu


softmax = named.softmax
