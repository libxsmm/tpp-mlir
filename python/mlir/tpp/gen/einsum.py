from typing import Union

from mlir import ir
from mlir.dialects import linalg, tensor

from .generic import add_bias as generic_add_bias, relu as generic_relu
from .utils import affine_map, get_outputs, get_weights


def times_weights(
  inputs: ir.Value,
  weights_or_weights_type: Union[ir.Value, ir.RankedTensorType],
  outputs_or_outputs_type: Union[ir.Value, ir.RankedTensorType],
) -> ir.Value:
  weights: ir.Value = get_weights(weights_or_weights_type)
  outputs: ir.Value = get_outputs(outputs_or_outputs_type)

  M, N, K, mb, nb, kb, vnni = [ir.AffineDimExpr.get(i) for i in range(7)]

  if weights.type.rank == 2:  # plain 2D weights
    affine_maps = [
      affine_map(3, [M, K]),
      affine_map(3, [K, N]),
      affine_map(3, [M, N]),
    ]
  elif weights.type.rank == 4:  # tiled weights, no vnni blocking
    affine_maps = [
      affine_map(6, [M, K, mb, kb]),
      affine_map(6, [N, K, nb, kb]),  # transposed K and N on B
      affine_map(6, [M, N, mb, nb]),
    ]
  elif weights.type.rank == 5:  # tiled weights with vnni blocking
    affine_maps = [
      affine_map(7, [M, K, mb, kb, vnni]),
      affine_map(7, [N, K, nb, kb, vnni]),  # transposed K and N on B
      affine_map(7, [M, N, mb, nb]),
    ]
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
  else:
    assert False

  return linalg.contract(
    inputs, weights, outs=[outputs], indexing_maps=affine_maps
  )


# TODO: enable python-bindings for elementwise ops and use add with affine_maps
add_bias = generic_add_bias


# TODO: enable python-bindings for elementwise ops and use max with affine_maps
relu = generic_relu
