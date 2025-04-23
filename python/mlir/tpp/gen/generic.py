from typing import Union

from mlir import ir
from mlir.dialects import linalg, arith, tensor

from .utils import (
  affine_map,
  get_bias,
  get_outputs,
  get_weights,
  parallel,
  reduction,
)


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
    iterator_types = [parallel, parallel, reduction]
  elif weights.type.rank == 4:  # tiled weights, no vnni blocking
    affine_maps = [
      affine_map(6, [M, K, mb, kb]),
      affine_map(6, [N, K, nb, kb]),  # transposed K and N on B
      affine_map(6, [M, N, mb, nb]),
    ]
    iterator_types = [parallel, parallel, reduction] * 2
  elif weights.type.rank == 5:  # tiled weights with vnni blocking
    affine_maps = [
      affine_map(7, [M, K, mb, kb, vnni]),
      affine_map(7, [N, K, nb, kb, vnni]),  # transposed K and N on B
      affine_map(7, [M, N, mb, nb]),
    ]
    iterator_types = [
      parallel,  # M
      parallel,  # N
      reduction,  # K
      parallel,  # mb
      parallel,  # nb
      reduction,  # kb
      reduction,  # vnni
    ]
    vnni_block = weights.type.get_dim_size(4)
    assert inputs.type.shape[-1] % vnni_block == 0

    expanded_shape = (
      inputs.type.shape[:-1]
      + [inputs.type.shape[-1] // vnni_block]
      + [vnni_block]
    )
    inputs = tensor.expand_shape(
      ir.RankedTensorType.get(expanded_shape, inputs.type.element_type),
      inputs,
      reassociation=[[0], [1], [2], [3, 4]],
      output_shape=[],
      static_output_shape=expanded_shape,
    )
  else:
    assert False

  @linalg.generic([inputs, weights], [outputs], affine_maps, iterator_types)
  def inputs_times_weights(a, b, c):
    prod = arith.MulFOp(a, b)
    return arith.AddFOp(prod.result, c)

  return inputs_times_weights


def add_bias(
  inputs: ir.Value, bias_or_bias_type: Union[ir.Value, ir.Type] = None
):
  bias: ir.Value = get_bias(bias_or_bias_type)

  M, N, mb, nb = [ir.AffineDimExpr.get(i) for i in range(4)]
  affine_maps, iterator_types = {
    2: ([affine_map(2, [N]), affine_map(2, [M, N])], [parallel] * 2),
    4: ([affine_map(4, [N, nb]), affine_map(4, [M, N, mb, nb])], [parallel] * 4)
  }[inputs.type.rank]

  @linalg.generic([bias], [inputs], affine_maps, iterator_types)
  def biased(a, b):
    return arith.AddFOp(a, b)

  return biased


def relu(inputs: ir.Value):
  zero = arith.constant(inputs.type.element_type, 0.0)

  M, N, mb, nb = [ir.AffineDimExpr.get(i) for i in range(4)]
  affine_maps, iterator_types = {
    2: ([affine_map(2, [M, N])], [parallel, parallel]),
    4: ([affine_map(4, [M, N, mb, nb])], [parallel, parallel] * 2),
  }[inputs.type.rank]

  @linalg.generic([], [inputs], affine_maps, iterator_types)
  def relu_ed(a):
    return arith.MaximumFOp(a, zero)

  return relu_ed
