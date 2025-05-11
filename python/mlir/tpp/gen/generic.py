from typing import Union

from mlir import ir
from mlir.dialects import linalg, arith, tensor, math

from .utils import (
    affine_map,
    get_bias,
    get_outputs,
    get_weights,
    parallel,
    reduction,
)


def affine_maps_and_iter_types(rank: int):
    M, N, K = [ir.AffineDimExpr.get(i) for i in range(3)]

    if rank == 2:  # plain 2D weights
        affine_maps = [
            affine_map(3, [M, K]),
            affine_map(3, [K, N]),
            affine_map(3, [M, N]),
        ]
        iterator_types = [parallel, parallel, reduction]
    elif rank == 4:  # tiled weights, no vnni blocking
        mb, nb, kb = [ir.AffineDimExpr.get(i) for i in range(3, 6)]
        affine_maps = [
            affine_map(6, [M, K, mb, kb]),
            affine_map(6, [N, K, kb, nb]),  # transposed K and N on B
            affine_map(6, [M, N, mb, nb]),
        ]
        iterator_types = [parallel, parallel, reduction] * 2
    elif rank == 5:  # tiled weights with vnni blocking
        # FIXME: due to replicating C++ code, vnni dim is in middle instead of at end.
        k_vnni, mb, nb, kb = [ir.AffineDimExpr.get(i) for i in range(3, 7)]

        affine_maps = [
            affine_map(7, [M, K, mb, kb, k_vnni]),
            # TODO(RM): check if kb and (k_)vnni _not_ being contiguous makes sense.
            affine_map(7, [N, K, kb, nb, k_vnni]),  # transposed K and N on B
            affine_map(7, [M, N, mb, nb]),
        ]
        iterator_types = [
            parallel,  # M
            parallel,  # N
            reduction,  # K
            reduction,  # vnni
            parallel,  # mb
            parallel,  # nb
            reduction,  # kb
        ]
    else:
        assert False

    return affine_maps, iterator_types


def times_weights(
    inputs: ir.Value,
    weights_or_weights_type: Union[ir.Value, ir.RankedTensorType],
    outputs_or_outputs_type: Union[ir.Value, ir.RankedTensorType],
) -> ir.Value:
    weights: ir.Value = get_weights(weights_or_weights_type)
    outputs: ir.Value = get_outputs(outputs_or_outputs_type)

    if weights.type.rank == 5:  # tiled weights with vnni blocking
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

    affine_maps, iterator_types = affine_maps_and_iter_types(weights.type.rank)

    @linalg.generic([inputs, weights], [outputs], affine_maps, iterator_types)
    def inputs_times_weights(a, b, c):
        prod = arith.MulFOp(a, b)
        return arith.AddFOp(prod.result, c)

    return inputs_times_weights


def add_bias(inputs: ir.Value, bias_or_bias_type: Union[ir.Value, ir.Type] = None):
    bias: ir.Value = get_bias(bias_or_bias_type)

    M, N, mb, nb = [ir.AffineDimExpr.get(i) for i in range(4)]
    affine_maps, iterator_types = {
        2: ([affine_map(2, [N]), affine_map(2, [M, N])], [parallel] * 2),
        4: ([affine_map(4, [N, nb]), affine_map(4, [M, N, mb, nb])], [parallel] * 4),
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


def softmax(
    inputs: ir.Value, softmax_buf_or_softmax_buf_type: Union[ir.Value, ir.Type]
) -> ir.Value:
    softmax_buf = get_outputs(softmax_buf_or_softmax_buf_type)

    shape, elem_type = inputs.type.shape, inputs.type.element_type
    exp_out_uninit = tensor.EmptyOp(shape, elem_type)

    dims = [ir.AffineDimExpr.get(i) for i in range(inputs.type.rank)]
    par_affine_map = affine_map(inputs.type.rank, dims)
    par_affine_maps = [par_affine_map] * inputs.type.rank
    par_iter_types = [parallel] * inputs.type.rank
    red_affine_map = affine_map(
        inputs.type.rank, [dims[0], ir.AffineConstantExpr.get(0)]
    )
    red_iter_types = [parallel, reduction] * (inputs.type.rank // 2)

    @linalg.generic([inputs], [exp_out_uninit.result], par_affine_maps, par_iter_types)
    def exped(input, _output):
        return math.exp(input)

    zero = arith.constant(elem_type, 0.0)
    reduction_out_uninit = tensor.EmptyOp((shape[0], 1), elem_type)
    reduction_out = linalg.fill(zero, outs=reduction_out_uninit)

    @linalg.generic(
        [exped], [reduction_out], [par_affine_map, red_affine_map], red_iter_types
    )
    def summed_exped(exped_input, redex):
        return arith.AddFOp(exped_input, redex)

    bcast_out_uninit = tensor.EmptyOp(shape, elem_type)

    @linalg.generic(
        [summed_exped],
        [bcast_out_uninit.result],
        [red_affine_map, par_affine_map],
        par_iter_types,
    )
    def bcasted_summed_exped(input, _output):
        return input

    @linalg.generic(
        [exped, bcasted_summed_exped],
        [softmax_buf],
        [par_affine_map] * 3,
        par_iter_types,
    )
    def dived_bcasted_summed_exped(exped_input, normalizing_term, _output):
        return arith.DivFOp(exped_input, normalizing_term)

    return dived_bcasted_summed_exped
