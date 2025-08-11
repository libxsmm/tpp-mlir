import random
import sys

from argparse import ArgumentParser
from typing import Sequence, Dict, Any, Optional
from collections import namedtuple

import numpy as np

from mlir import ir
from mlir.dialects import func

from . import named, generic, einsum, utils as gen_utils


BlockFactors = namedtuple("BlockFactors", "m n k vnni")


def config_from_args(args: Sequence[str]) -> Dict[str, Any]:
    def csints(s: str) -> Sequence[int]:
        return [int(n) for n in s.split(",")]

    parser = ArgumentParser(prog="mlir-gen.py", description="TODO")

    parser.add_argument(
        "--kernel",
        choices=("const", "args"),
        help="whether weights and outputs are generated as constants or provided via arguments",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="seed for random value generation for constants",
    )
    parser.add_argument(
        "--constants",
        choices=("ones", "distinct"),
        default="ones",
        help="in absence of --seed, the kind of constants to generate",
    )
    parser.add_argument(
        "--identity",
        action="store_true",
        help="whether to use a generic op for matmul instead of another linalg op",
    )

    parser.add_argument(
        "--output",
        choices=("generic", "einsum", "contract", "named"),
        default="generic",
        help="the category of linalg ops to use",
    )
    parser.add_argument(
        "--keep-generic-matmul",
        action="store_true",
        help="whether to use a generic op for matmul instead of another linalg op",
    )

    parser.add_argument(
        "--float-type",
        choices=("f32", "bf16", "f16"),
        default="f32",
        help="the float type to operate on",
    )
    parser.add_argument("--batch", type=int, default=256, help="the batch size")
    parser.add_argument(
        "--layers",
        type=csints,
        default=(128, 256, 512),
        help="the number of neurons in each layer - the first layer is the input layer and the last layer is the output layer",
    )

    parser.add_argument(
        "--tiles",
        type=csints,
        default=[0, 0, 0],
        help="tile sizes for M, N, and K dims",
    )
    parser.add_argument(
        "--vnni", choices=(0, 2, 4), type=int, default=0, help="VNNI blocking factor"
    )

    parser.add_argument(
        "--bias",
        action="store_true",
        help="whether to add a bias to the output of each layer",
    )
    parser.add_argument(
        "--relu",
        action="store_true",
        help="whether to apply relu to the output of each layer",
    )
    parser.add_argument(
        "--softmax",
        action="store_true",
        help="whether to apply softmax to the output of the last layer",
    )

    return vars(parser.parse_args(args))


class TensorType:
    def __init__(
        self, block_factors: BlockFactors, elem_type: Optional[ir.Type] = None
    ):
        self.block_factors = block_factors
        self.elem_type = elem_type

    def input(
        self, shape: Sequence[int], elem_type: Optional[ir.Type] = None
    ) -> ir.RankedTensorType:
        elem_type, block = elem_type or self.elem_type, self.block_factors

        if block.m and block.k:
            m_as_batch, k_as_num_inputs = shape
            assert m_as_batch % block.m == 0, "invalid tile size for M dim"
            assert k_as_num_inputs % block.k == 0, "invalid tile size for K dim"
            shape = (
                m_as_batch // block.m,
                k_as_num_inputs // block.k,
                block.m,
                block.k,
            )

        return ir.RankedTensorType.get(shape, elem_type)

    def weights(
        self, shape: Sequence[int], elem_type: Optional[ir.Type] = None
    ) -> ir.RankedTensorType:
        elem_type, block = elem_type or self.elem_type, self.block_factors

        if block.k and block.n:
            k_as_num_inputs, n_as_num_outputs = shape
            assert k_as_num_inputs % block.k == 0, "invalid tile size for K dim"
            assert n_as_num_outputs % block.n == 0, "invalid tile size for N dim"
            if block.vnni:
                assert (
                    block.n % block.vnni == 0
                ), "incompatible tile sizes for N and VNNI dims"
                shape = (
                    n_as_num_outputs // block.n,
                    k_as_num_inputs // block.k,
                    block.k // block.vnni,
                    block.n,
                    block.vnni,
                )
            else:
                shape = (
                    n_as_num_outputs // block.n,
                    k_as_num_inputs // block.k,
                    block.k,
                    block.n,
                )
        else:
            if block.vnni:
                assert False, "--vnni without --block-factors is not supported yet"

        return ir.RankedTensorType.get(shape, elem_type)

    def bias(
        self, shape: Sequence[int], elem_type: Optional[ir.Type] = None
    ) -> ir.RankedTensorType:
        elem_type, block = elem_type or self.elem_type, self.block_factors

        if block.n:
            (n_as_num_outputs,) = shape
            assert n_as_num_outputs % block.n == 0, "invalid tile size for N dim"
            shape = (n_as_num_outputs // block.n, block.n)

        return ir.RankedTensorType.get(shape, elem_type)

    def output(
        self, shape: Sequence[int], elem_type: Optional[ir.Type] = None
    ) -> ir.RankedTensorType:
        elem_type, block = elem_type or self.elem_type, self.block_factors
        if block.m and block.n:
            m_as_batch, n_as_num_outputs = shape
            assert m_as_batch % block.m == 0, "invalid tile size for M dim"
            assert n_as_num_outputs % block.n == 0, "invalid tile size for N dim"
            shape = (
                m_as_batch // block.m,
                n_as_num_outputs // block.n,
                block.m,
                block.n,
            )
        return ir.RankedTensorType.get(shape, elem_type)


def neural_net_as_func(
    overall_args_types: Sequence[ir.Type], config: Dict[str, Any]
) -> ir.Operation:
    keys = ["times_weights", "add_bias", "relu", "softmax"]
    times_weights, add_bias, relu, softmax = {
        "named": [getattr(named, key) for key in keys],
        "einsum": [getattr(einsum, key) for key in keys],
        "contract": [getattr(einsum, key) for key in keys],  # TODO(RM): remove alias
        "generic": [getattr(generic, key) for key in keys],
    }[config["output"]]

    if config["keep_generic_matmul"]:
        times_weights = generic.times_weights

    from_args = config["kernel"] == "args"

    func_args_types = overall_args_types if from_args else overall_args_types[:1]

    @func.func(*func_args_types, results=overall_args_types[-1:])
    def entry(*args):
        args_or_arg_types = iter(args if from_args else (args + overall_args_types[1:]))

        layer_inputs = next(args_or_arg_types)
        for _layer_num_outputs in config["layers"][1:]:
            weights_or_weights_type = next(args_or_arg_types)
            if config["bias"]:
                bias_or_bias_type = next(args_or_arg_types)
            outputs_or_outputs_type = next(args_or_arg_types)

            result = times_weights(
                layer_inputs, weights_or_weights_type, outputs_or_outputs_type
            )
            last_matmul = result

            if config["bias"]:
                result = add_bias(result, bias_or_bias_type)
            if config["relu"]:
                result = relu(result)

            layer_inputs = result

        if config["softmax"]:
            # NB: reuses same output buffer as last matmul.
            layer_inputs = softmax(layer_inputs, last_matmul.owner.operands[2])

        func.ReturnOp((layer_inputs,))

    return entry


def create_metadata(config: Dict[str, Any]) -> str:
    flops = 0
    for layer_num_neurons, next_layer_num_neurons in zip(
        config["layers"][:-1], config["layers"][1:]
    ):
        M = config["batch"]
        N = next_layer_num_neurons
        K = layer_num_neurons

        flops += (_matmul_flops := 2 * M * N * K)
        if config["bias"]:
            flops += (_bias_flops := M * N)
        if config["relu"]:
            flops += (_relu_flops := M * N)
    if config["softmax"]:
        flops += (_softmax_flops := 4 * M * N)

    return f"""\
// RUN: tpp-run %s -n 10 \\
// RUN:   -e entry -entry-point-result=void
// BENCH_TOTAL_FLOPS: {flops}
"""


def main(args: Sequence[str]) -> ir.Module:
    config = config_from_args(args)

    assert len(config["layers"]) >= 2, "at least one layer is required"

    if True:  # NB: Delimits modification of global state.
        gen_utils.CONSTANT_INIT_KIND = {
            "ones": gen_utils.CONSTANT_INIT_KIND.ones,
            "distinct": gen_utils.CONSTANT_INIT_KIND.distinct,
        }[config["constants"]]
        # TODO: does it make sense to allow constants to be an independent cmd flag?
        if config["identity"]:
            gen_utils.CONSTANT_INIT_KIND = gen_utils.CONSTANT_INIT_KIND.identity

        if config["seed"] != -1:
            random.seed(config["seed"])
            gen_utils.CONSTANT_INIT_KIND = gen_utils.CONSTANT_INIT_KIND.random
            gen_utils.RNG = np.random.default_rng(config["seed"])

    batch_size = config["batch"]
    num_inputs = config["layers"][0]
    blocking_factors = BlockFactors(*config["tiles"] + [config["vnni"]])

    with ir.Context(), ir.Location.name(" ".join(sys.argv)):
        elem_type = {
            "bf16": ir.BF16Type.get(),
            "f16": ir.F16Type.get(),
            "f32": ir.F32Type.get(),
        }[config["float_type"]]
        tensor_type = TensorType(blocking_factors, elem_type)

        overall_args_types = (tensor_type.input((batch_size, num_inputs)),)
        for layer_num_neurons, next_layer_num_neurons in zip(
            config["layers"][:-1], config["layers"][1:]
        ):
            bias_arg_type_if_any = (
                (tensor_type.bias((next_layer_num_neurons,)),) if config["bias"] else ()
            )
            overall_args_types += (
                (tensor_type.weights((layer_num_neurons, next_layer_num_neurons)),)
                + bias_arg_type_if_any
                + (tensor_type.output((batch_size, next_layer_num_neurons)),)
            )

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            neural_net_as_func(overall_args_types, config)

        print(create_metadata(config))
        print(module, end="")
