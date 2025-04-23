import random
import sys

from argparse import ArgumentParser
from typing import Sequence, Dict, Any

from mlir import ir
from mlir.dialects import func

from mlir_tpp import gen
from . import named, generic, einsum


def config_from_args(args: Sequence[str]) -> Dict[str, Any]:
    def csints(s: str) -> Sequence[int]:
        return [int(n) for n in s.split(",")]

    parser = ArgumentParser(prog="mlir-gen.py", description="TODO")
    parser.add_argument("--kernel", choices=("const", "args"))
    parser.add_argument(
        "--output", choices=("generic", "einsum", "named"), default="generic"
    )
    parser.add_argument("--float-type", choices=("f32", "bf16", "f16"), default="f32")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--layers", type=csints, default=(128, 256, 512))
    parser.add_argument(
        "--block-factors",
        type=csints,
        default=(None, None, None),
        help="tile sizes for M, N, and K dims",
    )
    parser.add_argument("--vnni", choices=(0, 2, 4), type=int, default=0)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--splats", action="store_true")

    parser.add_argument("--softmax", action="store_true")

    return vars(parser.parse_args(args))


def main(args: Sequence[str]) -> ir.Module:
    config = config_from_args(args)

    assert not config["softmax"], "not implemented yet"
    assert config["vnni"] == 0 or config["float_type"] == "bf16"

    if True:  # NB: delimits modification of global state
        gen.utils.CONSTANTS_AS_SPLATS = config["splats"]

        if config["seed"] != -1:
            random.seed(config["seed"])

    from_args = config["kernel"] == "args"
    batch_size = config["batch"]
    num_inputs = config["layers"][0]
    num_outputs = config["layers"][-1]

    m_block, n_block, k_block = config["block_factors"]
    vnni_block = config["vnni"]

    def input_tensor_type(shape: Sequence[int], elem_type: ir.Type):
        if m_block and k_block:
            m_as_batch, k_as_num_inputs = shape
            assert m_as_batch % m_block == 0, "invalid tile size for BATCH dim"
            assert k_as_num_inputs % k_block == 0, "invalid tile size for ??? dim"
            shape = (
                m_as_batch // m_block,
                k_as_num_inputs // k_block,
                m_block,
                k_block,
            )

        return ir.RankedTensorType.get(shape, elem_type)

    def weights_tensor_type(shape: Sequence[int], elem_type: ir.Type):
        if k_block and n_block:
            k_as_num_inputs, n_as_num_outputs = shape
            assert k_as_num_inputs % k_block == 0, "invalid tile size for BATCH dim"
            assert n_as_num_outputs % n_block == 0, "invalid tile size for ??? dim"
            if vnni_block:
                assert (
                    n_block % vnni_block == 0
                ), "incompatible tile sizes for N and VNNI dims"
                # TODO(RM): double check vs MLIRGen.cpp as this seems bonkers to me:
                shape = (
                    n_as_num_outputs // n_block,
                    k_as_num_inputs // k_block,
                    n_block,
                    k_block // vnni_block,
                    vnni_block,
                )
            else:
                # TODO(RM): double check vs MLIRGen.cpp as this seems bonkers to me:
                shape = (
                    n_as_num_outputs // n_block,
                    k_as_num_inputs // k_block,
                    n_block,
                    k_block,
                )
        else:
            if vnni_block:
                assert False, "--vnni without --block-factors is not supported yet"

        return ir.RankedTensorType.get(shape, elem_type)

    def bias_tensor_type(shape: Sequence[int], elem_type: ir.Type):
        if n_block:
            (n_as_num_outputs,) = shape
            assert n_as_num_outputs % n_block == 0, "invalid tile size for K dim"
            shape = (n_as_num_outputs // n_block, n_block)

        return ir.RankedTensorType.get(shape, elem_type)

    def output_tensor_type(shape: Sequence[int], elem_type: ir.Type):
        if m_block and n_block:
            m_as_batch, n_as_num_outputs = shape
            assert m_as_batch % m_block == 0, "invalid tile size for BATCH dim"
            assert n_as_num_outputs % n_block == 0, "invalid tile size for K dim"
            shape = (
                m_as_batch // m_block,
                n_as_num_outputs // n_block,
                m_block,
                n_block,
            )
        return ir.RankedTensorType.get(shape, elem_type)

    times_weights, add_bias, relu = {
        "named": (named.times_weights, named.add_bias, named.relu),
        "einsum": (einsum.times_weights, einsum.add_bias, einsum.relu),
        "generic": (generic.times_weights, generic.add_bias, generic.relu),
    }[config["output"]]

    with ir.Context(), ir.Location.name(" ".join(sys.argv)):
        elem_type = {"bf16": ir.BF16Type.get(), "f32": ir.F32Type.get()}[
            config["float_type"]
        ]

        overall_args_types = [input_tensor_type((batch_size, num_inputs), elem_type)]

        def overall_args_types_generator():
            yield input_tensor_type((batch_size, num_inputs), elem_type)
            for layer_num_neurons, next_layer_num_neurons in zip(
                config["layers"][:-1], config["layers"][1:]
            ):
                yield weights_tensor_type(
                    (layer_num_neurons, next_layer_num_neurons), elem_type
                )
                if config["bias"]:
                    yield bias_tensor_type((next_layer_num_neurons,), elem_type)
                yield output_tensor_type(
                    (batch_size, next_layer_num_neurons), elem_type
                )

        overall_args_types = tuple(overall_args_types_generator())

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            func_args_types = (
                overall_args_types if from_args else overall_args_types[:1]
            )
            func_outputs_type = (
                output_tensor_type((batch_size, num_outputs), elem_type),
            )

            @func.func(*func_args_types, results=func_outputs_type)
            def entry(*args):
                args_or_arg_types = iter(
                    args if from_args else (args + overall_args_types[1:])
                )

                layer_inputs = next(args_or_arg_types)
                for _layer_num_outputs in config["layers"][1:]:
                    weights_or_weights_type = next(args_or_arg_types)
                    if config["bias"]:
                        bias_or_bias_type = next(args_or_arg_types)
                    outputs_or_outputs_type = next(args_or_arg_types)

                    result = times_weights(
                        layer_inputs, weights_or_weights_type, outputs_or_outputs_type
                    )
                    if config["bias"]:
                        result = add_bias(result, bias_or_bias_type)
                    if config["relu"]:
                        result = relu(result)

                    layer_inputs = result

                func.ReturnOp((layer_inputs,))

        return module


if __name__ == "__main__":
    print(main(sys.argv[1:]))
