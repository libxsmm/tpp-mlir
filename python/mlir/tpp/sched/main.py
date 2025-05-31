import sys
from typing import Sequence, Dict, Any
from argparse import ArgumentParser

from mlir import ir, passmanager
from mlir.dialects import transform
from mlir.dialects import check, perf, xsmm  # Registers our TPP-MLIR dialects.
from mlir.dialects.transform import interpreter as transform_interpreter

from . import bundles
from .common import match
from .utils import PipelineInterrupt, GpuBackend


TRANSFORM_INTERPRETER_THROUGH_PASS_MANAGER = False

# Mapping of bundle names to their corresponding transform ops-generating funcs.
BUNDLES = dict(
    (name.replace("_", "-"), obj)  # Translate snake_case function names to kebab-case.
    for name, obj in bundles.__dict__.items()
    if name in bundles.__all__
)


def config_from_args(args: Sequence[str]) -> Dict[str, Any]:
    def comma_separated_ints(arg: str):
        return [int(n) for n in arg.split(",")]

    parser = ArgumentParser(prog="tpp-sched", description="TODO")
    parser.add_argument("input-file", type=str, help="input file", nargs="?")
    parser.add_argument(
        "--payload", type=str, help="payload file to print with schedule"
    )

    parser.add_argument("--split-input-file", action="store_true")

    def comma_separated_bundles(arg: str):
        return [BUNDLES[name] for name in arg.split(",")]

    parser.add_argument(
        "--bundles", type=comma_separated_bundles, default=[BUNDLES["default-pipeline"]]
    )

    dump_stages = ["early", "mid", "late", "llvm"]
    dump_stages_dict = dict(zip(dump_stages, dump_stages))

    def dump_args_parser(arg: str):
        return set(dump_stages_dict[name] for name in arg.split(","))

    parser.add_argument("--dump", type=dump_args_parser, dest="dump", default=set())

    # Options used by the bundles.
    parser.add_argument(
        "--gpu", choices=[o.value for o in GpuBackend], dest="gpu_backend"
    )
    parser.add_argument(
        "--parallel-task-grid", type=comma_separated_ints, default="2,8"
    )
    parser.add_argument(
        "--register-blocking", type=comma_separated_ints, default="8,32"
    )
    parser.add_argument("--def-parallel", action="store_true")
    parser.add_argument("--vector-to-xsmm", action="store_true")
    parser.add_argument("--vector-to-kernel", action="store_true")
    parser.add_argument("--linalg-to-vector", action="store_true")
    parser.add_argument("--linalg-to-loops", action="store_true")
    parser.add_argument("--lower-pack-unpack-without-transpose", action="store_true")

    key_value_mapping = vars(parser.parse_args(args))
    key_value_mapping["input_file"] = key_value_mapping.pop("input-file")

    return key_value_mapping


def overall_schedule(**config: Dict[str, Any]) -> ir.Module:
    module = ir.Module.create()
    module.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    with ir.InsertionPoint(module.body):
        named_sequence = transform.NamedSequenceOp(
            "__transform_main",
            [transform.AnyOpType.get()],  # input types
            [],  # output types
            arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
        )
        with ir.InsertionPoint(named_sequence.body):
            # The main handle that each of the bundles operates on is to the
            # module which directly contains the `func.func`s.
            func = match(named_sequence.bodyTarget, ops={"func.func"})
            mod = transform.get_parent_op(
                transform.AnyOpType.get(),
                func,
                op_name="builtin.module",
                deduplicate=True,
            )

            try:
                # Now pass the module handle through the bundles.
                # Each bundle returns a handle to the same module.
                for bundle in config["bundles"]:
                    mod = bundle(mod, **config)
            except PipelineInterrupt:
                # Allow for early stopping of schedule generation.
                # Useful for debugging purposes.
                pass
            finally:
                transform.YieldOp()
    return module


def main(args: Sequence[str]) -> ir.Module:
    config: Dict[str, Any] = config_from_args(args)

    if config["input_file"] and config["payload"]:
        print("Error: Cannot specify both input-file and payload.", file=sys.stderr)
        sys.exit(1)

    with ir.Context(), ir.Location.name("main_schedule"):
        # Derive the overall schedule from the specified config.
        schedule: ir.Module = overall_schedule(**config)

        if not config["input_file"] and not config["payload"]:
            # If no payload file is provided, just print the schedule.
            # TODO: add support for checking if there's incoming contents on stdin.
            print(schedule)
            return

        payload_path = config["input_file"] or config["payload"]

        # If an input file is provided, we load it and apply the schedule to it.
        container = ir.Module.create()
        if payload_path == "-":
            contents = sys.stdin.read()
        else:
            with open(payload_path, "r") as f:
                contents = f.read()

        if config["split_input_file"]:
            payloads = [ir.Module.parse(chunk) for chunk in contents.split("-----\n")]
        else:
            payloads = [ir.Module.parse(contents)]

        for payload in payloads:
            # Make payload IR and schedule IR children of a common parent module.
            container.body.append(payload.operation)
            container.body.append(schedule.operation)

            if config["payload"]:
                # When payload is provided via --payload, just dump it alongside the schedule.
                print(container)
                del payload  # Avoid double free error. See comment below.
                return

            if TRANSFORM_INTERPRETER_THROUGH_PASS_MANAGER:
                # NB: When running through the pass manager, the dependent dialects
                #     of transforms and passes cannot be loaded during pass manager
                #     execution. Expect errors when your transforms or passes
                #     require downstream dialects.
                pm = passmanager.PassManager()
                pm.add("transform-interpreter")
                pm.run(container.operation)
            else:
                main_named_sequence = schedule.body.operations[0]

                # Invoke the transform interpreter directly, without the pass manager being involved.
                transform_interpreter.apply_named_sequence(
                    payload_root=payload,
                    transform_root=main_named_sequence,
                    transform_module=container,
                )

            print(payload)

            # NB: MLIR is confused about ownership of the payload IR.
            #     Without explicitly deleting the python object we get
            #     a double free error when the python object is deleted.
            del payload
        del payloads
