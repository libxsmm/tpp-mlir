#!/usr/bin/env python3
"""Convert a PyTorch model Python file into MLIR using the Lighthouse ingress.

The input file must define a PyTorch ``nn.Module`` (class name ``Model`` by
default) and a function returning sample inputs (``get_inputs`` by default).
The resulting MLIR module is written to standard output, or to a file when the
``-o/--output`` option is given.

Run it inside the Lighthouse ``uv`` environment, e.g. from the tpp-mlir root:

    uv run --project third_party/lighthouse tools/pytorch/import.py model.py
"""

import argparse
import inspect
import sys
from pathlib import Path

import torch

from lighthouse.ingress.torch import import_from_model, import_model
from lighthouse.utils.importer import import_python_module

DIALECTS = ["linalg-on-tensors", "torch", "tosa", "stablehlo"]


def eval_arg(expr):
    """Evaluate a CLI value expression, falling back to the raw string."""
    try:
        return eval(expr, {"torch": torch})
    except (NameError, SyntaxError):
        return expr


def parse_named_args(extras):
    """Turn leftover '--name value' / '--name=value' / '--flag' args into a dict.

    Names have dashes normalized to underscores. Values are evaluated with
    'torch' in scope; bare flags become True.
    """
    named = {}
    i = 0
    while i < len(extras):
        tok = extras[i]
        if not tok.startswith("--"):
            raise SystemExit(f"import.py: unexpected argument: {tok}")
        key = tok[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            named[key.replace("-", "_")] = eval_arg(value)
            i += 1
        elif i + 1 < len(extras) and not extras[i + 1].startswith("--"):
            named[key.replace("-", "_")] = eval_arg(extras[i + 1])
            i += 2
        else:
            named[key.replace("-", "_")] = True
            i += 1
    return named


def apply_init_overrides(module, model_class, base_args, overrides):
    """Merge named overrides into the model constructor's positional init args.

    'base_args' is the default positional init list (e.g. from get_init_inputs);
    'overrides' maps constructor parameter names to values. Returns a positional
    list with the overridden values placed at their parameter positions. When an
    override name also exists as a module-level global, that global is updated
    too so helpers reading it (e.g. get_inputs) stay consistent.
    """
    cls = getattr(module, model_class, None)
    if cls is None:
        raise SystemExit(f"import.py: model class '{model_class}' not found")

    params = list(inspect.signature(cls.__init__).parameters.values())[1:]  # drop self
    names = [p.name for p in params]
    index = {name: i for i, name in enumerate(names)}

    module_globals = vars(module)
    merged = list(base_args)
    for name, value in overrides.items():
        if name not in index:
            raise SystemExit(
                f"import.py: '--{name}' is not a constructor argument of {model_class}"
            )
        pos = index[name]
        while len(merged) <= pos:
            default = params[len(merged)].default
            merged.append(None if default is inspect.Parameter.empty else default)
        merged[pos] = value
        if name in module_globals:
            setattr(module, name, value)
    return merged


def make_splat_hooks():
    """FX importer hook that emits uniform tensors as compact splat attributes.

    torch-mlir materializes every multi-element constant as a dense_resource
    blob, which bloats the output for large weights. When every element of a
    tensor is identical, this hook emits a splat ``dense<value>`` instead.
    """
    from torch_mlir import ir
    from torch_mlir.extras.fx_importer import FxImporterHooks, create_mlir_tensor_type

    class SplatHooks(FxImporterHooks):
        def resolve_literal(self, gni, literal, info):
            if not isinstance(literal, torch.Tensor) or literal.numel() <= 1:
                return None
            flat = literal.reshape(-1)
            if not bool(torch.all(flat == flat[0])):
                return None

            tensor_type = create_mlir_tensor_type(literal)
            element_type = tensor_type.element_type
            value = flat[0].item()
            if literal.is_floating_point():
                scalar = ir.FloatAttr.get(element_type, float(value))
            else:
                scalar = ir.IntegerAttr.get(element_type, int(value))

            elements = ir.DenseElementsAttr.get_splat(tensor_type, scalar)
            vtensor_type = gni._cc.tensor_to_vtensor_type(literal)
            return ir.Operation.create(
                name="torch.vtensor.literal",
                results=[vtensor_type],
                attributes={"value": elements},
            ).result

    return SplatHooks()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_file",
        help="Path to the Python file defining the PyTorch model.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write MLIR to FILE instead of standard output.",
    )
    parser.add_argument(
        "--model-class",
        default="Model",
        help="Name of the nn.Module class in the file (default: Model).",
    )
    parser.add_argument(
        "--dialect",
        choices=DIALECTS,
        default="linalg-on-tensors",
        help="Target MLIR dialect (default: linalg-on-tensors).",
    )
    parser.add_argument(
        "--init-args-fn",
        default="get_init_inputs",
        help="Function returning the model init args, or 'none' to skip "
        "(default: get_init_inputs).",
    )
    parser.add_argument(
        "--init-args",
        metavar="EXPR",
        help="Python expression evaluated to the model init args (an iterable), "
        "used instead of --init-args-fn. 'torch' is available. Example: '[64, 64]'.",
    )
    parser.add_argument(
        "--sample-args-fn",
        default="get_inputs",
        help="Function returning the sample inputs (default: get_inputs).",
    )
    parser.add_argument(
        "--sample-args",
        metavar="EXPR",
        help="Python expression evaluated to the sample inputs (an iterable), "
        "used instead of --sample-args-fn. 'torch' is available. "
        "Example: '(torch.ones(64, 64),)'.",
    )
    parser.add_argument(
        "--splat",
        action="store_true",
        help="Emit uniform weight tensors as compact splat 'dense<value>' "
        "attributes instead of full dense_resource blobs.",
    )
    # Unrecognized '--name value' args override the Model constructor by name.
    return parser.parse_known_args(argv)


def main(argv=None):
    args, extras = parse_args(argv)

    init_args_fn = None if args.init_args_fn.lower() == "none" else args.init_args_fn

    model_init_args = None
    if args.init_args is not None:
        model_init_args = eval(args.init_args, {"torch": torch})

    sample_args = None
    if args.sample_args is not None:
        sample_args = eval(args.sample_args, {"torch": torch})

    overrides = parse_named_args(extras)
    if overrides:
        module = import_python_module(Path(args.model_file))
        base = model_init_args
        if base is None and init_args_fn is not None:
            init_fn = getattr(module, init_args_fn, None)
            base = list(init_fn()) if init_fn is not None else []
        model_init_args = apply_init_overrides(
            module, args.model_class, base or [], overrides
        )
        # apply_init_overrides may update module globals; recompute the sample
        # inputs from the same module so their shapes track the overrides.
        if sample_args is None:
            sample_fn = getattr(module, args.sample_args_fn, None)
            if sample_fn is not None:
                sample_args = sample_fn()

    extra = {"hooks": make_splat_hooks()} if args.splat else {}

    nn_model, sample_args, sample_kwargs = import_model(
        args.model_file,
        model_class_name=args.model_class,
        init_args_fn_name=init_args_fn,
        model_init_args=model_init_args,
        sample_args_fn_name=args.sample_args_fn,
        sample_args=sample_args,
    )

    mlir = import_from_model(
        nn_model,
        sample_args,
        sample_kwargs=sample_kwargs,
        dialect=args.dialect,
        **extra,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(str(mlir))
            f.write("\n")
    else:
        print(mlir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
