from typing import Optional, Sequence

from mlir import ir
from mlir.dialects import transform
from .common import apply_registered_pass, match
from .utils import GpuBackend, PipelineInterrupt

from ..xsmm import utils as xsmm_utils


__all__ = []


# TODO: consider making into a NamedSequence to call with IncludeOp
def cleanup(op, **_config):
    op = apply_registered_pass(op, "canonicalize")
    op = apply_registered_pass(op, "cse")
    return op


__all__.append(cleanup.__name__)


# TODO: make bundle into a NamedSequence to call with IncludeOp
def tpp_mapping(mod, lower_pack_unpack_without_transpose: bool = False, **_config):
    "High-level transforms that map operations to TPP-compatible forms."

    # Canonicalize.
    func = match(mod, ops={"func.func"})
    mod = cleanup(mod)
    # Convert ops to packed layouts.
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "pack-matmul")
    apply_registered_pass(func, "pack-vnni")
    if lower_pack_unpack_without_transpose:
        mod = apply_registered_pass(mod, "lower-packs-unpacks-without-transpose")
    # Postprocess packing.
    # Run only canonicalizer at this stage as full cleanup (mostly CSE) can
    # mess up tensor producer-consumer chains used for analysis in the
    # following passes.
    func = match(mod, ops={"func.func"})
    apply_registered_pass(func, "propagate-pack-and-unpack")
    mod = apply_registered_pass(mod, "constant-fold-pack")
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "simplify-pack")
    apply_registered_pass(func, "linalg-generalize-named-ops")
    mod = cleanup(mod)
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "linalg-convert-compare-select-to-maximumf-pass")
    func = apply_registered_pass(func, "tile-consumer-and-fuse-producers")
    apply_registered_pass(func, "simplify-pack")
    mod = cleanup(mod)
    return mod


__all__.append(tpp_mapping.__name__)


# TODO: make bundle into a NamedSequence to call with IncludeOp
def linalg_lowering(mod, /, *, skip_operations: Sequence[str] = (), **_config):
    "Lower Linalg into combination of standard and local dialects."

    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(
        func,
        "convert-linalg-to-xsmm",
        options={"skip-operations": ",".join(skip_operations)},
    )
    func = apply_registered_pass(func, "combine-xsmm-op-optimization")
    func = apply_registered_pass(func, "fold-xsmm-flags")
    apply_registered_pass(func, "verify-xsmm-calls")
    return mod


__all__.append(linalg_lowering.__name__)


# TODO: make bundle into a NamedSequence to call with IncludeOp
def vector_to_xsmm(mod, **_config):
    """Vector-level transforms that map vector patterns to
    libxsmm call pairs (dispatch, invoke)."""

    mod = apply_registered_pass(mod, "vector-to-xsmm")
    return mod


__all__.append(vector_to_xsmm.__name__)


vector_to_xsmm_bundle = vector_to_xsmm  # Due to name clash with cmd option.


# TODO: make bundle into a NamedSequence to call with IncludeOp
def vector_to_kernel(mod, **_config):
    """Vector-level transforms which map vector patterns to
    specialized micro-kernels akin to libxsmm kernels."""

    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "vector-contract-to-bf16dp")
    func = apply_registered_pass(func, "hoist-vector-transfer")
    if xsmm_utils.has_amx():
        func = apply_registered_pass(func, "vector-contract-to-amx")
    func = apply_registered_pass(func, "canonicalize")
    apply_registered_pass(func, "vector-contract-to-fma")
    return mod


__all__.append(vector_to_kernel.__name__)


# TODO: make bundle into a NamedSequence to call with IncludeOp
def low_level_parallel(
    mod,
    /,
    *,
    parallel_task_grid: Sequence[int],  # NB: should be `Seq["certain pos ints"]`
    **_config,
):
    "Low-level parallelization, 2D blocking, AMX config"

    # Note that LICM should be performed before any function calls are generated
    # to ensure that ops which map directly to functions also get moved outside
    # of loops, if possible. This approach assumes that the function calls do
    # not have any side effects and can be safely moved outside of loop body.
    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "loop-invariant-code-motion")
    # Run cleanup after LICM to allow CSE to eliminate common operations now
    # that they are hoisted out of loops.
    mod = cleanup(mod)
    options = {"parallel-loop-tile-sizes": ",".join(map(str, parallel_task_grid))}
    mod = apply_registered_pass(mod, "scf-parallel-loop-tiling", options=options)
    return mod


__all__.append(low_level_parallel.__name__)


# TODO: make bundle into a NamedSequence to call with IncludeOp
def lower_local_dialects(mod, **_config):
    "Lower our Check and Perf dialects to standard dialects and function calls."

    func = match(mod, ops={"func.func"})
    func = apply_registered_pass(func, "convert-check-to-loops")
    apply_registered_pass(func, "convert-perf-to-loops")
    mod = apply_registered_pass(mod, "convert-perf-to-func")
    return mod


__all__.append(lower_local_dialects.__name__)


# TODO: make bundle into a NamedSequence to call with IncludeOp
def postprocess(mod, /, **_config):
    """Various post-processing transforms such as LICM, parallel loop fusion,
    buffer deallocation, general cleanup etc."""

    # Postprocess buffers.
    func = match(mod, ops={"func.func"})
    apply_registered_pass(func, "buffer-hoisting")
    mod = cleanup(mod)
    return mod


__all__.append(postprocess.__name__)


# TODO: make bundle into a NamedSequence to call with IncludeOp
def default_tpp_passes(
    mod,
    /,
    *,
    linalg_to_vector: bool = False,
    vector_to_xsmm: bool = False,
    vector_to_kernel: bool = False,
    linalg_to_loops: bool = False,
    register_blocking: Sequence[int] = [],
    **config,
):
    # We currently have four flows:
    #  * linalg-to-xsmm: linalg to XSMM-calls patterns -- the default.
    #  * linalg-to-vector: no changes at linalg level, lower to straight loops.
    #  * vector-to-xsmm: linalg-to-vector and vector to XSMM-calls patterns.
    #  * vector-to-kernel: linalg-to-vector and vector to XSMM-like micro-kernel
    #      patterns via specialized lowering of certain vector patterns.
    if vector_to_kernel and vector_to_xsmm:
        raise ValueError("XSMM and Kernel lowering are mutually exclusive")
    force_linalg_to_vector = vector_to_kernel or vector_to_xsmm

    # List of operations to skip when lowering Linalg to XSMM / Kernel.
    # This allows further passes to lower to vector, function, codegen
    skip_ops = set()
    # General linalg-to-vector choice needs to skip all XSMM matching at linalg
    # level.
    if linalg_to_vector or vector_to_kernel:
        skip_ops |= {"all"}
    if vector_to_xsmm:
        skip_ops = {"unary", "transpose", "vnni"}

    mod = apply_registered_pass(mod, "fold-add-into-dest")
    if linalg_to_loops:
        # Lower linalg directly to loops, skipping all TPP transformations.
        func = match(mod, ops={"func.func"})
        func = apply_registered_pass(func, "lower-packs-unpacks")
        apply_registered_pass(func, "decompose-aggregated-ops")
        mod = apply_registered_pass(mod, "bufferize")
        func = match(mod, ops={"func.func"})
        apply_registered_pass(func, "convert-linalg-to-loops")
        mod = cleanup(mod)
    else:
        mod = apply_registered_pass(mod, "fold-into-eltwise")
        func = match(mod, ops={"func.func"})
        func = apply_registered_pass(func, "convert-linalg-to-inplace")

        apply_registered_pass(func, "rewrite-batch-matmul-to-matmul")
        # Bundle of linalg-level passes to fuse and pack:
        mod = tpp_mapping(mod, **config)  # TODO: convert to called NamedSequence
        func = match(mod, ops={"func.func"})
        apply_registered_pass(func, "lower-packs-unpacks")
        mod = cleanup(mod)
        func = match(mod, ops={"func.func"})
        # Decompose Aggregated operations. These ops currently do not bufferize.
        # Once this is possible we can move this pass after bufferization.
        apply_registered_pass(func, "decompose-aggregated-ops")
        mod = apply_registered_pass(mod, "bufferize")
        mod = linalg_lowering(mod, skip_operations=skip_ops, **config)
        if linalg_to_vector or force_linalg_to_vector:
            func = match(mod, ops={"func.func"})
            options = {"registerTileShape": ",".join(map(str, register_blocking))}
            func = apply_registered_pass(func, "brgemm-linalg-tiling", options=options)
            func = apply_registered_pass(func, "loop-invariant-code-motion")
            apply_registered_pass(func, "vectorization-pass")
            # NB: canonicalizer should be after hoisting pass because
            # it fuses outer tiling loops and it results in no pattern
            # matching for hoisting pass. Moved inside VectorToKernel Path.
            if vector_to_xsmm:
                mod = vector_to_xsmm_bundle(mod)
            if vector_to_kernel:
                mod = vector_to_kernel(mod)
        mod = cleanup(mod)
    func = match(mod, ops={"func.func"})
    apply_registered_pass(func, "convert-forall-to-parallel")

    if linalg_to_vector:
        mod = apply_registered_pass(mod, "convert-vector-to-scf")
        mod = low_level_parallel(mod, **config)
    else:
        mod = low_level_parallel(mod, **config)
        # TODO: These passes have been moved out of low level parallelization
        # pass since these apply on xsmm dialect. They'll be moved back in
        # subsequent commits.
        func = match(mod, ops={"func.func"})
        func = apply_registered_pass(func, "intel-amx-tile-config-insertion-pass")
        func = apply_registered_pass(func, "canonicalize")
        func = apply_registered_pass(func, "loop-invariant-code-motion")
        func = apply_registered_pass(func, "canonicalize")
        apply_registered_pass(func, "intel-amx-tile-config-hoisting-pass")
        # TODO: This pass has been moved out of LocalDialectsLowering since it is
        # applicable to xsmm only. It'll be moved back in subsequent commits.
        mod = apply_registered_pass(mod, "convert-xsmm-to-func")
    # Convert all local TPP-related dialects.
    mod = lower_local_dialects(mod, **config)
    # Clean up after the default pipeline.
    mod = postprocess(mod, **config)
    return mod


__all__.append(default_tpp_passes.__name__)


# TODO: make bundle into a NamedSequence to call with IncludeOp
def default_pipeline(
    mod,
    /,
    *,
    def_parallel: bool = False,
    gpu_backend: Optional[GpuBackend] = None,
    **config,
):
    if "early" in config["dump"]:
        transform.PrintOp(target=mod, name="DUMP: stage-early")

    if gpu_backend:
        assert False, "GpuPipeline bundle not implemented for now"
    else:
        mod = default_tpp_passes(mod, **config)

    if "mid" in config["dump"]:
        transform.PrintOp(target=mod, name="DUMP: stage-mid")

    # Bail out early for Intel GPU.
    # The rest of the lowering is performed by IMEX.
    if gpu_backend == "intel":
        transform.PrintOp(target=mod)
        return mod

    # Partial lowering.
    mod = apply_registered_pass(mod, "expand-strided-metadata")
    mod = apply_registered_pass(mod, "convert-tensor-to-linalg")
    func = match(mod, ops={"func.func"})
    apply_registered_pass(func, "convert-linalg-to-loops")
    if def_parallel:
        mod = apply_registered_pass(mod, "convert-scf-to-openmp")
    mod = apply_registered_pass(mod, "convert-vector-to-scf")
    mod = apply_registered_pass(mod, "arith-expand")
    mod = apply_registered_pass(mod, "lower-affine")

    if "late" in config["dump"]:
        transform.PrintOp(target=mod, name="DUMP: stage-late")

    # Lower to LLVM
    # TODO: add support for detecting target architecture, i.e. to replicate:
    #     #if defined(__x86_64__)
    #     options.x86Vector = true;
    #     #endif
    options = {"enable-amx": int(xsmm_utils.has_amx())}
    mod = apply_registered_pass(mod, "convert-vector-to-llvm", options=options)
    mod = apply_registered_pass(mod, "finalize-memref-to-llvm")
    mod = apply_registered_pass(mod, "convert-scf-to-cf")

    if gpu_backend:
        func = match(mod, ops={"func.func"})
        apply_registered_pass(func, "gpu-async-region")
        assert False
        # gpu-to-llvm cannot be invoked from transform-interpreter as it
        # tries to load ... something while multi-threaded PassManager is running.
        mod = apply_registered_pass(mod, "gpu-to-llvm")
        options = {"compilation-target": "fatbin"}
        mod = apply_registered_pass(mod, "gpu-module-to-binary", options=options)
    mod = apply_registered_pass(mod, "convert-math-to-llvm")
    if gpu_backend:
        mod = apply_registered_pass(mod, "async-to-async-runtime")
        mod = apply_registered_pass(mod, "async-runtime-ref-counting")
        mod = apply_registered_pass(mod, "convert-async-to-llvm")

    mod = apply_registered_pass(mod, "convert-index-to-llvm")
    mod = apply_registered_pass(mod, "convert-func-to-llvm")
    mod = apply_registered_pass(mod, "convert-arith-to-llvm")
    mod = apply_registered_pass(mod, "convert-cf-to-llvm")
    if def_parallel:
        mod = apply_registered_pass(mod, "convert-openmp-to-llvm")
    mod = apply_registered_pass(mod, "convert-ub-to-llvm")
    mod = apply_registered_pass(mod, "canonicalize")
    mod = apply_registered_pass(mod, "cse")
    mod = apply_registered_pass(mod, "reconcile-unrealized-casts")

    # Anything useful has been lowered by now.
    # Cleanup IR by removing any dead symbols.
    # This step aims to avoid errors caused by frontend leftovers.
    # See issue: #704
    mod = apply_registered_pass(mod, "symbol-dce")

    if "llvm" in config["dump"]:
        transform.PrintOp(target=mod, name="DUMP: stage-llvm")

    return mod


__all__.append(default_pipeline.__name__)
