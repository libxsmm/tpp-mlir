#!/usr/bin/env python

import sys
from enum import Enum
from typing import Optional, Sequence
from argparse import ArgumentParser

from mlir import ir
from mlir.ir import Context, Location, InsertionPoint
from mlir.dialects import transform
from mlir.dialects.transform import structured


GpuBackend = Enum("GpuBackend", [("intel", "intel"), ("cuda", "cuda")])


# Wrapper to addresss verbosity
def ApplyRegisteredPass(*args, **kwargs):
  return transform.ApplyRegisteredPassOp(
    transform.AnyOpType.get(), *args, **kwargs
  )


# Wrapper to addresss verbosity
def Match(*args, **kwargs):
  return structured.MatchOp(transform.AnyOpType.get(), *args, **kwargs)


# TODO: consider making into a NamedSequence to call with IncludeOp
def CleanUp(op):
  op = ApplyRegisteredPass(op, "canonicalize")
  transform.ApplyCommonSubexpressionEliminationOp(op)
  return op


# TODO: make bundle into a NamedSequence to call with IncludeOp
def TppMapping(
  mod, lower_pack_unpack_without_transpose: bool = False, **_config
):
  # Preprocess convolutions.
  func = Match(mod, ops={"func.func"})
  ApplyRegisteredPass(func, "conv-init-simplify")
  mod = CleanUp(mod)
  # Convert ops to packed layouts.
  func = Match(mod, ops={"func.func"})
  func = ApplyRegisteredPass(func, "pack-conv2DNchwFchw")
  func = ApplyRegisteredPass(func, "pack-conv2DNhwcHwcf")
  func = ApplyRegisteredPass(func, "rewrite-conv-to-matmul-or-brgemm")
  func = ApplyRegisteredPass(func, "pack-matmul")
  ApplyRegisteredPass(func, "pack-vnni")
  if lower_pack_unpack_without_transpose:
    mod = ApplyRegisteredPass(mod, "lower-packs-unpacks-without-transpose")
  # Postprocess packing.
  # Run only canonicalizer at this stage as full cleanup (mostly CSE) can
  # mess up tensor producer-consumer chains used for analysis in the
  # following passes.
  func = Match(mod, ops={"func.func"})
  ApplyRegisteredPass(func, "propagate-pack-and-unpack")
  mod = ApplyRegisteredPass(mod, "constant-fold-pack")
  func = Match(mod, ops={"func.func"})
  func = ApplyRegisteredPass(func, "simplify-pack")
  ApplyRegisteredPass(func, "linalg-generalize-named-ops")
  mod = CleanUp(mod)
  func = Match(mod, ops={"func.func"})
  func = ApplyRegisteredPass(
    func, "linalg-convert-compare-select-to-maximumf-pass"
  )
  func = ApplyRegisteredPass(func, "tile-consumer-and-fuse-producers")
  ApplyRegisteredPass(func, "simplify-pack")
  mod = CleanUp(mod)
  return mod


# TODO: make bundle into a NamedSequence to call with IncludeOp
def LinalgLowering(mod, /, *, skip_operations: Sequence[str] = None, **_config):
  func = Match(mod, ops={"func.func"})
  func = ApplyRegisteredPass(
    func,
    "convert-linalg-to-xsmm",
    options="skip-operations=" + ",".join(skip_operations),
  )
  func = ApplyRegisteredPass(func, "combine-xsmm-op-optimization")
  func = ApplyRegisteredPass(func, "fold-xsmm-flags")
  ApplyRegisteredPass(func, "verify-xsmm-calls")
  return mod


# TODO: make bundle into a NamedSequence to call with IncludeOp
def VectorToXsmm(mod, **_config):
  mod = ApplyRegisteredPass(mod, "vector-to-xsmm")
  return mod


# TODO: make bundle into a NamedSequence to call with IncludeOp
def VectorToKernel(mod, **_config):
  func = Match(mod, ops={"func.func"})
  func = ApplyRegisteredPass(func, "hoist-vector-transfer")
  func = ApplyRegisteredPass(func, "canonicalize")
  ApplyRegisteredPass(func, "vector-contract-to-fma")
  return mod


# TODO: make bundle into a NamedSequence to call with IncludeOp
def LowLevelParallelization(
  mod,
  /,
  *,
  parallel_task_grid: Sequence[int],  # NB: should be `Seq["certain pos ints"]`
  **_config,
):
  # Note that LICM should be performed before any function calls are generated
  # to ensure that ops which map directly to functions also get moved outside
  # of loops, if possible. This approach assumes that the function calls do
  # not have any side effects and can be safely moved outside of loop body.
  func = Match(mod, ops={"func.func"})
  func = ApplyRegisteredPass(func, "loop-invariant-code-motion")
  ApplyRegisteredPass(func, "hoist-vector-transfer")
  # Run cleanup after LICM to allow CSE to eliminate common operations now
  # that they are hoisted out of loops.
  mod = CleanUp(mod)
  tile_sizes = ",".join(str(n) for n in parallel_task_grid)
  mod = ApplyRegisteredPass(
    mod,
    "scf-parallel-loop-tiling",
    options=f"parallel-loop-tile-sizes={tile_sizes}",
  )
  return mod


# TODO: make bundle into a NamedSequence to call with IncludeOp
def LocalDialectsLowering(mod, **_config):
  func = Match(mod, ops={"func.func"})
  func = ApplyRegisteredPass(func, "convert-check-to-loops")
  ApplyRegisteredPass(func, "convert-perf-to-loops")
  mod = ApplyRegisteredPass(mod, "convert-perf-to-func")
  return mod


# TODO: make bundle into a NamedSequence to call with IncludeOp
def PostProcessing(mod, /, **_config):
  # Postprocess buffers.
  func = Match(mod, ops={"func.func"})
  ApplyRegisteredPass(func, "buffer-hoisting")
  mod = CleanUp(mod)
  return mod


# TODO: make bundle into a NamedSequence to call with IncludeOp
def DefaultTpp(
  mod,
  /,
  *,
  linalg_to_vector: bool = False,
  vector_to_xsmm: bool = False,
  vector_to_kernel: bool = False,
  linalg_to_loops: bool = False,
  lhs_tile: Sequence[int] = None,  # NB: should be `Seq["certain pos ints"]`
  rhs_tile: Sequence[int] = None,  # NB: should be `Seq["certain pos ints"]`
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
  elif vector_to_xsmm:
    skip_ops |= {"transpose", "vnni"}

  mod = ApplyRegisteredPass(mod, "fold-add-into-dest")
  if linalg_to_loops:
    # Lower linalg directly to loops, skipping all TPP transformations.
    func = Match(mod, ops={"func.func"})
    func = ApplyRegisteredPass(func, "lower-packs-unpacks")
    ApplyRegisteredPass(func, "decompose-aggregated-ops")
    mod = ApplyRegisteredPass(mod, "bufferize")
    func = Match(mod, ops={"func.func"})
    ApplyRegisteredPass(func, "convert-linalg-to-loops")
    mod = CleanUp(mod)
  else:
    mod = ApplyRegisteredPass(mod, "fold-into-eltwise")
    func = Match(mod, ops={"func.func"})
    func = ApplyRegisteredPass(func, "convert-linalg-to-inplace")

    ApplyRegisteredPass(func, "rewrite-batch-matmul-to-matmul")
    # Bundle of linalg-level passes to fuse and pack:
    mod = TppMapping(mod, **config)  # TODO: convert to called NamedSequence
    func = Match(mod, ops={"func.func"})
    ApplyRegisteredPass(func, "lower-packs-unpacks")
    mod = CleanUp(mod)
    func = Match(mod, ops={"func.func"})
    ApplyRegisteredPass(func, "decompose-aggregated-ops")
    transform.PrintOp(target=mod, name="before-bufferize")
    mod = ApplyRegisteredPass(mod, "bufferize")
    mod = LinalgLowering(mod, skip_operations=skip_ops, **config)
    if linalg_to_vector or force_linalg_to_vector:
      func = Match(mod, ops={"func.func"})
      options = "lhsTile=" + ",".join(str(n) for n in lhs_tile)
      options += " " + "rhsTile=" + ",".join(str(n) for n in rhs_tile)
      func = ApplyRegisteredPass(func, "brgemm-linalg-tiling", options=options)
      func = ApplyRegisteredPass(func, "loop-invariant-code-motion")
      ApplyRegisteredPass(func, "vectorization-pass")
      # NB: canonicalizer should be after hoisting pass because
      # it fuses outer tiling loops and it results in no pattern
      # matching for hoisting pass. Moved inside VectorToKernel Path.
      if vector_to_xsmm:
        mod = VectorToXsmm(mod)
      if vector_to_kernel:
        mod = VectorToKernel(mod)
    mod = CleanUp(mod)
  func = Match(mod, ops={"func.func"})
  ApplyRegisteredPass(func, "convert-forall-to-parallel")

  if linalg_to_vector:
    mod = ApplyRegisteredPass(mod, "convert-vector-to-scf")
    mod = LowLevelParallelization(mod, **config)
  else:
    mod = LowLevelParallelization(mod, **config)
    # TODO: These passes have been moved out of low level parallelization
    # pass since these apply on xsmm dialect. They'll be moved back in
    # subsequent commits.
    func = Match(mod, ops={"func.func"})
    func = ApplyRegisteredPass(func, "intel-amx-tile-config-insertion-pass")
    func = ApplyRegisteredPass(func, "canonicalize")
    func = ApplyRegisteredPass(func, "loop-invariant-code-motion")
    func = ApplyRegisteredPass(func, "canonicalize")
    ApplyRegisteredPass(func, "intel-amx-tile-config-hoisting-pass")
    # TODO: This pass has been moved out of LocalDialectsLowering since it is
    # applicable to xsmm only. It'll be moved back in subsequent commits.
    mod = ApplyRegisteredPass(mod, "convert-xsmm-to-func")
  # Convert all local TPP-related dialects.
  mod = LocalDialectsLowering(mod, **config)
  # Clean up after the default pipeline.
  mod = PostProcessing(mod, **config)
  return mod


# TODO: make bundle into a NamedSequence to call with IncludeOp
def DefaultPipeline(
  mod,
  /,
  *,
  def_parallel: bool = False,
  gpu_backend: Optional[GpuBackend] = None,
  **config,
):
  transform.PrintOp(target=mod)
  if not gpu_backend:
    mod = DefaultTpp(mod, **config)
  else:
    assert False, "not implemented for now"
    # Bail out early for Intel GPU. The rest of the lowering is performed by IMEX.
    if gpu_backend == "intel":
      return mod

  # Partial lowering.
  mod = ApplyRegisteredPass(mod, "expand-strided-metadata")
  mod = ApplyRegisteredPass(mod, "convert-tensor-to-linalg")
  func = Match(mod, ops={"func.func"})
  ApplyRegisteredPass(func, "convert-linalg-to-loops")
  if def_parallel:
    mod = ApplyRegisteredPass(mod, "convert-scf-to-openmp")
  mod = ApplyRegisteredPass(mod, "convert-vector-to-scf")
  mod = ApplyRegisteredPass(mod, "arith-expand")
  mod = ApplyRegisteredPass(mod, "lower-affine")

  transform.PrintOp(target=mod, name="HERE!")

  # Lower to LLVM
  mod = ApplyRegisteredPass(mod, "convert-vector-to-llvm")
  mod = ApplyRegisteredPass(mod, "finalize-memref-to-llvm")
  mod = ApplyRegisteredPass(mod, "convert-scf-to-cf")
  if def_parallel:
    mod = ApplyRegisteredPass(mod, "convert-openmp-to-llvm")
  mod = ApplyRegisteredPass(mod, "convert-math-to-llvm")

  if gpu_backend:
    func = Match(mod, ops={"func.func"})
    ApplyRegisteredPass(func, "gpu-async-region")
    assert False
    # gpu-to-llvm cannot be invoked from transform-interpreter as it
    # tries to load ... something while multi-threaded PassManager is running.
    mod = ApplyRegisteredPass(mod, "gpu-to-llvm")
    mod = ApplyRegisteredPass(
      mod, "gpu-module-to-binary", options="compilation-target=fatbin"
    )
    mod = ApplyRegisteredPass(mod, "async-to-async-runtime")
    mod = ApplyRegisteredPass(mod, "async-runtime-ref-counting")
    mod = ApplyRegisteredPass(mod, "convert-async-to-llvm")

  mod = ApplyRegisteredPass(mod, "convert-func-to-llvm")
  # FIXME: once llvm-project is updated, add -convert-arith-to-llvm and -convert-cf-to-llvm here
  func = Match(mod, ops={"func.func"})
  func = ApplyRegisteredPass(func, "convert-arith-to-llvm")
  # func = ApplyRegisteredPass(func, "convert-cf-to-llvm")
  func = ApplyRegisteredPass(func, "canonicalize")
  transform.ApplyCommonSubexpressionEliminationOp(func)
  mod = ApplyRegisteredPass(mod, "reconcile-unrealized-casts")

  # Anything useful has been lowered by now.
  # Cleanup IR by removing any dead symbols.
  # This step aims to avoid errors caused by frontend leftovers.
  # See issue: #704
  transform.ApplyDeadCodeEliminationOp(mod)

  return mod


def MainSchedule(**config):
  module = ir.Module.create()
  module.operation.attributes["transform.with_named_sequence"] = (
    ir.UnitAttr.get()
  )
  with InsertionPoint(module.body):
    named_sequence = transform.NamedSequenceOp(
      "__transform_main",
      [transform.AnyOpType.get()],  # input types
      [transform.AnyOpType.get()],  # output types
      arg_attrs=[{"transform.readonly": ir.UnitAttr.get()}],
    )
    with InsertionPoint(named_sequence.body):
      func = Match(named_sequence.bodyTarget, ops={"func.func"})
      mod = transform.GetParentOp(
        transform.AnyOpType.get(),
        func,
        op_name="builtin.module",
        deduplicate=True,
      )

      mod = DefaultPipeline(mod, **config)
      transform.YieldOp(mod)
  return module


def config_from_args(args: Sequence[str]):
  def csints(s):
    return [int(n) for n in s.split(",")]

  parser = ArgumentParser(prog="tpp-opt.py", description="TODO")
  parser.add_argument(
    "--gpu", choices=[o.value for o in GpuBackend], dest="gpu_backend"
  )
  parser.add_argument("--parallel-task-grid", type=csints, default="2,8")
  parser.add_argument("--lhs-tile", type=csints, default="8,8")
  parser.add_argument("--rhs-tile", type=csints, default="8,16")
  parser.add_argument("--def-parallel", action="store_true")
  parser.add_argument("--vector-to-xsmm", action="store_true")
  parser.add_argument("--vector-to-kernels", action="store_true")
  parser.add_argument("--linalg-to-vector", action="store_true")
  parser.add_argument(
    "--lower-pack-unpack-without-transpose", action="store_true"
  )

  return vars(parser.parse_args(args))


def main():
  config = config_from_args(sys.argv[1:])
  print(config, file=sys.stderr)
  with Context(), Location.name("main_schedule"):
    module = MainSchedule(**config)

  print(module)


if __name__ == "__main__":
  main()
