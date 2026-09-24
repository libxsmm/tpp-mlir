#!/usr/bin/env python3
"""Emit an MLIR batch-reduce matmul via ``linalg.contract`` (no torch-mlir).

The generated ``@entry`` function contracts A (br x M x K) and B (br x K x N)
into C (M x N) with the batch dim folded into ``linalg.contract`` as a reduction
dimension (via 4-D indexing maps d0=batch, d1=M, d2=N, d3=K). Both ``tensor``
and ``memref`` containers are supported (``--container``), and the
register-blocking hint is attached as a ``#dlti.target_system_spec``
``reg_gemm_unroll`` attribute built from ``--vtM/--vtN/--vtK``.

When the C element type equals the f32/i32 accumulator type the contract
accumulates directly into C. When C is a narrower float (e.g. bf16 out, f32
accumulate) the kernel contracts into an f32 temporary and a ``linalg.generic``
epilogue adds the incoming C and down-converts (extf + addf + truncf).

Run inside the lighthouse uv env, e.g.:

    uv run --project third_party/lighthouse --extra ingress_torch_cpu \
        tools/pytorch/emit_brgemm.py brgemm --M 64 --N 64 --K 64 \
        --br_count 8 --aType bf16 --bType bf16 --cType f32 --container tensor
"""

import argparse

from mlir import ir
from mlir.dialects import func, linalg, tensor, arith, memref

# ---------------------------------------------------------------------------
# linalg.contract batch-reduce matmul
# ---------------------------------------------------------------------------
# bf8/hf8 are libxsmm's FP8 names; they map to the MLIR f8E5M2 / f8E4M3FN
# element types in the emitted IR.
_ALIASES = {"bf8": "f8E5M2", "hf8": "f8E4M3FN"}
# Supported user-facing element types for the A/B operands and the C output.
_AB_TYPES = ("f64", "i64", "f32", "f16", "bf16", "i16", "bf8", "hf8", "i8")
_C_TYPES = ("i64", "i32", "i16", "i8", "bf8", "hf8", "bf16", "f16", "f32", "f64")
# C types narrower than the f32/i32 accumulator: emitted via the down-convert
# epilogue. F16/BF16/BF8/HF8 accumulate in f32 then truncf; I16/I8 accumulate
# in i32 then trunci. The wide C types (f32/f64/i32) accumulate straight into C.
_NARROW_C = {"f16", "bf16", "i16", "bf8", "hf8", "i8"}
# Computation (accumulator) types selectable via --compType (libxsmm "comp").
_COMP_TYPES = ("i32", "i64", "f32", "f64")
_CONTAINERS = ("tensor", "memref")


def _get_type(name):
    name = _ALIASES.get(name, name)
    if name == "f64":
        return ir.F64Type.get()
    if name == "f32":
        return ir.F32Type.get()
    if name == "f16":
        return ir.F16Type.get()
    if name == "bf16":
        return ir.BF16Type.get()
    if name[0] == "i" and name[1:].isdigit():
        return ir.IntegerType.get_signless(int(name[1:]))
    # f8E4M3FN / f8E5M2 and any other named element type.
    return ir.Type.parse(name)


def _is_int(name):
    return name.startswith("i")


def _acc_name(c_name):
    """Accumulator element type for a given C output type: narrow C accumulates
    in i32 (integer) or f32 (float); wide C (f32/f64/i32) accumulates at its own
    precision (the contract writes straight into C)."""
    if c_name in _NARROW_C:
        return "i32" if _is_int(c_name) else "f32"
    return c_name


# VNNI reduction-packing factor per element type (f32/f64/i64 have no VNNI).
_VNNI_FACTOR = {"bf16": 2, "f16": 2, "i16": 2, "i8": 4, "bf8": 4, "hf8": 4}

# Element bit widths (resolved names), used to pick ext vs trunc when converting
# between the computation type and C.
_BITS = {"f8E5M2": 8, "f8E4M3FN": 8, "f16": 16, "bf16": 16, "f32": 32,
         "f64": 64, "i8": 8, "i16": 16, "i32": 32, "i64": 64}


def build_contract(m, n, k, br_count, a_name, b_name, c_name,
                   vtm, vtn, vtk, container, lda=None, ldb=None, ldc=None,
                   trans_a=False, trans_b=False, vnni_a=False, vnni_b=False,
                   alpha=1.0, beta=1.0, comp_name=None, fn_name="entry"):
    """Build an ``@entry`` batch-reduce matmul using ``linalg.contract``.

    A is (br_count x M x K), B is (br_count x K x N) and C is (M x N). The batch
    dim is a reduction dim folded into the contraction via 4-D maps
    (d0=batch, d1=M, d2=N, d3=K). ``container`` selects tensor or memref types.
    ``vtm/vtn/vtk`` only populate the ``reg_gemm_unroll`` dlti hint.
    When ``c_name`` equals the accumulator type (f32/i32) the contract writes
    straight into C; otherwise (bf16/i8 out) it contracts into an f32/i32 temp
    and a ``linalg.generic`` epilogue adds C and down-converts to the narrower
    type (ext + add + trunc).
    trans_a/trans_b store A as K x M / B as N x K (map + shape swap). vnni_a/
    vnni_b pack the reduction (K) dim as (K/vf, vf) with vf from _VNNI_FACTOR,
    adding a 5th (reduction) contraction dim; VNNI must be on for both or
    neither operand.
    lda/ldb/ldc are optional leading (outer row) dimensions (libxsmm LDA/LDB/LDC)
    and are memref-only: each defaults to the packed row stride and must be >=
    it; a larger value makes the argument a strided view into a padded buffer
    while the logical extents stay M/N/K.
    alpha/beta scale the result as ``alpha*(A*B) + beta*C``. When both are 1.0
    (default) no scaling ops are emitted; otherwise the epilogue multiplies the
    accumulator by alpha and the incoming C by beta (mulf for float element
    types, muli for integer ones) before summing.
    comp_name (``--compType``) sets the computation/accumulator element type
    (libxsmm ``comp``); it defaults to i32 for integer C and f32 for float C.
    When it differs from C the epilogue converts between the two (ext/trunc
    chosen by bit width).
    """
    if br_count < 1:
        raise ValueError(f"br_count must be >= 1, got {br_count}")
    if container not in _CONTAINERS:
        raise ValueError(f"container must be one of {_CONTAINERS}, got {container}")
    if a_name not in _AB_TYPES:
        raise ValueError(
            f"unsupported A type {a_name}; allowed: {', '.join(_AB_TYPES)}")
    if b_name not in _AB_TYPES:
        raise ValueError(
            f"unsupported B type {b_name}; allowed: {', '.join(_AB_TYPES)}")
    if c_name not in _C_TYPES:
        raise ValueError(
            f"unsupported C type {c_name}; allowed: {', '.join(_C_TYPES)}")
    if vnni_a != vnni_b:
        raise ValueError("VNNI must be enabled for both A and B or neither")
    vnni = vnni_a
    if vnni and trans_b:
        raise ValueError("VNNI does not support transposing B (only A may be transposed)")
    if vnni and a_name not in _VNNI_FACTOR:
        raise ValueError(f"no VNNI layout for element type {a_name}")
    vf = _VNNI_FACTOR[a_name] if vnni else 1
    if vnni and k % vf != 0:
        raise ValueError(f"K={k} not divisible by the VNNI factor {vf}")
    kp = k // vf

    # Per-batch (non-batched) operand shapes: transpose swaps the 2D extents,
    # VNNI splits K into (K/vf, vf) with vf innermost.
    if vnni:
        a_inner = [kp, m, vf] if trans_a else [m, kp, vf]
        b_inner = [n, kp, vf] if trans_b else [kp, n, vf]
    else:
        a_inner = [k, m] if trans_a else [m, k]
        b_inner = [n, k] if trans_b else [k, n]
    a_lead_def = a_inner[1] * (vf if vnni else 1)
    b_lead_def = b_inner[1] * (vf if vnni else 1)

    if container != "memref" and any(v is not None for v in (lda, ldb, ldc)):
        raise ValueError("LDA/LDB/LDC are only supported for --container memref")
    a_lead = a_lead_def if lda is None else lda
    b_lead = b_lead_def if ldb is None else ldb
    c_lead = n if ldc is None else ldc
    for label, val, floor in (("LDA", a_lead, a_lead_def),
                              ("LDB", b_lead, b_lead_def),
                              ("LDC", c_lead, n)):
        if val < floor:
            raise ValueError(f"{label}={val} must be >= {floor} (packed row stride)")
    if comp_name is None:
        comp_name = _acc_name(c_name)
    elif comp_name not in _COMP_TYPES:
        raise ValueError(
            f"unsupported comp type {comp_name}; allowed: {', '.join(_COMP_TYPES)}")
    if _is_int(comp_name) != _is_int(c_name):
        raise ValueError(
            f"comp type {comp_name} and C type {c_name} must both be integer or "
            "both floating-point")
    acc_name = comp_name
    is_int = _is_int(comp_name)
    comp_res = _ALIASES.get(comp_name, comp_name)
    c_res = _ALIASES.get(c_name, c_name)

    def _cvt(val, src_res, dst_res, dst_ty):
        # Convert between the computation type and C: ext or trunc by bit width.
        if src_res == dst_res:
            return val
        wider = _BITS[dst_res] > _BITS[src_res]
        if is_int:
            return (arith.ExtSIOp(dst_ty, val).result if wider
                    else arith.TruncIOp(dst_ty, val).result)
        return (arith.ExtFOp(dst_ty, val).result if wider
                else arith.TruncFOp(dst_ty, val).result)

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        ta, tb, tc = _get_type(a_name), _get_type(b_name), _get_type(c_name)
        tacc = _get_type(acc_name)
        is_tensor = container == "tensor"

        def cont(shape, ty):
            return (ir.RankedTensorType.get(shape, ty) if is_tensor
                    else ir.MemRefType.get(shape, ty))

        def mref(shape, name, strides):
            # Explicit leading dim -> strided memref view; else contiguous.
            if strides is None:
                return ir.MemRefType.get(shape, _get_type(name))
            dims = "x".join(str(s) for s in shape)
            joined = ", ".join(str(s) for s in strides)
            return ir.Type.parse(f"memref<{dims}x{name}, strided<[{joined}]>>")

        def batch_strides(inner, lead):
            # Row-major strides for [br] + inner with outer row stride `lead`.
            s = [inner[0] * lead, lead]
            if vnni:
                s.append(vf)
            s.append(1)
            return s

        if is_tensor:
            a_ty = ir.RankedTensorType.get([br_count] + a_inner, ta)
            b_ty = ir.RankedTensorType.get([br_count] + b_inner, tb)
            c_ty = ir.RankedTensorType.get([m, n], tc)
        else:
            a_str = None if a_lead == a_lead_def else batch_strides(a_inner, a_lead)
            b_str = None if b_lead == b_lead_def else batch_strides(b_inner, b_lead)
            c_str = None if c_lead == n else [c_lead, 1]
            a_ty = mref([br_count] + a_inner, a_name, a_str)
            b_ty = mref([br_count] + b_inner, b_name, b_str)
            c_ty = mref([m, n], c_name, c_str)

        # Contraction dims: d0=batch, d1=M, d2=N, d3=K (+ d4=vf for VNNI); batch,
        # K and vf are all reduction dims.
        ndims = 5 if vnni else 4
        d = [ir.AffineDimExpr.get(i) for i in range(ndims)]
        a_dims = [d[0]] + ([d[3], d[1]] if trans_a else [d[1], d[3]]) + ([d[4]] if vnni else [])
        b_dims = [d[0]] + ([d[2], d[3]] if trans_b else [d[3], d[2]]) + ([d[4]] if vnni else [])
        maps = [
            ir.AffineMap.get(ndims, 0, a_dims),  # A: batch, M, K (+vf)
            ir.AffineMap.get(ndims, 0, b_dims),  # B: batch, K, N (+vf)
            ir.AffineMap.get(ndims, 0, [d[1], d[2]]),  # C: M, N
        ]
        dlti_attr = ir.Attribute.parse(
            f'#dlti.target_system_spec<"CPU" = '
            f'#dlti.target_device_spec<"reg_gemm_unroll" = [{vtm}, {vtn}, {vtk}]>>'
        )
        scaled = alpha != 1.0 or beta != 1.0
        direct = (comp_res == c_res) and not scaled

        with ir.InsertionPoint(module.body):
            fn = func.FuncOp(fn_name, ir.FunctionType.get([a_ty, b_ty, c_ty], [c_ty]))
            fn.attributes["dlti.target_system_spec"] = dlti_attr
            entry = fn.add_entry_block()

            with ir.InsertionPoint(entry):
                A, B, C = entry.arguments
                if direct:
                    # C is the accumulator type: contract straight into C.
                    res = linalg.contract(A, B, outs=[C], indexing_maps=maps)
                    func.ReturnOp([res if is_tensor else C])
                    return module

                # Narrow C and/or alpha/beta scaling: contract into an
                # accumulator temp, then a generic epilogue computes
                # alpha*acc + beta*C, down-converting when C is narrower.
                acc_ty = cont([m, n], tacc)
                if is_int:
                    zero = arith.ConstantOp(tacc, ir.IntegerAttr.get(tacc, 0))
                else:
                    zero = arith.ConstantOp(tacc, ir.FloatAttr.get(tacc, 0.0))
                if is_tensor:
                    tmp = tensor.EmptyOp([m, n], tacc)
                    filled = linalg.fill(zero, outs=[tmp])
                    contracted = linalg.contract(A, B, outs=[filled], indexing_maps=maps)
                else:
                    tmp = memref.alloc(acc_ty, [], [], alignment=64)
                    linalg.fill(zero, outs=[tmp])
                    linalg.contract(A, B, outs=[tmp], indexing_maps=maps)
                    contracted = tmp

                id2 = ir.AffineMap.get(
                    2, 0, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
                emaps = ir.ArrayAttr.get([ir.AffineMapAttr.get(id2)] * 3)
                par = ir.Attribute.parse("#linalg.iterator_type<parallel>")
                iters = ir.ArrayAttr.get([par, par])
                results = [c_ty] if is_tensor else []
                g = linalg.GenericOp(results, [contracted, C], [C], emaps, iters)
                blk = g.regions[0].blocks.append(tacc, tc, tc)
                with ir.InsertionPoint(blk):
                    acc_in, c_in, _out = blk.arguments
                    a_term = acc_in
                    if alpha != 1.0:
                        attr = (ir.IntegerAttr.get(tacc, int(alpha)) if is_int
                                else ir.FloatAttr.get(tacc, alpha))
                        ac = arith.ConstantOp(tacc, attr)
                        a_term = (arith.MulIOp if is_int else arith.MulFOp)(ac, acc_in).result
                    c_val = _cvt(c_in, c_res, comp_res, tacc)  # C -> comp
                    if beta != 1.0:
                        attr = (ir.IntegerAttr.get(tacc, int(beta)) if is_int
                                else ir.FloatAttr.get(tacc, beta))
                        bc = arith.ConstantOp(tacc, attr)
                        c_val = (arith.MulIOp if is_int else arith.MulFOp)(bc, c_val).result
                    summed = (arith.AddIOp if is_int else arith.AddFOp)(a_term, c_val).result
                    out_val = _cvt(summed, comp_res, c_res, tc)  # comp -> C
                    linalg.YieldOp([out_val])
                func.ReturnOp([g.results[0] if is_tensor else C])

        return module


def _add_matmul_args(p, br_default):
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--vtM", type=int, default=16,
                   help="M register-blocking hint for the reg_gemm_unroll dlti attr")
    p.add_argument("--vtN", type=int, default=16,
                   help="N register-blocking hint for the reg_gemm_unroll dlti attr")
    p.add_argument("--vtK", type=int, default=32,
                   help="K register-blocking hint for the reg_gemm_unroll dlti attr")
    p.add_argument("--aType", choices=_AB_TYPES, default="bf16")
    p.add_argument("--bType", choices=_AB_TYPES, default="bf16")
    p.add_argument("--cType", choices=_C_TYPES, default="f32")
    p.add_argument("--container", choices=_CONTAINERS, default="tensor",
                   help="emit tensor or memref types (default: tensor)")
    p.add_argument("--LDA", type=int, default=None,
                   help="A leading (outer row) stride; memref only, default K")
    p.add_argument("--LDB", type=int, default=None,
                   help="B leading (outer row) stride; memref only, default N")
    p.add_argument("--LDC", type=int, default=None,
                   help="C leading (outer row) stride; memref only, default N")
    p.add_argument("--transA", action="store_true", help="store A transposed (K x M)")
    p.add_argument("--transB", action="store_true", help="store B transposed (N x K)")
    p.add_argument("--vnniA", action="store_true", help="VNNI-pack A on the K dim")
    p.add_argument("--vnniB", action="store_true", help="VNNI-pack B on the K dim")
    p.add_argument("--br_count", type=int, default=br_default,
                   help="batch-reduce count: leading batch dim of A/B, folded "
                        "into linalg.contract as a reduction dim")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="scalar multiplier on A*B (default 1.0 = no scaling)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="scalar multiplier on C (default 1.0 = plain accumulate)")
    p.add_argument("--compType", choices=_COMP_TYPES, default=None,
                   help="computation/accumulator type (libxsmm comp); default "
                        "i32 for integer C, f32 for float C")
    p.add_argument("-o", "--output", metavar="FILE")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="gen", required=True)
    bg = sub.add_parser(
        "brgemm", help="linalg.contract batch-reduce matmul (br_count default 8)")
    _add_matmul_args(bg, br_default=8)
    mm = sub.add_parser("matmul", help="linalg.contract matmul (br_count default 1)")
    _add_matmul_args(mm, br_default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    module = build_contract(
        args.M, args.N, args.K, args.br_count,
        args.aType, args.bType, args.cType,
        args.vtM, args.vtN, args.vtK,
        args.container,
        args.LDA, args.LDB, args.LDC,
        args.transA, args.transB,
        args.vnniA, args.vnniB,
        args.alpha, args.beta,
        args.compType,
    )
    text = str(module)
    if getattr(args, "output", None):
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
