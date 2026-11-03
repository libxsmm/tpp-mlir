#!/usr/bin/env python3
"""Emit an MLIR batch-reduce matmul via ``linalg.contract``.

The generated ``@entry`` function contracts A (br x M x K) and B (br x K x N)
into C (M x N) with the batch dim folded into ``linalg.contract`` as a reduction
dimension (via 4-D indexing maps d0=batch, d1=M, d2=N, d3=K). 

When the C element type equals the f32/i32 accumulator type the contract
accumulates directly into C. When C is a narrower float (e.g. bf16 out, f32
accumulate) the kernel contracts into an f32 temporary and a ``linalg.generic``
epilogue adds the incoming C and down-converts (extf + addf + truncf).

Run inside the lighthouse uv env, e.g.:

    #Run:
    uv run --project third_party/lighthouse --extra ingress_torch_cpu \
        tools/pytorch/emit_brgemm.py brgemm --M 64 --N 64 --K 64 \
        --br_count 8 --aType bf16 --bType bf16 --compType f32 --cType bf16

    #Generated IR:
    #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
    #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
    #map2 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
    #map3 = affine_map<(d0, d1) -> (d0, d1)>

    func.func @entry(%arg0: tensor<8x64x64xbf16>, %arg1: tensor<8x64x64xbf16>, %arg2: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = tensor.empty() : tensor<64x64xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x64xf32>) -> tensor<64x64xf32>
      // The gemm kernel
      %2 = linalg.contract indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<8x64x64xbf16>, 
            tensor<8x64x64xbf16>) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
      // element-wise down-convert.
      %3 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} 
            ins(%2, %arg2 : tensor<64x64xf32>, tensor<64x64xbf16>) outs(%arg2 : tensor<64x64xbf16>) {
      ^bb0(%in: f32, %in_0: bf16, %out: bf16):
        %4 = arith.extf %in_0 : bf16 to f32
        %5 = arith.addf %in, %4 : f32
        %6 = arith.truncf %5 : f32 to bf16
        linalg.yield %6 : bf16
      } -> tensor<64x64xbf16>

      return %3 : tensor<64x64xbf16>
    }

"""

import argparse
import platform

from mlir import ir
from mlir.dialects import func, linalg, tensor, arith

from lighthouse.ingress.mlir_gen.utils import affine_map, get_mlir_elem_type

# ---------------------------------------------------------------------------
# linalg.contract
# ---------------------------------------------------------------------------
# bf8/hf8 are libxsmm's FP8 names; they map to the MLIR f8E5M2 / f8E4M3FN
# element types in the emitted IR.
_ALIASES = {"bf8": "f8E5M2", "hf8": "f8E4M3FN"}
# Supported user-facing element types for the A/B operands and the C output.
_AB_TYPES = ("f64", "i64", "f32", "f16", "bf16", "i16", "bf8", "hf8", "i8")

# --- MX microscaling (linalg.scaled_contract) ---------------------------------
# Block size: one shared E8M0 (f8E8M0FNU) scale per this many K elements.
_MX_BLOCK = 32
# MXINT8 elements carry an implicit 2^-6 per-element exponent bias, so a
# product of two terms accumulates 2^-12 = 1/4096 (matches libxsmm MXINT8 ref).
_MX_INT_BIAS = 1.0 / 4096.0
# User A/B type -> (MLIR element type name, cast TypeFn). Signedness of integer
# operands is carried by the scaled_contract `cast` attr (element stays i8).
_MX_TYPES = {
    "i8": ("i8", "cast_signed"),         # MXINT8
    "bf8": ("f8E5M2", "cast_signed"),    # MXBF8
    "hf8": ("f8E4M3FN", "cast_signed"),  # MXHF8
}
# A/B operand types accepted by --scale-kind mx (MXINT8/MXBF8/MXHF8).
_MX_AB_TYPES = ("i8", "bf8", "hf8")
_C_TYPES = ("i64", "i32", "i16", "i8", "bf8", "hf8", "bf16", "f16", "f32", "f64")
# Computation (accumulator) types selectable via --compType.
_COMP_TYPES = ("i32", "i64", "f32", "f64")
# Scale element types accepted on the CLI; --scale-kind scalar narrows these to
# f32 or f8E8M0FNU and throws for any other scale type.
_SCALE_TYPES = ("f32", "f8E8M0FNU", "f16", "bf16", "bf8", "hf8")

# C types narrower than the f32/i32 accumulator: emitted via the down-convert
# epilogue. F16/BF16/BF8/HF8 accumulate in f32 then truncf; I16/I8 accumulate
# in i32 then trunci. The wide C types (f32/f64/i32) accumulate straight into C.
_NARROW_C = {"f16", "bf16", "i16", "bf8", "hf8", "i8"}


def _get_type(name):
    name = _ALIASES.get(name, name)
    # f8E5M2 / f8E4M3FN aren't covered by the shared helper.
    if name.startswith("f8"):
        return ir.Type.parse(name)
    return get_mlir_elem_type(name)


def _is_int(name):
    return name.startswith("i")


def _acc_name(c_name):
    """Accumulator element type for a given C output type: narrow C accumulates
    in i32 (integer) or f32 (float); wide C (f32/f64/i32) accumulates at its own
    precision (the contract writes straight into C)."""
    if c_name in _NARROW_C:
        return "i32" if _is_int(c_name) else "f32"
    return c_name


# VNNI reduction-packing factor per architecture and element type (types absent
# from an arch's table have no VNNI layout, e.g. f32/f64/i64). x86 packs bf16 in
# blocks of 2 (AVX512-BF16); ARM uses 4 (BFMMLA). 
_VNNI_FACTOR = {
    "x86": {"bf16": 2, "f16": 2, "i16": 2, "i8": 4, "bf8": 4, "hf8": 4},
    "arm": {"bf16": 4},
}

# Element types supported per architecture. x86 supports all declared A/B/C
# types; ARM is restricted to f32/bf16.
_SUPPORTED_TYPES = {
    "x86": set(_AB_TYPES) | set(_C_TYPES),
    "arm": {"f32", "bf16"},
}


def _detect_arch():
    # platform.machine() -> x86_64/AMD64 on x86, aarch64/arm64 on ARM.
    m = platform.machine().lower()
    return "arm" if ("arm" in m or "aarch64" in m) else "x86"

# Element bit widths (resolved names), used to pick ext vs trunc when converting
# between the computation type and C.
_BITS = {"f8E5M2": 8, "f8E4M3FN": 8, "f16": 16, "bf16": 16, "f32": 32,
         "f64": 64, "i8": 8, "i16": 16, "i32": 32, "i64": 64}


def _dlti_unroll_attr(vtm, vtn, vtk):
    # reg_gemm_unroll hint consumed by the tpp-mlir register-unroll pass.
    return ir.Attribute.parse(
        f'#dlti.target_system_spec<"CPU" = '
        f'#dlti.target_device_spec<"reg_gemm_unroll" = [{vtm}, {vtn}, {vtk}]>>'
    )


def build_contract(m, n, k, br_count, a_name, b_name, c_name,
                   vtm, vtn, vtk,
                   trans_a=False, trans_b=False, vnni_a=False, vnni_b=False,
                   alpha=1.0, beta=1.0, comp_name=None, fn_name="entry"):
    if br_count < 1:
        raise ValueError(f"br_count must be >= 1, got {br_count}")
    if a_name not in _AB_TYPES:
        raise ValueError(
            f"unsupported A type {a_name}; allowed: {', '.join(_AB_TYPES)}")
    if b_name not in _AB_TYPES:
        raise ValueError(
            f"unsupported B type {b_name}; allowed: {', '.join(_AB_TYPES)}")
    if c_name not in _C_TYPES:
        raise ValueError(
            f"unsupported C type {c_name}; allowed: {', '.join(_C_TYPES)}")
    arch = _detect_arch()
    allowed = _SUPPORTED_TYPES[arch]
    bad = {t for t in (a_name, b_name, c_name) if t not in allowed}
    if bad:
        raise ValueError(
            f"{arch.upper()} only supports {', '.join(sorted(allowed))} types, "
            f"got: {', '.join(sorted(bad))}")
    if vnni_a and not vnni_b:
        raise ValueError("VNNI must be enabled for B if it is enabled for A")

    # vnni_b only: A's argument stays non-VNNI and is expanded into VNNI before
    # the contract via tensor.expand_shape.
    expand_a = vnni_b and not vnni_a
    vnni = vnni_b
    if expand_a and trans_a:
        raise ValueError("VNNI expand of A is not supported with --transA")
    if vnni and trans_b:
        raise ValueError("VNNI does not support transposing B (only A may be transposed)")
    if vnni and a_name not in _VNNI_FACTOR[arch]:
        raise ValueError(f"no VNNI layout for element type {a_name}")
    vf = _VNNI_FACTOR[arch][a_name] if vnni else 1
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

    # When only B is VNNI-packed, A's argument stays non-VNNI (M x K); it is
    # expanded into the contract's VNNI shape (M x K/vf x vf) inside the body.
    a_arg_inner = [m, k] if expand_a else a_inner

    if comp_name is None:
        comp_name = _acc_name(c_name)
    elif comp_name not in _COMP_TYPES:
        raise ValueError(
            f"unsupported comp type {comp_name}; allowed: {', '.join(_COMP_TYPES)}")
    if _is_int(comp_name) != _is_int(c_name):
        raise ValueError(
            f"comp type {comp_name} and C type {c_name} must both be integer "
            "or both floating-point")

    acc_name = comp_name
    is_int = _is_int(comp_name)
    comp_res = _ALIASES.get(comp_name, comp_name)
    c_res = _ALIASES.get(c_name, c_name)

    # Function to up/down convert input types, when the compute type and C
    # type differs
    def _cvt(val, src_res, dst_res, dst_ty):
        # Convert a scalar between two element types of the same category:
        # ext/trunc within ints or floats.
        if src_res == dst_res:
            return val
        if _is_int(src_res):
            return (arith.ExtSIOp(dst_ty, val).result
                    if _BITS[dst_res] > _BITS[src_res]
                    else arith.TruncIOp(dst_ty, val).result)
        return (arith.ExtFOp(dst_ty, val).result
                if _BITS[dst_res] > _BITS[src_res]
                else arith.TruncFOp(dst_ty, val).result)

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        ta, tb, tc = _get_type(a_name), _get_type(b_name), _get_type(c_name)
        tacc = _get_type(acc_name)

        a_ty = ir.RankedTensorType.get([br_count] + a_arg_inner, ta)
        b_ty = ir.RankedTensorType.get([br_count] + b_inner, tb)
        c_ty = ir.RankedTensorType.get([m, n], tc)

        # affine map creation.
        # Contraction dims: d0=batch, d1=M, d2=N, d3=K (+ d4=vf for VNNI); batch,
        # K and vf are all reduction dims.
        ndims = 5 if vnni else 4
        d = [ir.AffineDimExpr.get(i) for i in range(ndims)]
        a_dims = [d[0]] + ([d[3], d[1]] if trans_a else [d[1], d[3]]) + ([d[4]] if vnni else [])
        b_dims = [d[0]] + ([d[2], d[3]] if trans_b else [d[3], d[2]]) + ([d[4]] if vnni else [])
        maps = [
            affine_map(ndims, a_dims),  # A: batch, M, K (+vf)
            affine_map(ndims, b_dims),  # B: batch, K, N (+vf)
            affine_map(ndims, [d[1], d[2]]),  # C: M, N
        ]
   
        dlti_attr = _dlti_unroll_attr(vtm, vtn, vtk)
        scaled = alpha != 1.0 or beta != 1.0
        direct = (comp_res == c_res) and not scaled

        # module and function creation.
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp(fn_name, ir.FunctionType.get([a_ty, b_ty, c_ty], [c_ty]))
            fn.attributes["dlti.target_system_spec"] = dlti_attr
            entry = fn.add_entry_block()

            with ir.InsertionPoint(entry):
                A, B, C = entry.arguments

                # if B is in VNNI, pack A into VNNI
                if expand_a:
                    # Split A's K dim into (K/vf, vf) to match B's VNNI layout.
                    a_vnni_ty = ir.RankedTensorType.get([br_count] + a_inner, ta)
                    A = tensor.expand_shape(
                        a_vnni_ty, A, [[0], [1], [2, 3]], [],
                        [br_count] + a_inner)

                # if compute and C type are same and no alpha/beta, emit the 
                # contract directly with C as the outs.
                if direct:
                    # C is the accumulator type: contract straight into C.
                    res = linalg.contract(A, B, outs=[C], indexing_maps=maps)
                    func.ReturnOp([res])
                    return module

                # The else case:
                # Narrow C and/or alpha/beta scaling: contract into an
                # accumulator temp, then a generic epilogue computes
                # alpha*acc + beta*C, down-converting when C is narrower.
                if is_int:
                    zero = arith.ConstantOp(tacc, ir.IntegerAttr.get(tacc, 0))
                else:
                    zero = arith.ConstantOp(tacc, ir.FloatAttr.get(tacc, 0.0))

                # Create an empty tensor filled with zero for accumulation.
                tmp = tensor.EmptyOp([m, n], tacc)
                filled = linalg.fill(zero, outs=[tmp])

                # contraction operation creation.
                contracted = linalg.contract(A, B, outs=[filled], indexing_maps=maps)

                # The epilouge code to multiply the alpha/beta parameters and to up/down
                # convert input types, accordingly.
                id2 = affine_map(
                    2, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
                emaps = ir.ArrayAttr.get([ir.AffineMapAttr.get(id2)] * 3)
                par = ir.Attribute.parse("#linalg.iterator_type<parallel>")
                iters = ir.ArrayAttr.get([par, par])
                g = linalg.GenericOp([c_ty], [contracted, C], [C], emaps, iters)
                blk = g.regions[0].blocks.append(tacc, tc, tc)
                with ir.InsertionPoint(blk):
                    acc_in, c_in, _out = blk.arguments
                    a_term = acc_in
                    if alpha != 1.0:
                        attr = (ir.IntegerAttr.get(tacc, int(alpha)) if is_int
                                else ir.FloatAttr.get(tacc, alpha))
                        ac = arith.ConstantOp(tacc, attr)
                        a_term = (arith.MulIOp if is_int else arith.MulFOp)(ac, a_term).result
                    c_val = _cvt(c_in, c_res, comp_res, tacc)  # C -> accumulator
                    if beta != 1.0:
                        attr = (ir.IntegerAttr.get(tacc, int(beta)) if is_int
                                else ir.FloatAttr.get(tacc, beta))
                        bc = arith.ConstantOp(tacc, attr)
                        c_val = (arith.MulIOp if is_int else arith.MulFOp)(bc, c_val).result
                    summed = (arith.AddIOp if is_int else arith.AddFOp)(a_term, c_val).result
                    out_val = _cvt(summed, comp_res, c_res, tc)  # accumulator -> C
                    linalg.YieldOp([out_val])
                func.ReturnOp([g.results[0]])

        return module


def build_scalar_scaled(m, n, k, br_count, a_name, b_name, c_name, scale_name,
                        comp_name, vtm, vtn, vtk, vnni_a=False, vnni_b=False,
                        trans_a=False, trans_b=False, fn_name="entry"):
    """Integer GEMM with per-dimension (per-row/per-column) scale vectors.

    A (i8) * B (i8) accumulates in i32; an epilogue applies a per-M scale
    ``sA[m]`` and a per-N scale ``sB[n]`` (constant across K, so they factor out
    of the reduction) and adds the incoming C. ``scale_name`` selects the scale
    element type: ``f32`` (used directly) or ``f8E8M0FNU`` (power-of-two, decoded
    via extf). Three variants, matching the validated reference IRs:

      * ``--cType f32 --scaleType f32``        -> out = (f32)acc*sA*sB + C
      * ``--cType f32 --scaleType f8E8M0FNU``  -> same, E8M0 scales
      * ``--cType i8`` (quantized output)      -> a per-N output scale ``sC[n]``
        divides the result, which is then clamped to [-128, 127] and converted
        to i8: out = clamp((f32)acc*sA*sB/sC + C, -128, 127).

    ``vnni_b`` (optionally ``vnni_a``) VNNI-packs the K reduction dim into
    ``(K/vf, vf)`` blocks (vf = the arch i8 VNNI factor); when only B is packed,
    A is expanded to match. The scales are independent of K, so the epilogue is
    unchanged.
    """
    if br_count < 1:
        raise ValueError(f"br_count must be >= 1, got {br_count}")
    if a_name != "i8" or b_name != "i8":
        raise ValueError("--scale-kind scalar requires --aType i8 and --bType i8")
    if comp_name not in (None, "i32"):
        raise ValueError("--scale-kind scalar requires --compType i32")
    if scale_name not in ("f32", "f8E8M0FNU"):
        raise ValueError(
            "--scaleType must be f32 or f8E8M0FNU for --scale-kind scalar")
    if c_name not in ("f32", "i8"):
        raise ValueError("--scale-kind scalar expects --cType f32 or i8")
    if trans_a or trans_b:
        raise ValueError("--scale-kind scalar does not support transpose")
    # Quantized output: emit a per-N output scale sC, clamp and convert to i8.
    quant = c_name == "i8"

    # VNNI packs K into (K/vf, vf) blocks (vf innermost); vnni_a-only is invalid,
    # and vnni_b-only expands A to the packed layout inside the body.
    arch = _detect_arch()
    if vnni_a and not vnni_b:
        raise ValueError("VNNI must be enabled for B if it is enabled for A")
    vnni = vnni_b
    expand_a = vnni_b and not vnni_a
    if vnni and a_name not in _VNNI_FACTOR[arch]:
        raise ValueError(f"no VNNI layout for element type {a_name}")
    vf = _VNNI_FACTOR[arch][a_name] if vnni else 1
    if vnni and k % vf != 0:
        raise ValueError(f"K={k} not divisible by the VNNI factor {vf}")
    kp = k // vf

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        ta, tb = _get_type(a_name), _get_type(b_name)
        ti32 = _get_type("i32")
        tf32 = _get_type("f32")
        tscale = (_get_type("f32") if scale_name == "f32"
                  else ir.Type.parse("f8E8M0FNU"))
        tc = _get_type(c_name)

        # VNNI splits K into (K/vf, vf) with vf innermost; when only B is packed,
        # A's argument stays (M x K) and is expanded in the body.
        a_inner = [m, kp, vf] if vnni else [m, k]
        b_inner = [kp, n, vf] if vnni else [k, n]
        a_arg_inner = [m, k] if expand_a else a_inner
        a_ty = ir.RankedTensorType.get([br_count] + a_arg_inner, ta)
        b_ty = ir.RankedTensorType.get([br_count] + b_inner, tb)
        sa_ty = ir.RankedTensorType.get([m], tscale)   # per-M input scale
        sb_ty = ir.RankedTensorType.get([n], tscale)   # per-N input scale
        sc_ty = ir.RankedTensorType.get([n], tscale)   # per-N output scale
        c_ty = ir.RankedTensorType.get([m, n], tc)

        # Contraction dims: d0=batch, d1=M, d2=N, d3=K (+ d4=vf for VNNI); batch,
        # K and vf are reduction dims.
        ndims = 5 if vnni else 4
        d = [ir.AffineDimExpr.get(i) for i in range(ndims)]
        a_dims = [d[0], d[1], d[3]] + ([d[4]] if vnni else [])
        b_dims = [d[0], d[3], d[2]] + ([d[4]] if vnni else [])
        maps = [
            affine_map(ndims, a_dims),          # A: batch, M, K(/vf)(, vf)
            affine_map(ndims, b_dims),          # B: batch, K(/vf), N(, vf)
            affine_map(ndims, [d[1], d[2]]),    # C: M, N
        ]
        dlti_attr = _dlti_unroll_attr(vtm, vtn, vtk)

        # Epilogue maps (output is M x N): acc/C are elementwise (d0=M, d1=N),
        # sA is per-M (d0), sB and the output scale sC are per-N (d1).
        e0, e1 = ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)
        id2 = affine_map(2, [e0, e1])
        per_m = affine_map(2, [e0])
        per_n = affine_map(2, [e1])

        in_types = ([a_ty, sa_ty, b_ty, sb_ty]
                    + ([sc_ty] if quant else []) + [c_ty])

        with ir.InsertionPoint(module.body):
            fn = func.FuncOp(fn_name, ir.FunctionType.get(in_types, [c_ty]))
            fn.attributes["dlti.target_system_spec"] = dlti_attr
            entry = fn.add_entry_block()
            with ir.InsertionPoint(entry):
                if quant:
                    A, SA, B, SB, SC, C = entry.arguments
                else:
                    A, SA, B, SB, C = entry.arguments
                    SC = None

                # Expand A (M x K) into the VNNI layout (M x K/vf x vf) to match
                # a VNNI-packed B.
                if expand_a:
                    a_vnni_ty = ir.RankedTensorType.get(
                        [br_count] + a_inner, ta)
                    A = tensor.expand_shape(
                        a_vnni_ty, A, [[0], [1], [2, 3]], [],
                        [br_count] + a_inner)

                zero = arith.ConstantOp(ti32, ir.IntegerAttr.get(ti32, 0))
                tmp = tensor.EmptyOp([m, n], ti32)
                filled = linalg.fill(zero, outs=[tmp])
                acc = linalg.contract(A, B, outs=[filled], indexing_maps=maps)

                scale_ins = [acc, SA, SB] + ([SC] if quant else [])
                emap_list = ([id2, per_m, per_n]
                             + ([per_n] if quant else []) + [id2])
                emaps = ir.ArrayAttr.get(
                    [ir.AffineMapAttr.get(mp) for mp in emap_list])
                par = ir.Attribute.parse("#linalg.iterator_type<parallel>")
                iters = ir.ArrayAttr.get([par, par])
                g = linalg.GenericOp([c_ty], scale_ins, [C], emaps, iters)
                blk_types = ([ti32, tscale, tscale]
                             + ([tscale] if quant else []) + [tc])
                blk = g.regions[0].blocks.append(*blk_types)
                with ir.InsertionPoint(blk):
                    if quant:
                        acc_in, sa_in, sb_in, sc_in, c_in = blk.arguments
                    else:
                        acc_in, sa_in, sb_in, c_in = blk.arguments
                        sc_in = None

                    def _scale_f32(val):
                        # E8M0 scales decode to f32 via extf; f32 scales as-is.
                        return (val if scale_name == "f32"
                                else arith.ExtFOp(tf32, val).result)

                    accf = arith.SIToFPOp(tf32, acc_in).result
                    prod = arith.MulFOp(
                        _scale_f32(sa_in), _scale_f32(sb_in)).result
                    if quant:
                        prod = arith.DivFOp(prod, _scale_f32(sc_in)).result
                    scaled = arith.MulFOp(accf, prod).result
                    if quant:
                        c_val = arith.SIToFPOp(tf32, c_in).result
                        summed = arith.AddFOp(scaled, c_val).result
                        lo = arith.ConstantOp(
                            tf32, ir.FloatAttr.get(tf32, -128.0))
                        hi = arith.ConstantOp(
                            tf32, ir.FloatAttr.get(tf32, 127.0))
                        clamped = arith.MaximumFOp(summed, lo).result
                        clamped = arith.MinimumFOp(clamped, hi).result
                        out_val = arith.FPToSIOp(tc, clamped).result
                    else:
                        out_val = arith.AddFOp(scaled, c_in).result
                    linalg.YieldOp([out_val])
                func.ReturnOp([g.results[0]])
        return module


def build_mx_scaled(m, n, k, br_count, a_name, b_name, c_name, scale_name,
                    comp_name, block, vtm, vtn, vtk, fn_name="entry"):
    """MX block-scaled GEMM via ``linalg.scaled_contract``: A and B carry one
    E8M0 (f8E8M0FNU) scale per ``block`` K elements. Integer operands
    (MXINT8) additionally apply the implicit 2^-12 exponent bias.
    A is transposed and VNNI-packed (batch, K/vf, M, vf) and B is VNNI-packed
    (batch, K/vf, N, vf). The contraction accumulates in f32 and C is f32.
    Mixed A/B types are not supported."""
    if br_count < 1:
        raise ValueError(f"br_count must be >= 1, got {br_count}")
    if scale_name != "f8E8M0FNU":
        raise ValueError("--scale-kind mx requires --scaleType f8E8M0FNU")
    if a_name not in _MX_AB_TYPES or b_name not in _MX_AB_TYPES:
        raise ValueError(
            f"--scale-kind mx requires A/B types in {_MX_AB_TYPES}, "
            f"got {a_name}/{b_name}")
    if a_name != b_name:
        raise ValueError(
            "--scale-kind mx requires matching A/B types; mixed types "
            "are not supported")
    if c_name != "f32":
        raise ValueError("--scale-kind mx requires --cType f32")
    if comp_name not in (None, "f32"):
        raise ValueError("--scale-kind mx requires --compType f32")
    if k % block != 0:
        raise ValueError(f"K={k} must be divisible by the MX block size {block}")

    arch = _detect_arch()
    if a_name not in _VNNI_FACTOR[arch]:
        raise ValueError(f"no VNNI layout for MX type {a_name} on {arch}")
    vf = _VNNI_FACTOR[arch][a_name]
    if block % vf != 0:
        raise ValueError(
            f"MX block size {block} must be divisible by the VNNI factor {vf}")

    elem_name, cast_fn = _MX_TYPES[a_name]
    is_int = _is_int(elem_name)
    kb = k // block     # number of E8M0 scale blocks along K
    kp = k // vf        # VNNI-packed (outer) K extent
    bpv = block // vf   # VNNI groups sharing one scale block

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        telem = _get_type(elem_name)
        tscale = ir.Type.parse("f8E8M0FNU")
        tf32 = _get_type("f32")
        tc = _get_type(c_name)

        # A is transposed + VNNI-packed (batch, K/vf, M, vf); B is VNNI-packed
        # (batch, K/vf, N, vf). Scales carry one E8M0 per `block` K elements.
        a_ty = ir.RankedTensorType.get([br_count, kp, m, vf], telem)
        b_ty = ir.RankedTensorType.get([br_count, kp, n, vf], telem)
        sa_ty = ir.RankedTensorType.get([br_count, m, kb], tscale)
        sb_ty = ir.RankedTensorType.get([br_count, n, kb], tscale)
        # scaled_contract accumulates in f32 (C is f32).
        c_ty = ir.RankedTensorType.get([m, n], tc)

        # Iteration dims: d0=batch, d1=M, d2=N, d3=K/vf, d4=vf (batch, K/vf and
        # vf reduced). A scale block spans bpv=block/vf consecutive d3 groups, so
        # the scale block index is d3 floordiv bpv.
        d = [ir.AffineDimExpr.get(i) for i in range(5)]
        blkidx = ir.AffineFloorDivExpr.get(
            d[3], ir.AffineConstantExpr.get(bpv))
        maps = ir.ArrayAttr.get([
            ir.AffineMapAttr.get(affine_map(5, [d[0], d[3], d[1], d[4]])),  # A
            ir.AffineMapAttr.get(affine_map(5, [d[0], d[1], blkidx])),      # sA
            ir.AffineMapAttr.get(affine_map(5, [d[0], d[3], d[2], d[4]])),  # B
            ir.AffineMapAttr.get(affine_map(5, [d[0], d[2], blkidx])),      # sB
            ir.AffineMapAttr.get(affine_map(5, [d[1], d[2]])),             # C
        ])
        cast = ir.Attribute.parse(f"#linalg.type_fn<{cast_fn}>")
        dlti_attr = _dlti_unroll_attr(vtm, vtn, vtk)

        with ir.InsertionPoint(module.body):
            fn = func.FuncOp(
                fn_name,
                ir.FunctionType.get([a_ty, sa_ty, b_ty, sb_ty, c_ty], [c_ty]))
            fn.attributes["dlti.target_system_spec"] = dlti_attr
            entry = fn.add_entry_block()
            with ir.InsertionPoint(entry):
                A, SA, B, SB, C = entry.arguments

                # FP8: contract straight into the f32 C.
                if not is_int:
                    op = linalg.ScaledContractOp(
                        result_tensors=[c_ty], inputs=[A, SA, B, SB],
                        outputs=[C], indexing_maps=maps, cast=cast)
                    linalg.fill_builtin_region(op.operation)
                    func.ReturnOp([op.results[0]])
                    return module

                # MXINT8: accumulate into an f32 temp, then an epilogue applies
                # the implicit 2^-12 bias and adds C.
                zero = arith.ConstantOp(tf32, ir.FloatAttr.get(tf32, 0.0))
                tmp = tensor.EmptyOp([m, n], tf32)
                init = linalg.fill(zero, outs=[tmp])
                op = linalg.ScaledContractOp(
                    result_tensors=[c_ty], inputs=[A, SA, B, SB],
                    outputs=[init], indexing_maps=maps, cast=cast)
                linalg.fill_builtin_region(op.operation)
                acc = op.results[0]

                id2 = affine_map(
                    2, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)])
                emaps = ir.ArrayAttr.get([ir.AffineMapAttr.get(id2)] * 3)
                par = ir.Attribute.parse("#linalg.iterator_type<parallel>")
                iters = ir.ArrayAttr.get([par, par])
                g = linalg.GenericOp([c_ty], [acc, C], [C], emaps, iters)
                blk = g.regions[0].blocks.append(tf32, tf32, tf32)
                with ir.InsertionPoint(blk):
                    acc_in, c_in, _out = blk.arguments
                    bias = arith.ConstantOp(
                        tf32, ir.FloatAttr.get(tf32, _MX_INT_BIAS))
                    biased = arith.MulFOp(acc_in, bias).result
                    summed = arith.AddFOp(biased, c_in).result
                    linalg.YieldOp([summed])
                func.ReturnOp([g.results[0]])
        return module


def _add_matmul_args(p):
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--vtM", type=int, default=16,
                   help="M register-blocking hint for the reg_gemm_unroll dlti attr")
    p.add_argument("--vtN", type=int, default=16,
                   help="N register-blocking hint for the reg_gemm_unroll dlti attr")
    p.add_argument("--vtK", type=int, default=32,
                   help="K register-blocking hint for the reg_gemm_unroll dlti attr")
    p.add_argument("--aType", choices=_AB_TYPES, default="bf16")
    p.add_argument("--bType", choices=_AB_TYPES, default="bf16")
    p.add_argument("--compType", choices=_COMP_TYPES, default=None,
                   help="computation/accumulator type (libxsmm comp); default "
                        "i32 for integer C, f32 for float C")
    p.add_argument("--cType", choices=_C_TYPES, default="bf16")
    p.add_argument("--transA", action="store_true", help="store A transposed (K x M)")
    p.add_argument("--transB", action="store_true", help="store B transposed (N x K)")
    p.add_argument("--vnniA", action="store_true", help="VNNI-pack A on the K dim")
    p.add_argument("--vnniB", action="store_true", help="VNNI-pack B on the K dim")
    p.add_argument("--br_count", type=int, default=1,
                   help="batch-reduce count: leading batch dim of A/B, folded "
                        "into linalg.contract as a reduction dim")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="scalar multiplier on A*B (default 1.0 = no scaling)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="scalar multiplier on C (default 1.0 = plain accumulate)")
    p.add_argument("--scale-kind", dest="scale_kind",
                   choices=("scalar", "mx"), default=None,
                   help="emit a scaled GEMM: 'scalar' = int8 A*B with per-M/per-N "
                        "scale vectors (sA, sB) and optional quantized i8 output "
                        "(per-N output scale sC); 'mx' = linalg.scaled_contract "
                        "with per-block E8M0 scale tensors "
                        "(MXINT8/MXBF8/MXHF8)")
    p.add_argument("--scaleType", dest="scale_type",
                   choices=_SCALE_TYPES, default="f32",
                   help="scale element type for --scale-kind scalar (sA/sB/sC): "
                        "f32 or f8E8M0FNU (power-of-two); other types throw")
    p.add_argument("--mxBlock", type=int, default=_MX_BLOCK,
                   help="MX microscaling block size (K elements per E8M0 scale)")
    p.add_argument("-o", "--output", metavar="FILE")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="gen", required=True)
    bg = sub.add_parser(
        "brgemm", help="linalg.contract batch-reduce matmul (br_count default 1)")
    _add_matmul_args(bg)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.scale_kind == "mx":
        module = build_mx_scaled(
            args.M, args.N, args.K, args.br_count,
            args.aType, args.bType, args.cType, args.scale_type,
            args.compType, args.mxBlock,
            args.vtM, args.vtN, args.vtK,
        )
    elif args.scale_kind == "scalar":
        module = build_scalar_scaled(
            args.M, args.N, args.K, args.br_count,
            args.aType, args.bType, args.cType, args.scale_type,
            args.compType, args.vtM, args.vtN, args.vtK,
            args.vnniA, args.vnniB,
            args.transA, args.transB,
        )
    else:
        module = build_contract(
            args.M, args.N, args.K, args.br_count,
            args.aType, args.bType, args.cType,
            args.vtM, args.vtN, args.vtK,
            args.transA, args.transB,
            args.vnniA, args.vnniB,
            args.alpha, args.beta,
            args.compType,
        )
    # Embed the benchmark with FLOPS count so that the tpp-mlir benchmarking
    # infra is re-used.
    total_flops = 2 * args.br_count * args.M * args.N * args.K
    header = (
        "// RUN: tpp-run %s -n 10 \\\n"
        "// RUN:  -e entry -entry-point-result=void\n"
        f"// BENCH_TOTAL_FLOPS: {total_flops}\n"
    )
    text = header + str(module)
    if getattr(args, "output", None):
        with open(args.output, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
