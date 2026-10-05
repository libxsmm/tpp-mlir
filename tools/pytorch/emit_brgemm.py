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
_C_TYPES = ("i64", "i32", "i16", "i8", "bf8", "hf8", "bf16", "f16", "f32", "f64")
# Computation (accumulator) types selectable via --compType.
_COMP_TYPES = ("i32", "i64", "f32", "f64")

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


# VNNI reduction-packing factor per element type (f32/f64/i64 have no VNNI).
_VNNI_FACTOR = {"bf16": 2, "f16": 2, "i16": 2, "i8": 4, "bf8": 4, "hf8": 4}

# Element bit widths (resolved names), used to pick ext vs trunc when converting
# between the computation type and C.
_BITS = {"f8E5M2": 8, "f8E4M3FN": 8, "f16": 16, "bf16": 16, "f32": 32,
         "f64": 64, "i8": 8, "i16": 16, "i32": 32, "i64": 64}


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
    if vnni_a and not vnni_b:
        raise ValueError("VNNI must be enabled for B, if it is enabled for B")

    # vnni_b only: A's argument stays non-VNNI and is expanded into VNNI before
    # the contract via tensor.expand_shape.
    expand_a = vnni_b and not vnni_a
    vnni = vnni_b
    if expand_a and trans_a:
        raise ValueError("VNNI expand of A is not supported with --transA")
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
   
        # dlti attribute creation. Used by unroll pass to unroll contract operation.
        dlti_attr = ir.Attribute.parse(
            f'#dlti.target_system_spec<"CPU" = '
            f'#dlti.target_device_spec<"reg_gemm_unroll" = [{vtm}, {vtn}, {vtk}]>>'
        )
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
    module = build_contract(
        args.M, args.N, args.K, args.br_count,
        args.aType, args.bType, args.cType,
        args.vtM, args.vtN, args.vtK,
        args.transA, args.transB,
        args.vnniA, args.vnniB,
        args.alpha, args.beta,
        args.compType,
    )
    # Mirror mlir-gen: embed the benchmark harness directives so the tpp-mlir
    # harness can scrape the entry point and FLOP count from the IR itself.
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
