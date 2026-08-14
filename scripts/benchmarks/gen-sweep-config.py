#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a benchmark JSON sweep config for mlir-gen GEMM over (M,N,K).

All flags/environment/extensions are copied from
benchmarks/config/omp/mlir-i8-bf16-gemm-amx.json. Only --batch=M and
--layers=K,N vary across the cartesian product of SHAPES.
"""

import argparse
import itertools
import json
import os
import sys

SHAPES = [512, 1024, 2048, 4096, 8192]
THREADS = [64]
OMP_ENV = {
    "OMP_NUM_THREADS": "64",
    "KMP_AFFINITY": "granularity=fine,verbose,compact,1,0",
}

DTYPES = {
    "i8": {
        "group": "gemm_i8_i8_quant_mlir_vector_large_amx",
        "name_fmt": "i8_i8_quant_{M}x{N}x{K}_omp_{T}_mlir",
        "gen_flags": (
            "--kernel=args --data-type=i8 --batch={M} "
            "--layers={K},{N} --tiles=32,32,64 --vnni=4 "
            "--quant"
        ),
        "run_args": (
            "--def-parallel --nano-kernels --gemm-unroll=16,16,16 "
            "--registerBlocking=32,32,64 --bench-replication-gb=5 "
            "--init-type=quant --splat-to-random --seed=123"
        ),
        "extensions": ["amx_int8"],
    },
    "bf16": {
        "group": "gemm_bf16_f32_dequant_mlir_vector_large_amx",
        "name_fmt": "bf16_f32_dequant_{M}x{N}x{K}_omp_{T}_mlir",
        "gen_flags": (
            "--kernel=args --data-type=bf16-f32 --batch={M} "
            "--layers={K},{N} --tiles=32,32,32 --vnni=2"
        ),
        "run_args": (
            "--def-parallel --nano-kernels --gemm-unroll=16,16,16 "
            "--registerBlocking=32,32,32 --bench-replication-gb=5 "
            "--init-type=normal --splat-to-random --seed=123"
        ),
        "extensions": ["amx_bf16"],
    },
}


def build_run(dtype_cfg, M, N, K, threads, iters):
    name = dtype_cfg["name_fmt"].format(M=M, N=N, K=K, T=threads)
    gen_flags = dtype_cfg["gen_flags"].format(M=M, N=N, K=K)
    env = dict(OMP_ENV)
    env["OMP_NUM_THREADS"] = str(threads)
    return name, {
        "type": "IR-GEN",
        "benchmark": ["mlir-gen", gen_flags],
        "environment": env,
        "flags": [
            "-n", str(iters),
            "-run-args='" + dtype_cfg["run_args"] + "'",
        ],
        "extensions": list(dtype_cfg["extensions"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-o", "--output", required=True,
        help="Output JSON path",
    )
    p.add_argument(
        "--iters", type=int, default=100,
        help="Number of timed iterations per run (default: 100)",
    )
    p.add_argument(
        "--shapes", default=",".join(str(s) for s in SHAPES),
        help="Comma-separated dim list (default: %(default)s)",
    )
    p.add_argument(
        "--dtypes", default="i8,bf16",
        help="Comma-separated dtype keys (i8, bf16) (default: %(default)s)",
    )
    p.add_argument(
        "--threads", default=",".join(str(t) for t in THREADS),
        help="Comma-separated OMP_NUM_THREADS list (default: %(default)s)",
    )
    args = p.parse_args()

    shapes = [int(s) for s in args.shapes.split(",") if s]
    threads = [int(t) for t in args.threads.split(",") if t]
    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    if any(t <= 0 for t in threads):
        sys.exit("All --threads values must be positive integers")
    for d in dtypes:
        if d not in DTYPES:
            sys.exit(f"Unknown dtype '{d}'. Choices: {list(DTYPES)}")

    cfg = []
    for d in dtypes:
        dcfg = DTYPES[d]
        runs = {}
        for M, N, K in itertools.product(shapes, repeat=3):
            for t in threads:
                name, run = build_run(dcfg, M, N, K, t, args.iters)
                # name, run = build_run(dcfg, 8192, N, K, t, args.iters)
                runs[name] = run
        cfg.append({dcfg["group"]: runs})

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cfg, f, indent=2)
    total = sum(len(list(g.values())[0]) for g in cfg)
    print(f"Wrote {args.output} with {len(cfg)} group(s), {total} run(s)")


if __name__ == "__main__":
    main()
