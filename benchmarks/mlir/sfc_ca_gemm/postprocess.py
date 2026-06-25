#!/usr/bin/env python3
"""Post-process raw sfc_ca_gemm timings.

Reads a raw CSV with columns ``M,N,K,runtime_s`` produced by
``run_sfc_ca_gemm.sh`` and writes a CSV that:

  * computes the arithmetic complexity (FLOPs = 2 * M * N * K) of each GEMM,
  * computes the achieved performance in TFLOP/s next to the runtime,
  * is sorted by ascending arithmetic complexity.

Usage:
    postprocess.py <raw_csv> <out_csv>
"""

import csv
import sys


def main(argv):
    if len(argv) != 3:
        print(f"usage: {argv[0]} <raw_csv> <out_csv>", file=sys.stderr)
        return 1

    raw_path, out_path = argv[1], argv[2]

    rows = []
    with open(raw_path, newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            M = int(rec["M"])
            N = int(rec["N"])
            K = int(rec["K"])
            try:
                runtime_s = float(rec["runtime_s"])
            except (ValueError, KeyError):
                runtime_s = float("nan")

            # Arithmetic complexity of a single M x N x K GEMM.
            flops = 2.0 * M * N * K

            if runtime_s > 0.0:
                tflops = flops / runtime_s / 1.0e12
            else:
                tflops = float("nan")

            rows.append(
                {
                    "M": M,
                    "N": N,
                    "K": K,
                    "flops": int(flops),
                    "runtime_s": runtime_s,
                    "tflops": tflops,
                }
            )

    # Sort by arithmetic complexity (ascending), then by dimensions for ties.
    rows.sort(key=lambda r: (r["flops"], r["M"], r["N"], r["K"]))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "N", "K", "flops", "runtime_s", "tflops"])
        for r in rows:
            writer.writerow(
                [
                    r["M"],
                    r["N"],
                    r["K"],
                    r["flops"],
                    f"{r['runtime_s']:.9g}",
                    f"{r['tflops']:.6g}",
                ]
            )

    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
