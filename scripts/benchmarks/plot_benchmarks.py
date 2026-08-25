#!/usr/bin/env python3
"""
Plot TPP-MLIR GEMM sweep throughput (TFLOPS vs shape) from sorted CSV(s).

Each input is a sorted throughput CSV with columns
``M,N,K,flops,runtime_s,tflops`` (as produced by postprocess.py). One line is
drawn per CSV, labeled by the file name (e.g. throughput_i8_i8_quant.csv ->
i8_i8_quant).

Usage:
    python3 plot_benchmarks.py throughput_i8_i8_quant.csv -o tflops.png
    python3 plot_benchmarks.py throughput_*.csv -o tflops.png
"""

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


def parse_csv_data(path):
    """Parse a sorted throughput CSV (columns M,N,K,flops,runtime_s,tflops)
    into (group, {(M, N, K): tflops}). The group name is derived from the file
    name (e.g. throughput_i8_i8_quant.csv -> i8_i8_quant)."""
    stem = Path(path).stem
    group = stem[len("throughput_"):] if stem.startswith("throughput_") else stem
    cases = OrderedDict()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            try:
                M, N, K = int(rec["M"]), int(rec["N"]), int(rec["K"])
                tflops = float(rec["tflops"])
            except (KeyError, ValueError):
                continue
            cases[(M, N, K)] = tflops
    return group, cases


def plot_lines_tflops(series, output_path):
    """Plot TFLOPS vs (M,N,K) shape, one line per group. The x-axis is the
    shared list of MxNxK shapes sorted by arithmetic complexity (M*N*K)."""
    all_shapes = set()
    for cases in series.values():
        all_shapes.update(cases)

    if not all_shapes:
        print("ERROR: no MxNxK data found in input", file=sys.stderr)
        sys.exit(1)

    shapes_sorted = sorted(all_shapes, key=lambda mnk: (mnk[0] * mnk[1] * mnk[2], mnk))
    labels = [f"{m}x{n}x{k}" for (m, n, k) in shapes_sorted]
    x = np.arange(len(shapes_sorted))

    fig, ax = plt.subplots(figsize=(max(12, 0.18 * len(shapes_sorted)), 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(series), 1)))
    for idx, (group, sh_to_tf) in enumerate(series.items()):
        y = [sh_to_tf.get(s, np.nan) for s in shapes_sorted]
        ax.plot(x, y, marker="o", linewidth=1.5, markersize=4,
                label=group, color=colors[idx])

    ax.set_xlabel("Shape (MxNxK)", fontsize=11)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=11)
    ax.set_title("tpp-run GEMM sweep — TFLOPS vs shape (nano-kernels)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend(loc="best", fontsize=9)
    # Fine-grained y-axis ticks every 5 TFLOPS (minor every 1).
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which="major", alpha=0.3, linestyle="--")
    ax.grid(True, which="minor", axis="y", alpha=0.15, linestyle=":")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved line plot: {output_path}")
    plt.close()


def print_summary(series):
    """Print a text summary of TFLOP/s per group."""
    print("\n" + "=" * 70)
    print("THROUGHPUT SUMMARY (TFLOPS)")
    print("=" * 70)
    for group, cases in series.items():
        print(f"\n{group}:  {len(cases)} shapes")
        if cases:
            vals = list(cases.values())
            print(f"  {'Peak:':<10s} {max(vals):>10,.2f} TFLOPS")
            print(f"  {'Average:':<10s} {sum(vals) / len(vals):>10,.2f} TFLOPS")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Plot TPP-MLIR GEMM sweep TFLOPS vs shape from sorted CSV(s)"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="Sorted throughput CSV(s) (columns M,N,K,flops,runtime_s,tflops)",
    )
    parser.add_argument(
        "-o", "--output",
        default="tflops.png",
        help="Output image path (default: tflops.png)",
    )
    args = parser.parse_args()

    # One line per CSV, keyed by group (file-name stem).
    series = OrderedDict()
    for inp in args.input:
        group, cases = parse_csv_data(inp)
        series.setdefault(group, OrderedDict()).update(cases)

    if not any(series.values()):
        print("ERROR: No throughput data found in input!", file=sys.stderr)
        sys.exit(1)

    print_summary(series)

    out = args.output
    if not out.lower().endswith((".png", ".svg", ".pdf")):
        out = f"{out}.png"
    plot_lines_tflops(series, out)


if __name__ == "__main__":
    main()
