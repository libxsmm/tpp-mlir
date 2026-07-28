#!/usr/bin/env python3
"""
Parse TPP-MLIR benchmark output and plot GFLOPS results.

Usage:
    # From file
    python3 plot_benchmarks.py results.txt

    # From stdin
    cat results.txt | python3 plot_benchmarks.py -

    # Custom output path
    python3 plot_benchmarks.py results.txt -o my_plot.png
"""

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_benchmark_data(text):
    """Parse benchmark text into a nested dict: {benchmark_name: {test_case: gflops}}."""
    data = OrderedDict()
    current_bench = None

    bench_re = re.compile(r"^Benchmark:\s*(.+)$")
    # Matches: "i8_f32_dequant_3x1024_omp_2_mlir:    23.221 gflops"
    case_re = re.compile(r"^(\S+):\s+([\d.]+)\s+gflops", re.IGNORECASE)

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        m = bench_re.match(line)
        if m:
            current_bench = m.group(1).strip()
            data[current_bench] = OrderedDict()
            continue

        m = case_re.match(line)
        if m and current_bench:
            test_name = m.group(1)
            gflops = float(m.group(2))
            data[current_bench][test_name] = gflops

    return data


def extract_thread_count(test_name):
    """Extract OMP thread count from a test name like 'i8_f32_dequant_3x1024_omp_8_mlir'."""
    m = re.search(r"omp_(\d+)", test_name)
    return int(m.group(1)) if m else 0


def extract_size(test_name):
    """Extract problem size from test name (e.g., '3x1024', '4096x8192', or '512x1024x2048')."""
    m = re.search(r"(\d+x\d+(?:x\d+)?)", test_name)
    return m.group(1) if m else "unknown"


def extract_mnk(test_name):
    """Extract (M, N, K) tuple from a test name with an MxNxK token.
    Returns None if not a 3-D shape."""
    m = re.search(r"(\d+)x(\d+)x(\d+)", test_name)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def plot_lines_tflops(data, output_path):
    """Plot TFLOPS vs (M,N,K) shape, one line per top-level benchmark group.

    The x-axis is the shared sorted list of MxNxK shape labels found across
    all groups; each group contributes one line (e.g., one for i8, one for
    bf16). Values are converted from GFLOPS (as parsed) to TFLOPS by /1000.
    """
    # Collect (shape_tuple -> {group_name: tflops})
    series = OrderedDict()  # group_name -> {shape_tuple: tflops}
    all_shapes = set()
    for group, cases in data.items():
        series[group] = {}
        for tc, gflops in cases.items():
            mnk = extract_mnk(tc)
            if mnk is None:
                continue
            tflops = gflops / 1000.0
            series[group][mnk] = tflops
            all_shapes.add(mnk)

    if not all_shapes:
        print("ERROR: no MxNxK test names found in input", file=sys.stderr)
        sys.exit(1)

    shapes_sorted = sorted(all_shapes)
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
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved line plot: {output_path}")
    plt.close()


def group_by_size(data):
    """Group benchmarks by problem size for clearer comparison."""
    grouped = OrderedDict()
    for bench_name, cases in data.items():
        # Determine the size from the first test case
        if not cases:
            continue
        first_case = next(iter(cases))
        size = extract_size(first_case)
        if size not in grouped:
            grouped[size] = OrderedDict()
        grouped[size][bench_name] = cases
    return grouped


def plot_grouped_bars(data, output_path, title_suffix=""):
    """Plot a grouped bar chart of GFLOPS per benchmark per thread count."""
    grouped = group_by_size(data)
    n_groups = len(grouped)

    fig, axes = plt.subplots(
        n_groups, 1,
        figsize=(12, 6 * n_groups),
        squeeze=False
    )

    for idx, (size, benchmarks) in enumerate(grouped.items()):
        ax = axes[idx][0]

        # Collect all unique thread counts across benchmarks of this size
        all_threads = sorted({
            extract_thread_count(tc)
            for cases in benchmarks.values()
            for tc in cases
        })

        x = np.arange(len(all_threads))
        n_bench = len(benchmarks)
        width = 0.8 / max(n_bench, 1)

        # Color palette
        colors = plt.cm.tab10(np.linspace(0, 1, n_bench))

        for bidx, (bench_name, cases) in enumerate(benchmarks.items()):
            # Map thread count -> gflops for this benchmark
            tc_to_gflops = {extract_thread_count(tc): g for tc, g in cases.items()}
            values = [tc_to_gflops.get(t, 0) for t in all_threads]

            offset = (bidx - n_bench / 2) * width + width / 2
            bars = ax.bar(
                x + offset, values, width,
                label=bench_name, color=colors[bidx],
                edgecolor='black', linewidth=0.5
            )

            # Annotate bars with values
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f'{val:,.0f}',
                        ha='center', va='bottom', fontsize=8, rotation=0
                    )

        ax.set_xlabel("OMP Threads", fontsize=11)
        ax.set_ylabel("Performance (GFLOPS)", fontsize=11)
        ax.set_title(
            f"GEMM i8→f32 Dequant — Size {size}{title_suffix}",
            fontsize=12, fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t} threads" for t in all_threads])
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax.set_yscale('log')  # Log scale due to large dynamic range

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved bar chart: {output_path}")
    plt.close()


def plot_scaling(data, output_path):
    """Plot performance scaling vs thread count for each benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Group by size for separate scaling plots
    grouped = group_by_size(data)

    for ax_idx, (size, benchmarks) in enumerate(grouped.items()):
        if ax_idx >= 2:
            break  # Only handle first 2 sizes
        ax = axes[ax_idx]

        for bench_name, cases in benchmarks.items():
            threads_gflops = sorted(
                [(extract_thread_count(tc), g) for tc, g in cases.items()]
            )
            if not threads_gflops:
                continue
            threads, gflops = zip(*threads_gflops)
            ax.plot(threads, gflops, marker='o', linewidth=2,
                    markersize=8, label=bench_name)

        ax.set_xlabel("OMP Threads", fontsize=11)
        ax.set_ylabel("Performance (GFLOPS)", fontsize=11)
        ax.set_title(f"Performance Scaling — Size {size}",
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved scaling plot: {output_path}")
    plt.close()


def plot_speedup(data, output_path):
    """Plot speedup of vectorized (AMX) vs non-vectorized variants."""
    grouped = group_by_size(data)

    fig, ax = plt.subplots(figsize=(12, 6))

    for size, benchmarks in grouped.items():  # FIX: use 'in', not '='
        # Find pairs of (baseline, vectorized) benchmarks
        baseline = None
        vectorized = None
        for bn in benchmarks:
            if "vector" in bn:
                vectorized = bn
            else:
                baseline = bn

        if not (baseline and vectorized):
            continue

        base_cases = benchmarks[baseline]
        vec_cases = benchmarks[vectorized]

        # Match by thread count
        common_tcs = set(base_cases.keys()) & set(vec_cases.keys())
        speedups = []
        threads = []
        for tc in sorted(common_tcs, key=extract_thread_count):
            speedup = vec_cases[tc] / base_cases[tc]
            speedups.append(speedup)
            threads.append(extract_thread_count(tc))

        if speedups:
            ax.plot(threads, speedups, marker='s', linewidth=2,
                    markersize=10, label=f"Size {size}")

    ax.set_xlabel("OMP Threads", fontsize=11)
    ax.set_ylabel("Speedup (Vectorized AMX / Baseline)", fontsize=11)
    ax.set_title("AMX Vectorization Speedup", fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log', base=2)
    ax.axhline(y=1, color='red', linestyle=':', alpha=0.5, label='No speedup')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved speedup plot: {output_path}")
    plt.close()


def print_summary(data):
    """Print a text summary of the parsed data."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for bench_name, cases in data.items():
        print(f"\n{bench_name}:")
        for tc, gflops in cases.items():
            threads = extract_thread_count(tc)
            print(f"  {threads:3d} threads: {gflops:>12,.2f} GFLOPS")

        # Compute peak and average
        if cases:
            peak = max(cases.values())
            avg = sum(cases.values()) / len(cases)
            print(f"  {'Peak:':<12s} {peak:>12,.2f} GFLOPS")
            print(f"  {'Average:':<12s} {avg:>12,.2f} GFLOPS")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Parse and plot TPP-MLIR benchmark results"
    )
    parser.add_argument(
        "input",
        help="Input file with benchmark data (use '-' for stdin)"
    )
    parser.add_argument(
        "-o", "--output-prefix",
        default="benchmark",
        help="Output file prefix (or full path for single-file modes)"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["all", "lines-tflops"],
        help="Plot mode (default: all)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display plots interactively"
    )
    args = parser.parse_args()

    # Read input
    if args.input == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.input).read_text()

    # Parse
    data = parse_benchmark_data(text)
    if not data:
        print("ERROR: No benchmark data found in input!", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print_summary(data)

    if args.mode == "lines-tflops":
        out = args.output_prefix
        if not out.lower().endswith((".png", ".svg", ".pdf")):
            out = f"{out}_tflops.png"
        plot_lines_tflops(data, out)
        return

    # Generate plots
    plot_grouped_bars(data, f"{args.output_prefix}_bars.png")
    plot_scaling(data, f"{args.output_prefix}_scaling.png")
    plot_speedup(data, f"{args.output_prefix}_speedup.png")

    print(f"\nGenerated 3 plots with prefix '{args.output_prefix}':")
    print(f"  - {args.output_prefix}_bars.png    (grouped bar chart)")
    print(f"  - {args.output_prefix}_scaling.png (thread scaling)")
    print(f"  - {args.output_prefix}_speedup.png (vectorization speedup)")


if __name__ == "__main__":
    main()
