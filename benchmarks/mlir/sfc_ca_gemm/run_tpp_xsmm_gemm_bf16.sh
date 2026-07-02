#!/bin/bash
#
# Cold-cache GEMM sweep for tpp-run, modeled after libxsmm's sfc_ca_gemm
# benchmark_sample.sh. For every (M, N, K) configuration it:
#   1. generates an mx-bf16 GEMM kernel with mlir-gen,
#   2. runs tpp-run with -bench-replication-gb=5.0 so the kernel arguments are
#      replicated to a 5 GiB footprint and iterated inside the timing loop
#      (cold-cache measurement),
#   3. records the reported mean per-call runtime (seconds).
#
# The raw timings are written to a CSV (M,N,K,runtime_s) which is then handed to
# postprocess.py to compute TFLOP/s and sort by arithmetic complexity.
#
# Run this from the build directory (so that ./bin/mlir-gen and ./bin/tpp-run
# resolve), or set BIN_DIR to point at the directory holding those tools.

set -euo pipefail

#==============================================================================
# Environment settings (adopted from libxsmm sfc_ca_gemm/benchmark_sample.sh)
#==============================================================================
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=64
export LIBXSMM_X86_AMX_GEMM_STREAMING_A=1
export LIBXSMM_X86_AMX_GEMM_STREAMING_B=1

#==============================================================================
# Configuration
#==============================================================================
# Directory containing the mlir-gen / tpp-run binaries (default: ./bin).
BIN_DIR="${BIN_DIR:-./bin}"
# Directory for this script (used to locate postprocess.py).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Output directory for generated MLIR and result CSVs.
OUT_DIR="${OUT_DIR:-$(pwd)/sfc_ca_gemm_results}"
# GEMM block tiles (bm, bn, bk). K-blocking and K-layers are fixed to 1 since
# the generated MLIR does not model them.
TILES="${TILES:-32,32,32}"
# Number of timed loops handed to tpp-run.
N_ITERS="${N_ITERS:-100}"
# Target replication footprint in GiB for cold-cache benchmarking.
REPL_GIB="${REPL_GIB:-5.0}"
# Enable parallel (multi-threaded / OpenMP) execution in tpp-run. Set to 0 for
# a serial run. Parallelism is driven by OMP_NUM_THREADS (set above).
PARALLEL="${PARALLEL:-1}"
# Initialize the replicated cold-cache buffers with random FP values (matching
# the element precision) instead of zeros. All-zero inputs let the FMA units
# run at an unrealistically high clock. Set to 0 to keep zero initialization.
RANDOM_INIT="${RANDOM_INIT:-1}"

# Assemble the parallel-execution flags for tpp-run.
PARALLEL_FLAGS=()
if [[ "${PARALLEL}" != "0" ]]; then
  PARALLEL_FLAGS=(--def-parallel)
fi

# Assemble the random-initialization flag for tpp-run.
RANDOM_FLAGS=()
if [[ "${RANDOM_INIT}" != "0" ]]; then
  RANDOM_FLAGS=(--splat-to-random --seed=555)
fi

# GEMM dimensions to sweep (full cross-product), matching benchmark_sample.sh.
SIZES=(512 1024 2048 4096 8192)

MLIR_GEN="${BIN_DIR}/mlir-gen"
TPP_RUN="${BIN_DIR}/tpp-run"

#==============================================================================
# Sanity checks
#==============================================================================
for tool in "${MLIR_GEN}" "${TPP_RUN}"; do
  if [[ ! -x "${tool}" ]]; then
    echo "error: cannot find executable '${tool}'." >&2
    echo "       Run from the build directory or set BIN_DIR." >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"
RAW_CSV="${OUT_DIR}/sfc_ca_gemm_raw.csv"
echo "M,N,K,runtime_s" > "${RAW_CSV}"

#==============================================================================
# Sweep
#==============================================================================
for M in "${SIZES[@]}"; do
  for N in "${SIZES[@]}"; do
    for K in "${SIZES[@]}"; do
      mlir_file="${OUT_DIR}/gemm_sfc_${M}_${N}_${K}_mx-bf16.mlir"

      echo "=== Generating M=${M} N=${N} K=${K} ==="
      "${MLIR_GEN}" --kernel=args --float-type=bf16 \
        --batch="${N}" --layers="${K},${M}" --tiles="${TILES}" --vnni=2 \
        > "${mlir_file}"

      echo "=== Running   M=${M} N=${N} K=${K} ==="
      run_out="$("${TPP_RUN}" "${mlir_file}" \
        -e entry -entry-point-result=void \
        -n "${N_ITERS}" --bench-replication-gb="${REPL_GIB}" \
        "${RANDOM_FLAGS[@]}" \
        "${PARALLEL_FLAGS[@]}")"

      # tpp-run prints the mean per-call time (seconds) as a single float.
      runtime="$(printf '%s\n' "${run_out}" \
        | grep -oE '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?' \
        | tail -n 1)"

      if [[ -z "${runtime}" ]]; then
        echo "warning: could not parse runtime for M=${M} N=${N} K=${K}" >&2
        runtime="nan"
      fi

      echo "${M},${N},${K},${runtime}" >> "${RAW_CSV}"
      echo "    runtime = ${runtime} s"
    done
  done
done

echo
echo "Raw timings written to ${RAW_CSV}"

#==============================================================================
# Post-process: compute TFLOP/s and sort by arithmetic complexity
#==============================================================================
SORTED_CSV="${OUT_DIR}/sfc_ca_gemm_results.csv"
python3 "${SCRIPT_DIR}/postprocess.py" "${RAW_CSV}" "${SORTED_CSV}"

echo "Sorted results with TFLOP/s written to ${SORTED_CSV}"
