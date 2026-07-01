#!/bin/bash
#
# Correctness test sweep for MX-I8-F32 GEMM kernels. For every (M, N, K)
# configuration it:
#   1. generates an mx-i8-f32 GEMM kernel with mlir-gen,
#   2. runs tpp-run with --linalg-to-loops (reference implementation),
#   3. runs tpp-run with --vector-to-kernels (optimized implementation),
#   4. compares outputs using fpcmp with relative tolerance 0.001.
#
# Run this from the build directory (so that ./bin/mlir-gen, ./bin/tpp-run,
# and ./bin/fpcmp resolve), or set BIN_DIR to point at the directory holding
# those tools.

set -euo pipefail

#==============================================================================
# Environment settings
#==============================================================================
export KMP_AFFINITY=granularity=fine,compact,1,0
export OMP_NUM_THREADS=64

#==============================================================================
# Configuration
#==============================================================================
# Directory containing the mlir-gen / tpp-run / fpcmp binaries (default: ./bin).
BIN_DIR="${BIN_DIR:-./bin}"
# Directory for this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Output directory for generated MLIR and temp files.
OUT_DIR="${OUT_DIR:-$(pwd)/sfc_ca_gemm_i8_correctness_results}"
# GEMM block tiles (bm, bn, bk).
TILES="${TILES:-32,32,64}"
# Enable parallel (multi-threaded / OpenMP) execution in tpp-run. Set to 0 for
# a serial run.
PARALLEL="${PARALLEL:-0}"
# Relative tolerance for fpcmp comparison.
TOLERANCE="${TOLERANCE:-0.001}"
# Initialize buffers with random FP values (matching the element precision)
# instead of zeros. Set to 0 to keep zero initialization.
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

# GEMM dimensions to sweep. Use smaller sizes for correctness testing.
SIZES=(512 1024 2048 4096 8192)

MLIR_GEN="${BIN_DIR}/mlir-gen"
TPP_RUN="${BIN_DIR}/tpp-run"
FPCMP="${BIN_DIR}/fpcmp"

#==============================================================================
# Sanity checks
#==============================================================================
for tool in "${MLIR_GEN}" "${TPP_RUN}" "${FPCMP}"; do
  if [[ ! -x "${tool}" ]]; then
    echo "error: cannot find executable '${tool}'." >&2
    echo "       Run from the build directory or set BIN_DIR." >&2
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"

#==============================================================================
# Counters for summary
#==============================================================================
PASS_COUNT=0
FAIL_COUNT=0

#==============================================================================
# Sweep
#==============================================================================
for M in "${SIZES[@]}"; do
  for N in "${SIZES[@]}"; do
    for K in "${SIZES[@]}"; do
      mlir_file="${OUT_DIR}/gemm_sfc_${M}_${N}_${K}_mx-i8-f32.mlir"
      ref_out="${OUT_DIR}/ref_${M}_${N}_${K}.out"
      opt_out="${OUT_DIR}/opt_${M}_${N}_${K}.out"

      echo "=== Testing M=${M} N=${N} K=${K} ==="

      # Generate the MLIR kernel
      "${MLIR_GEN}" --kernel=args --float-type=mx-i8-f32 \
        --batch="${N}" --layers="${K},${M}" --tiles="${TILES}" --vnni=4 \
        --quant-type=dequantize \
        > "${mlir_file}"

      # Run reference implementation (linalg-to-loops)
      echo "  Running reference (--linalg-to-loops)..."
      "${TPP_RUN}" "${mlir_file}" \
        -e entry -entry-point-result=void \
        --linalg-to-loops \
        --init-type=quant --print \
        "${RANDOM_FLAGS[@]}" \
        "${PARALLEL_FLAGS[@]}" \
        > "${ref_out}" 2>&1 || true

      # Run optimized implementation (vector-to-kernels)
      echo "  Running optimized (--vector-to-kernels)..."
      "${TPP_RUN}" "${mlir_file}" \
        "${RANDOM_FLAGS[@]}" \
        -e entry -entry-point-result=void \
        --vector-to-kernels --registerBlocking=32,32,64 \
        --init-type=quant --print \
        "${PARALLEL_FLAGS[@]}" \
        > "${opt_out}" 2>&1 || true

      # Compare outputs
      echo "  Comparing outputs (tolerance=${TOLERANCE})..."
      if "${FPCMP}" -r "${TOLERANCE}" "${ref_out}" "${opt_out}" > /dev/null 2>&1; then
        echo "  PASS: M=${M} N=${N} K=${K}"
        PASS_COUNT=$((PASS_COUNT + 1))
      else
        echo "  FAIL: M=${M} N=${N} K=${K}"
        echo "    Reference output: ${ref_out}"
        echo "    Optimized output: ${opt_out}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
      fi
      echo
    done
  done
done

#==============================================================================
# Summary
#==============================================================================
echo "============================================================"
echo "Correctness Test Summary"
echo "============================================================"
echo "PASSED: ${PASS_COUNT}"
echo "FAILED: ${FAIL_COUNT}"
echo "TOTAL:  $((PASS_COUNT + FAIL_COUNT))"
echo

if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  echo "Some tests FAILED. Check the output files in ${OUT_DIR}"
  exit 1
else
  echo "All tests PASSED!"
  exit 0
fi
