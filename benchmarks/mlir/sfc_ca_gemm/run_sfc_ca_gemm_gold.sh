#!/bin/bash
#
# Cold-cache GEMM sweep for libxsmm's reference sfc_ca_gemm binary (the "gold"
# reference for the tpp-run sweep in run_sfc_ca_gemm.sh). For every (M, N, K)
# configuration it:
#   1. runs the sfc_ca_gemm binary with cold-cache replication (n_layers=-1
#      auto-sizes the replicated footprint to ~5 GiB, matching the original
#      benchmark_sample.sh),
#   2. parses the reported GFLOP/s from the binary's MEASURE line,
#   3. derives the effective per-GEMM runtime so the result is directly
#      comparable to the tpp-run sweep (runtime_s = 2*M*N*K / (GFLOPS * 1e9)).
#
# The raw timings are written to a CSV (M,N,K,runtime_s) which is then handed to
# the shared postprocess.py to compute TFLOP/s and sort by arithmetic
# complexity -- the exact same post-processing used by run_sfc_ca_gemm.sh, so
# the two result CSVs line up row-for-row.
#
# The sfc_ca_gemm binary is built from https://github.com/libxsmm/sfc_ca_gemm.
# Point SFC_BIN at it (default: ./sfc_ca_gemm).
#
# CLI of the reference binary:
#   sfc_ca_gemm M N K bm bn bk kbf K_layers n_layers n_iters check [DTYPE]
# K-blocking (kbf) and K-layers are fixed to 1 to match the MLIR sweep, which
# does not model them.

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
# Directory for this script (used to locate postprocess.py).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Path to the libxsmm sfc_ca_gemm reference binary (default: ./sfc_ca_gemm).
SFC_BIN="${SFC_BIN:-${SCRIPT_DIR}/sfc_ca_gemm-main/sfc_ca_gemm}"
# Output directory for the result CSVs.
OUT_DIR="${OUT_DIR:-$(pwd)/sfc_ca_gemm_gold_results}"
# GEMM block tiles (bm, bn, bk). K-blocking and K-layers are fixed to 1 since
# the MLIR sweep does not model them.
TILES="${TILES:-32,32,32}"
# Number of timed iterations handed to the binary.
N_ITERS="${N_ITERS:-100}"
# Replication layer count. -1 lets the binary auto-size the cold-cache footprint
# to ~5 GiB (matching benchmark_sample.sh). Set a positive value to override.
N_LAYERS="${N_LAYERS:--1}"
# K-blocking factor and K-layers, fixed to 1 to match the MLIR sweep.
KBF="${KBF:-1}"
K_LAYERS="${K_LAYERS:-1}"
# Run the binary's built-in correctness check (0=no, 1=yes). Off by default for
# faster timing-only runs.
CHECK="${CHECK:-0}"
# Element data type: BF16, BF8 or FP32 (matches the MLIR sweep's bf16 default).
DTYPE="${DTYPE:-BF16}"
# numactl prefix for NUMA/affinity pinning (as in benchmark_sample.sh). Set to
# an empty string to run without numactl.
NUMACTL="${NUMACTL:-numactl -m 0 -C 0-63}"

# Split TILES (bm,bn,bk) into the positional block-size arguments.
IFS=',' read -r BM BN BK <<< "${TILES}"

# Assemble the optional numactl prefix as an argv array.
NUMACTL_PREFIX=()
if [[ -n "${NUMACTL}" ]]; then
  # shellcheck disable=SC2206
  NUMACTL_PREFIX=(${NUMACTL})
fi

# GEMM dimensions to sweep (full cross-product), matching run_sfc_ca_gemm.sh.
SIZES=(512 1024 2048 4096 8192)

#==============================================================================
# Sanity checks / auto-build
#==============================================================================
# If the sfc_ca_gemm binary is missing, fetch the upstream sources into this
# script's directory, build LIBXSMM and the benchmark with icx, and point
# SFC_BIN at the freshly built binary.
if [[ ! -x "${SFC_BIN}" ]]; then
  echo "sfc_ca_gemm binary not found at '${SFC_BIN}'; building from source." >&2

  SFC_SRC_URL="https://github.com/libxsmm/sfc_ca_gemm/archive/refs/heads/main.zip"
  SFC_SRC_DIR="${SCRIPT_DIR}/sfc_ca_gemm-main"
  SFC_ZIP="${SCRIPT_DIR}/sfc_ca_gemm-main.zip"

  if [[ ! -d "${SFC_SRC_DIR}" ]]; then
    echo "  Downloading ${SFC_SRC_URL}" >&2
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL "${SFC_SRC_URL}" -o "${SFC_ZIP}"
    elif command -v wget >/dev/null 2>&1; then
      wget -q "${SFC_SRC_URL}" -O "${SFC_ZIP}"
    else
      echo "error: need 'curl' or 'wget' to download the sfc_ca_gemm sources." >&2
      exit 1
    fi

    echo "  Unpacking $(basename "${SFC_ZIP}")" >&2
    unzip -q -o "${SFC_ZIP}" -d "${SCRIPT_DIR}"
    rm -f "${SFC_ZIP}"
  fi

  echo "  Preparing LIBXSMM (SFC_CA_GEMM_COMPILER=clang)" >&2
  ( cd "${SFC_SRC_DIR}" && SFC_CA_GEMM_COMPILER=clang ./prepare_libxsmm.sh )

  echo "  Building sfc_ca_gemm (SFC_CA_GEMM_COMPILER=clang make)" >&2
  ( cd "${SFC_SRC_DIR}" && SFC_CA_GEMM_COMPILER=clang make )

  SFC_BIN="${SFC_SRC_DIR}/sfc_ca_gemm"

  # Make the freshly built LIBXSMM shared library visible for this run only.
  SFC_LIBXSMM_LIB="${SFC_SRC_DIR}/libxsmm/lib"
  export LD_LIBRARY_PATH="${SFC_LIBXSMM_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ ! -x "${SFC_BIN}" ]]; then
  echo "error: cannot find executable '${SFC_BIN}' after build attempt." >&2
  echo "       Build it from https://github.com/libxsmm/sfc_ca_gemm and set" >&2
  echo "       SFC_BIN to its path." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
RAW_CSV="${OUT_DIR}/sfc_ca_gemm_gold_raw.csv"
echo "M,N,K,runtime_s" > "${RAW_CSV}"

#==============================================================================
# Sweep
#==============================================================================
for M in "${SIZES[@]}"; do
  for N in "${SIZES[@]}"; do
    for K in "${SIZES[@]}"; do
      echo "=== Running   M=${M} N=${N} K=${K} ==="
      run_out="$("${NUMACTL_PREFIX[@]}" "${SFC_BIN}" \
        "${M}" "${N}" "${K}" "${BM}" "${BN}" "${BK}" \
        "${KBF}" "${K_LAYERS}" "${N_LAYERS}" "${N_ITERS}" "${CHECK}" "${DTYPE}")"

      # The binary prints "MEASURE <gflops> SFC_CA_GEMM_...". Pull the GFLOP/s.
      gflops="$(printf '%s\n' "${run_out}" \
        | awk '/^MEASURE/ {print $2; exit}')"

      if [[ -z "${gflops}" ]]; then
        echo "warning: could not parse GFLOP/s for M=${M} N=${N} K=${K}" >&2
        runtime="nan"
      else
        # Effective per-GEMM time, comparable to the tpp-run per-call time:
        #   runtime_s = (2 * M * N * K) / (GFLOPS * 1e9)
        runtime="$(awk -v m="${M}" -v n="${N}" -v k="${K}" -v g="${gflops}" \
          'BEGIN { printf "%.10g", (2.0 * m * n * k) / (g * 1.0e9) }')"
      fi

      echo "${M},${N},${K},${runtime}" >> "${RAW_CSV}"
      echo "    GFLOP/s = ${gflops:-nan}, runtime = ${runtime} s"
    done
  done
done

echo
echo "Raw timings written to ${RAW_CSV}"

#==============================================================================
# Post-process: compute TFLOP/s and sort by arithmetic complexity
#==============================================================================
SORTED_CSV="${OUT_DIR}/sfc_ca_gemm_gold_results.csv"
python3 "${SCRIPT_DIR}/postprocess.py" "${RAW_CSV}" "${SORTED_CSV}"

echo "Sorted results with TFLOP/s written to ${SORTED_CSV}"
