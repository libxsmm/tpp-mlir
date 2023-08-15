#!/usr/bin/env bash
#
# Script for Buildkite automation only.
# Environment variables must have been declared already.
#
# Run GPU benchmarks after building TPP-MLIR.
# shellcheck disable=SC1091

SCRIPT_DIR=$(realpath "$(dirname "$0")/..")
source "${SCRIPT_DIR}/ci/common.sh"

BENCH_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH:-.}/benchmarks
BUILD_DIR=$(realpath "${BUILD_DIR:-build-${COMPILER}}")
CONFIG_DIR=$(realpath "${BENCH_DIR}/config")

LOGFILE=$(mktemp)
trap 'rm ${LOGFILE}' EXIT

if [ ! -z "${GPU}" ]; then
  GPU_OPTION="${GPU}"
else
  echo "Benchmark GPU target not specified"
  exit 1
fi

# Build
eval "GPU=${GPU_OPTION} ${SCRIPT_DIR}/buildkite/build_tpp.sh"

# Benchmark
benchmark () {
  JSON=$1
  if [ ! -f "${CONFIG_DIR}/${JSON}" ]; then
    echo "Cannot find benchmark configuration '${JSON}'"
    exit 1
  fi
  NAME=$2
  if [ ! "${NAME}" ]; then
    echo "Invalid benchmark name '${NAME}'"
    exit 1
  fi

  echo "--- BENCHMARK '${NAME}'"
  pushd "${BENCH_DIR}" || exit 1
  echo_run ./driver.py -v \
           -n 100 \
           -c "${CONFIG_DIR}/${JSON}" \
           --build "${BUILD_DIR}" \
           | tee -a "${LOGFILE}"
  popd || exit 1
}

# CUDA Benchmarks
if [ "${GPU_OPTION}" == "cuda" ]; then
  benchmark GPU/cuda.json "CUDA kernels"
fi

# Vulkan Benchmarks
if [ "${GPU_OPTION}" == "vulkan" ]; then
  echo "No Vulkan benchmarks"
  exit 1
fi

# Summary report for all benchmarks
echo "+++ REPORT"
if [ "main" = "${BUILDKITE_BRANCH}" ]; then
  export LOGRPTBRN=main
fi
eval "${LIBXSMMROOT}/scripts/tool_logrept.sh ${LOGFILE}"
