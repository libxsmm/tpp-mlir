#!/usr/bin/env bash
#
# Script for automation only.
# Environment variables must have been declared already.
#
# Run long benchmarks after building TPP-MLIR.
# shellcheck disable=SC1091

SCRIPT_DIR=$(realpath "$(dirname "$0")/..")
source "${SCRIPT_DIR}/ci/common.sh"

die_syntax() {
  echo "Syntax: $0 [-b] [-p] [-o] [-m] [-f]"
  echo ""
  echo "  -b: Optional, runs Base benchmarks"
  echo "  -p: Optional, runs MLIR of PyTorch models as benchmarks"
  echo "  -o: Optional, runs OpenMP benchmarks"
  echo "  -m: Optional, runs Matmul benchmarks"
  echo "  -f: Optional, runs Fully-Connected benchmarks"
  exit 1
}

# Options
BENCH_BASE=
BENCH_PT=
BENCH_OMP=
BENCH_MM=
BENCH_FC=
while getopts "bomfp" arg; do
  case ${arg} in
    b)
      BENCH_BASE=1
      ;;
    o)
      BENCH_OMP=1
      ;;
    m)
      BENCH_MM=1
      ;;
    f)
      BENCH_FC=1
      ;;
    p)
      BENCH_PT=1
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

# At least one must be enabled
if [ ! "$BENCH_BASE" ] && [ ! "$BENCH_PT" ] && [ ! "$BENCH_OMP" ] &&
   [ ! "$BENCH_MM" ] && [ ! "$BENCH_FC" ]; then
  echo "At least one benchmark must be enabled"
  exit 1
fi

BENCH_DIR=benchmarks
BUILD_DIR=$(realpath "${BUILD_DIR:-build-${COMPILER}}")
CONFIG_DIR=$(realpath "${BENCH_DIR}/config")

# CI jobs can make the run extra long
NUM_ITER=100
if [ "${BENCHMARK_NUM_ITER}" ]; then
  NUM_ITER=${BENCHMARK_NUM_ITER}
fi

# Build
"${SCRIPT_DIR}/github/build_tpp.sh"
if [ "$?" != 0 ]; then
  exit 1
fi

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
           -n ${NUM_ITER} \
           -c "${CONFIG_DIR}/${JSON}" \
           --build "${BUILD_DIR}" || exit 1

  popd || exit 1
}

# Base Benchmarks
if [ "$BENCH_BASE" ]; then
  benchmark base/base.json "Base Benchmarks"
  benchmark base/pack.json "Pack Benchmarks"
  benchmark base/mha.json "MHA Benchmarks"
  benchmark base/named-ops.json "Named Ops Benchmarks"
fi

# PyTorch model benchmarks
if [ "$BENCH_PT" ]; then
  benchmark pytorch/torch_dynamo.json "PyTorch-Dynamo Benchmarks"
fi

# OpenMP Benchmarks
if [ "$BENCH_OMP" ]; then
  benchmark omp/dnn-fp32.json "OpenMP XSMM-DNN FP32"
  benchmark omp/dnn-bf16.json "OpenMP XSMM-DNN BF16"
  benchmark omp/mlir-fp32.json "OpenMP TPP-MLIR FP32"
  benchmark omp/mlir-fp32-vector-to-kernel.json "OpenMP TPP-MLIR VECTOR-TO-KERNEL FP32"
  benchmark omp/mlir-bf16.json "OpenMP TPP-MLIR BF16"
  benchmark omp/torch-dynamo.json "OpenMP TPP-MLIR PyTorch"
  benchmark omp/torch-dynamo-vector-to-kernel.json "OpenMP TPP-MLIR VECTOR-TO-KERNEL PyTorch"
fi

# Matmul Benchmarks
if [ "$BENCH_MM" ]; then
  benchmark matmul/256x1024x1024.json "Matmul 256x1024x1024"
  benchmark matmul/256x1024x4096.json "Matmul 256x1024x4096"
  benchmark matmul/256x4096x1024.json "Matmul 256x4096x1024"
  benchmark matmul/128x1024x4096.json "Matmul 128x1024x4096"
  benchmark matmul/128x4096x1024.json "Matmul 128x4096x1024"
  benchmark matmul/128x1024x1024.json "Matmul 128x1024x1024"
  benchmark matmul/256x768x768.json "Matmul 256x768x768"
  benchmark matmul/128x768x768.json "Matmul 128x768x768"
  benchmark matmul/128x3072x768.json "Matmul 128x3072x768"
  benchmark matmul/128x768x3072.json "Matmul 128x768x3072"
  benchmark matmul/256x3072x768.json "Matmul 256x3072x768"
  benchmark matmul/256x768x3072.json "Matmul 256x768x3072"
  benchmark matmul/128x768x2304.json "Matmul 128x768x2304"
  benchmark matmul/1024x2560x1024.json "Matmul 1024x2560x1024"
  benchmark matmul/1024x1024x512.json "Matmul 1024x1024x512"
  benchmark matmul/1024x352x512.json "Matmul 1024x352x512"
  benchmark matmul/1024x512x256.json "Matmul 1024x512x256"
fi

# FC Benchmarks
if [ "$BENCH_FC" ]; then
  benchmark fc/256x1024x1024.json "FC 256x1024x1024"
  benchmark fc/256x1024x4096.json "FC 256x1024x4096"
  benchmark fc/256x4096x1024.json "FC 256x4096x1024"
  benchmark fc/128x1024x4096.json "FC 128x1024x4096"
  benchmark fc/128x4096x1024.json "FC 128x4096x1024"
  benchmark fc/128x1024x1024.json "FC 128x1024x1024"
  benchmark fc/256x768x768.json "FC 256x768x768"
  benchmark fc/128x768x768.json "FC 128x768x768"
  benchmark fc/128x3072x768.json "FC 128x3072x768"
  benchmark fc/128x768x3072.json "FC 128x768x3072"
  benchmark fc/256x3072x768.json "FC 256x3072x768"
  benchmark fc/256x768x3072.json "FC 256x768x3072"
  benchmark fc/128x768x2304.json "FC 128x768x2304"
  benchmark fc/1024x2560x1024.json "FC 1024x2560x1024"
  benchmark fc/1024x1024x512.json "FC 1024x1024x512"
  benchmark fc/1024x352x512.json "FC 1024x352x512"
  benchmark fc/1024x512x256.json "FC 1024x512x256"
fi
