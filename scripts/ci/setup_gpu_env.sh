#!/usr/bin/env bash
#
# Sets up GPU environment.
# Usage: source setup_gpu_env.sh

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

# Env CUDA setup
if [[ ${GPU,,} =~ "cuda" ]]; then
  echo "Setting up CUDA environment"
  echo "Hard-coding MLIR-compatible CUDA version (12.9)"
  source /swtools/cuda/12.9.0/cuda_vars.sh
  check_program nvcc
fi

# Env Intel setup
if [[ ${GPU,,} =~ "intel" ]]; then
  echo "Setting up Intel XeGPU environment"
  VERSION="25.44.36015.5"
  echo "Using driver version ${VERSION}"
  source /swtools/intel-gpu/$VERSION/intel_gpu_vars.sh
  check_program iga64
fi
