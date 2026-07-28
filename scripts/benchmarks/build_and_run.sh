#!/usr/bin/env bash
#
# This script is meant to be used in a new machine, to build LLVM, TPP-MLIR
# and run all benchmarks. This should work in the same way as our local tests
# and reproduce our numbers on local/cloud machines.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

# Install packages needed
if [ "$(is_linux_distro Ubuntu)" == "YES" ]; then
  sudo apt update && \
  sudo apt install -y build-essential \
                      cmake clang lld ninja-build \
                      unzip python3-pip libomp-dev git
elif [ "$(is_linux_distro Amazon)" == "YES" ]; then
  sudo dnf install -y cmake clang lld ninja-build \
                      unzip python3-pip libomp-devel git
else
  echo "Not Ubuntu distro, tools may not be available"
fi

# Environment used by the scripts
SOURCE_DIR=$(git_root)
export KIND=Release
export COMPILER=clang
export LINKER=lld

# Build LLVM
export LLVMROOT=${HOME}/installs/llvm
export LLVM_VERSION=$(llvm_version)
export LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}
export LLVM_TAR_DIR=${SOURCE_DIR}/llvm
export LLVM_BUILD_DIR=${SOURCE_DIR}/llvm/build
if [ ! -f "${LLVM_INSTALL_DIR}/bin/mlir-opt" ]; then
  ${SCRIPT_DIR}/github/build_llvm.sh
else
  echo "LLVM already built on ${LLVM_INSTALL_DIR}"
fi

# Build TPP-MLIR
export BUILDKITE_BUILD_CHECKOUT_PATH=${SOURCE_DIR}
export BUILD_DIR=${SOURCE_DIR}/build-${COMPILER}
${SCRIPT_DIR}/github/build_tpp.sh

# Run benchmarks
export BENCH_DIR=${BUILDKITE_BUILD_CHECKOUT_PATH:-.}/benchmarks
export CONFIG_DIR=$(realpath "${BENCH_DIR}/config")
export NUM_ITER=${NUM_ITER:=1000}

pushd ${BENCH_DIR}

run_bench_on_dir() {
  local DIR=$1
  if [ ! -d "${DIR}" ]; then
    echo "Directory ${DIR} does not exist"
    exit 1
  fi
  for cfg in ${DIR}/*.json ; do
    echo_run ./driver.py -vv \
             -n ${NUM_ITER} \
             -c "${cfg}" \
             --build "${BUILD_DIR}"
  done
}

echo " ========= Base Benchmarks ==========="
run_bench_on_dir "${CONFIG_DIR}/base"

echo " ========= PyTorch Benchmarks ==========="
run_bench_on_dir "${CONFIG_DIR}/pytorch"

echo " ========= OpenMP Benchmarks ==========="
run_bench_on_dir "${CONFIG_DIR}/omp"

popd
