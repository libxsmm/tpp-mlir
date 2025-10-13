#!/usr/bin/env bash
#
# Script for automation only.
# Environment variables must have been declared already.
#
# CMake and build TPP-MLIR.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=${HOME}/installs/llvm
if [ ! -d ${LLVMROOT} ]; then
  mkdir -p ${LLVMROOT}
fi

# Find LLVM_VERSION
echo "--- LLVM"
LLVM_VERSION=$(llvm_version)

LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}
LLVM_INSTALL_DIR=$(add_device_extensions ${LLVM_INSTALL_DIR} ${GPU})

if [ ! -d "${LLVM_INSTALL_DIR}" ]; then
  echo "LLVM ${LLVM_VERSION} not found"
  exit 1
else
  echo "Found LLVM ${LLVM_VERSION}"
fi

echo "--- ENVIRONMENT"
if [ ! "${COMPILER}" ]; then
  COMPILER=clang
fi
if [ "${SANITIZERS}" ]; then
  SANITIZERS="-S"
fi
if [ "${INSTALL}" ]; then
  if [ -d "${INSTALL_PREFIX}" ]; then
    INSTALL_OPTION="-i ${INSTALL_PREFIX}"
  else
    echo "Installation requires INSTALL_PREFIX (=${INSTALL_PREFIX}) to be a valid directory"
    exit 1
  fi
fi
if [ ! "${LINKER}" ]; then
  LINKER=lld
fi
if [ "${GPU}" ]; then
  GPU_OPTION="-G ${GPU}"
  source ${SCRIPT_DIR}/ci/setup_gpu_env.sh
fi
# Always build OpenMP in CI
EXTENSIONS="-O"
# Enable OneDNN build
if [ "${ONEDNN}" ]; then
  EXTENSIONS="${EXTENSIONS} -D"
fi

if [ "${CLEAN}" ]; then
  BUILD_DIR_RM=-R
fi

# Defaults when lacking CI environment
PROJECT_DIR=.
if [ ! "${BUILD_DIR}" ]; then
  BUILD_DIR="build-${COMPILER}"
fi
BUILD_DIR=$(realpath ${BUILD_DIR})
BUILD_DIR=${BUILD_DIR:-build-${COMPILER}}
mkdir -p ${BUILD_DIR}

echo "--- CONFIGURE"
if ! ${SCRIPT_DIR}/ci/cmake.sh \
  -s ${PROJECT_DIR} \
  -b ${BUILD_DIR} ${BUILD_DIR_RM} \
  -m ${LLVM_INSTALL_DIR}/lib/cmake/mlir \
  ${INSTALL_OPTION} \
  -t ${KIND} \
  ${SANITIZERS} \
  ${EXTENSIONS} \
  ${GPU_OPTION} \
  -c ${COMPILER} \
  -l ${LINKER} \
  -n ${NPROCS_LIMIT_LINK:-1}
then
  exit 1
fi

# Build
echo "--- BUILD"
if ! ${SCRIPT_DIR}/ci/build.sh \
  -b ${BUILD_DIR}
then
  exit 1
fi

# Check
if [ "1" == "${CHECK}" ]; then
  echo "--- CHECK"
  if ! ${SCRIPT_DIR}/ci/build.sh \
    -b ${BUILD_DIR} \
    -c
  then
    exit 1
  fi
fi

# Install
if [ "1" == "${INSTALL}" ]; then
  echo "--- INSTALL"
  if ! ${SCRIPT_DIR}/ci/build.sh \
    -b ${BUILD_DIR} \
    -i
  then
    exit 1
  fi
fi
