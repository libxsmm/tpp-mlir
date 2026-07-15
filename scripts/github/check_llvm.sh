#!/usr/bin/env bash
#
# Script for automation only.
# Environment variables must have been declared already.
#
# Check LLVM installation.
# Optionally, trigger a build if LLVM is not found.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=${HOME}/installs/llvm
mkdir -p ${LLVMROOT}

# Find LLVM_VERSION
LLVM_VERSION=$(llvm_version)

LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}

if [ -f "${LLVM_INSTALL_DIR}/bin/mlir-opt" ]; then
  echo "Found $LLVM_VERSION"
  exit 0
else
  echo "Not Found 'mlir-opt' in ${LLVM_INSTALL_DIR}"
fi

# LLVM not found.
# Trigger a build if requested.
# Otherwise, return an error.
if [ "1" == "${BUILD}" ]; then
  COMMIT_SHA=$(git_commit)
  ${SCRIPT_DIR}/ci/trigger.sh -p tpp-llvm -c ${COMMIT_SHA}
else
  exit 1
fi
