#!/usr/bin/env bash
#
# Script for automation only.
# Environment variables must have been declared already.
#
# CMake and build LLVM.

# Include common utils
SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

LLVMROOT=${HOME}/installs/llvm
mkdir -p ${LLVMROOT}

# LLVM setup
echo "--- LLVM"
LLVM_VERSION=$(llvm_version)

echo "LLVM version: ${LLVM_VERSION}"

LLVM_INSTALL_DIR=${LLVMROOT}/${LLVM_VERSION}
LLVM_INSTALL_DIR=$(add_device_extensions ${LLVM_INSTALL_DIR} ${GPU})

# If there's another process building it, wait.
# Otherwise, make the dir quickly so others don't attempt at the same time
if [ -d ${LLVM_INSTALL_DIR} ]; then
  echo "LLVM being built by another process, results will be in ${LLVM_INSTALL_DIR}"
  wait_for_file "${LLVM_INSTALL_DIR}" "bin/mlir-opt"
  exit 0
else
  mkdir -p ${LLVM_INSTALL_DIR}
fi

# Destination for tar balls
if [ ! "${LLVM_TAR_DIR}" ]; then
  LLVM_TAR_DIR="/scratch/tpp-llvm-tar"
fi
LLVM_TAR_DIR=$(add_device_extensions ${LLVM_TAR_DIR} ${GPU})
mkdir -p ${LLVM_TAR_DIR}

# Fetch specific LLVM version
LLVM_TAR_NAME=${LLVM_VERSION}.zip
LLVM_TAR_FILE=${LLVM_TAR_DIR}/${LLVM_TAR_NAME}
if [ ! -f "${LLVM_TAR_FILE}" ]; then
  echo_run wget -nv -P ${LLVM_TAR_DIR} https://github.com/llvm/llvm-project/archive/${LLVM_TAR_NAME}
  if [ $? != 0 ]; then
    echo "Failed to fetch repo"
    rm -r ${LLVM_INSTALL_DIR}
    exit 1
  fi
fi

# Unzip the fetched repo
echo_run unzip -uqn ${LLVM_TAR_FILE} -d ${LLVM_TAR_DIR}
if [ $? != 0 ]; then
  echo "Failed to unpack repo"
  rm -r ${LLVM_INSTALL_DIR}
  exit 1
fi

# Cleanup tar ball to save device space
rm ${LLVM_TAR_FILE}

LLVM_PROJECT_DIR=${LLVM_TAR_DIR}/llvm-project-${LLVM_VERSION}

# Environment setup
echo "--- ENVIRONMENT"
if [ ! "${COMPILER}" ]; then
  COMPILER=clang
fi
if [ "${COMPILER}" == "clang" ]; then
  check_program clang
  check_program clang++
  CC=clang
  CXX=clang++
elif [ "${COMPILER}" == "gcc" ]; then
  check_program gcc
  check_program g++
  CC=gcc
  CXX=g++
else
  echo "Compiler "${COMPILER}" not recognized"
  rm -r ${LLVM_INSTALL_DIR}
  exit 1
fi

if [ ! "${LINKER}" ]; then
  LINKER=lld
fi
check_program ${LINKER}

if [ ! "${LLVM_BUILD_DIR}" ]; then
  LLVM_BUILD_DIR="/scratch/tpp-llvm"
fi
LLVM_BUILD_DIR=$(add_device_extensions ${LLVM_BUILD_DIR} ${GPU})
LLVM_BUILD_DIR=$(realpath ${LLVM_BUILD_DIR})
LLVM_BUILD_DIR=${LLVM_BUILD_DIR:-build-${COMPILER}}
mkdir -p ${LLVM_BUILD_DIR}

if [ "${GPU}" ]; then
  source ${SCRIPT_DIR}/ci/setup_gpu_env.sh
fi

# Necessary for building Python-bindings.
pip install --upgrade --user nanobind pybind11

echo "Environment configured successfully"

 # Configure LLVM
echo "--- CONFIGURE"

LLVM_PROJECTS="mlir"
LLVM_TARGETS="host"
if [ ! "${KIND}" ]; then
  KIND=RelWithDebInfo
fi

# LLVM CUDA setup
if [[ ${GPU,,} =~ "cuda" ]]; then
  LLVM_BUILD_EXTENSIONS="${LLVM_BUILD_EXTENSIONS} -DCMAKE_CUDA_COMPILER=nvcc -DMLIR_ENABLE_CUDA_RUNNER=ON -DMLIR_ENABLE_CUDA_CONVERSIONS=ON"
  LLVM_TARGETS="${LLVM_TARGETS};NVPTX"
fi
# LLVM SPIRV setup
if [ "${GPU}" ]; then
  LLVM_BUILD_EXTENSIONS="${LLVM_BUILD_EXTENSIONS} -DMLIR_ENABLE_SPIRV_CPU_RUNNER=ON"
fi

echo_run cmake -Wno-dev -G Ninja \
    -B${LLVM_BUILD_DIR} -S${LLVM_PROJECT_DIR}/llvm \
    -DLLVM_ENABLE_PROJECTS=${LLVM_PROJECTS} \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD=${LLVM_TARGETS} \
    -DCMAKE_BUILD_TYPE=${KIND} \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX} \
    -DLLVM_USE_LINKER=${LINKER} \
    -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR} \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    ${LLVM_BUILD_EXTENSIONS}

if [ $? != 0 ]; then
  rm -r ${LLVM_INSTALL_DIR}
  rm -r ${LLVM_BUILD_DIR}
  exit 1
fi

# Build LLVM
echo "--- BUILD"
echo_run ninja -C ${LLVM_BUILD_DIR} all
if [ $? != 0 ]; then
  rm -r ${LLVM_INSTALL_DIR}
  rm -r ${LLVM_BUILD_DIR}
  exit 1
fi

# Check LLVM
echo "--- CHECK"
echo_run ninja -C ${LLVM_BUILD_DIR} check-all
if [ $? != 0 ]; then
  rm -r ${LLVM_INSTALL_DIR}
  rm -r ${LLVM_BUILD_DIR}
  exit 1
fi

# Install LLVM
echo "--- INSTALL"
mkdir -p ${LLVM_INSTALL_DIR}
echo_run ninja -C ${LLVM_BUILD_DIR} install
if [ $? != 0 ]; then
  rm -r ${LLVM_INSTALL_DIR}
  rm -r ${LLVM_BUILD_DIR}
  exit 1
fi

# Cleanup
echo "--- CLEANUP"
rm -r ${LLVM_BUILD_DIR}
if [ $? != 0 ]; then
  echo "Failed to cleanup"
  exit 1
fi
