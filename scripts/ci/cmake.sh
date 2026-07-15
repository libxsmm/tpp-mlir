#!/usr/bin/env bash
#
# Runs CMake on the source / build directories

# Include common utils
source $(realpath $(dirname $0))/common.sh

die_syntax() {
  echo "Syntax: $0 -s SRC_DIR -b BLD_DIR -m MLIR_DIR [-i INST_DIR]"
  echo "          [-t (Release|Debug|RelWithDebInfo)] [-c (clang|gcc)] [-g (gcc toolchain)]"
  echo "          [-l (ld|lld|gold|mold)] [-R] [-S] [-O] [-D] [-n N]"
  echo ""
  echo "  -i: Optional install dir, defaults to system"
  echo "  -t: Optional build type flag, defaults to Release"
  echo "  -c: Optional compiler flag, defaults to clang"
  echo "  -l: Optional linker flag, defaults to system linker"
  echo "  -R: Optional request to remove BLD_DIR before CMake"
  echo "  -S: Optional sanitizer flag, defaults to none"
  echo "  -O: Optional OpenMP support flag, defaults to none"
  echo "  -n: Optional link job flag, defaults to nproc"
  exit 1
}

# Cmd-line opts
while getopts "s:b:i:m:t:c:l:n:G:RSOD" arg; do
  case ${arg} in
    s)
      SRC_DIR=$(realpath ${OPTARG})
      if [ ! -d ${SRC_DIR}/.git ]; then
        echo "Source '${OPTARG}' not a Git directory"
        die_syntax
      fi
      ;;
    b)
      BLD_DIR=$(realpath ${OPTARG})
      if ! mkdir -p ${BLD_DIR}; then
        echo "Error creating build directory '${OPTARG}'"
        die_syntax
      fi
      ;;
    i)
      INST_DIR=$(realpath ${OPTARG})
      if ! mkdir -p ${INST_DIR}; then
        echo "Error creating install directory '${OPTARG}'"
        die_syntax
      fi
      ;;
    m)
      MLIR_DIR=$(realpath ${OPTARG})
      if [ ! -f ${MLIR_DIR}/MLIRConfig.cmake ]; then
        echo "MLIR '${OPTARG}' not a CMake directory"
        die_syntax
      fi
      ;;
    c)
      if [ "${OPTARG}" == "clang" ]; then
        check_program clang
        check_program clang++
        CC=clang
        CXX=clang++
      elif [ "${OPTARG}" == "gcc" ]; then
        check_program gcc
        check_program g++
        CC=gcc
        CXX=g++
      else
        echo "Compiler "${OPTARG}" not recognized"
        die_syntax
      fi
      ;;
    t)
      if [ "${OPTARG}" == "Release" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=${OPTARG}"
      elif [ "${OPTARG}" == "Debug" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=${OPTARG}"
      elif [ "${OPTARG}" == "RelWithDebInfo" ]; then
        BUILD_OPTIONS="${BUILD_OPTIONS} -DCMAKE_BUILD_TYPE=${OPTARG}"
      else
        echo "Build type "${OPTARG}" not recognized"
        die_syntax
      fi
      ;;
    l)
      if [ "${OPTARG}" == "ld" ]; then
        check_program ld
      elif [ "${OPTARG}" == "lld" ]; then
        check_program lld
      elif [ "${OPTARG}" == "gold" ]; then
        check_program gold
      elif [ "${OPTARG}" == "mold" ]; then
        check_program mold
      else
        echo "Linker "${OPTARG}" not recognized"
        die_syntax
      fi
      LINKER_OPTIONS="${LINKER_OPTIONS} -DLLVM_USE_LINKER=${OPTARG}"
      ;;
    R)
      REMOVE_BLD_DIR=1
      ;;
    S)
      SAN_OPTIONS="-DUSE_SANITIZER=\"Address;Memory;Leak;Undefined\""
      ;;
    O)
      BUILD_OPTIONS="${BUILD_OPTIONS} -DUSE_OpenMP=ON"
      ;;
    n)
      PROCS=$(nproc)
      if [ "${OPTARG}" -gt "0" ] && [ "${OPTARG}" -lt "${PROCS}" ]; then
        LINKER_OPTIONS="${LINKER_OPTIONS} -DCMAKE_JOB_POOL_LINK=link -DCMAKE_JOB_POOLS=link=${OPTARG}"
      else
        echo "Invalid value for number of linker jobs '${OPTARG}'"
        die_syntax
      fi
      ;;
    ?)
      echo "Invalid option: ${OPTARG}"
      die_syntax
      ;;
  esac
done

# Mandatory arguments
if [ ! "${BLD_DIR}" ] || [ ! "${SRC_DIR}" ] || [ ! "${MLIR_DIR}" ]; then
  die_syntax
fi

if [ ! "${BUILD_OPTIONS}" ]; then
  BUILD_OPTIONS="-DCMAKE_BUILD_TYPE=Release"
fi

if [ ! "${CC}" ] || [ ! "${CXX}" ]; then
  check_program clang
  check_program clang++
  CC=clang
  CXX=clang++
fi

# Check deps
check_program cmake
check_program ninja
check_program pip
pip install --upgrade --user lit
pip install --upgrade --user -r ${SRC_DIR}/benchmarks/harness/requirements.txt
TPP_LIT=$HOME/.local/bin/lit
if [ ! -x "${TPP_LIT}" ]; then
  echo "LIT at ${TPP_LIT} not executable"
  exit 1
fi

# Consider to remove BLD_DIR shortly before running CMake
if [ "${REMOVE_BLD_DIR}" ] && [ "0" != "${REMOVE_BLD_DIR}" ]; then
  echo_run rm -rf ${BLD_DIR}
fi

# CMake
echo_run cmake -Wno-dev -G Ninja \
    -B${BLD_DIR} -S${SRC_DIR} \
    -DCMAKE_C_COMPILER=${CC} -DCMAKE_CXX_COMPILER=${CXX} \
    -DCMAKE_INSTALL_PREFIX=${INST_DIR} \
    -DMLIR_DIR=${MLIR_DIR} \
    -DLLVM_EXTERNAL_LIT=${TPP_LIT} \
    ${BUILD_OPTIONS} \
    ${LINKER_OPTIONS} \
    ${SAN_OPTIONS}    
