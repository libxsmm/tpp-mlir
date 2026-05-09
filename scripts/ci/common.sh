#!/usr/bin/env bash
#
# Common functions to all scripts
# Usage: source common.sh

# Find the git root directory
git_root() {
  if [ "$(command -v git)" ]; then
    git rev-parse --show-toplevel
  else
    echo "ERROR: missing prerequisites!"
    exit 1
  fi
}

# Find the current git commit SHA
git_commit() {
  if [ "$(command -v git)" ]; then
    git rev-parse HEAD
  else
    echo "ERROR: missing prerequisites!"
    exit 1
  fi
}

# Check if a program is in the PATH
check_program() {
  local PROG=${1}
  if ! which ${PROG} > /dev/null; then
    echo "ERROR: '${PROG}' not found!"
    exit 1
  fi
}

# Echoes and runs a program
echo_run() {
  local PROGRAM=$*
  echo "${PROGRAM}"
  ${PROGRAM}
}

# Get the LLVM version for this build
llvm_version() {
  local LLVM_VERSION_FILE=$(git_root)/build_tools/llvm_version.txt
  if [ ! -f "${LLVM_VERSION_FILE}" ]; then
    echo "ERROR: cannot find ${LLVM_VERSION_FILE} for ${PWD}!"
    exit 1
  fi
  local LLVM_VERSION=$(cat "${LLVM_VERSION_FILE}")
  if [ ! "${LLVM_VERSION}" ]; then
    echo "ERROR: cannot find LLVM version in ${LLVM_VERSION_FILE}!"
    exit 1
  fi
  echo "${LLVM_VERSION}"
}

# Add known device extension suffixes to a base string
add_device_extensions() {
  local BASE=${1}
  local DEVICE_LIST=${2}

  # GPU extensions
  if [[ ${DEVICE_LIST,,} =~ "cuda" ]]; then
    BASE=${BASE}-cuda
  elif [[ ${DEVICE_LIST,,} =~ "intel" ]]; then
    BASE=${BASE}-intel
  fi

  echo ${BASE}
}

# Wait for a file to appear on an existing directory
wait_for_file() {
  local DIR="${1}"
  local FILE="${1}/${2}"

  if [ ! -d ${DIR} ]; then
    echo "ERROR: Directory ${DIR} not found"
  fi
  echo "Waiting for ${FILE}..."
  while [ ! -f ${FILE} ]; do
    sleep 30
    echo "."
  done
  echo "Found"
}

# Check if Linux is of a particular distro
is_linux_distro() {
  local NAME="${1}"

  for file in /etc/os-release /etc/lsb-release /etc/redhat-release; do
    if [ -f "${file}" ] && grep -qi "${NAME}" ${file}; then
      echo "YES"
      return
    fi
  done
  echo "NO"
}
