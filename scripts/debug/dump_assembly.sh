#!/usr/bin/env bash

SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

TMP_DIR=$(mktemp -d)
ROOT_DIR=$(git_root)
BIN_DIR=$ROOT_DIR/build/bin
TPP_RUN="${BIN_DIR}/tpp-run"
check_program ${TPP_RUN}
OBJDUMP=llvm-objdump
check_program ${OBJDUMP}
DUMP_FILE="$(mktemp).o"

# "Parse" the command line arguments for clues about the asm name
# Example:
# $ dump_assembly.sh matmul.mlir -e entry -entry-point-result=void --target-feature=avx512
for arg in "$@"; do
  if [[ "$arg" =~ \.mlir ]]; then
    MLIR_FILE=$(realpath "$arg")
    continue
  fi
  if [[ "$arg" =~ target-feature=(.*) ]]; then
    TARGET_FEATURE=${BASH_REMATCH[1]}
    continue
  fi
done
if [ -z "${MLIR_FILE}" ]; then
  echo "ERROR: No MLIR file specified!"
  exit 1
fi
if [ -z "${TARGET_FEATURE}" ]; then
  ASM_FILE=$(basename ${MLIR_FILE%.mlir}.s)
else
  ASM_FILE=$(basename ${MLIR_FILE%.mlir}-${TARGET_FEATURE}.s)
fi

# Compile and dump the assembly
TPP_RUN_CMDLINE="${TPP_RUN} $@ --dump-object-file --object-filename=${DUMP_FILE}"

echo "Running the program:"
echo " $ \"${TPP_RUN_CMDLINE}\""
${TPP_RUN_CMDLINE} > /dev/null

if [ $? -ne 0 ]; then
  echo "Error while running the program!"
  exit 1
fi

# Dump the assembly and remove the temporary object file
OBJDUMP_COMMAND="${OBJDUMP} -d ${DUMP_FILE}"

echo "Dumping the assembly:"
echo " $ \"${OBJDUMP_COMMAND} > ${ASM_FILE}\""
${OBJDUMP_COMMAND} > ${ASM_FILE}

if [ $? -ne 0 ]; then
  echo "Error dumping the assembly!"
  echo "Object file: ${DUMP_FILE}"
  exit 1
fi

rm -f ${DUMP_FILE}
