#!/usr/bin/env bash

SCRIPT_DIR=$(realpath $(dirname $0)/..)
source ${SCRIPT_DIR}/ci/common.sh

TMP_DIR=$(mktemp -d)
BASELINE=${TMP_DIR}/baseline.out
OUTPUT=${TMP_DIR}/tpp-run.out
ENTRY=entry
DELTA=0.01

ROOT_DIR=$(git_root)
BIN_DIR=$ROOT_DIR/build/bin
while getopts "b:o:d:e:i:" arg; do
  case ${arg} in
    b)
      BIN_DIR=$(realpath ${OPTARG})
      ;;
    o)
      TPP_RUN_FLAGS=${OPTARG}
      ;;
    d)
      DELTA=${OPTARG}
      ;;
    e)
      ENTRY=${OPTARG}
      ;;
    i)
      INPUT_FILE=$(realpath ${OPTARG})
      if [ ! -f ${INPUT_FILE} ]; then
        echo "'${OPTARG}' not a file"
        exit 1
      fi
      ;;
    *)
      echo "Invalid option: ${OPTARG}"
      exit 1
  esac
done
if [ ! -x ${BIN_DIR}/tpp-run ]; then
  echo "'${OPTARG}' not a bin directory"
  exit 1
fi
TPP_RUN=${BIN_DIR}/tpp-run
DIFF_TOOL=${BIN_DIR}/fpcmp
if [ ! -f ${INPUT_FILE} ]; then
  echo "Invalid input file '${INPUT_FILE}'"
  exit 1
fi

## Get baseline (linalg-to-loops)
echo "Producing baseline at ${BASELINE} (linalg-to-loops)"
${TPP_RUN} \
  -e ${ENTRY} -entry-point-result=void \
  -linalg-to-loops \
  "${INPUT_FILE}" \
  > ${BASELINE}

if [ ! -s ${BASELINE} ]; then
  echo "Baseline file '${BASELINE}' is empty"
  exit 1
fi

## Get payload IR
echo "Producing dump at ${OUTPUT} (${TPP_RUN_FLAGS})"
${TPP_RUN} \
  -e ${ENTRY} -entry-point-result=void \
  ${TPP_RUN_FLAGS} \
  "${INPUT_FILE}" \
  > ${OUTPUT}

## Diff the outputs
echo "Diffing the files with ${DIFF_TOOL}"
${DIFF_TOOL} -a ${DELTA} -r ${DELTA} -i ${BASELINE} ${OUTPUT}

if [ $? -eq 0 ]; then
  echo "The outputs are compatible to DELTA=${DELTA}"
else
  exit 1
fi
