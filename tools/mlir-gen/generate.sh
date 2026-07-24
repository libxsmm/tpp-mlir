#!/usr/bin/env bash
#
# This script emulates the libxsmm-dnn command line to call the MLIR generator
# in the same way in hopes it will generate the same network. The IR is generated
# by mlir-gen, which is then passed to tpp-run to optimize and run the benchmark.
# See: https://github.com/plaidml/tpp-mlir/issues/341

# Comment to disable debug messages
debug () {
  LINE=$*
  echo -e "$LINE"
}

# Helper functions
getNumber() {
  ARG=$1
  if [ ! -z "${ARG##*[!0-9]*}" ]; then
    echo $ARG
  fi
}

run() {
  CMD=$1
  OUTPUT=$2
  echo " $ $CMD"
  if [ -n "$OUTPUT" ]; then
    $CMD > $OUTPUT 2> /dev/null
    ls -l $OUTPUT
  else
    $CMD 2> /dev/null
  fi
}

die() {
  echo "Syntax: generate.sh [ITER] [I/O] [FUSE] [PASS] [N-tile] [C-tile] [K-tile] [layers N] ..."
  echo
  echo "Where:"
  echo "  - [ITER] is the number of iterations to benchmark (after warmup). ITER > 1"
  echo "  - [MB] is the mini batch size"
  echo "  - [FUSE] is 0 (None), 1 (Bias), 2 (ReLU), 3 (Bias+ReLU) to fuse int the GEMM"
  echo "  - [PASS] is either F (Forward), B (Backward), A (Both) - Only 'F' supported now"
  echo "  - [*-tile] is the tile size for each matmul dimension - to pass the compiler"
  echo "  - [layers N] are the N layer sizes (input, hidden, output) [N >= 2]"
  echo "  - [-bf16] at the end, uses BF16"
  echo
  exit 1
}

# Parse args
ITER=$(getNumber $1)
if [ -z "$ITER" ]; then echo "Invalid ITER"; die; fi
shift
MB=$(getNumber $1)
if [ -z "$MB" ]; then echo "Invalid MiniBatch"; die; fi
shift
# Note: We ignore this argument as we always want to fuse
# but right now, we're not fusing yet
FUSE=$(getNumber $1)
if [ -z "$FUSE" ]; then echo "Invalid FUSE"; die; fi
shift
# Note: We don't implement the backward pass, so it must be 'F'
PASS=$1
if [ "$PASS" != "F" ]; then echo "Invalid PASS"; die; fi
shift
NTILE=$(getNumber $1)
if [ -z "$NTILE" ]; then echo "Invalid NTILE"; die; fi
shift
CTILE=$(getNumber $1)
if [ -z "$CTILE" ]; then echo "Invalid CTILE"; die; fi
shift
KTILE=$(getNumber $1)
if [ -z "$KTILE" ]; then echo "Invalid KTILE"; die; fi
shift
TILE_SIZES="$NTILE,$CTILE,$KTILE"
LAYER_SIZES=
while [ $# -gt 0 ]; do
  HIDDEN=$(getNumber $1)
  if [ -z "$HIDDEN" ]; then break; fi
  shift
  LAYER_SIZES="$LAYER_SIZES,$HIDDEN"
done
LAYER_SIZES=${LAYER_SIZES#","}
debug "CmdLine: $ITER $MB $FUSE $PASS $TILE_SIZES $LAYER_SIZES"

# Parse other optional arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -bf16)
      BF16=1
      shift # past argument
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      shift # past argument
      ;;
  esac
done

# BF16
FLOAT_TYPE="f32"
if [ BF16 ]; then FLOAT_TYPE="bf16"; fi

# Find binaries
ROOT=$(git rev-parse --show-toplevel)
if [ ! -d $ROOT ]; then
  ROOT=$PWD
fi
MLIR_GEN=$(find $ROOT -type f -name mlir-gen | head -n1)
TPP_OPT=$(find $ROOT -type f -name tpp-opt | head -n1)
TPP_RUN=$(find $ROOT -type f -name tpp-run | head -n1)
if [ -z "$MLIR_GEN" ] || [ -z "$TPP_OPT" ] || [ -z "$TPP_RUN" ]; then
  echo "Could not find binaries"
  exit 1
fi
debug "Generator: $MLIR_GEN"
debug "Optimizer: $TPP_OPT"
debug "Runner: $TPP_RUN"

# Pick a random seed (and print, for interoperability)
SEED=$(date +%s)
debug "Random seed: $SEED"

# Defaults

# Command line to extract and run
MLIR_GEN_ARGS="--data-type=$FLOAT_TYPE --seed=$SEED --batch=$MB --layers=$LAYER_SIZES --tiles=$TILE_SIZES"
ORIG_MLIR="mlir-gen-original.mlir"
debug "\nCreating the original MLIR model:"
run "$MLIR_GEN $MLIR_GEN_ARGS" $ORIG_MLIR

# FIXME: --pack-matmul isn't quite working with the rest of the pipeline
# Once it works, we can pass the TILE variables to it
TPP_OPT_ARGS="--default-tpp-passes"
if [ "$FLOAT_TYPE" == "bf16" ]; then
  TPP_OPT_ARGS="--pack-vnni $TPP_OPT_ARGS"
fi
OPT_MLIR="mlir-gen-optimized.mlir"
debug "\nOptimizing model:"
run "$TPP_OPT $TPP_OPT_ARGS $ORIG_MLIR" $OPT_MLIR

TPP_RUN_ARGS="-e entry -entry-point-result=void --seed=$SEED --n $ITER"
debug "\nBenchmarking model:"
run "$TPP_RUN $TPP_RUN_ARGS $OPT_MLIR"
