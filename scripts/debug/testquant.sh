#!/bin/env bash
set -euo pipefail

# Base check: output is "correct" with base pass
for test in 3 8 16 32 64; do
  direct=direct-$test
  quantized=quant-$test
  echo "Running $direct and $quantized..."
  ./bin/mlir-gen --seed=123 --kernel=const --float-type=mx-f32-i8 --batch=$test --layers=$test,$test --quant-type=testquant > $direct.mlir
  ./bin/tpp-run -e entry -entry-point-result=void -print --splat-to-random --seed=123 $direct.mlir > $direct.out
  ./bin/mlir-gen --seed=123 --kernel=const --float-type=mx-f32-i8 --batch=$test --layers=$test,$test --quant-type=testquant > $quantized.mlir
  ./bin/tpp-run -e entry -entry-point-result=void -print --splat-to-random --seed=123 $quantized.mlir > $quantized.out
  ./bin/fpcmp -a 0.002 -i $direct.out $quantized.out
  echo "Direct & Quantize have compatible results up to 0.002"
done

# IR check: full pipeline
for test in 3 8 16 32 64; do
  direct=direct-$test
  quantized=quant-$test
  echo "Optimizing $direct and $quantized..."
  ./bin/tpp-opt --default-tpp-passes="linalg-to-vector" $direct.mlir -o $direct-opt.mlir
  ./bin/tpp-run -e entry -entry-point-result=void -print --splat-to-random --seed=123 $direct-opt.mlir > $direct-opt.out
  ./bin/tpp-opt --default-tpp-passes="linalg-to-vector" $quantized.mlir -o $quantized-opt.mlir
  ./bin/tpp-run -e entry -entry-point-result=void -print --splat-to-random --seed=123 $quantized-opt.mlir > $quantized-opt.out
done



# IR check: full pipeline
# loc("direct-3-opt.mlir":114:5): error: memory free side-effect on MemRef value not supported!
for test in 3 8 16 32 64; do
  direct=direct-$test
  quantized=quant-$test
  echo "Optimizing $direct and $quantized..."
  ./bin/tpp-opt --default-tpp-passes="linalg-to-vector" $direct.mlir -o $direct-opt.mlir
  ./bin/tpp-run -e entry -entry-point-result=void -print --splat-to-random --seed=123 $direct-opt.mlir > $direct-opt.out
  ./bin/tpp-opt --default-tpp-passes="linalg-to-vector" $quantized.mlir -o $quantized-opt.mlir
  ./bin/tpp-run -e entry -entry-point-result=void -print --splat-to-random --seed=123 $quantized-opt.mlir > $quantized-opt.out
done

