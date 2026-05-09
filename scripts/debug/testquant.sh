#!/bin/env bash
set -exuo pipefail

# Base check: output is "correct" with base pass
for test in 3 8 16 32 64; do
  direct=direct-$test
  quantized=quant-$test
  echo "Running $direct and $quantized..."
  ./bin/mlir-gen --seed=123 --kernel=const --float-type=f32 --batch=$test --layers=$test,$test > $direct.mlir
  ./bin/tpp-run -e entry -entry-point-result=void -print --splat-to-random --seed=123 $direct.mlir > $direct.out
  ./bin/mlir-gen --seed=123 --kernel=const --float-type=mx-f32-i8 --batch=$test --layers=$test,$test --quant-type=testquant > $quantized.mlir
  ./bin/tpp-run -e entry -entry-point-result=void -print --splat-to-random --seed=123 $quantized.mlir > $quantized.out
  ./bin/fpcmp -a 0.02 -i $direct.out $quantized.out
  echo "Direct & Quantize have compatible results up to 0.002"
done

# IR check: LOOPS pipeline
for test in 3 8 16 32 64; do
  direct=direct-$test
  quantized=quant-$test
  echo "Linalg To Loops: $direct and $quantized..."
  ./bin/tpp-run -linalg-to-loops -e entry -entry-point-result=void -print --splat-to-random --seed=123 $direct.mlir > $direct-loop.out
  diff -q $direct.out $direct-loop.out
  ./bin/tpp-run -linalg-to-loops -e entry -entry-point-result=void -print --splat-to-random --seed=123 $quantized.mlir > $quantized-loop.out
  diff -q $quantized.out $quantized-loop.out
done

# IR check: VECTOR pipeline
for test in 3 8 16 32 64; do
  direct=direct-$test
  quantized=quant-$test
  echo "Linalg To Vector: $direct and $quantized..."
  ./bin/tpp-run -linalg-to-vector -e entry -entry-point-result=void -print --splat-to-random --seed=123 $direct.mlir > $direct-vector.out
  diff -q $direct.out $direct-vector.out
  ./bin/tpp-run -linalg-to-vector -e entry -entry-point-result=void -print --splat-to-random --seed=123 $quantized.mlir > $quantized-vector.out
  diff -q $quantized.out $quantized-vector.out
done

