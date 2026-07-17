// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=16 --layers=16,16 \
// RUN:  --tiles=16,16,16 --float-type=bf16 | \
// RUN: tpp-opt --pack-vnni | \
// RUN: tpp-run -print -seed 123 \
// RUN:  -e entry -entry-point-result=void > %t.xsmm

// RUN: mlir-gen --kernel=const --bias --relu --seed=123 --batch=16 --layers=16,16 \
// RUN:  --tiles=16,16,16 --float-type=bf16 | \
// RUN: tpp-opt --pack-vnni | \
// RUN: tpp-run -print -seed 123 -linalg-to-loops \
// RUN:  -e entry -entry-point-result=void > %t.loops

// RUN: fpcmp -r 0.01 %t.xsmm %t.loops
