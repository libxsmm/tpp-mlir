# Scripts

These scripts help build and maintain the project.

The `env.sh` script is sourcing the TPP-environment (`enable-tpp`) as well as the LLVM-runtime environment matching TPP-mlir (`$TPP_LLVM_DIR` or `$CUSTOM_LLVM_ROOT`).

```bash
cd tpp-mlir
source scripts/env.sh
```

For example, no path to `mlir-vulkan-runner` needs to be specified (`$PATH`).

```bash
mlir-vulkan-runner ~/llvm-project/mlir/test/mlir-vulkan-runner/addf.mlir -e main -entry-point-result=void \
  -shared-libs=$TPP_LLVM_DIR/lib/libvulkan-runtime-wrappers.so \
  -shared-libs=$TPP_LLVM_DIR/lib/libmlir_c_runner_utils.so \
  -shared-libs=$TPP_LLVM_DIR/lib/libmlir_runner_utils.so
```

## CI

Generic CI scripts that check for dependencies, prepare environments (Conda, Virtualenv, etc).
They can be used by multiple CI environments to do generic setup, not specific setup.
They can also be used by developers on their machines.

Scripts to CMake and build the project, run benchmarks etc.
These should be generic to all environments, including developers' own machines.
Given the appropriate dependencies are installed, these should work everywhere.

## Github

Scripts executed by the Github CI. There should be one script per rule, used by all different builds.

To run local tests like in the CI-environment (relies at least on the TPP-environment):

```bash
cd tpp-mlir
source scripts/env.sh
KIND=Debug COMPILER=clang LINKER=lld CHECK=1 CLEAN=1 scripts/github/build_tpp.sh
```

Above scripts (scripts/env.sh) must be sourced from inside of the tpp-mlir directory (Git repository).
