# TPP MLIR

This is an experiment in using MLIR to automatically select the best [Tensor Processing Primitives](https://arxiv.org/abs/2104.05755) for linear algebra.

This repository contains an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as an `opt`-like tool to operate on that dialect and a `runner`-like tool to execute and benchmark MLIR kernels.

It also contains the recipes to use [LIBXSMM](https://github.com/libxsmm/libxsmm) from inside MLIR and can be used by other tools to drive our passes.

This repository was previously called `tpp-sandbox`.
If you have a checkout with the previous name, please follow [these instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository) to rename the remote locally.

## Build Status

| Build | Status |
| ----- | ------ |
| Tests | [![TPP-mlir](https://badge.buildkite.com/7c04eb392db7ba16b30684d80e0e4320254f7cf61558c6336f.svg?branch=main)](https://buildkite.com/intel/tpp-mlir) |
| Benchmarks | [![TPP-benchmark](https://badge.buildkite.com/087a1980507200f059ce3661f6ddb33c227db858d115691bf9.svg?branch=main)](https://buildkite.com/intel/tpp-benchmark) |

## How to setup the environment

In order to build LLVM and TPP-MLIR, several software development tools such as git, cmake, compilers, etc. are needed. As each operating system has its own package 
manager and package names, we opted for providing instructions for the user-level package manager ```conda```. This environment has been successfully tested on top of a Fedora Server
minimal installation with less than 400 system-wide packages being installed.

Initial Setup (using Conda):
```sh
export TPPMLIR_WORKSPACE_DIR=/foo
cd ${TPPMLIR_WORKSPACE_DIR}
export ARCH_NAME=$(uname -m)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH_NAME}.sh
bash Miniconda3-latest-Linux-${ARCH_NAME}.sh -b -p ${TPPMLIR_WORKSPACE_DIR}/miniconda3
eval "$(${TPPMLIR_WORKSPACE_DIR}/miniconda3/bin/conda shell.bash hook)"
conda activate

conda install -y cmake ninja git clang clangxx llvm lld llvm-openmp llvm-tools binutils
if [ "${ARCH_NAME}" == "aarch64" ]; then
   conda install -y gcc_linux-aarch64 gxx_linux-aarch64
elif [ "${ARCH_NAME}" == "x86_64" ]; then
   conda install -y gcc_linux-64 gxx_linux-64
fi
python -m pip install coloredlogs
```

Reloading the environment  after conda deactivate/logout/reboot:
```sh
export TPPMLIR_WORKSPACE_DIR=/foo
cd ${TPPMLIR_WORKSPACE_DIR}
eval "$(${TPPMLIR_WORKSPACE_DIR}/miniconda3/bin/conda shell.bash hook)"
conda activate
```

### Formatting Tools

Our project requires Python and C++ source format to be consistent on every commit.
For that, we use two tools on the `check-all` Ninja target to verify the formatting:
 * `clang-format` version 16
 * Python's `black` lint checker

Due to `clang-format`'s instability and non-backwards compatibility, we require that the version used be 16.
If you have another version of `clang` installed, make sure you install `clang-format-16` on your system.

Our CI will fail if these tools find formatting checks, as it uses the target `check-all`.
Having those tools locally will make it easier to have PRs passing CI and being merged.

## How to build LLVM

```sh
# Clone
git clone https://github.com/llvm/llvm-project.git

# checking out a tpp-mlir compatible version of llvm-project
wget https://raw.githubusercontent.com/plaidml/tpp-mlir/main/build_tools/llvm_version.txt
pushd llvm-project
git checkout `cat ../llvm_version.txt`
popd
rm llvm_version.txt

# create build dir
mkdir llvm-project/build
pushd llvm-project/build

# This is important for the next step
export CUSTOM_LLVM_ROOT=`pwd`
echo $CUSTOM_LLVM_ROOT
export PATH=$CUSTOM_LLVM_ROOT/bin:$PATH

# Configure Build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_USE_LINKER=lld

# Build
ninja 

popd
```

## How to build TPP MLIR

This setup assumes that you have built LLVM and MLIR in `$CUSTOM_LLVM_ROOT` as above.

_Note: OpenMP is a requirement to get multi-threaded performance on our code.
If you don't want to build with OpenMP, disable with the CMake flag `-DUSE_OpenMP=False`._

_Note: OneDNN is a requirement to get performance comparisons against our code.
If you don't want to build with OneDNN, disable with the CMake flag `-DUSE_OneDNN=False`._

```sh
# Clone
git clone https://github.com/plaidml/tpp-mlir.git
mkdir tpp-mlir/build
pushd tpp-mlir/build

# Build & test
# Please, make sure to use clang to build TPP-MLIR
cmake -G Ninja .. \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DMLIR_DIR=$CUSTOM_LLVM_ROOT/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$CUSTOM_LLVM_ROOT/bin/llvm-lit \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ 
cmake --build . --target check-all

popd
```

To build the documentation from the TableGen description of the dialect
operations, run:

```sh
cmake --build . --target mlir-doc
```

To enable experimental GPU support see: [GPU/README.md](lib/TPP/GPU/README.md)

## License

This project is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.

## References

BRGEMM: [High-Performance Deep Learning via a Single Building Block (2019)](https://arxiv.org/abs/1906.06440)

TPP: [Tensor Processing Primitives: A Programming Abstraction for Efficiency and Portability in Deep Learning & HPC Workloads (2021)](https://arxiv.org/abs/2104.05755)

PARLOOPER: [Harnessing Deep Learning and HPC Kernels via High-Level Loop and Tensor Abstractions on CPU Architectures (2023)](https://arxiv.org/abs/2304.12576)
