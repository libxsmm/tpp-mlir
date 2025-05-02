message(STATUS "TPP Project CMakeLists.txt BEGIN")

cmake_minimum_required(VERSION 3.13.4)

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(tpp-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(TPP_GPU "" CACHE STRING "Enables GPU runtime (default: '')")
set_property(CACHE TPP_GPU PROPERTY STRINGS "" "cuda")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(STATUS "TPP-MLIR out-of-tree build.")

  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  message(STATUS "TPP-MLIR in-tree build.")

  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

# Nanobind for Python bindings
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(PYTHON_DEV_MODULE Development)
else()
  set(PYTHON_DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${PYTHON_DEV_MODULE} REQUIRED)

# Nanobind docs say "Without this addition, binding code may run slowly and produce large binaries.":
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the pip/conda-installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
message (STATUS "Nanobind found")

set(TPP_MAIN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set(TPP_GEN_INCLUDE_DIR "${PROJECT_BINARY_DIR}/include")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${TPP_MAIN_INCLUDE_DIR})
include_directories(${TPP_GEN_INCLUDE_DIR})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
include(openmp)

# libxsmm
include(xsmm)
message (STATUS "LIBXSMM Include dir: ${XSMM_INCLUDE_DIRS}")
include(xsmm-dnn)
message (STATUS "LIBXSMM-DNN Include dir: ${XSMM_DNN_INCLUDE_DIRS}")

# onednn
include(one-dnn)

# Sanitizers (include as late as possible aka append)
include(sanitizers)

# Sub directories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(runtime)
add_subdirectory(tools)
add_subdirectory(test)

# Benchmarks
set(BENCHMARK_DIR "${PROJECT_SOURCE_DIR}/benchmarks")
set(CONFIG_DIR "${BENCHMARK_DIR}/config")

# Run baseline benchmarks with default iterations to track simple performance
set(BENCH_CFGS
  ${CONFIG_DIR}/base/base.json
  ${CONFIG_DIR}/base/pack.json
  ${CONFIG_DIR}/base/mha.json
  ${CONFIG_DIR}/base/named-ops.json
)
string(JOIN ',' BENCH_CFGS_STR ${BENCH_CFGS})
# Run a small set of benchmarks with small iterations to test the benchmarks and run locally on small machines
add_custom_target(benchmarks ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR} -n 10
                  -c ${BENCH_CFGS_STR}
                  DEPENDS tpp-opt tpp-run xsmm_dnn_mlp
                  WORKING_DIRECTORY ${BENCHMARK_DIR}
                  COMMENT Run Base Benchmarks)

# Run OpenMP benchmarks with default iterations to track simple performance
set(BENCH_OMP_CFGS
  ${CONFIG_DIR}/omp/dnn-fp32.json
  ${CONFIG_DIR}/omp/dnn-bf16.json
  ${CONFIG_DIR}/omp/mlir-fp32.json
  ${CONFIG_DIR}/omp/mlir-bf16.json
  ${CONFIG_DIR}/omp/mlir-fp32-vector-to-kernel.json
  ${CONFIG_DIR}/omp/torch-dynamo.json
  ${CONFIG_DIR}/omp/torch-dynamo-vector-to-kernel.json
)
string(JOIN ',' BENCH_OMP_CFGS_STR ${BENCH_OMP_CFGS})
add_custom_target(benchmarks-omp ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR} -n 10
                  -c ${BENCH_OMP_CFGS_STR}
                  DEPENDS tpp-opt tpp-run xsmm_dnn_mlp
                  WORKING_DIRECTORY ${BENCHMARK_DIR}
                  COMMENT Run Benchmarks)

# GPU Benchmarks
if (TPP_GPU)
  if (TPP_GPU MATCHES "cuda")
    set(BENCH_CUDA_CFGS
      ${CONFIG_DIR}/GPU/cuda.json
    )
    set(BENCH_CUDA_DEPS
      bench_cuda_matmul
    )
  endif()

  string(JOIN ',' BENCH_GPU_CFGS_STR ${BENCH_CUDA_CFGS})

  # Run a small set of benchmarks with small iterations to test the benchmarks and run locally on small machines
  add_custom_target(benchmarks-gpu ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR} -n 10
  -c ${BENCH_GPU_CFGS_STR}
  DEPENDS tpp-opt tpp-run ${BENCH_CUDA_DEPS}
  WORKING_DIRECTORY ${BENCHMARK_DIR}
  COMMENT Run Quick GPU Benchmarks)
endif()

# Run PyTorch Dynamo generated models as benchmarks
set(BENCH_PT_CFGS
  ${CONFIG_DIR}/pytorch/torch_dynamo.json
)
string(JOIN ',' BENCH_PT_CFGS_STR ${BENCH_PT_CFGS})
add_custom_target(benchmarks-pt ${BENCHMARK_DIR}/driver.py -v --build ${PROJECT_BINARY_DIR} -n 10
                  -c ${BENCH_PT_CFGS_STR}
                  DEPENDS tpp-opt tpp-run
                  WORKING_DIRECTORY ${BENCHMARK_DIR}
                  COMMENT Run PyTorch Models)

# Code reformat
set(CI_DIR "${PROJECT_SOURCE_DIR}/scripts/ci")
add_custom_target(reformat-python ${CI_DIR}/lint-python.sh -i)
add_custom_target(reformat-clang  ${CI_DIR}/lint-clang.sh -i)
add_custom_target(reformat DEPENDS reformat-clang reformat-python)

# Code format checks
add_custom_target(check-python ${CI_DIR}/lint-python.sh)
add_custom_target(check-clang  ${CI_DIR}/lint-clang.sh)
add_custom_target(check-format DEPENDS check-clang check-python)
