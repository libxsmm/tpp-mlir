include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(TPPPythonSources)
declare_mlir_python_sources(TPPPythonSources.Dialects
  ADD_TO_PARENT TPPPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TPPPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/CheckOps.td
  SOURCES
  dialects/check.py
  DIALECT_NAME check
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TPPPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/PerfOps.td
  SOURCES
  dialects/perf.py
  DIALECT_NAME perf
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT TPPPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/XsmmOps.td
  SOURCES
  dialects/xsmm.py
  DIALECT_NAME xsmm
)

declare_mlir_python_extension(TPPPythonSources.DialectExtension
  MODULE_NAME _tppDialects
  ADD_TO_PARENT TPPPythonSources
  SOURCES
  TPPDialects.cpp
  EMBED_CAPI_LINK_LIBS
  TPPCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)

declare_mlir_python_sources(TPPPythonSources.Sched
  ADD_TO_PARENT TPPPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  SOURCES
  tpp/sched/__init__.py
  tpp/sched/__main__.py
  tpp/sched/bundles.py
  tpp/sched/common.py
  tpp/sched/main.py
  tpp/sched/utils.py
)

declare_mlir_python_sources(TPPPythonSources.Xsmm
  ADD_TO_PARENT TPPPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  SOURCES
  tpp/xsmm/utils.py
)

################################################################################
# Common CAPI
################################################################################
add_mlir_python_common_capi_library(TPPPythonCAPI
  INSTALL_COMPONENT TPPPythonModules
  INSTALL_DESTINATION python_packages/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
  TPPPythonSources
  MLIRPythonExtension.RegisterEverything
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine
)

################################################################################
# Instantiation of all Python modules
################################################################################
add_mlir_python_modules(TPPPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/mlir"
  INSTALL_PREFIX "python_packages/mlir"
  DECLARED_SOURCES
  TPPPythonSources # These are the TPP-MLIR specific bits of the Python bindings
  LLVMSupport
  MLIRPythonExtension.RegisterEverything # Generates whole mlir Python bindings
  MLIRPythonSources.Core # Generates whole mlir Python bindings
  MLIRPythonSources.Dialects # Generates whole mlir Python bindings
  MLIRPythonSources.ExecutionEngine # Generates whole mlir Python bindings
  COMMON_CAPI_LINK_LIBS
  TPPPythonCAPI
  TPPTransforms
  TPPPipeline
  TPPRunner  # Only here because of weird dependency elsewhere...
)
