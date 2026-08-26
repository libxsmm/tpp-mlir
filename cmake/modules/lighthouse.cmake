# Optional integration with the LLVM Lighthouse project.
#
# Lighthouse (https://github.com/llvm/lighthouse) is a pure Python project that
# exposes the `lighthouse` Python module. It is vendored as a git submodule in
# third_party/lighthouse and installed into a Python virtual environment.
#
# The MLIR Python bindings and Torch-MLIR that Lighthouse pulls from PyPI are
# independent from the LLVM revision pinned in build_tools/llvm_version.txt: they
# are prebuilt wheels fetched at install time and need not match the LLVM used to
# build tpp-mlir.
#
# This integration is opt-in. Enable it with:
#   cmake -DTPP_ENABLE_LIGHTHOUSE=ON ...
# and then build the `lighthouse` target:
#   ninja lighthouse
#
# Options:
#   TPP_ENABLE_LIGHTHOUSE     Enable the Lighthouse integration (default OFF).
#   LIGHTHOUSE_TORCH_INGRESS  Torch ingress extra to install, e.g. cpu, nvidia,
#                             rocm or xpu. Empty to skip torch ingress (default).
#   LIGHTHOUSE_VENV           Path to the virtual environment to create/use
#                             (default: <lighthouse source>/.venv).

option(TPP_ENABLE_LIGHTHOUSE "Enable the LLVM Lighthouse Python integration" OFF)

if(NOT TPP_ENABLE_LIGHTHOUSE)
  return()
endif()

set(LIGHTHOUSE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/lighthouse")

if(NOT EXISTS "${LIGHTHOUSE_SOURCE_DIR}/pyproject.toml")
  message(FATAL_ERROR
    "Lighthouse submodule not found at ${LIGHTHOUSE_SOURCE_DIR}. "
    "Run: git submodule update --init --recursive")
endif()

set(LIGHTHOUSE_TORCH_INGRESS "" CACHE STRING
    "Lighthouse torch ingress extra (cpu, nvidia, rocm, xpu). Empty to skip.")
set(LIGHTHOUSE_VENV "${LIGHTHOUSE_SOURCE_DIR}/.venv" CACHE PATH
    "Virtual environment to install Lighthouse into")

# Prefer the `uv` package manager (the way Lighthouse recommends). Fall back to
# `pip` if `uv` is not available.
find_program(UV_EXECUTABLE uv)

if(UV_EXECUTABLE)
  message(STATUS "Lighthouse: using uv (${UV_EXECUTABLE})")

  set(_lighthouse_sync_cmd
      ${UV_EXECUTABLE} venv "${LIGHTHOUSE_VENV}"
      COMMAND ${UV_EXECUTABLE} sync)
  if(LIGHTHOUSE_TORCH_INGRESS)
    list(APPEND _lighthouse_sync_cmd
         COMMAND ${UV_EXECUTABLE} sync --extra ingress_torch_${LIGHTHOUSE_TORCH_INGRESS})
  endif()

  add_custom_target(lighthouse
    ${_lighthouse_sync_cmd}
    WORKING_DIRECTORY "${LIGHTHOUSE_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Installing Lighthouse Python package via uv into ${LIGHTHOUSE_VENV}")
else()
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  message(STATUS "Lighthouse: uv not found, using pip (${Python3_EXECUTABLE})")

  if(LIGHTHOUSE_TORCH_INGRESS)
    set(_lighthouse_spec ".[ingress_torch_${LIGHTHOUSE_TORCH_INGRESS}]")
  else()
    set(_lighthouse_spec ".")
  endif()

  add_custom_target(lighthouse
    ${Python3_EXECUTABLE} -m venv "${LIGHTHOUSE_VENV}"
    COMMAND "${LIGHTHOUSE_VENV}/bin/python" -m pip install ${_lighthouse_spec}
            --find-links https://llvm.github.io/eudsl/
            --find-links https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
            --extra-index-url https://download.pytorch.org/whl
            --only-binary :all:
    WORKING_DIRECTORY "${LIGHTHOUSE_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Installing Lighthouse Python package via pip into ${LIGHTHOUSE_VENV}")
endif()

# Run the Lighthouse pre-commit checks and LIT tests. precommit.sh drives
# everything through `uv run`, so it requires uv to be available.
if(UV_EXECUTABLE)
  add_custom_target(check-lighthouse
    ${UV_EXECUTABLE} run bash precommit.sh
    DEPENDS lighthouse
    WORKING_DIRECTORY "${LIGHTHOUSE_SOURCE_DIR}"
    USES_TERMINAL
    COMMENT "Running Lighthouse pre-commit checks and tests")
else()
  message(STATUS "Lighthouse: uv not found, 'check-lighthouse' target unavailable")
endif()

message(STATUS "Lighthouse integration enabled (venv: ${LIGHTHOUSE_VENV})")
