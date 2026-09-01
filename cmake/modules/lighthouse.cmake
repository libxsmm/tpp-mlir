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
# Lighthouse is installed with the `uv` package manager (the way Lighthouse
# recommends); `uv` must be available on PATH.
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

# Lighthouse is installed with `uv`; it is required for this integration.
find_program(UV_EXECUTABLE uv REQUIRED)
message(STATUS "Lighthouse: using uv (${UV_EXECUTABLE})")

set(_lighthouse_sync_cmd
    COMMAND ${UV_EXECUTABLE} venv "${LIGHTHOUSE_VENV}"
    COMMAND ${UV_EXECUTABLE} sync)
if(LIGHTHOUSE_TORCH_INGRESS)
  list(APPEND _lighthouse_sync_cmd
       COMMAND ${UV_EXECUTABLE} sync --extra ingress_torch_${LIGHTHOUSE_TORCH_INGRESS})
endif()

# pyvenv.cfg is written by `uv venv` when the environment is created, at a stable
# path. Driving the install through add_custom_command(OUTPUT ...) lets Ninja/Make
# skip it once the venv exists, so `ninja lighthouse` is a no-op on a second run;
# a failed edge is still re-run. Force a reinstall with: rm -rf ${LIGHTHOUSE_VENV}
add_custom_command(
  OUTPUT "${LIGHTHOUSE_VENV}/pyvenv.cfg"
  ${_lighthouse_sync_cmd}
  WORKING_DIRECTORY "${LIGHTHOUSE_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Installing Lighthouse Python package via uv into ${LIGHTHOUSE_VENV}")

add_custom_target(lighthouse DEPENDS "${LIGHTHOUSE_VENV}/pyvenv.cfg")

# Run the Lighthouse pre-commit checks and LIT tests. precommit.sh drives
# everything through `uv run`.
add_custom_target(check-lighthouse
  ${UV_EXECUTABLE} run bash precommit.sh
  DEPENDS lighthouse
  WORKING_DIRECTORY "${LIGHTHOUSE_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Running Lighthouse pre-commit checks and tests")

message(STATUS "Lighthouse integration enabled (venv: ${LIGHTHOUSE_VENV})")
