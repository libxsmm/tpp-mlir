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

option(LIGHTHOUSE_TRACK_REMOTE
       "Update the Lighthouse submodule to the tip of its tracked branch (see \
.gitmodules) at configure time instead of the committed revision. This keeps the \
MLIR-bindings nightly wheels current with the upstream rolling feed so the pin \
never ages out; turn OFF to build the exact committed submodule revision." ON)

# Auto-checkout (and, when tracking is enabled, auto-advance) the submodule so
# users need not run git manually. Only attempt this inside a git work tree; a
# source tarball ships the sources directly.
set(_lighthouse_need_checkout FALSE)
if(NOT EXISTS "${LIGHTHOUSE_SOURCE_DIR}/pyproject.toml")
  set(_lighthouse_need_checkout TRUE)
endif()
if(_lighthouse_need_checkout OR LIGHTHOUSE_TRACK_REMOTE)
  find_package(Git QUIET)
  if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(_lighthouse_update_args submodule update --init --recursive)
    if(LIGHTHOUSE_TRACK_REMOTE)
      # --remote fetches and checks out the tip of the branch configured in
      # .gitmodules rather than the frozen gitlink.
      set(_lighthouse_update_args submodule update --init --remote --recursive)
    endif()
    message(STATUS "Lighthouse: syncing submodule at ${LIGHTHOUSE_SOURCE_DIR}")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} ${_lighthouse_update_args} -- third_party/lighthouse
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RESULT_VARIABLE _lighthouse_submodule_result)
    if(NOT _lighthouse_submodule_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to update the Lighthouse submodule "
        "(git exit code ${_lighthouse_submodule_result}). "
        "Run manually: git submodule update --init --recursive")
    endif()
  endif()
endif()

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
# Besides PATH, also look in uv's standard install locations: the standalone
# installer and `pip install --user` drop the binary in ~/.local/bin, the Cargo
# install in ~/.cargo/bin, and UV_INSTALL_DIR overrides both. CI shells that
# invoke cmake without a login profile (e.g. srun) often omit ~/.local/bin.
find_program(UV_EXECUTABLE
  NAMES uv
  HINTS
    ENV UV_INSTALL_DIR
    "$ENV{HOME}/.local/bin"
    "$ENV{HOME}/.cargo/bin"
  REQUIRED)
message(STATUS "Lighthouse: using uv (${UV_EXECUTABLE})")

# `uv venv --clear` recreates the environment even if a previous run left a
# partial venv behind (e.g. a failed `uv sync`), so the install is idempotent.
set(_lighthouse_sync_cmd
    COMMAND ${UV_EXECUTABLE} venv --clear "${LIGHTHOUSE_VENV}"
    COMMAND ${UV_EXECUTABLE} sync)
if(LIGHTHOUSE_TORCH_INGRESS)
  list(APPEND _lighthouse_sync_cmd
       COMMAND ${UV_EXECUTABLE} sync --extra ingress_torch_${LIGHTHOUSE_TORCH_INGRESS})
endif()

# pyvenv.cfg is written by `uv venv` when the environment is created, at a stable
# path. Driving the install through add_custom_command(OUTPUT ...) lets Ninja/Make
# skip it once the venv exists, so `ninja lighthouse` is a no-op on a second run.
# Depending on pyproject.toml re-triggers the install when the submodule advances
# to a revision with new dependency pins, so the venv never lags the sources.
# Force a full reinstall with: rm -rf ${LIGHTHOUSE_VENV}
add_custom_command(
  OUTPUT "${LIGHTHOUSE_VENV}/pyvenv.cfg"
  ${_lighthouse_sync_cmd}
  WORKING_DIRECTORY "${LIGHTHOUSE_SOURCE_DIR}"
  DEPENDS "${LIGHTHOUSE_SOURCE_DIR}/pyproject.toml"
  USES_TERMINAL
  COMMENT "Installing Lighthouse Python package via uv into ${LIGHTHOUSE_VENV}")

add_custom_target(lighthouse ALL DEPENDS "${LIGHTHOUSE_VENV}/pyvenv.cfg")

# Run the Lighthouse pre-commit checks and LIT tests. precommit.sh drives
# everything through `uv run`.
add_custom_target(check-lighthouse
  ${UV_EXECUTABLE} run bash precommit.sh
  DEPENDS lighthouse
  WORKING_DIRECTORY "${LIGHTHOUSE_SOURCE_DIR}"
  USES_TERMINAL
  COMMENT "Running Lighthouse pre-commit checks and tests")

message(STATUS "Lighthouse integration enabled (venv: ${LIGHTHOUSE_VENV})")
