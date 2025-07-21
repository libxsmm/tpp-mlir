set(CHECKER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/check-cpuid.c)

add_executable(check-cpuid ${CHECKER_SRC})
target_compile_options(check-cpuid PRIVATE -O2)

# Optional: ensure it outputs to a known build dir
set_target_properties(check-cpuid PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Export the full path to parent scope
set(CHECKER_BIN ${CMAKE_BINARY_DIR}/bin/check-cpuid PARENT_SCOPE)

