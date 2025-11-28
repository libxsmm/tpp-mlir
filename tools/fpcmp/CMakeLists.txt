set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# These warnings are not enabled in llvm-test-suite by default.
add_compile_options(-Wno-cast-qual -Wno-bitwise-instead-of-logical)

add_llvm_executable(fpcmp
  fpcmp.c)

llvm_update_compile_flags(fpcmp)

install(TARGETS fpcmp RUNTIME DESTINATION bin)
