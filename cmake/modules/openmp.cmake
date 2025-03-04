option(USE_OpenMP "Use OpenMP" ON)

# We don't want GOMP because its performance sinks for large core count, so we force libomp
# This finds the library path from the system's clang for OpenMP
#
# On Fedora, it's at the same place as others, so we don't need to look elsewhere
# On Ubuntu, it's in /usr/lib/llvm-${version}, so GCC finds GOMP instead.

if(USE_OpenMP)
  # Only bypass for GCC
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Only if we found an "llvm" path that we need to add
    find_library(LLVM_OMP NAMES omp libomp libomp.so libomp.so.5)
    find_library(PTHREAD NAMES pthread libpthread libpthread.a libpthread.so)
    if (NOT LLVM_OMP OR NOT PTHREAD)
      message(FATAL_ERROR "LLVM OpenMP required but not found")
    endif()
    set(OPENMP_FOUND ON)
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_C_LIBRARIES "${LLVM_OMP};${PTHREAD}")
    set(OpenMP_C_LIB_NAMES "omp;pthread")

    message(STATUS "OpenMP found")
    message(STATUS "OpenMP C flags: ${OpenMP_C_FLAGS}")
    message(STATUS "OpenMP C libs: ${OpenMP_C_LIBRARIES}")
    message(STATUS "OpenMP C libnames: ${OpenMP_C_LIB_NAMES}")

  else() # GNU / LLVM

    # Clang always prefers its own OpenMP
    find_package(OpenMP)
    if(OPENMP_FOUND)
      message(STATUS "OpenMP found")
      message(STATUS "OpenMP version: ${OpenMP_C_VERSION}")
      message(STATUS "OpenMP C flags: ${OpenMP_C_FLAGS}")
      message(STATUS "OpenMP C libs: ${OpenMP_C_LIBRARIES}")
      message(STATUS "OpenMP C libnames: ${OpenMP_C_LIB_NAMES}")
    else()
      message(FATAL_ERROR "LLVM OpenMP required but not found")
    endif()

  endif() # GNU / LLVM
endif() # USE_OpenMP
