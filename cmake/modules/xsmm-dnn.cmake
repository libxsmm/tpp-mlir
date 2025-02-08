# This needs to run *after* xsmm.cmake
if (NOT LIBXSMMROOT)
  message(FATAL_ERROR "LIBXSMM is a hard dependency for LIBXSMM-DNN")
endif()

# Use LIBXSMM_DNN (make PREFIX=/path/to/libxsmm-dnn) given by LIBXSMM_DNNROOT
set(LIBXSMM_DNNROOT $ENV{LIBXSMM_DNNROOT})
# Fetch LIBXSMM_DNN (even if LIBXSMM_DNNROOT is present)
set(LIBXSMM_DNNFETCH $ENV{LIBXSMM_DNNFETCH})

if(LIBXSMM_DNNROOT AND NOT LIBXSMM_DNNFETCH)
  message(STATUS "Found LIBXSMM_DNN (${LIBXSMM_DNNROOT})")
else()
  message(STATUS "Fetching LIBXSMM_DNN")
  include(FetchContent)

  FetchContent_Declare(
    xsmm_dnn
    URL https://github.com/libxsmm/libxsmm-dnn/archive/6757af7d214a49b06e5289b9784b4b99b4a35aaf.tar.gz
    URL_HASH SHA256=f7c879f2aed6a1562f3e15ea57637845181c6ae4267ec71975ab7b11cd16dfa4
  )

  FetchContent_GetProperties(xsmm_dnn)
  if(NOT xsmm_dnn_POPULATED)
    FetchContent_Populate(xsmm_dnn)
  endif()

  set(LIBXSMM_DNNROOT ${xsmm_dnn_SOURCE_DIR})
endif()

# Global settings
set(XSMM_DNN_INCLUDE_DIRS ${LIBXSMM_DNNROOT}/include)
file(GLOB XSMM_DNN_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMM_DNNROOT}/src/*.c)

# Create the MLP runner
add_executable(xsmm_dnn_mlp
  ${XSMM_DNN_SRCS}
  ${LIBXSMM_DNNROOT}/tests/mlp/mlp_example.c
)
set_property(TARGET xsmm_dnn_mlp PROPERTY COMPILE_WARNING_AS_ERROR ON)

set_Target_properties(xsmm_dnn_mlp PROPERTIES RUNTIME_OUTPUT_DIRECTORY bin)
target_include_directories(xsmm_dnn_mlp PRIVATE ${XSMM_INCLUDE_DIRS})
target_include_directories(xsmm_dnn_mlp PRIVATE ${XSMM_INCLUDE_DIRS}/../src/template)
target_include_directories(xsmm_dnn_mlp PRIVATE ${XSMM_DNN_INCLUDE_DIRS})
target_link_libraries(xsmm_dnn_mlp PRIVATE xsmm)
if (OPENMP_FOUND)
  target_compile_options(xsmm_dnn_mlp PRIVATE ${OpenMP_C_FLAGS})
  target_link_libraries(xsmm_dnn_mlp PRIVATE omp)
endif()
install(TARGETS xsmm_dnn_mlp RUNTIME DESTINATION bin)
