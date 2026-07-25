# Make to pass full path to LIBXSMMROOT
if(EXISTS ${LIBXSMMROOT})
  message(STATUS "Found LIBXSMM (${LIBXSMMROOT})")
  file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/include/libxsmm/*.c)
  list(REMOVE_ITEM XSMM_SRCS ${LIBXSMMROOT}/include/libxsmm/libxsmm_generator_gemm_driver.c)
else()
  message(STATUS "Fetching LIBXSMM")
  include(FetchContent)

  FetchContent_Declare(
    xsmm
    URL https://github.com/libxsmm/libxsmm/archive/7944bf36cf847c846b3fa0eb194789295e00b624.tar.gz
    URL_HASH SHA256=da74a1ca71d64b6e89e6e81ad6f337aab0d9ac9525fc87bc2d910e6bb058dfb4
  )

  FetchContent_GetProperties(xsmm)
  if(NOT xsmm_POPULATED)
    FetchContent_Populate(xsmm)
  endif()

  set(LIBXSMMROOT ${xsmm_SOURCE_DIR})
endif()

if(NOT XSMM_SRCS)
  file(GLOB XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
  list(REMOVE_ITEM XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
endif()

set(XSMM_INCLUDE_DIRS ${LIBXSMMROOT}/include)

add_mlir_library(xsmm STATIC ${XSMM_SRCS})
target_include_directories(xsmm PUBLIC
  $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
  $<INSTALL_INTERFACE:include/xsmm>
)
add_definitions(-DLIBXSMM_DEFAULT_CONFIG -U_DEBUG -D__BLAS=0)

set_property(TARGET xsmm PROPERTY COMPILE_WARNING_AS_ERROR ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(xsmm PUBLIC Threads::Threads)
target_link_libraries(xsmm PUBLIC ${CMAKE_DL_LIBS})

include(CheckLibraryExists)
check_library_exists(m sqrt "" XSMM_LIBM)
if(XSMM_LIBM)
  target_link_libraries(xsmm PUBLIC m)
endif()
check_library_exists(rt sched_yield "" XSMM_LIBRT)
if(XSMM_LIBRT)
  target_link_libraries(xsmm PUBLIC rt)
endif()
#target_link_libraries(xsmm PUBLIC c)
