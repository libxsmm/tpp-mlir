option(USE_OneDNN "Use OneDNN" OFF)

if(USE_OneDNN)
  include (ExternalProject)

  set(DNNL_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/dnnl/src/dnnl/src)
  set(DNNL_INSTALL ${CMAKE_CURRENT_BINARY_DIR}/dnnl/install)
  set(DNNL_LIB_DIR ${DNNL_INSTALL}/${CMAKE_INSTALL_LIBDIR})
  set(DNNL_INCLUDE_DIR ${DNNL_INSTALL}/include)
  message (STATUS "ONE-DNN Include dir: ${DNNL_INCLUDE_DIR}")

  ExternalProject_Add(project_dnnl
      PREFIX dnnl
      GIT_REPOSITORY https://github.com/oneapi-src/onednn.git
      GIT_TAG 03691d7898b5a4fd1f471c8e30c97d394a5fdec2
      SOURCE_DIR ${DNNL_SOURCE}
      INSTALL_DIR ${DNNL_INSTALL}
      GIT_PROGRESS true
      USES_TERMINAL_BUILD true
      CMAKE_ARGS -DDNNL_BUILD_TESTS=OFF -DDNNL_ENABLE_CONCURRENT_EXEC=ON -DDNNL_BUILD_EXAMPLES=OFF
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${DNNL_INSTALL}
    )

  include_directories(${DNNL_INCLUDE_DIR})
  link_directories(${DNNL_LIB_DIR})
endif()
