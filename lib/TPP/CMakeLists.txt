add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(GPU)
add_subdirectory(IR)
add_subdirectory(PassBundles)
add_subdirectory(Runner)
add_subdirectory(Transforms)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(TPPPipeline
  DefaultPipeline.cpp
  DefaultTppPasses.cpp
  LoadTppDialects.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPCompilerPassIncGen
    TPPCompilerPassBundleIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    ${mlir_dialect_libs}
    ${conversion_libs}
    TPPGPU
    TPPPassBundles
  )
