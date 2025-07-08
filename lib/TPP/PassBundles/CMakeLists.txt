get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_mlir_library(TPPPassBundles
  Cleanup.cpp
  LinalgLowering.cpp
  LocalDialectsLowering.cpp
  LowLevelParallelization.cpp
  PostProcessing.cpp
  TppMapping.cpp
  VectorToXSMM.cpp
  VectorToKernel.cpp
  X86Vectorizer.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
    TPPCompilerPassIncGen
    TPPCompilerPassBundleIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    TPPTransforms
    ${mlir_dialect_libs}
    ${conversion_libs}
  )
