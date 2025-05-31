add_mlir_dialect_library(TuneTransformOps
  TuneTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TuneTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
)
