add_subdirectory(TransformOps)

add_mlir_dialect_library(TPPTuneDialect
  # Ops and dialects
    TuneDialect.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TPP

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
