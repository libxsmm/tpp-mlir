# FFI transform dialect extension
add_mlir_extension_library(
  TransformFfiExtensionTransformOps

  TransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/TPP

  DEPENDS
  TransformFfiExtensionTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
)
