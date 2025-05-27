add_mlir_public_c_api_library(TPPCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  TPPCheckDialect
  TPPPerfDialect
  TPPTuneDialect
  TPPXsmmDialect
  TPPTransforms
  TPPPipeline
  tpp_xsmm_runner_utils
)
