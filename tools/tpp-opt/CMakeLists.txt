set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

if(USE_OneDNN)
  set(ONEDNN_LIBS "tpp_dnnl_runner_utils")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_source_files_properties(
    tpp-opt.cpp
    PROPERTIES COMPILE_FLAGS "-Wno-error=subobject-linkage")
endif()


set(LIBS
        MLIRRegisterAllExtensions
        MLIRRegisterAllDialects
        MLIRRegisterAllPasses
        MLIRToLLVMIRTranslationRegistration
        MLIROptLib
        TPPPipeline
        TPPRunner
        TPPTransforms
        tpp_xsmm_runner_utils
        ${ONEDNN_LIBS}
        TPPTestLib
        )
add_llvm_executable(tpp-opt tpp-opt.cpp)

llvm_update_compile_flags(tpp-opt)
target_link_libraries(tpp-opt PRIVATE ${LIBS})
install(TARGETS tpp-opt RUNTIME DESTINATION bin)
