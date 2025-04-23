set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRIR
        MLIROptLib
        MLIRParser
        MLIRSupport
        TPPTransformsUtils
        )

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_llvm_executable(mlir-gen
  MLIRGen.cpp
  mlir-gen.cpp)

llvm_update_compile_flags(mlir-gen)

target_link_libraries(mlir-gen PRIVATE ${LIBS})

install(TARGETS mlir-gen RUNTIME DESTINATION bin)

configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/mlir-gen.py
        ${CMAKE_BINARY_DIR}/bin/mlir-gen.py
        COPYONLY)
