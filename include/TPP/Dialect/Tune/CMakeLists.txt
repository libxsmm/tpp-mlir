#add_mlir_dialect(TuneOps tune)
add_mlir_doc(TuneDialect TuneDialect TPP/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TuneTransformOps.td)
mlir_tablegen(TuneTransformOps.h.inc -gen-op-decls)
mlir_tablegen(TuneTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TuneTransformOpsIncGen)

add_mlir_doc(TuneTransformOps TuneTransformOps ./ -gen-op-doc)
