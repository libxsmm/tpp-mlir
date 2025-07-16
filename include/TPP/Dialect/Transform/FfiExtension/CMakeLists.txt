set(LLVM_TARGET_DEFINITIONS TransformOps.td)
mlir_tablegen(TransformOps.h.inc -gen-op-decls)
mlir_tablegen(TransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TransformFfiExtensionTransformOpsIncGen)

add_mlir_doc(FfiExtensionTransformOps FfiExtensionTransformOps ./ -gen-op-doc)
