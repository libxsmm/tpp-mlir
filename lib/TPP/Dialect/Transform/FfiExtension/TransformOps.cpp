#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "TPP/Dialect/Transform/FfiExtension/TransformOps.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "TPP/Dialect/Transform/FfiExtension/TransformOps.cpp.inc"

namespace mlir {
namespace transform {
namespace ffi {
Handler handler = nullptr;
} // namespace ffi
} // namespace transform
} // namespace mlir

//===----------------------------------------------------------------------===//
// CallbackOp
//===----------------------------------------------------------------------===//

void transform::ffi::CallbackOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getPayloadsMutable(), // TODO: Make specifiable on the op.
                  effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

DiagnosedSilenceableFailure
transform::ffi::CallbackOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {
  if (transform::ffi::handler == nullptr)
    return emitDefiniteFailure()
           << "callback called without a registered callback handler";

  SmallVector<SmallVector<MappedValue>> payloads;
  transform::detail::prepareValueMappings(payloads, getPayloads(), state);

  SmallVector<SmallVector<MappedValue>> res = transform::ffi::handler(
      getName().getRootReference().getValue(), payloads);

  for (auto &&[result, resPayload] : zip_equal(getResults(), res))
    results.setMappedValues(llvm::cast<OpResult>(result), resPayload);

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class TransformFfiDialectExtension
    : public transform::TransformDialectExtension<
          TransformFfiDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformFfiDialectExtension)

  TransformFfiDialectExtension() {
    registerTransformOps<
#define GET_OP_LIST
#include "TPP/Dialect/Transform/FfiExtension/TransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::transform::ffi::registerDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TransformFfiDialectExtension>();
}
