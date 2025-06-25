#include "TPP/Dialect/Tune/TuneTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "TPP/Dialect/Tune/TuneTransformOps.cpp.inc"

namespace mlir {
namespace tune {
Handler handler = nullptr;
} // namespace tune
} // namespace mlir

//===----------------------------------------------------------------------===//
// TuneSelectOp
//===----------------------------------------------------------------------===//

void transform::TuneSelectOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  producesHandle(getOperation()->getOpResults(), effects);
  onlyReadsPayload(effects);
}

DiagnosedSilenceableFailure
transform::TuneSelectOp::apply(transform::TransformRewriter &rewriter,
                               transform::TransformResults &results,
                               transform::TransformState &state) {
  return emitDefiniteFailure()
         << "this op does not have interpreted semantics!";
}

//===----------------------------------------------------------------------===//
// TuneCallbackOp
//===----------------------------------------------------------------------===//

void transform::TuneCallbackOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getPayloadsMutable(), // TODO: Make specifiable on the op.
                  effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

DiagnosedSilenceableFailure
transform::TuneCallbackOp::apply(transform::TransformRewriter &rewriter,
                                 transform::TransformResults &results,
                                 transform::TransformState &state) {
  if (tune::handler == nullptr)
    return emitDefiniteFailure()
           << "callback called without a registered callback handler";

  SmallVector<SmallVector<MappedValue>> payloads;
  detail::prepareValueMappings(payloads, getPayloads(), state);

  SmallVector<SmallVector<MappedValue>> res =
      tune::handler(getName().getRootReference().getValue(), payloads);

  for (auto &&[result, resPayload] : zip_equal(getResults(), res))
    results.setMappedValues(llvm::cast<OpResult>(result), resPayload);

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class TuneTransformDialectExtension
    : public transform::TransformDialectExtension<
          TuneTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TuneTransformDialectExtension)

  TuneTransformDialectExtension() {
    registerTransformOps<
#define GET_OP_LIST
#include "TPP/Dialect/Tune/TuneTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::tune::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TuneTransformDialectExtension>();
}
