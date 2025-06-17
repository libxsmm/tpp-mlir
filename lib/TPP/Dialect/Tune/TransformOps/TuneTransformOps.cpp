#include "TPP/Dialect/Tune/TuneTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "TPP/Dialect/Tune/TuneTransformOps.cpp.inc"

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
  if (getOptions().size() == 1) {
    results.setParams(getOperation()->getOpResults()[0], getOptions()[0]);
    return DiagnosedSilenceableFailure::success();
  }

  return emitDefiniteFailure()
         << "this op does not resolve non-deterministic choice!";
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
