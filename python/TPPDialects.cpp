#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"

#include "mlir/Dialect/Transform/IR/TransformTypes.h"

#include "TPP/Dialect/Check/CheckDialect.h"
#include "TPP/Dialect/Perf/PerfDialect.h"
#include "TPP/Dialect/Transform/FfiExtension/TransformOps.h"
#include "TPP/Dialect/Xsmm/XsmmDialect.h"
#include "TPP/PassBundles.h"
#include "TPP/Passes.h"

namespace nb = nanobind;
using namespace mlir;

// Global to hold the python callback handler so that it is avialable to be
// called by the C++-callback handler.
nb::object callback_handler;

NB_MODULE(_tppDialects, m) {
  auto checkModule = m.def_submodule("check");

  checkModule.def(
      "register_dialect",
      [](MlirDialectRegistry wrappedRegistry) {
        DialectRegistry *registry = unwrap(wrappedRegistry);
        registry->insert<check::CheckDialect, perf::PerfDialect,
                         xsmm::XsmmDialect>();
      },
      "registry");

  auto perfModule = m.def_submodule("perf");

  perfModule.def(
      "register_dialect",
      [](MlirDialectRegistry wrappedRegistry) {
        DialectRegistry *registry = unwrap(wrappedRegistry);
        registry->insert<perf::PerfDialect>();
      },
      "registry");

  auto xsmmModule = m.def_submodule("xsmm");

  xsmmModule.def(
      "register_dialect",
      [](MlirDialectRegistry wrappedRegistry) {
        DialectRegistry *registry = unwrap(wrappedRegistry);
        registry->insert<xsmm::XsmmDialect>();
      },
      "registry");

  auto transformModule = m.def_submodule("transform");
  auto transformFfiModule = transformModule.def_submodule("ffi");

  transformFfiModule.def(
      "register_dialect_extension",
      [](MlirDialectRegistry wrappedRegistry) {
        DialectRegistry *registry = unwrap(wrappedRegistry);
        transform::ffi::registerDialectExtension(*registry);
      },
      "registry");

  transformFfiModule.def(
      "register_callback_handler", [&](nb::callable callable) {
        callback_handler =
            nb::borrow(callable); // TODO: should we ever release this?

        // Register a C++ callback that will
        // 1) wrap its arguments,
        // 2) call a Python callback with the wrapped-up arguments,
        // 3) and unwrap the results that the Python callback returned.
        transform::ffi::handler =
            [&](StringRef name,
                SmallVector<SmallVector<transform::MappedValue>> args)
            -> SmallVector<SmallVector<transform::MappedValue>> {
          // Wrap up the arguments to prepare for passing them to Python.
          nb::list pyArgs;
          for (auto handleAssociatedValues : args) {
            nb::list pyAssociatedValues;

            for (auto associatedValue : handleAssociatedValues) {
              if (auto *op = dyn_cast<Operation *>(associatedValue)) {
                // std::cout << "CALLBACK: pushing op\n";
                pyAssociatedValues.append(wrap(op));
              } else if (auto value = dyn_cast<Value>(associatedValue)) {
                pyAssociatedValues.append(wrap(value));
              } else if (auto paramAttr =
                             dyn_cast<transform::Param>(associatedValue)) {
                pyAssociatedValues.append(wrap(paramAttr));
              }
            }

            pyArgs.append(pyAssociatedValues);
          }

          // The callback to Python code.
          auto res = callback_handler(nb::str(name.data()), *pyArgs);

          // Needing to do this import here is ... not ideal.
          // The below commented-out code is potentially a better solution...
          nb::handle mlir_ir = nb::module_::import_("mlir.ir");
          nb::handle Operation = mlir_ir.attr("Operation");
          nb::handle Value = mlir_ir.attr("Value");
          nb::handle Attribute = mlir_ir.attr("Attribute");

          // Unwrap the results to prepare for passing them to C++.
          SmallVector<SmallVector<transform::MappedValue>> results;
          if (nb::isinstance<nb::list>(res) || nb::isinstance<nb::tuple>(res)) {
            for (auto assocList : res) {
              SmallVector<transform::MappedValue> associatedValues;
              for (auto elt : assocList) {
                // The following is probably preferable but is broken...
                // if (nb::isinstance<MlirValue>(elt)) {
                // If `elt` is of the wrong type, isinstance call will crash.
                if (nb::isinstance(elt, Value)) {
                  auto val = nb::cast<MlirValue>(elt);
                  associatedValues.push_back(unwrap(val));
                  // The following is probably preferable but is broken...
                  //} else if (nb::isinstance<MlirOperation>(elt)) {
                  // If `elt` is of the wrong type, isinstance call will crash.
                } else if (nb::isinstance(elt, Operation)) {
                  auto op = nb::cast<MlirOperation>(elt);
                  associatedValues.push_back(unwrap(op));
                  // The following is probably preferable but is broken...
                  //} else if (nb::isinstance<MlirAttribute>(elt)) {
                  // If `elt` is of the wrong type, isinstance call will crash.
                } else if (nb::isinstance(elt, Attribute)) {
                  auto param = nb::cast<MlirAttribute>(elt);
                  associatedValues.push_back(unwrap(param));
                }
              }
              results.push_back(associatedValues);
            }
          }
          return results;
        };
      });

  tpp::registerTppCompilerPasses();
  tpp::registerTppPassBundlePasses();
}
