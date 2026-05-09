//===- Builder Utils - Helper for builder patterns ------------------------===//
// Utilities to help build MLIR
//
//===----------------------------------------------------------------------===//
#ifndef TPP_TRANSFORMS_UTILS_BUILDERUTILS_H
#define TPP_TRANSFORMS_UTILS_BUILDERUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "TPP/Transforms/Utils/TensorInit.h"

namespace mlir {
class DenseElementsAttr;
class ModuleOp;
class MemRefType;
class TensorType;
class ShapedType;
class Type;
class TypeRange;
class OpBuilder;
class Operation;
class Value;
class FloatType;
namespace func {
class FuncOp;
} // namespace func

// Creates a function, its entry basic block and sets the entry point
// inside that block
func::FuncOp createFunction(OpBuilder &builder, ModuleOp module,
                            llvm::StringRef name, TypeRange args, TypeRange ret,
                            bool createBody = true);

// Create a local constant dense tensor
Value createDenseTensor(OpBuilder &, TensorInitType, TensorType, int);

// Create a global dense memref
Value createDenseMemref(OpBuilder &, ModuleOp, TensorInitType, MemRefType, int,
                        MemRefType, bool isScaleArgument = false);

// Return a ConstantOp of a certain type with a certain initializer
Value getConstIndex(OpBuilder &, int);
Value getConstInt(OpBuilder &, int, int);
Value getConstFloat(OpBuilder &, float, FloatType);

// Return a typed attribute of specified type and value.
// For integer types, the value is rounded toward zero.
TypedAttr getTypedAttr(OpBuilder &builder, Type type, double value);

} // namespace mlir

#endif // TPP_TRANSFORMS_UTILS_BUILDERUTILS_H
