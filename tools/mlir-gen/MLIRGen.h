//===- MLIRGen.h MLIR Generator -------------------------------------------===//
//
// Class that handles MLIR generation for the MLIR options.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "TPP/Transforms/Utils/BuilderUtils.h"

namespace mlir {
class ModuleOp;
class MemRefType;
class Operation;
class Value;
namespace func {
class FuncOp;
} // namespace func

/// MLIR Generator: produces MLIR linalg-on-tensor dialect for an MLIR model
/// with the appropriate number of hidden layers and other properties selected.
class MLIRGenerator {
  /// MLIR Context
  MLIRContext context;

  /// MLIR OpBuilder
  OpBuilder builder;

  /// Unknown location, since all this code is auto-generated
  Location loc;

  /// Main module
  ModuleOp module;

  /// Batch size
  unsigned batch;

  /// Layer sizes
  SmallVector<int64_t> layers;

  /// Tile sizes
  SmallVector<int64_t> tiles;

  /// Data type (element type of all tensors)
  SmallVector<Type> dataTypes;

  /// Random seed
  int seed;

  /// Identity weights
  bool identity;

  /// Generated model's flops
  int64_t flops;

  /// Tensor init type
  TensorInitType initType;

  // ============================ Code Generation Options

  /// Lower bias add on every layer
  bool enableBias;

  /// Lower ReLU on every layer
  bool enableRelu;

  /// Lower softmax at the last layer
  bool enableSoftmax;

  /// List of linalg output Op kind which can be generated
  enum class OutputOpKind { Generic, Contract, NamedOp };

  /// Kind of linalg output Op to be generated
  OutputOpKind outputOpKind;

  /// Allow emitting generic matmul when OutputOpKind is named op
  bool keepGenericMatmul;

  /// List of supported kernel types that can be generated
  ///  * Const: Generates weights and biases as constant (RO).
  ///  * Args: Generates weights and biaseds as arguments (RW).
  enum class KernelType { Const, Args };

  /// Type of kernel to be generated
  KernelType kernelType;

  /// List of supported quantization ops types that can be generated
  enum class QuantizationType { None, Quant, Dequant, QuantDequant };

  /// Type of quantization ops to be generated
  QuantizationType quantType;

  /// VNNI packing factor (0, 2, 4)
  int vnniFactor;

  /// Apply VNNI packing
  bool vnniPacked;

  // ============================ Helpers

  /// Return current random seed, update next
  int getRand();

  /// Type of packing (NxC, KxC, NxK). Extended to include scales.
  enum PackingType {
    PACK_INPUT,
    PACK_WEIGHT,
    PACK_OUTPUT,
    PACK_INTERMEDIATE,
    INPUT_SCALE,
    WEIGHT_SCALE
  };

  /// Return shaped type (packed if requested)
  TensorType getShape(ArrayRef<int64_t>, PackingType);

  /// Return a zero-init tensor for matmul outputs
  Value getZeroInitTensor(TensorType);

  /// Computes required flops for matmul
  void computeMatmulFlops(ShapedType inputShape, ShapedType outputShape);

  /// Computes required flops for bias/relu
  void computeBiasOrReluFlops(ShapedType outputShape);

  /// Affine expressions for maps
  SmallVector<AffineExpr, 6> affineExprs;

  enum MapType {
    MAP_PARALLEL,
    MAP_REDUCTION,
    MAP_BROADCAST,
    MAP_MATMUL_INPUT,
    MAP_MATMUL_WEIGHT,
    MAP_MATMUL_OUTPUT,
    MAP_MATMUL // Alias for iterator type
  };

  /// Types are created first, values are created from the types if inside the
  /// function, or populated later from function arguments if external.
  struct Arg {
    Value value;
    TensorType type;
  };

  /// There could be multiple layers, each with its own weights and biases
  /// Input of one layer is the output of the previous
  /// Input of the model is the input of the first layer
  /// Output of the model is the output of the last layer
  struct LayerArgs {
    unsigned index;
    Arg input;
    Arg inputScale;
    Arg weight;
    Arg weightScale;
    Arg bias;
    Arg intermediate; // For quantdequant validation
    Arg output;
  };

  /// Return affine map (packed if requested)
  /// If order is not empty, re-order the dims in that order
  /// If dims is passed, force number of dims, otherwise, take from tensor
  /// If reduction is true, emit zeroExpr for the tail reduction
  AffineMap getMap(Value, MapType);

  /// Return the iterator types for a particular map type
  /// Add iterators if the types are packed
  SmallVector<utils::IteratorType> getIterators(MapType);

  // ============================ Core Logic
  // To avoid allocating new tensors, we bind the output of the matmul to the
  // input of the bias add, make it in-place and bind that to the input of
  // the ReLU, also making it in-place, and returning the first alloc.

  /// Creates a matmul in the current function
  /// Args: Contains A, B, C
  /// Boolean indicates if mixed type (quantization) is used.
  /// Returns the chain value to be used in the next op
  Value lowerMatmul(LayerArgs &args, bool);

  /// Creates linalg generic matmul
  Value lowerGenericMatmul(Value, Value, Value);

  /// Creates linalg named matmul
  Value lowerNamedMatmul(Value, Value, Value);

  /// Creates linalg contract
  Value lowerContract(Value, Value, Value);

  /// Computes scaling factor for the given input. Returns the scaling factor of
  /// same shape as input.
  SmallVector<Value> computeScalingFactor(Value input);

  /// Creates a matmul quantization kernel
  Value quantizeGemm(LayerArgs &args, Value chain, Value scale);

  /// Creates a matmul dequantization kernel
  Value dequantizeGemm(LayerArgs &args, Value chain);

  Value testQuantDequant(LayerArgs &args, Value input);

  /// Creates a bias add in the current function
  /// Args: Input, Output (same for in-place)
  /// Returns the chain value to be used in the next op
  Value lowerBiasAdd(Value, Value, Value);

  /// Creates linalg named bias add
  Value lowerNamedBiasAdd(Value, Value, Value);

  /// Creates a relu in the current function
  /// Args: Input, Output (same for in-place)
  /// Returns the chain value to be used in the next op
  Value lowerRelu(Value, Value);

  /// Creates linalg named relu
  Value lowerNamedRelu(Value, Value);

  /// Creates a softmax in the current function
  /// Args: Input, Output (same for in-place)
  /// Returns the chain value to be used in the next op
  Value lowerSoftmax(Value, Value);

  /// Creates linalg named softmax
  Value lowerNamedSoftmax(Value, Value);

  // ============================ Main API

  /// Creates metadata string containing run command, flops info etc.
  std::string createMetadata();

  /// Some arguments are optional, so we use this struct to simplify the
  /// argument handling in createLayer.
  typedef SmallVector<LayerArgs, 3> KernelArgs;

  /// Creates the kernel types from layer definitions and options. Boolean
  /// indicates if mixed type (quantization) is used.
  void getKernelTypes(KernelArgs &);

  /// Creates a layer function, to be called by the kernel. Boolean indicates
  /// if mixed type (quantization) is used.
  Value createLayer(LayerArgs &, bool hasMixedType = false);

  /// Creates a kernel (N * {GEMM + AddBias + ReLU} + Softmax)
  /// AddBias, ReLU and Softmax are optional. Boolean indicates if mixed type
  /// (quantization) is used.
  void createKernel(bool hasMixedType = false);

public:
  /// Creates a specific module. Different configurations need different modules
  /// so should create new objects to not have to share / cleanup existing MLIR
  /// modules.
  MLIRGenerator(StringRef, StringRef, unsigned, StringRef, StringRef, StringRef,
                StringRef, StringRef, int, bool, bool, bool, bool, bool, int);

  ~MLIRGenerator() { module->destroy(); }

  /// Generates the whole IR and write to file
  /// Return 0 on success, 1 on failure. 'hasMixedType' indicates simple mixed
  /// type without quant.
  int generate(StringRef filename, bool hasMixedType = false);
};

} // namespace mlir
