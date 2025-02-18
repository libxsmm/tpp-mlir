//===- StructuredOpMatcher.h -------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_IR_STRUCTUREDOPMATCHER_H
#define TPP_IR_STRUCTUREDOPMATCHER_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <utility>

namespace mlir {
class Operation;
namespace structured_match {

struct KindAdd {
  static bool classof(const Operation *op) {
    return isa<arith::AddFOp>(op) || isa<arith::AddIOp>(op);
  }
};

struct KindMul {
  static bool classof(const Operation *op) {
    return isa<arith::MulFOp>(op) || isa<arith::MulIOp>(op);
  }
};

// Base class for the matcher predicates selection tag.
struct MatchSelector {
  MatchSelector() = delete;
  size_t getLowerBound() const { return lowerBound; };
  size_t getUpperBound() const { return upperBound; };

protected:
  explicit MatchSelector(size_t lowerBound, size_t upperBound)
      : lowerBound(lowerBound), upperBound(upperBound) {
    assert(upperBound > lowerBound);
  }

private:
  const size_t lowerBound;
  const size_t upperBound;
};

// Selector which specifies that predicate should apply on all values.
struct MatchAll : public MatchSelector {
  MatchAll() : MatchSelector(0, std::numeric_limits<size_t>::max()) {}
};

// Selector which specifies that predicate should apply only on one value at
// the position `idx`.
struct MatchOne : public MatchSelector {
  MatchOne() = delete;
  MatchOne(size_t idx) : MatchSelector(idx, idx + 1) {}
};

// Selector which specifies that predicate should apply only on range of values
// at positions from `lowerBound` up to - but not including - `upperBound`.
struct MatchRange : public MatchSelector {
  MatchRange() = delete;
  MatchRange(size_t lowerBound, size_t upperBound)
      : MatchSelector(lowerBound, upperBound) {}
};

// Callable object to check if the number of loops in `op` satisfies `fun`.
struct NumOfLoops {
  NumOfLoops() = delete;
  explicit NumOfLoops(std::function<bool(size_t)> fun) : fun(std::move(fun)){};

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op)) {
      auto numberOfLoops = linalgOp.getNumLoops();
      return fun(numberOfLoops);
    }
    return false;
  }
  std::function<bool(size_t)> fun;
};

// Callable object to check if the `operand` of `op` has a map that satisfies
// `fun`.
struct HasMap {
  HasMap() = delete;
  explicit HasMap(std::function<bool(AffineMap)> fun) : fun(std::move(fun)){};
  explicit HasMap(std::function<bool(AffineMap)> fun, AffineMap *ptrMap)
      : fun(std::move(fun)), ptrMap(ptrMap){};

  bool operator()(OpOperand *operand, Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op)) {
      auto map = linalgOp.getMatchingIndexingMap(operand);
      assert(fun && "must be a callable target");
      if (!fun(map))
        return false;
      if (ptrMap)
        *ptrMap = std::move(map);
      return true;
    }
    return false;
  }
  std::function<bool(AffineMap map)> fun;
  AffineMap *ptrMap = nullptr;
};

// Callble object to verify if `map` is a broadcastable projected permutation
// map. We require the dimensions to be in sorted order this avoid filtering
// projected permutation without broadcasting semantics, for example
// affine_map<(d0, d1) -> (d1, d0)> is rejected.
struct BroadcastableProjectedPermutation {
  BroadcastableProjectedPermutation() = default;

  bool operator()(AffineMap map) const {
    if (map.getNumSymbols() > 0 || map.getNumResults() > map.getNumInputs())
      return false;

    SmallVector<bool> seen(map.getNumInputs(), false);
    SmallVector<int64_t> pos;
    for (auto expr : map.getResults()) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        if (seen[dim.getPosition()])
          return false;
        seen[dim.getPosition()] = true;
        pos.push_back(dim.getPosition());
      } else if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
        if (constExpr.getValue() != 0)
          return false;
      } else {
        return false;
      }
    }
    return llvm::is_sorted(pos);
  }
};

// Callable object to verify if `map` is a projected permutation.
struct ProjectedPermutation {
  ProjectedPermutation() = default;

  bool operator()(AffineMap map) const { return map.isProjectedPermutation(); }
};

// Callable object to verify if `map` is an identity map.
struct Identity {
  Identity() = default;

  bool operator()(AffineMap map) const { return map.isIdentity(); }
};

// Callable object to capture any map.
struct Any {
  Any() = default;

  bool operator()(AffineMap map) const { return true; }
};

// Callable object to verify if `operand` has static shape.
struct HasStaticShape {
  HasStaticShape() = default;
  HasStaticShape(SmallVectorImpl<int64_t> *shape) : shape(shape){};

  bool operator()(OpOperand *operand, Operation *op) const {
    auto operandType = operand->get().getType();
    if (auto shapedType = dyn_cast_or_null<ShapedType>(operandType)) {
      if (!shapedType.hasStaticShape())
        return false;
      if (shape) {
        for (int64_t shapeOnDim : shapedType.getShape())
          shape->push_back(shapeOnDim);
      }
    }
    return true;
  }
  SmallVectorImpl<int64_t> *shape = nullptr;
};

// Callable object to verify if `operand` has static strides.
// If `operand` is a tensor type or a scalar, return true.
struct HasStaticStrides {
  HasStaticStrides() = default;
  HasStaticStrides(SmallVector<int64_t> *strides) : strides(strides){};

  bool operator()(OpOperand *operand, Operation *op) const {
    auto operandType = operand->get().getType();
    SmallVector<int64_t> strides;
    if (auto memRefType = dyn_cast_or_null<MemRefType>(operandType)) {
      int64_t offset;
      if (failed(memRefType.getStridesAndOffset(strides, offset)))
        return false;
      if (llvm::any_of(strides, [](int64_t stride) {
            return stride == ShapedType::kDynamic;
          })) {
        return false;
      }
      if (this->strides)
        this->strides->append(strides.begin(), strides.end());
    }
    return true;
  }
  SmallVectorImpl<int64_t> *strides = nullptr;
};

// Callable object to verify `operand` to have a rank in `ranks`.
struct HasRank {
  HasRank() = delete;
  explicit HasRank(std::initializer_list<int64_t> ranks) : ranks(ranks){};

  bool operator()(OpOperand *operand, Operation *op) const {
    auto operandType = operand->get().getType();
    if (!isa<ShapedType>(operandType))
      return llvm::is_contained(ranks, HasRank::SCALAR);
    int64_t rank = cast<ShapedType>(operandType).getRank();
    return llvm::any_of(
        ranks, [=](int64_t expectedRank) { return expectedRank == rank; });
  }

  // There are multiple way to represent a scalar: f32, tensor<f32>.
  // SCALAR means f32.
  static constexpr int64_t SCALAR = -1;
  std::vector<int64_t> ranks;
};

// Callable object to verify `operand` to have an element type `T`.
template <typename T> struct HasElementType {
  bool operator()(OpOperand *operand, Operation *op) const {
    auto operandType = getElementTypeOrSelf(operand->get().getType());
    return isa<T>(operandType);
  }
};

// Callable object to check if the input is equal to specified `value`.
template <typename T> struct EqualsTo {
  EqualsTo() = delete;
  explicit EqualsTo(T value) : value(value){};

  const T value;

  bool operator()(T value) const { return value == this->value; }
};
template <typename T> EqualsTo(T) -> EqualsTo<T>;

// Callable object to check if the input is less than or equal to specified
// `value`.
struct LessThanOrEqualTo {
  LessThanOrEqualTo() = delete;
  explicit LessThanOrEqualTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) const { return value <= this->value; }
};

// Callable object to check if the input is greater than or equal to specified
// `value`.
struct GreaterThanOrEqualTo {
  GreaterThanOrEqualTo() = delete;
  explicit GreaterThanOrEqualTo(size_t value) : value(value){};
  const size_t value;

  bool operator()(size_t value) const { return value >= this->value; }
};

// Callable object to check if `op` has tensor semantics.
struct HasTensorSemantics {
  HasTensorSemantics() = default;

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return linalgOp.hasPureTensorSemantics();
    return false;
  }
};

// Callable object to check if `op` buffer semantics.
struct HasBufferSemantics {
  HasBufferSemantics() = default;

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return linalgOp.hasPureBufferSemantics();
    return false;
  }
};

// Callable object to validate number of init operands for `op`.
struct NumDpsInits {
  NumDpsInits() = delete;
  explicit NumDpsInits(std::function<bool(size_t)> fun) : fun(std::move(fun)){};

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return fun(linalgOp.getNumDpsInits());
    return false;
  }

  std::function<bool(size_t)> fun;
};

// Callable object to check the number of affine map for `op`.
struct NumAffineMaps {
  NumAffineMaps() = delete;
  explicit NumAffineMaps(std::function<bool(size_t)> fun) : fun(std::move(fun)){};

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return fun(linalgOp.getIndexingMapsArray().size());
    return false;
  }

  std::function<bool(size_t)> fun;
};

// Callable object to validate number of input operands for `op`.
struct NumDpsInputs {
  NumDpsInputs() = delete;
  explicit NumDpsInputs(std::function<bool(size_t)> fun) : fun(std::move(fun)){};

  bool operator()(Operation *op) {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return fun(linalgOp.getNumDpsInputs());
    return false;
  }

  std::function<bool(size_t)> fun;
};

// Callable object to validate number of regions for `op`.
struct NumRegions {
  NumRegions() = delete;
  explicit NumRegions(std::function<bool(size_t)> fun) : fun(std::move(fun)){};

  bool operator()(Operation *op) const {
    if (auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op))
      return fun(linalgOp->getNumRegions());
    return false;
  }

  std::function<bool(size_t)> fun;
};

// Logical OR between two predicates.
struct _OR {
  _OR() = delete;
  _OR(std::function<bool(size_t)> lhs, std::function<bool(size_t)> rhs)
      : lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  bool operator()(size_t num) { return (lhs(num) || rhs(num)); }

  std::function<bool(size_t)> lhs;
  std::function<bool(size_t)> rhs;
};

// Callable object to check if `op` adheres to a given property passed
// as an std::function object.
struct VerifyOpProperty {
  VerifyOpProperty() = delete;
  explicit VerifyOpProperty(std::function<LogicalResult(Operation *op)> fun)
      : fun(std::move(fun)){};

  bool operator()(Operation *op) {
    if (succeeded(fun(op)))
      return true;
    return false;
  }

  std::function<LogicalResult(Operation *op)> fun;
};

// Work-around for template specialization.
struct WithSingleOpImpl {
  WithSingleOpImpl() = default;

  bool withSingleOpImpl(StringRef, Region *, Operation *,
                        SmallVectorImpl<Value> *);
};

// Callable object to check the `op` region for a single scalar operation OpTy.
template <typename OpTy> struct WithSingleOp {
  WithSingleOp() : WithSingleOp(nullptr){};
  WithSingleOp(SmallVectorImpl<Value> *captures) : captures(captures){};

  bool operator()(Region *region, Operation *op) {
    return WithSingleOpImpl().withSingleOpImpl(OpTy::getOperationName(), region,
                                               op, captures);
  }

private:
  SmallVectorImpl<Value> *captures;
};

// Implemenation to allow definition in cpp file
using TypeCheckFunc = std::function<bool(Operation *)>;
bool withOpChainImpl(Region *region, Operation *op, SmallVectorImpl<Value> *,
                     SmallVectorImpl<TypeCheckFunc> &);

// Callable object to check the region for a chain of operations.
template <typename... OpTy> struct WithOpChain {
  WithOpChain() : WithOpChain(nullptr){};
  WithOpChain(SmallVectorImpl<Value> *captures) : captures(captures) {
    (typeChecks.push_back([](Operation *op) { return isa<OpTy>(op); }), ...);
  };

  bool operator()(Region *region, Operation *op) {
    return withOpChainImpl(region, op, captures, typeChecks);
  }

private:
  SmallVectorImpl<Value> *captures;
  SmallVector<TypeCheckFunc> typeChecks;
};

class StructuredOpMatcher {
  using PredicateFn = std::function<bool(linalg::LinalgOp)>;

public:
  StructuredOpMatcher() = default;

  StructuredOpMatcher(PredicateFn &&firstPredicate) {
    predicates.push_back(std::move(firstPredicate));
  }

  template <typename OpTy> static StructuredOpMatcher make() {
    return StructuredOpMatcher(
        [](linalg::LinalgOp op) { return isa<OpTy>(op.getOperation()); });
  }

  // Match given `op` using stored predicates.
  bool match(Operation *op);

  // Predicates on operation.
  StructuredOpMatcher &operation(std::function<bool(Operation *)>);

  // Predicate on OpOperands.
  StructuredOpMatcher &input(MatchSelector range,
                             std::function<bool(OpOperand *, Operation *)>);

  // Predicates on OpOperands.
  StructuredOpMatcher &output(MatchSelector range,
                              std::function<bool(OpOperand *, Operation *)>);

  // Predicates on Iterators.
  StructuredOpMatcher &dim(MatchSelector range,
                           SmallVector<mlir::utils::IteratorType> kinds);
  StructuredOpMatcher &dim(MatchSelector range, mlir::utils::IteratorType kind);

  // Predicates on region.
  StructuredOpMatcher &region(MatchSelector range,
                              std::function<bool(Region *, Operation *op)>);

private:
  llvm::SmallVector<PredicateFn> predicates;
};

} // namespace structured_match
} // namespace mlir

#endif // TPP_IR_STRUCTUREDOPMATCHER_H
