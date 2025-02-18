//===- ConvertToBlockLayoutAndBack.cpp ---------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TPP/Passes.h"
#include "TPP/Transforms/Transforms.h"
#include "TPP/Transforms/Utils/TransformUtils.h"
#include "TPP/Transforms/Utils/VNNIUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir {
namespace tpp {
#define GEN_PASS_DEF_PACKVNNI
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_PACKMATMUL
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_PACKCONV2DNCHWFCHW
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_PACKCONV2DNHWCHWCF
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_PROPAGATEPACKUNPACK
#include "TPP/Passes.h.inc"
#define GEN_PASS_DEF_SIMPLIFYANDCANONICALIZEPACK
#include "TPP/Passes.h.inc"
} // namespace tpp
} // namespace mlir

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// Helper function to create the pack operation.
static Value toPackLayoutImpl(OpBuilder &builder, Location loc, Value input,
                              ArrayRef<OpFoldResult> tiles,
                              ArrayRef<int64_t> innerDimsPos,
                              ArrayRef<int64_t> outerDimsPerm) {
  SmallVector<Value> dynamicTiles;
  SmallVector<int64_t> staticTiles;
  dispatchIndexOpFoldResults(tiles, dynamicTiles, staticTiles);
  RankedTensorType result =
      linalg::PackOp::inferPackedType(cast<RankedTensorType>(input.getType()),
                                      staticTiles, innerDimsPos, outerDimsPerm);
  auto inputType = cast<RankedTensorType>(input.getType());
  ArrayRef<int64_t> shape = result.getShape();
  Value output =
      builder.create<tensor::EmptyOp>(loc, shape, inputType.getElementType());
  return builder.create<linalg::PackOp>(loc, input, output, innerDimsPos, tiles,
                                        /*paddingValue=*/std::nullopt,
                                        outerDimsPerm);
}

// Helper function to create the unpack operation.
static Value toUnPackLayoutImpl(OpBuilder &builder, Location loc, Value input,
                                Value output, ArrayRef<OpFoldResult> tiles,
                                ArrayRef<int64_t> innerDimPos,
                                ArrayRef<int64_t> outerDimsPerm) {
  if (auto fillOp = output.getDefiningOp<linalg::FillOp>())
    output = fillOp.getOutputs()[0];
  return builder.create<linalg::UnPackOp>(loc, input, output, innerDimPos,
                                          tiles, outerDimsPerm);
}

static Value handleLayout_VNNI(OpBuilder &builder, Location loc, Value input,
                               ArrayRef<OpFoldResult> tiles, int64_t kDimPos) {
  assert(tiles.size() == 1 && "expect 1 block for VNNI");
  return toPackLayoutImpl(builder, loc, input, tiles,
                          SmallVector<int64_t>{kDimPos},
                          /*outerDimsPerm=*/{});
}

static Value handleBRGemmLayout_VNNI(OpBuilder &builder, Location loc,
                                     Value input, ArrayRef<OpFoldResult> tiles,
                                     int64_t kDimPos) {
  assert(tiles.size() == 1 && "expect 1 block for VNNI");
  return toPackLayoutImpl(builder, loc, input, tiles,
                          SmallVector<int64_t>{kDimPos},
                          /*outerDimsPerm=*/{});
}

// Helper function to pack from [outer][K][inner] to [outer][K/2][inner][2].
static Value toPackLayout_VNNI(OpBuilder &builder, Location loc, Value input,
                               ArrayRef<OpFoldResult> tiles, int64_t kDimPos) {
  return handleLayout_VNNI(builder, loc, input, tiles, kDimPos);
}

// Helper function to pack from [outer][K][inner] to [outer][K/2][inner][2].
static Value toPackBRGemmLayout_VNNI(OpBuilder &builder, Location loc,
                                     Value input, ArrayRef<OpFoldResult> tiles,
                                     int64_t kDimPos) {
  return handleBRGemmLayout_VNNI(builder, loc, input, tiles, kDimPos);
}

static Value handleLayoutNCHW_NCHWc(OpBuilder &builder, Location loc,
                                    Value input, Value output,
                                    ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 1 && "expect one tile size for NCHW_NCHWc");
  SmallVector<int64_t> innerDimPos = {1};
  if (!output)
    return toPackLayoutImpl(builder, loc, input, tiles, innerDimPos,
                            /*outerDimsPerm=*/{});
  return toUnPackLayoutImpl(builder, loc, input, output, tiles, innerDimPos,
                            /*outerDimsPerm=*/{});
}

// Helper function to pack from NCHW to NCHWc.
static Value toPackLayoutNCHW_NCHWc(OpBuilder &builder, Location loc,
                                    Value input, ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNCHW_NCHWc(builder, loc, input, nullptr, tiles);
}

// Helper function to unpack from NCHWc to NCHW.
static Value fromPackLayoutNCHWc_NCHW(OpBuilder &builder, Location loc,
                                      Value input, Value output,
                                      ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNCHW_NCHWc(builder, loc, input, output, tiles);
}

static Value handleLayoutNPQK_NKPQk(OpBuilder &builder, Location loc,
                                    Value input, Value output,
                                    ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 1 && "expect one tile size for NPQK_NKPQk");
  SmallVector<int64_t> innerDimsPos = {3};
  SmallVector<int64_t> outerDimsPerm = {0, 3, 1, 2};
  if (!output)
    return toPackLayoutImpl(builder, loc, input, tiles, innerDimsPos,
                            outerDimsPerm);
  return toUnPackLayoutImpl(builder, loc, input, output, tiles, innerDimsPos,
                            outerDimsPerm);
}

// Helper function to pack NPQK to NKPQk.
static Value toPackLayoutNPQK_NKPQk(OpBuilder &builder, Location loc,
                                    Value input, ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNPQK_NKPQk(builder, loc, input, nullptr, tiles);
}

// Helper function to unpack NKPQk to NPQK.
static Value fromPackLayoutNKPQk_NPQK(OpBuilder &builder, Location loc,
                                      Value input, Value output,
                                      ArrayRef<OpFoldResult> tiles) {
  return handleLayoutNPQK_NKPQk(builder, loc, input, output, tiles);
}

// Helper function to pack from RSCK to KCRSck.
static Value toPackLayoutRSCK_KCRSck(OpBuilder &builder, Location loc,
                                     Value input,
                                     ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 2 && "expect two tiles for RSCK_KCRSck");
  SmallVector<int64_t> innerDimsPos = {2, 3};
  SmallVector<int64_t> outerDimsPerm = {3, 2, 0, 1};
  return toPackLayoutImpl(builder, loc, input, tiles, innerDimsPos,
                          outerDimsPerm);
}

// Helper function to pack from KCRS to KCRSck.
static Value toPackLayoutKCRS_KCRSck(OpBuilder &builder, Location loc,
                                     Value input,
                                     ArrayRef<OpFoldResult> tiles) {
  assert(tiles.size() == 2 && "expect two tiles size for KCRS_KCRSck");
  SmallVector<int64_t> innerDimPos = {1, 0};
  return toPackLayoutImpl(builder, loc, input, tiles, innerDimPos,
                          /*outerDimsPerm=*/{});
}

template <typename OpTy>
static FailureOr<linalg::GenericOp>
packConvolutions(RewriterBase &rewriter, OpTy convOp,
                 ArrayRef<OpFoldResult> tiles) {
  static_assert(llvm::is_one_of<OpTy, linalg::Conv2DNhwcHwcfOp,
                                linalg::Conv2DNchwFchwOp>::value,
                "applies to only pack or unpack operations");

  if (tiles.size() != 2)
    return rewriter.notifyMatchFailure(convOp, "require 2 tile factors");
  if (convOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(convOp, "require static shape");
  if (convOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(convOp, "require tensor semantics");

  bool isConv2DNhwcHwcfOp =
      static_cast<bool>(std::is_same<OpTy, linalg::Conv2DNhwcHwcfOp>::value);

  Location loc = convOp.getLoc();
  MLIRContext *ctx = convOp.getContext();

  SmallVector<Value> inputOperands = convOp.getDpsInputs();
  SmallVector<Value> outputOperands = convOp.getDpsInits();

  // pack the image and the filter.
  Value image = inputOperands[0];
  Value packedImage =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutNPQK_NKPQk(rewriter, loc, image, tiles[0])
          : toPackLayoutNCHW_NCHWc(rewriter, loc, image, tiles[0]);
  Value filter = inputOperands[1];
  Value packedFilter =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutRSCK_KCRSck(rewriter, loc, filter, tiles)
          : toPackLayoutKCRS_KCRSck(rewriter, loc, filter, tiles);
  SmallVector<Value, 2> packedInputs = {packedImage, packedFilter};

  // pack the output.
  Value output = outputOperands[0];
  Value packedOutput =
      (isConv2DNhwcHwcfOp)
          ? toPackLayoutNPQK_NKPQk(rewriter, loc, output, tiles[0])
          : toPackLayoutNCHW_NCHWc(rewriter, loc, output, tiles[0]);

  SmallVector<int64_t, 2> strides = {1, 1};
  if (DenseIntElementsAttr stridesAttr = convOp.getStrides()) {
    auto strideValues = stridesAttr.getValues<int64_t>();
    assert(strideValues.size() == 2 && "expect two stride values");
    strides[0] = strideValues[0];
    strides[1] = strideValues[1];
  }

  // Swap convolution with generic.
  //         N   K   P   Q   k   C   R   S   c
  AffineExpr p1, p2, p3, p4, p5, r1, r2, r3, r4;
  bindDims(ctx, p1, p2, p3, p4, p5, r1, r2, r3, r4);
  AffineMap mapOut =
      AffineMap::get(/*dims=*/9, /*symbols=*/0, {p1, p2, p3, p4, p5}, ctx);
  AffineMap mapImg = AffineMap::get(
      /*dims=*/9, /*symbols=*/0,
      {p1, r1, p3 * strides[0] + r2, p4 * strides[1] + r3, r4}, ctx);
  AffineMap mapFil =
      AffineMap::get(/*dims=*/9, /*symbols=*/0, {p2, r1, r2, r3, r4, p5}, ctx);
  linalg::GenericOp replacementOp = rewriter.create<linalg::GenericOp>(
      loc, packedOutput.getType(), packedInputs, ValueRange{packedOutput},
      ArrayRef<AffineMap>{mapImg, mapFil, mapOut},
      ArrayRef<utils::IteratorType>{
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::parallel,
          utils::IteratorType::parallel, utils::IteratorType::reduction,
          utils::IteratorType::reduction, utils::IteratorType::reduction,
          utils::IteratorType::reduction},
      /*doc=*/"", /*libraryCall=*/"");
  rewriter.inlineRegionBefore(convOp->getRegion(0), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());
  if (auto metadata = convOp->getAttr("metadata"))
    replacementOp->setAttr("metadata", metadata);

  // convert back from pack layout.
  Value outPackedTensor = replacementOp.getResult(0);
  Value outUnPackedTensor = outputOperands[0];
  Value outReplacement =
      (isConv2DNhwcHwcfOp)
          ? fromPackLayoutNKPQk_NPQK(rewriter, loc, outPackedTensor,
                                     outUnPackedTensor, tiles[0])
          : fromPackLayoutNCHWc_NCHW(rewriter, loc, outPackedTensor,
                                     outUnPackedTensor, tiles[0]);
  rewriter.replaceOp(convOp, outReplacement);
  return replacementOp;
}

//===----------------------------------------------------------------------===//
// Conv2DNhwcHwcfOp
//===----------------------------------------------------------------------===//
// Original layout: [N][P][Q][K] += [N][H][W][C] * [R][S][C][K]
// New      layout: [N][K'][P][Q][k] += [N][C'][H][W][c] * [K'][C'][R][S][c][k]
FailureOr<linalg::GenericOp>
mlir::linalgx::packConv2DNhwcHwcfOp(RewriterBase &rewriter,
                                    linalg::Conv2DNhwcHwcfOp convOp,
                                    ArrayRef<OpFoldResult> tiles) {
  if (!linalgx::utils::validateFullTilesOnDims(
          cast<TilingInterface>(convOp.getOperation()), tiles,
          {/*Kidx=*/3, /*Cidx=*/6}))
    return rewriter.notifyMatchFailure(convOp, "expect full tiles only");
  return packConvolutions(rewriter, convOp, tiles);
}

//===----------------------------------------------------------------------===//
// Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//
// Original layout: [N][K][P][Q] += [N][C][H][W] * [K][C][R][S]
// New      layout: [N][K'][P][Q][k] += [N][C'][H][W][c] + [K'][C'][R][S][c][k]
FailureOr<linalg::GenericOp>
mlir::linalgx::packConv2DNchwFchwOp(RewriterBase &rewriter,
                                    linalg::Conv2DNchwFchwOp convOp,
                                    ArrayRef<OpFoldResult> tiles) {
  if (!linalgx::utils::validateFullTilesOnDims(
          cast<TilingInterface>(convOp.getOperation()), tiles,
          {/*Kidx=*/1, /*Cidx=*/4}))
    return rewriter.notifyMatchFailure(convOp, "expect full tiles only");
  return packConvolutions(rewriter, convOp, tiles);
}

//===----------------------------------------------------------------------===//
// MatmulOp (VNNI packing)
//===----------------------------------------------------------------------===//
// Original layout:
//      [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb][jb]
// New      layout:
//      [IB][JB][ib][jb] += [IB][KB][ib][kb] * [JB][KB][kb/VNNI][jb][VNNI]
FailureOr<linalg::GenericOp>
mlir::linalgx::packVNNIMatmulOp(RewriterBase &rewriter,
                                linalg::GenericOp matmulOp) {
  if (matmulOp.getInputs().size() > 0) {
    auto elementType = getElementTypeOrSelf(matmulOp.getInputs()[0].getType());
    if (!elementType.isBF16())
      return rewriter.notifyMatchFailure(matmulOp, "require bf16 type");
  }

  if (matmulOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(matmulOp, "require static shape");

  if (matmulOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(matmulOp, "require tensor semantics");

  auto dims = linalgx::utils::isContraction(matmulOp);
  if (failed(dims))
    return rewriter.notifyMatchFailure(matmulOp, "require matmul semantics");

  OpOperand &operandB = matmulOp->getOpOperand(1);
  auto blockingFactor =
      vnni::utils::getVnniBlockingFactor(operandB.get().getType(), matmulOp);
  if (!blockingFactor) {
    return rewriter.notifyMatchFailure(matmulOp,
                                       "unsupported blocking factor for type");
  }

  if (vnni::utils::isInVnniLayout(matmulOp)) {
    return rewriter.notifyMatchFailure(matmulOp, "already packed to VNNI");
  }

  Location loc = matmulOp.getLoc();
  SmallVector<OpFoldResult> tilesOnSmallK = {
      rewriter.getI64IntegerAttr(blockingFactor)};
  SmallVector<std::pair<Value, unsigned>> kOperands;
  matmulOp.mapIterationSpaceDimToAllOperandDims(dims->k.back(), kOperands);
  if (kOperands.size() != 2)
    return rewriter.notifyMatchFailure(matmulOp,
                                       "Invalid reduction dim operands");
  // Reshape input A.
  Value packedMatrixA =
      toPackLayout_VNNI(rewriter, loc, matmulOp.getInputs()[0], tilesOnSmallK,
                        kOperands[0].second);
  // Reshape input B.
  Value packedMatrixB = toPackLayout_VNNI(rewriter, loc, operandB.get(),
                                          tilesOnSmallK, kOperands[1].second);

  MLIRContext *ctx = matmulOp.getContext();
  AffineExpr p1, p2, r1, p3, p4, r2, r3;
  SmallVector<Value> packedInputs = {packedMatrixA, packedMatrixB};
  AffineMap mapA, mapB, mapC;
  Value matrixC = matmulOp.getOutputs()[0];

  //            IB  JB  KB  ib  jb  kb  VNNI
  bindDims(ctx, p1, p2, r1, p3, p4, r2, r3);
  mapA = AffineMap::get(/*dims=*/7, /*symbols=*/0, {p1, r1, p3, r2, r3}, ctx);
  mapB = AffineMap::get(/*dims=*/7, /*symbols=*/0, {p2, r1, r2, p4, r3}, ctx);
  mapC = AffineMap::get(/*dims=*/7, /*symbols=*/0, {p1, p2, p3, p4}, ctx);
  auto replacementOp = rewriter.create<linalg::GenericOp>(
      loc, matrixC.getType(), packedInputs, ValueRange{matrixC},
      ArrayRef<AffineMap>{mapA, mapB, mapC},
      ArrayRef<mlir::utils::IteratorType>{mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::reduction,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::parallel,
                                          mlir::utils::IteratorType::reduction,
                                          mlir::utils::IteratorType::reduction},
      /*doc=*/"", /*libraryCall=*/"");

  rewriter.inlineRegionBefore(matmulOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  rewriter.replaceOp(matmulOp, replacementOp.getResult(0));
  return replacementOp;
}

//===----------------------------------------------------------------------===//
// BrgemmOp (VNNI layout)
//===----------------------------------------------------------------------===//
// Original layout: [I][J] += [R][I][K] * [R][K][J]
// New      layout: [I][J] += [R][I][K] * [R][K/VNNI][J][VNNI]
FailureOr<linalg::GenericOp>
mlir::linalgx::packVNNIBRGemmOp(RewriterBase &rewriter,
                                linalg::BatchReduceMatmulOp brgemmOp) {
  auto elementType = getElementTypeOrSelf(brgemmOp.getInputs()[0].getType());
  if (!elementType.isBF16())
    return rewriter.notifyMatchFailure(brgemmOp, "require bf16 type");

  if (brgemmOp.hasDynamicShape())
    return rewriter.notifyMatchFailure(brgemmOp, "require static shape");

  if (brgemmOp.hasPureBufferSemantics())
    return rewriter.notifyMatchFailure(brgemmOp, "require tensor semantics");

  Value operandB = brgemmOp.getInputs()[1];
  // Blocking factor on the `k` dimension.
  auto blockingFactor =
      vnni::utils::getVnniBlockingFactor(operandB.getType(), brgemmOp);
  if (!blockingFactor) {
    return rewriter.notifyMatchFailure(brgemmOp,
                                       "unsupported blocking factor for type");
  }
  SmallVector<OpFoldResult> tilesOnK = {
      rewriter.getI64IntegerAttr(blockingFactor)};

  Location loc = brgemmOp.getLoc();
  // Reshape input A.
  Value packedMatrixA = toPackBRGemmLayout_VNNI(
      rewriter, loc, brgemmOp.getInputs()[0], tilesOnK, 2);
  // Reshape input B.
  Value packedMatrixB =
      toPackBRGemmLayout_VNNI(rewriter, loc, operandB, tilesOnK, 1);

  MLIRContext *ctx = brgemmOp.getContext();
  AffineExpr r1, p1, p2, r3, r4;
  AffineMap mapA, mapB, mapC;
  bindDims(ctx, r1, p1, p2, r3, r4);
  mapA = AffineMap::get(/*dims=*/5, /*symbols=*/0, {r1, p1, r3, r4}, ctx);
  mapB = AffineMap::get(/*dims=*/5, /*symbols=*/0, {r1, r3, p2, r4}, ctx);
  mapC = AffineMap::get(/*dims=*/5, /*symbols=*/0, {p1, p2}, ctx);

  auto replacementOp = rewriter.create<linalg::GenericOp>(
      loc, brgemmOp.getOutputs()[0].getType(),
      ValueRange{packedMatrixA, packedMatrixB},
      ValueRange{brgemmOp.getOutputs()[0]},
      ArrayRef<AffineMap>{mapA, mapB, mapC},
      ArrayRef<mlir::utils::IteratorType>{
          mlir::utils::IteratorType::reduction,  // b
          mlir::utils::IteratorType::parallel,   // i
          mlir::utils::IteratorType::parallel,   // j
          mlir::utils::IteratorType::reduction,  // k
          mlir::utils::IteratorType::reduction}, // k/VNNI
      /*doc=*/"", /*libraryCall=*/"");

  rewriter.inlineRegionBefore(brgemmOp.getRegion(), replacementOp.getRegion(),
                              replacementOp.getRegion().begin());

  rewriter.replaceOp(brgemmOp, replacementOp.getResult(0));
  return replacementOp;
}

namespace {

static SmallVector<int64_t>
getDefaultBlockingFactors(linalg::LinalgOp linalgOp) {
  assert(linalgOp && "expect a valid linalgOp");
  if (isa<linalg::Conv2DNchwFchwOp>(linalgOp) ||
      isa<linalg::Conv2DNhwcHwcfOp>(linalgOp)) {
    return {32, 32};
  }
  assert(isa<linalg::MatmulOp>(linalgOp) ||
         isa<linalg::BatchMatmulOp>(linalgOp) ||
         isa<linalg::MatmulTransposeAOp>(linalgOp) ||
         isa<linalg::MatmulTransposeBOp>(linalgOp));
  return {32, 32, 32};
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

// Entry point for packing a matmul operation.
// Pack MatmulOp as following:
// [NB][KB][nb][kb] += [NB][CB][nb][cb] * [KB][CB][cb][kb]
// CB = batch reduce dimension.
// Pack a BatchMatmulOp as following:
// [B][IB][JB][ib][jb] += [B][IB][KB][ib][kb] * [B][JB][KB][kb][jb]
// KB is the batch reduce dimension.
struct PackMatmul : public tpp::impl::PackMatmulBase<PackMatmul> {
  using PackMatmulBase::PackMatmulBase;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);

    // TODO: Add a cost function that decides whether to pack at all.
    auto packControlFn = [&](linalg::LinalgOp linalgOp)
        -> std::optional<linalg::BlockPackMatmulOptions> {
      linalg::BlockPackMatmulOptions options;

      // Pack only these named matmul variants.
      if (!(isa<linalg::MatmulOp>(linalgOp) ||
            isa<linalg::MatmulTransposeAOp>(linalgOp) ||
            isa<linalg::MatmulTransposeBOp>(linalgOp) ||
            isa<linalg::BatchMatmulOp>(linalgOp))) {
        return std::nullopt;
      }

      // Enforce user defined blocking factors or use defaults.
      if (!blockingFactors.empty()) {
        SmallVector<int64_t, 3> blockFactors{*blockingFactors};
        options.blockFactors = blockFactors;
      } else {
        options.blockFactors = getDefaultBlockingFactors(linalgOp);
      }

      // Allow padding to avoid double checks.
      options.allowPadding = true;

      // Adjust block factors to smaller dimensions.
      // If a dimension is smaller than the blocking factor, then
      // try to block by the dimension size.
      auto dims = linalg::inferContractionDims(linalgOp);
      if (failed(dims))
        return std::nullopt;

      OpBuilder builder(linalgOp);
      auto tileOp = cast<TilingInterface>(linalgOp.getOperation());
      SmallVector<Range> iterationDomain = tileOp.getIterationDomain(builder);

      if (std::optional<int64_t> dimM =
              linalgx::utils::getConstantRange(iterationDomain[dims->m.back()]))
        options.blockFactors[0] = std::min(*dimM, options.blockFactors[0]);
      if (std::optional<int64_t> dimN =
              linalgx::utils::getConstantRange(iterationDomain[dims->n.back()]))
        options.blockFactors[1] = std::min(*dimN, options.blockFactors[1]);
      if (std::optional<int64_t> dimK =
              linalgx::utils::getConstantRange(iterationDomain[dims->k.back()]))
        options.blockFactors[2] = std::min(*dimK, options.blockFactors[2]);

      // Apply more restrictive packing validation.
      SmallVector<OpFoldResult> tiles =
          getAsOpFoldResult(builder.getI64ArrayAttr(options.blockFactors));
      OpFoldResult tileOnI = tiles[0];
      OpFoldResult tileOnJ = tiles[1];
      OpFoldResult tileOnK = tiles[2];
      bool isBatchMatmulOp = isa<linalg::BatchMatmulOp>(linalgOp);
      size_t inc = isBatchMatmulOp ? 1 : 0;
      size_t posI = 0 + inc;
      size_t posJ = 1 + inc;
      size_t posK = 2 + inc;
      if (!linalgx::utils::validateFullTilesOnDims(
              cast<TilingInterface>(linalgOp.getOperation()),
              {tileOnI, tileOnJ, tileOnK}, {posI, posJ, posK},
              /*minTileFactor=*/1)) {
        return std::nullopt;
      }

      // Apply XSMM packing with block transpose only.
      options.lhsTransposeOuterBlocks = false;
      options.lhsTransposeInnerBlocks = false;
      options.rhsTransposeOuterBlocks = true;
      options.rhsTransposeInnerBlocks = false;

      return options;
    };
    linalg::populateBlockPackMatmulPatterns(patterns, packControlFn);
    linalg::populateLinalgDeGeneralizationPatterns(patterns);

    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct DoItOnConv2DNchwFchw
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  DoItOnConv2DNchwFchw(MLIRContext *context, ArrayRef<int64_t> blockingFactors,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::Conv2DNchwFchwOp>(context, benefit),
        blockingFactors(blockingFactors) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (blockingFactors.empty())
      blockingFactors = getDefaultBlockingFactors(linalgOp);
    FailureOr<linalg::GenericOp> genericOp =
        mlir::linalgx::packConv2DNchwFchwOp(
            rewriter, linalgOp,
            getAsOpFoldResult(rewriter.getI64ArrayAttr(blockingFactors)));
    if (failed(genericOp))
      return failure();
    return success();
  }

private:
  mutable SmallVector<int64_t> blockingFactors;
};

struct PackConv2DNchwFchw
    : public tpp::impl::PackConv2DNchwFchwBase<PackConv2DNchwFchw> {
  using PackConv2DNchwFchwBase::PackConv2DNchwFchwBase;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DoItOnConv2DNchwFchw>(ctx, blockingFactors);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct DoItOnConv2DNhwcHwcf
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  DoItOnConv2DNhwcHwcf(MLIRContext *context, ArrayRef<int64_t> blockingFactors,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::Conv2DNhwcHwcfOp>(context, benefit),
        blockingFactors(blockingFactors) {}

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (blockingFactors.empty())
      blockingFactors = getDefaultBlockingFactors(linalgOp);
    FailureOr<linalg::GenericOp> maybeGeneric =
        mlir::linalgx::packConv2DNhwcHwcfOp(
            rewriter, linalgOp,
            getAsOpFoldResult(rewriter.getI64ArrayAttr(blockingFactors)));
    if (failed(maybeGeneric))
      return failure();
    return success();
  }

private:
  mutable SmallVector<int64_t> blockingFactors;
};

struct PackConv2DNhwcHwcf
    : tpp::impl::PackConv2DNhwcHwcfBase<PackConv2DNhwcHwcf> {
  using PackConv2DNhwcHwcfBase::PackConv2DNhwcHwcfBase;

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DoItOnConv2DNhwcHwcf>(ctx, blockingFactors);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

// Pack MatmulOp to VNNI.
struct VNNIOnMatmul : public OpRewritePattern<linalg::GenericOp> {
  VNNIOnMatmul(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::GenericOp>(context, benefit) {}
  LogicalResult matchAndRewrite(linalg::GenericOp matmulOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> packedMatmul =
        mlir::linalgx::packVNNIMatmulOp(rewriter, matmulOp);
    if (failed(packedMatmul))
      return failure();
    return success();
  }
};

// Pack BRGemmOp to VNNI.
struct VNNIOnBRGemm : public OpRewritePattern<linalg::BatchReduceMatmulOp> {
  VNNIOnBRGemm(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::BatchReduceMatmulOp>(context, benefit) {}
  LogicalResult matchAndRewrite(linalg::BatchReduceMatmulOp brgemmOp,
                                PatternRewriter &rewriter) const override {
    FailureOr<linalg::GenericOp> packedBRGemm =
        mlir::linalgx::packVNNIBRGemmOp(rewriter, brgemmOp);
    if (failed(packedBRGemm))
      return failure();
    return success();
  }
};

// Entry point for packing a matmul/brgemm operation to vnni format.
struct PackVNNI : public tpp::impl::PackVNNIBase<PackVNNI> {

  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    linalg::populateLinalgDeGeneralizationPatterns(patterns);
    patterns.add<VNNIOnMatmul, VNNIOnBRGemm>(ctx);
    tensor::populateSimplifyPackAndUnpackPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

struct PropagatePackUnPack
    : public tpp::impl::PropagatePackUnPackBase<PropagatePackUnPack> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    linalg::populateDataLayoutPropagationPatterns(
        patterns, [](OpOperand *operand) { return true; });
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

// Fold a linalg.unpack into an scf.parallel_insert.
//
// The pattern looks like:
//
// %p = linalg.pack %a into %b
// %l = scf.forall ... iter_args(%0 = %p) {
// ...
// }
// %u = linalg.unpack %l into %c
//
// We will rewrite as:
//
// %l = scf.forall ... iter_args(%0 = %a) {
// ...
// }
struct FoldUnPackIntoInsertSlice : public OpRewritePattern<linalg::UnPackOp> {
  using OpRewritePattern<linalg::UnPackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    if (!unPackOp.getOuterDimsPerm().empty())
      return failure();
    SmallVector<int64_t> innerDimsPos =
        llvm::to_vector(unPackOp.getInnerDimsPos());
    SmallVector<int64_t> expectedDimsPos = llvm::to_vector(
        llvm::seq<int64_t>(0, unPackOp.getDestType().getRank()));
    if (innerDimsPos != expectedDimsPos)
      return failure();

    Operation *loop = unPackOp.getSource().getDefiningOp();
    if (!isa_and_nonnull<scf::ForallOp>(loop))
      return failure();
    auto forallOp = cast<scf::ForallOp>(loop);
    if (!forallOp->hasOneUse() || forallOp->getNumResults() != 1)
      return failure();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(forallOp);

    // Create a new scf.forall operation, updating its output.
    Value loopOperand =
        forallOp.getTiedOpOperand(forallOp->getResult(0))->get();
    linalg::PackOp packOp =
        dyn_cast_or_null<linalg::PackOp>(loopOperand.getDefiningOp());
    if (!packOp)
      return failure();
    Value newLoopOperand = packOp.getSource();
    SmallVector<Value> newOuts(forallOp.getOutputs());
    if (newOuts.size() != 1)
      return failure();

    newOuts.push_back(newLoopOperand);
    auto newForallOp = rewriter.create<scf::ForallOp>(
        forallOp.getLoc(), forallOp.getMixedLowerBound(),
        forallOp.getMixedUpperBound(), forallOp.getMixedStep(), newOuts,
        forallOp.getMapping());
    rewriter.eraseBlock(newForallOp.getBody());
    newForallOp.getRegion().takeBody(forallOp.getRegion());
    newForallOp.getBody()->addArgument(newOuts.back().getType(),
                                       newOuts.back().getLoc());

    ArrayRef<BlockArgument> bbArgs = newForallOp.getRegionIterArgs();
    assert(bbArgs.size() == 2);

    rewriter.setInsertionPointToStart(newForallOp.getBody());
    AffineExpr dim0;
    bindDims(rewriter.getContext(), dim0);
    AffineExpr s0 = rewriter.getAffineSymbolExpr(0);
    auto mulMap = AffineMap::get(1, 1, {dim0 * s0});
    SmallVector<OpFoldResult> newMixedOffsets;
    for (auto ivs : llvm::enumerate(newForallOp.getInductionVars())) {
      OpFoldResult applied = affine::makeComposedFoldedAffineApply(
          rewriter, newForallOp.getLoc(), mulMap,
          {ivs.value(), unPackOp.getMixedTiles()[ivs.index()]});
      newMixedOffsets.push_back(applied);
    }

    for (Operation *operation : bbArgs.front().getUsers()) {
      if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(operation)) {
        rewriter.setInsertionPoint(extractSliceOp);

        int64_t rank = unPackOp.getDestType().getRank();
        auto mixedStrides = extractSliceOp.getMixedStrides();
        auto newMixedStrides = SmallVector<OpFoldResult>(
            mixedStrides.begin() + rank, mixedStrides.end());

        auto mixedSizes = extractSliceOp.getMixedSizes();
        auto newMixedSizes = SmallVector<OpFoldResult>(
            mixedSizes.begin() + rank, mixedSizes.end());

        auto newExtractSliceOp = rewriter.create<tensor::ExtractSliceOp>(
            extractSliceOp.getLoc(), bbArgs.back(), newMixedOffsets,
            newMixedSizes, newMixedStrides);

        rewriter.replaceAllUsesWith(extractSliceOp->getResults(),
                                    newExtractSliceOp->getResults());
        continue;
      }
      if (auto parallelInsertSlice =
              dyn_cast<tensor::ParallelInsertSliceOp>(operation)) {
        rewriter.setInsertionPoint(parallelInsertSlice);

        int64_t rank = unPackOp.getDestType().getRank();
        auto mixedStrides = parallelInsertSlice.getMixedStrides();
        auto newMixedStrides = SmallVector<OpFoldResult>(
            mixedStrides.begin() + rank, mixedStrides.end());

        auto mixedSizes = parallelInsertSlice.getMixedSizes();
        auto newMixedSizes = SmallVector<OpFoldResult>(
            mixedSizes.begin() + rank, mixedSizes.end());

        auto newInsertSliceOp = rewriter.create<tensor::ParallelInsertSliceOp>(
            parallelInsertSlice.getLoc(), parallelInsertSlice.getSource(),
            bbArgs.back(), newMixedOffsets, newMixedSizes, newMixedStrides);
        rewriter.replaceAllUsesWith(parallelInsertSlice->getResults(),
                                    newInsertSliceOp->getResults());
        rewriter.eraseOp(parallelInsertSlice);
        continue;
      }
      return failure();
    }

    rewriter.replaceOp(unPackOp, newForallOp->getResults()[1]);
    return success();
  }
};

struct SimplifyAndCanonicalizePack
    : public tpp::impl::SimplifyAndCanonicalizePackBase<
          SimplifyAndCanonicalizePack> {
  void runOnOperation() override {
    MLIRContext *ctx = getOperation().getContext();
    RewritePatternSet patterns(ctx);
    tpp::populateSimplifyPacking(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // end namespace

void mlir::tpp::populateSimplifyPacking(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  tensor::populateSimplifyPackAndUnpackPatterns(patterns);
  tensor::populateFoldTensorEmptyPatterns(patterns);
  linalg::PackOp::getCanonicalizationPatterns(patterns, ctx);
  linalg::UnPackOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::EmptyOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::PadOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ParallelInsertSliceOp::getCanonicalizationPatterns(patterns, ctx);
  scf::ForallOp::getCanonicalizationPatterns(patterns, ctx);
  // Propagate packs/unpacks only through expand shapes at this point.
  // This captures the transformation scope of the replaced downstream pass.
  linalg::populateDataLayoutPropagationPatterns(
      patterns, [](OpOperand *operand) {
        return isa<tensor::ExpandShapeOp>(operand->get().getDefiningOp());
      });
  ctx->getLoadedDialect<tensor::TensorDialect>()->getCanonicalizationPatterns(
      patterns);
  patterns.add<FoldUnPackIntoInsertSlice>(ctx);
  tensor::populateReassociativeReshapeFoldingPatterns(patterns);
}
