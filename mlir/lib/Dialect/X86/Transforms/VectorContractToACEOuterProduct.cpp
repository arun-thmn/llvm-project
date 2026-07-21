//===- VectorContractToACEOuterProduct.cpp---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"
#include "mlir/Dialect/X86/X86Dialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::x86;

namespace {

// Recursively follows single-use values through scf.yield operations
// and returns the first non-yield user result in the contraction chain.
static Value contractionUsersAfterYield(Value v) {

  if (v.getNumUses() != 1)
    return nullptr;

  OpOperand &use = *v.use_begin();
  Operation *user = use.getOwner();

  if (!isa<scf::YieldOp>(user))
    return v;

  auto yield = cast<scf::YieldOp>(user);
  Operation *parent = yield->getParentOp();
  unsigned idx = use.getOperandNumber();

  return contractionUsersAfterYield(parent->getResult(idx));
}

// Function to collapse the last two dimension (vnni and k) to help the
// amx.tile_load to correctly load the packed element type.
static Value collapseInnerDims(OpBuilder &builder, mlir::Location loc,
                               Value input) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  int64_t firstDimToCollapse = inputType.getRank() - 2;

  if (inputType.getRank() == 1)
    return input;

  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < firstDimToCollapse; ++i)
    reassociation.push_back(ReassociationIndices{i});

  ReassociationIndices collapsedIndices;
  for (int64_t i = firstDimToCollapse; i < inputType.getRank(); ++i)
    collapsedIndices.push_back(i);

  reassociation.push_back(collapsedIndices);
  return memref::CollapseShapeOp::create(builder, loc, input, reassociation);
}

// Get the MemRef source and offset index for the operands of
// vector.contract.
static FailureOr<std::pair<Value, SmallVector<Value>>>
getSrcIndxValue(OpBuilder &rewriter, Location loc, Value operand,
                bool isNotAcc) {
  Operation *defOp = operand.getDefiningOp();
  if (!defOp)
    return failure();

  Value srcBuff;
  SmallVector<OpFoldResult> indexVals;
  llvm::TypeSwitch<Operation *>(operand.getDefiningOp())
      .Case<TransferReadOp, LoadOp>([&](auto readOp) {
        indexVals = SmallVector<OpFoldResult>(readOp.getIndices().begin(),
                                              readOp.getIndices().end());
        srcBuff = readOp.getOperand(0);
      });

  if (!srcBuff)
    return failure();

  if (isNotAcc)
    indexVals.pop_back();

  SmallVector<Value> indices;
  indices.reserve(indexVals.size());

  for (OpFoldResult ofr : indexVals) {
    indices.push_back(
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, ofr));
  }

  if (isNotAcc) {
    srcBuff = collapseInnerDims(rewriter, loc, srcBuff);
  }

  return std::make_pair(srcBuff, indices);
}

// Function to validate the vector.contract operation.
static LogicalResult validateContractOps(OpBuilder &rewriter,
                                         vector::ContractionOp contractOp,
                                         unsigned int blockingFactor,
                                         Value srcBuffLhs, Value srcBuffRhs,
                                         bool srcValidate, Type ipType) {

  if (srcValidate) {
    // Get the MemRef buffer of LHS operand.
    auto srcIndxLhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                      contractOp.getLhs(), false);
    if (failed(srcIndxLhs))
      return failure();
    auto [buffLhs, indicesLhs] = *srcIndxLhs;

    // Get the MemRef buffer of RHS operand.
    auto srcIndxRhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                      contractOp.getRhs(), false);
    if (failed(srcIndxRhs))
      return failure();
    auto [buffRhs, indicesRhs] = *srcIndxRhs;

    // Return failure if the Memref buff didn't match.
    if (buffLhs != srcBuffLhs)
      return failure();

    if (buffRhs != srcBuffRhs)
      return failure();
  }

  if (!contractionUsersAfterYield(contractOp.getResult()))
    return failure();

  VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
  if (!accTy)
    return failure();

  // The Accumulator dims should be 16 or 1. Like <1x16x16> or <16x16>.
  ArrayRef<int64_t> accShape = accTy.getShape();
  llvm::SmallVector<int64_t> nonUnitDimAcc;
  llvm::copy_if(accShape, std::back_inserter(nonUnitDimAcc),
                [](int64_t dim) { return (dim != 16 && dim != 1); });

  if (nonUnitDimAcc.size() != 0)
    return failure();

  // The LHS dims should be 16 or vnni or 1. Like <1x16x16x2> or
  // <16x16x4>. The vnni dims should be 2 or 4.
  VectorType lhsTy = contractOp.getLhsType();
  ArrayRef<int64_t> lhsShape = lhsTy.getShape();
  llvm::SmallVector<int64_t> nonUnitDimLhs;
  llvm::copy_if(lhsShape, std::back_inserter(nonUnitDimLhs),
                [](int64_t dim) { return (dim != 16 && dim != 1); });

  if (ipType.isBF16() && nonUnitDimLhs.size() != 1)
    return failure();

  if (ipType.isBF16() && nonUnitDimLhs[0] != blockingFactor)
    return failure();

  if (ipType.isSignlessInteger(8) && nonUnitDimLhs.size() != 0)
    return failure();

  // The RHS dims should be 16 or vnni or 1. Like <1x16x16x2> or
  // <16x16x4>. The vnni dims should be 2 or 4.
  VectorType rhsTy = contractOp.getRhsType();
  ArrayRef<int64_t> rhsShape = rhsTy.getShape();
  llvm::SmallVector<int64_t> nonUnitDimRhs;
  llvm::copy_if(rhsShape, std::back_inserter(nonUnitDimRhs),
                [](int64_t dim) { return (dim != 16 && dim != 1); });

  if (ipType.isBF16() && nonUnitDimRhs.size() != 1)
    return failure();

  if (ipType.isBF16() && nonUnitDimRhs[0] != (blockingFactor))
    return failure();

  if (ipType.isSignlessInteger(8) && nonUnitDimRhs.size() != 0)
    return failure();

  return success();
}

// Returns the loop index position to get mapped during the
// MemRef type clone.
static unsigned getIndexPosition(Value operand, scf::ForOp loop) {
  Value iv = loop.getInductionVar();

  Value srcBuff;
  llvm::TypeSwitch<Operation *>(operand.getDefiningOp())
      .Case<TransferReadOp, LoadOp>(
          [&](auto readOp) { srcBuff = readOp.getOperand(0); });

  auto subview = srcBuff.getDefiningOp<memref::SubViewOp>();
  if (!subview)
    return 0;

  auto offsets = subview.getOffsets();

  for (auto it : llvm::enumerate(offsets)) {
    if (it.value() == iv)
      return it.index();
  }

  return 0;
}

static SmallVector<Value> createTileZeros(OpBuilder &rewriter, Location loc,
                                          Type opType, scf::ForOp loop,
                                          int64_t size) {
  rewriter.setInsertionPoint(loop);
  // ace::BSRInitOp::create(rewriter, loc);

  SmallVector<Value> loopItrArgs;
  auto zeroTileType = amx::TileType::get({16, 16}, opType);

  for (int i = 0; i < size; i++) {
    auto zeroTile = amx::TileZeroOp::create(rewriter, loc, zeroTileType);
    loopItrArgs.push_back(zeroTile);
  }

  return loopItrArgs;
}

static SmallVector<Value> loadMasks(OpBuilder &rewriter, Location loc,
                                    MemRefType memrefTy) {

  SmallVector<Value> mask;

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto vecTy = VectorType::get({16}, rewriter.getI32Type());

  Value mask1 = memref::GetGlobalOp::create(rewriter, loc, memrefTy, "mask_1");
  mask1 = vector::LoadOp::create(rewriter, loc, vecTy, mask1, ValueRange{c0});
  mask.push_back(mask1);

  Value mask2 = memref::GetGlobalOp::create(rewriter, loc, memrefTy, "mask_2");
  mask2 = vector::LoadOp::create(rewriter, loc, vecTy, mask2, ValueRange{c0});
  mask.push_back(mask2);

  Value mask3 = memref::GetGlobalOp::create(rewriter, loc, memrefTy, "mask_3");
  mask3 = vector::LoadOp::create(rewriter, loc, vecTy, mask3, ValueRange{c0});
  mask.push_back(mask3);

  return mask;
}

// Function to load the next row vector by incrementing the row offset, shuffles
// it into the target vector layout, and conditionally selects it based on
// the given comparison mask.
static Value loadShuffleSelect(OpBuilder &rewriter, Location loc,
                               Value currentVec, Value cmp, Value memref,
                               SmallVector<Value> &loadOffsets, Value c1,
                               VectorType vecTy, VectorType vecTy4) {
  int rowIdx = loadOffsets.size() - 2;
  loadOffsets[rowIdx] =
      arith::AddIOp::create(rewriter, loc, loadOffsets[rowIdx], c1);

  Value v = vector::LoadOp::create(rewriter, loc, vecTy4, memref, loadOffsets);

  Value shuffled = vector::ShuffleOp::create(
      rewriter, loc, vecTy, v, v,
      ArrayRef<int64_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});

  return arith::SelectOp::create(rewriter, loc, cmp, shuffled, currentVec);
}

static SmallVector<Value>
transposeShuffle(OpBuilder &rewriter, Location loc, Value matrix, Type ipType,
                 Type opType, Value mask1, Value mask2, Value mask3,
                 int64_t upperBound, int64_t blockingFactor) {

  SmallVector<Value> transposedValue;

  Operation *defOp = matrix.getDefiningOp();
  auto subview = dyn_cast<memref::SubViewOp>(defOp);

  ValueRange offsets = subview.getOffsets();
  auto myMemref = defOp->getOperand(0);
  Type llvmPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());

  auto castOp =
      UnrealizedConversionCastOp::create(rewriter, loc, llvmPtrType, myMemref);

  auto sourceMemRefType = cast<MemRefType>(myMemref.getType());

  llvm::ArrayRef<int64_t> shape = sourceMemRefType.getShape();
  SmallVector<int64_t> newShape(shape.begin(), shape.end());
  newShape[newShape.size() - 1] =
      newShape[newShape.size() - 1] / blockingFactor;

  auto targetMemRefType = MemRefType::get(newShape, rewriter.getI32Type());

  auto castOpFinal = UnrealizedConversionCastOp::create(
      rewriter, loc, targetMemRefType, ValueRange{castOp.getResult(0)});

  SmallVector<Value> loadOffsets(offsets.begin(), offsets.end());
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto vecTy = VectorType::get({16}, rewriter.getI32Type());
  auto vecTy_4 = VectorType::get({4}, rewriter.getI32Type());
  Value cst_one = arith::ConstantOp::create(
      rewriter, loc, vecTy, DenseIntElementsAttr::get(vecTy, 1));

  Value blk = arith::ConstantIndexOp::create(rewriter, loc, blockingFactor);
  Value offset_blk = arith::DivUIOp::create(
      rewriter, loc, loadOffsets[loadOffsets.size() - 1], blk);
  loadOffsets[loadOffsets.size() - 1] = offset_blk;

  for (int i = 0; i < upperBound; i = i + 16) {

    Value v0 = vector::LoadOp::create(rewriter, loc, vecTy,
                                      castOpFinal.getResult(0), loadOffsets);

    Value offset_b0 = arith::AddIOp::create(
        rewriter, loc, loadOffsets[loadOffsets.size() - 2], c1);
    loadOffsets[loadOffsets.size() - 2] = offset_b0;

    Value v1 = vector::LoadOp::create(rewriter, loc, vecTy,
                                      castOpFinal.getResult(0), loadOffsets);

    Value offset_c0 = arith::AddIOp::create(
        rewriter, loc, loadOffsets[loadOffsets.size() - 2], c1);
    loadOffsets[loadOffsets.size() - 2] = offset_c0;

    Value v2 = vector::LoadOp::create(rewriter, loc, vecTy,
                                      castOpFinal.getResult(0), loadOffsets);

    Value offset_d0 = arith::AddIOp::create(
        rewriter, loc, loadOffsets[loadOffsets.size() - 2], c1);
    loadOffsets[loadOffsets.size() - 2] = offset_d0;

    Value v3 = vector::LoadOp::create(rewriter, loc, vecTy,
                                      castOpFinal.getResult(0), loadOffsets);

    SmallVector<Value> vecs = {v0, v1, v2, v3};

    SmallVector<Value> masks = {mask1, mask2, mask3};

    for (Value mask : masks) {
      Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                        mask, cst_one);

      for (Value &vec : vecs) {
        vec =
            loadShuffleSelect(rewriter, loc, vec, cmp, castOpFinal.getResult(0),
                              loadOffsets, c1, vecTy, vecTy_4);
      }
    }

    v0 = vecs[0];
    v1 = vecs[1];
    v2 = vecs[2];
    v3 = vecs[3];

    Value s4;
    Value s5;
    Value s6;
    Value s7;

    int elemNumber = 64;
    if (ipType.isBF16())
      elemNumber = 32;

    v0 = vector::BitCastOp::create(rewriter, loc,
                                   VectorType::get({elemNumber}, ipType), v0);
    v1 = vector::BitCastOp::create(rewriter, loc,
                                   VectorType::get({elemNumber}, ipType), v1);
    v2 = vector::BitCastOp::create(rewriter, loc,
                                   VectorType::get({elemNumber}, ipType), v2);
    v3 = vector::BitCastOp::create(rewriter, loc,
                                   VectorType::get({elemNumber}, ipType), v3);
    if (ipType.isBF16()) {
      auto s0 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v0, v1,
          ArrayRef<int64_t>{0,  1,  32, 33, 2,  3,  34, 35, 8,  9,  40,
                            41, 10, 11, 42, 43, 16, 17, 48, 49, 18, 19,
                            50, 51, 24, 25, 56, 57, 26, 27, 58, 59});
      auto s1 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v0, v1,
          ArrayRef<int64_t>{4,  5,  36, 37, 6,  7,  38, 39, 12, 13, 44,
                            45, 14, 15, 46, 47, 20, 21, 52, 53, 22, 23,
                            54, 55, 28, 29, 60, 61, 30, 31, 62, 63});
      auto s2 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v2, v3,
          ArrayRef<int64_t>{0,  1,  32, 33, 2,  3,  34, 35, 8,  9,  40,
                            41, 10, 11, 42, 43, 16, 17, 48, 49, 18, 19,
                            50, 51, 24, 25, 56, 57, 26, 27, 58, 59});
      auto s3 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v2, v3,
          ArrayRef<int64_t>{4,  5,  36, 37, 6,  7,  38, 39, 12, 13, 44,
                            45, 14, 15, 46, 47, 20, 21, 52, 53, 22, 23,
                            54, 55, 28, 29, 60, 61, 30, 31, 62, 63});

      s4 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), s0, s2,
          ArrayRef<int64_t>{0,  1,  2,  3,  32, 33, 34, 35, 8,  9,  10,
                            11, 40, 41, 42, 43, 16, 17, 18, 19, 48, 49,
                            50, 51, 24, 25, 26, 27, 56, 57, 58, 59});
      s5 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), s0, s2,
          ArrayRef<int64_t>{4,  5,  6,  7,  36, 37, 38, 39, 12, 13, 14,
                            15, 44, 45, 46, 47, 20, 21, 22, 23, 52, 53,
                            54, 55, 28, 29, 30, 31, 60, 61, 62, 63});
      s6 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), s1, s3,
          ArrayRef<int64_t>{0,  1,  2,  3,  32, 33, 34, 35, 8,  9,  10,
                            11, 40, 41, 42, 43, 16, 17, 18, 19, 48, 49,
                            50, 51, 24, 25, 26, 27, 56, 57, 58, 59});
      s7 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), s1, s3,
          ArrayRef<int64_t>{4,  5,  6,  7,  36, 37, 38, 39, 12, 13, 14,
                            15, 44, 45, 46, 47, 20, 21, 22, 23, 52, 53,
                            54, 55, 28, 29, 30, 31, 60, 61, 62, 63});
    } else {

      auto s0 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), v0, v1,
          ArrayRef<int64_t>{
              0,   1,   2,   3,   64, 65,  66,  67,  4,   5,   6,   7,  68,
              69,  70,  71,  16,  17, 18,  19,  80,  81,  82,  83,  20, 21,
              22,  23,  84,  85,  86, 87,  32,  33,  34,  35,  96,  97, 98,
              99,  36,  37,  38,  39, 100, 101, 102, 103, 48,  49,  50, 51,
              112, 113, 114, 115, 52, 53,  54,  55,  116, 117, 118, 119});

      auto s1 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), v0, v1,
          ArrayRef<int64_t>{
              8,   9,   10,  11,  72, 73,  74,  75,  12,  13,  14,  15,  76,
              77,  78,  79,  24,  25, 26,  27,  88,  89,  90,  91,  28,  29,
              30,  31,  92,  93,  94, 95,  40,  41,  42,  43,  104, 105, 106,
              107, 44,  45,  46,  47, 108, 109, 110, 111, 56,  57,  58,  59,
              120, 121, 122, 123, 60, 61,  62,  63,  124, 125, 126, 127});

      auto s2 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), v2, v3,
          ArrayRef<int64_t>{
              0,   1,   2,   3,   64, 65,  66,  67,  4,   5,   6,   7,  68,
              69,  70,  71,  16,  17, 18,  19,  80,  81,  82,  83,  20, 21,
              22,  23,  84,  85,  86, 87,  32,  33,  34,  35,  96,  97, 98,
              99,  36,  37,  38,  39, 100, 101, 102, 103, 48,  49,  50, 51,
              112, 113, 114, 115, 52, 53,  54,  55,  116, 117, 118, 119});

      auto s3 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), v2, v3,
          ArrayRef<int64_t>{
              8,   9,   10,  11,  72, 73,  74,  75,  12,  13,  14,  15,  76,
              77,  78,  79,  24,  25, 26,  27,  88,  89,  90,  91,  28,  29,
              30,  31,  92,  93,  94, 95,  40,  41,  42,  43,  104, 105, 106,
              107, 44,  45,  46,  47, 108, 109, 110, 111, 56,  57,  58,  59,
              120, 121, 122, 123, 60, 61,  62,  63,  124, 125, 126, 127});

      s4 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), s0, s2,
          ArrayRef<int64_t>{0,   1,   2,   3,   4,   5,   6,   7,   64, 65, 66,
                            67,  68,  69,  70,  71,  16,  17,  18,  19, 20, 21,
                            22,  23,  80,  81,  82,  83,  84,  85,  86, 87, 32,
                            33,  34,  35,  36,  37,  38,  39,  96,  97, 98, 99,
                            100, 101, 102, 103, 48,  49,  50,  51,  52, 53, 54,
                            55,  112, 113, 114, 115, 116, 117, 118, 119});

      s5 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), s0, s2,
          ArrayRef<int64_t>{
              8,  9,   10,  11,  12,  13,  14,  15,  72,  73,  74,  75, 76,
              77, 78,  79,  24,  25,  26,  27,  28,  29,  30,  31,  88, 89,
              90, 91,  92,  93,  94,  95,  40,  41,  42,  43,  44,  45, 46,
              47, 104, 105, 106, 107, 108, 109, 110, 111, 56,  57,  58, 59,
              60, 61,  62,  63,  120, 121, 122, 123, 124, 125, 126, 127});

      s6 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), s1, s3,
          ArrayRef<int64_t>{0,   1,   2,   3,   4,   5,   6,   7,   64, 65, 66,
                            67,  68,  69,  70,  71,  16,  17,  18,  19, 20, 21,
                            22,  23,  80,  81,  82,  83,  84,  85,  86, 87, 32,
                            33,  34,  35,  36,  37,  38,  39,  96,  97, 98, 99,
                            100, 101, 102, 103, 48,  49,  50,  51,  52, 53, 54,
                            55,  112, 113, 114, 115, 116, 117, 118, 119});

      s7 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({64}, ipType), s1, s3,
          ArrayRef<int64_t>{
              8,  9,   10,  11,  12,  13,  14,  15,  72,  73,  74,  75, 76,
              77, 78,  79,  24,  25,  26,  27,  28,  29,  30,  31,  88, 89,
              90, 91,  92,  93,  94,  95,  40,  41,  42,  43,  44,  45, 46,
              47, 104, 105, 106, 107, 108, 109, 110, 111, 56,  57,  58, 59,
              60, 61,  62,  63,  120, 121, 122, 123, 124, 125, 126, 127});
    }

    transposedValue.push_back(s4);
    transposedValue.push_back(s5);
    transposedValue.push_back(s6);
    transposedValue.push_back(s7);

    Value offset_res = arith::AddIOp::create(
        rewriter, loc, loadOffsets[loadOffsets.size() - 2], c1);
    loadOffsets[loadOffsets.size() - 2] = offset_res;
  }

  return transposedValue;
}

/// Helper to generate a strided prefetch. Updates the selected index in
/// `loadOffsets`, emits `memref.prefetch`, and restores the original offset.
static void createStridedPrefetch(bool isStrided, bool leastOffset,
                                  int64_t offset, int64_t preOffset,
                                  Value restoreOffset, OpBuilder &rewriter,
                                  Location loc, Value matrix,
                                  SmallVectorImpl<Value> &loadOffsets) {
  if (!isStrided)
    return;

  size_t dim = leastOffset ? loadOffsets.size() - 1 : loadOffsets.size() - 2;
  Value pOffset =
      arith::ConstantIndexOp::create(rewriter, loc, preOffset + offset);
  loadOffsets[dim] = pOffset;
  memref::PrefetchOp::create(rewriter, loc, matrix, loadOffsets, false, 3,
                             true);
  loadOffsets[dim] = restoreOffset;
}

static SmallVector<Value> vnniShuffle(OpBuilder &rewriter, Location loc,
                                      Value matrix, Type ipType, Type opType,
                                      int64_t upperBound, int64_t offsetA,
                                      int group) {

  Operation *defOp = matrix.getDefiningOp();
  auto subview = dyn_cast<memref::SubViewOp>(defOp);

  int loopStep = 32;
  if (!ipType.isBF16()) {
    offsetA = offsetA * 2;
    loopStep = 64;
  }

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  ValueRange offsets = subview.getOffsets();
  SmallVector<Value> loadOffsets(offsets.size(), c0);

  SmallVector<Value> shuffledValue;

  Value offset_0 = arith::ConstantIndexOp::create(rewriter, loc, offsetA);
  Value offset_1 = arith::ConstantIndexOp::create(rewriter, loc, (offsetA + 1));
  Value offset_2 = arith::ConstantIndexOp::create(rewriter, loc, (offsetA + 2));
  Value offset_3 = arith::ConstantIndexOp::create(rewriter, loc, (offsetA + 3));

  bool isStrided = true;
  auto subviewTy = llvm::cast<mlir::MemRefType>(matrix.getType());
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(subviewTy.getStridesAndOffset(strides, offset)))
    isStrided = false;

  int64_t preOffset = 0;
  bool leastOffset = false;

  if (isStrided) {
    if (strides[strides.size() - 1] != 1)
      isStrided = false;

    if (strides[strides.size() - 2] < 4096) {
      preOffset = 4096 / strides[strides.size() - 2];
    } else {
      leastOffset = true;
      preOffset = 4096;
    }
  }

  for (int i = 0; i < upperBound; i = i + loopStep) {
    Value offset_i = arith::ConstantIndexOp::create(rewriter, loc, i);
    loadOffsets[loadOffsets.size() - 1] = offset_i;
    loadOffsets[loadOffsets.size() - 2] = offset_0;

    Value v0 = vector::LoadOp::create(rewriter, loc,
                                      VectorType::get({loopStep}, ipType),
                                      matrix, loadOffsets);
    createStridedPrefetch(isStrided, leastOffset, leastOffset ? i : offsetA,
                          preOffset, leastOffset ? offset_i : offset_0,
                          rewriter, loc, matrix, loadOffsets);

    loadOffsets[loadOffsets.size() - 2] = offset_1;
    Value v1 = vector::LoadOp::create(rewriter, loc,
                                      VectorType::get({loopStep}, ipType),
                                      matrix, loadOffsets);
    createStridedPrefetch(isStrided, leastOffset, leastOffset ? i : offsetA + 1,
                          preOffset, leastOffset ? offset_i : offset_1,
                          rewriter, loc, matrix, loadOffsets);

    Value s0;
    Value s1;
    Value s2;
    Value s3;

    if (ipType.isBF16()) {
      s0 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v0, v1,
          ArrayRef<int64_t>{0,  32, 1,  33, 2,  34, 3,  35, 8,  40, 9,
                            41, 10, 42, 11, 43, 16, 48, 17, 49, 18, 50,
                            19, 51, 24, 56, 25, 57, 26, 58, 27, 59});
      s1 = vector::ShuffleOp::create(
          rewriter, loc, VectorType::get({32}, ipType), v0, v1,
          ArrayRef<int64_t>{4,  36, 5,  37, 6,  38, 7,  39, 12, 44, 13,
                            45, 14, 46, 15, 47, 20, 52, 21, 53, 22, 54,
                            23, 55, 28, 60, 29, 61, 30, 62, 31, 63});
    } else {

      loadOffsets[loadOffsets.size() - 2] = offset_2;
      Value v2 = vector::LoadOp::create(rewriter, loc,
                                        VectorType::get({loopStep}, ipType),
                                        matrix, loadOffsets);
      createStridedPrefetch(isStrided, leastOffset,
                            leastOffset ? i : offsetA + 2, preOffset,
                            leastOffset ? offset_i : offset_2, rewriter, loc,
                            matrix, loadOffsets);

      loadOffsets[loadOffsets.size() - 2] = offset_3;
      Value v3 = vector::LoadOp::create(rewriter, loc,
                                        VectorType::get({loopStep}, ipType),
                                        matrix, loadOffsets);
      createStridedPrefetch(isStrided, leastOffset,
                            leastOffset ? i : offsetA + 3, preOffset,
                            leastOffset ? offset_i : offset_3, rewriter, loc,
                            matrix, loadOffsets);

      if (group == 4) {
        auto a0 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v0, v2,
            ArrayRef<int64_t>{0,  64, 1,  65, 2,  66, 3,  67, 4,  68, 5,
                              69, 6,  70, 7,  71, 8,  72, 9,  73, 10, 74,
                              11, 75, 12, 76, 13, 77, 14, 78, 15, 79, 16,
                              80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85,
                              22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27,
                              91, 28, 92, 29, 93, 30, 94, 31, 95});

        auto a1 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v0, v2,
            ArrayRef<int64_t>{
                32,  96,  33,  97,  34,  98,  35,  99,  36,  100, 37,  101, 38,
                102, 39,  103, 40,  104, 41,  105, 42,  106, 43,  107, 44,  108,
                45,  109, 46,  110, 47,  111, 48,  112, 49,  113, 50,  114, 51,
                115, 52,  116, 53,  117, 54,  118, 55,  119, 56,  120, 57,  121,
                58,  122, 59,  123, 60,  124, 61,  125, 62,  126, 63,  127});

        auto a2 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v1, v3,
            ArrayRef<int64_t>{0,  64, 1,  65, 2,  66, 3,  67, 4,  68, 5,
                              69, 6,  70, 7,  71, 8,  72, 9,  73, 10, 74,
                              11, 75, 12, 76, 13, 77, 14, 78, 15, 79, 16,
                              80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85,
                              22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27,
                              91, 28, 92, 29, 93, 30, 94, 31, 95});

        auto a3 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v1, v3,
            ArrayRef<int64_t>{
                32,  96,  33,  97,  34,  98,  35,  99,  36,  100, 37,  101, 38,
                102, 39,  103, 40,  104, 41,  105, 42,  106, 43,  107, 44,  108,
                45,  109, 46,  110, 47,  111, 48,  112, 49,  113, 50,  114, 51,
                115, 52,  116, 53,  117, 54,  118, 55,  119, 56,  120, 57,  121,
                58,  122, 59,  123, 60,  124, 61,  125, 62,  126, 63,  127});

        s0 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a0, a2,
            ArrayRef<int64_t>{0,  1,  64, 65, 2,  3,  66, 67, 4,  5,  68,
                              69, 6,  7,  70, 71, 8,  9,  72, 73, 10, 11,
                              74, 75, 12, 13, 76, 77, 14, 15, 78, 79, 16,
                              17, 80, 81, 18, 19, 82, 83, 20, 21, 84, 85,
                              22, 23, 86, 87, 24, 25, 88, 89, 26, 27, 90,
                              91, 28, 29, 92, 93, 30, 31, 94, 95});

        s1 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a0, a2,
            ArrayRef<int64_t>{
                32,  33,  96,  97,  34,  35,  98,  99,  36,  37,  100, 101, 38,
                39,  102, 103, 40,  41,  104, 105, 42,  43,  106, 107, 44,  45,
                108, 109, 46,  47,  110, 111, 48,  49,  112, 113, 50,  51,  114,
                115, 52,  53,  116, 117, 54,  55,  118, 119, 56,  57,  120, 121,
                58,  59,  122, 123, 60,  61,  124, 125, 62,  63,  126, 127});

        s2 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a1, a3,
            ArrayRef<int64_t>{0,  1,  64, 65, 2,  3,  66, 67, 4,  5,  68,
                              69, 6,  7,  70, 71, 8,  9,  72, 73, 10, 11,
                              74, 75, 12, 13, 76, 77, 14, 15, 78, 79, 16,
                              17, 80, 81, 18, 19, 82, 83, 20, 21, 84, 85,
                              22, 23, 86, 87, 24, 25, 88, 89, 26, 27, 90,
                              91, 28, 29, 92, 93, 30, 31, 94, 95});

        s3 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a1, a3,
            ArrayRef<int64_t>{
                32,  33,  96,  97,  34,  35,  98,  99,  36,  37,  100, 101, 38,
                39,  102, 103, 40,  41,  104, 105, 42,  43,  106, 107, 44,  45,
                108, 109, 46,  47,  110, 111, 48,  49,  112, 113, 50,  51,  114,
                115, 52,  53,  116, 117, 54,  55,  118, 119, 56,  57,  120, 121,
                58,  59,  122, 123, 60,  61,  124, 125, 62,  63,  126, 127});
      } else {
        auto a0 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v0, v1,
            ArrayRef<int64_t>{
                0,  64,  1,   65,  2,   66,  3,   67,  4,   68,  5,   69, 6,
                70, 7,   71,  16,  80,  17,  81,  18,  82,  19,  83,  20, 84,
                21, 85,  22,  86,  23,  87,  32,  96,  33,  97,  34,  98, 35,
                99, 36,  100, 37,  101, 38,  102, 39,  103, 48,  112, 49, 113,
                50, 114, 51,  115, 52,  116, 53,  117, 54,  118, 55,  119});

        auto a1 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v0, v1,
            ArrayRef<int64_t>{
                8,   72,  9,   73,  10,  74,  11,  75,  12,  76,  13,  77,  14,
                78,  15,  79,  24,  88,  25,  89,  26,  90,  27,  91,  28,  92,
                29,  93,  30,  94,  31,  95,  40,  104, 41,  105, 42,  106, 43,
                107, 44,  108, 45,  109, 46,  110, 47,  111, 56,  120, 57,  121,
                58,  122, 59,  123, 60,  124, 61,  125, 62,  126, 63,  127});

        auto a2 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v2, v3,
            ArrayRef<int64_t>{
                0,  64,  1,   65,  2,   66,  3,   67,  4,   68,  5,   69, 6,
                70, 7,   71,  16,  80,  17,  81,  18,  82,  19,  83,  20, 84,
                21, 85,  22,  86,  23,  87,  32,  96,  33,  97,  34,  98, 35,
                99, 36,  100, 37,  101, 38,  102, 39,  103, 48,  112, 49, 113,
                50, 114, 51,  115, 52,  116, 53,  117, 54,  118, 55,  119});

        auto a3 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), v2, v3,
            ArrayRef<int64_t>{
                8,   72,  9,   73,  10,  74,  11,  75,  12,  76,  13,  77,  14,
                78,  15,  79,  24,  88,  25,  89,  26,  90,  27,  91,  28,  92,
                29,  93,  30,  94,  31,  95,  40,  104, 41,  105, 42,  106, 43,
                107, 44,  108, 45,  109, 46,  110, 47,  111, 56,  120, 57,  121,
                58,  122, 59,  123, 60,  124, 61,  125, 62,  126, 63,  127});

        s0 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a0, a2,
            ArrayRef<int64_t>{
                0,  1,  64,  65,  2,   3,  66,  67,  4,   5,  68,  69,  6,
                7,  70, 71,  16,  17,  80, 81,  18,  19,  82, 83,  20,  21,
                84, 85, 22,  23,  86,  87, 32,  33,  96,  97, 34,  35,  98,
                99, 36, 37,  100, 101, 38, 39,  102, 103, 48, 49,  112, 113,
                50, 51, 114, 115, 52,  53, 116, 117, 54,  55, 118, 119});
        s1 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a0, a2,
            ArrayRef<int64_t>{
                8,   9,  72,  73,  10,  11, 74,  75,  12,  13,  76,  77,  14,
                15,  78, 79,  24,  25,  88, 89,  26,  27,  90,  91,  28,  29,
                92,  93, 30,  31,  94,  95, 40,  41,  104, 105, 42,  43,  106,
                107, 44, 45,  108, 109, 46, 47,  110, 111, 56,  57,  120, 121,
                58,  59, 122, 123, 60,  61, 124, 125, 62,  63,  126, 127});

        s2 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a1, a3,
            ArrayRef<int64_t>{
                0,  1,  64,  65,  2,   3,  66,  67,  4,   5,  68,  69,  6,
                7,  70, 71,  16,  17,  80, 81,  18,  19,  82, 83,  20,  21,
                84, 85, 22,  23,  86,  87, 32,  33,  96,  97, 34,  35,  98,
                99, 36, 37,  100, 101, 38, 39,  102, 103, 48, 49,  112, 113,
                50, 51, 114, 115, 52,  53, 116, 117, 54,  55, 118, 119});
        s3 = vector::ShuffleOp::create(
            rewriter, loc, VectorType::get({64}, ipType), a1, a3,
            ArrayRef<int64_t>{
                8,   9,  72,  73,  10,  11, 74,  75,  12,  13,  76,  77,  14,
                15,  78, 79,  24,  25,  88, 89,  26,  27,  90,  91,  28,  29,
                92,  93, 30,  31,  94,  95, 40,  41,  104, 105, 42,  43,  106,
                107, 44, 45,  108, 109, 46, 47,  110, 111, 56,  57,  120, 121,
                58,  59, 122, 123, 60,  61, 124, 125, 62,  63,  126, 127});
      }
    }

    shuffledValue.push_back(s0);
    shuffledValue.push_back(s1);

    if (!ipType.isBF16()) {
      shuffledValue.push_back(s2);
      shuffledValue.push_back(s3);
    }
  }

  return shuffledValue;
}

/// Loads vectors from the input memref along the row dimension in fixed-size
/// chunks and returns the loaded vectors as a collection.
static SmallVector<Value> loadData(OpBuilder &rewriter, Location loc,
                                   Value matrix, Type ipType, Type opType,
                                   int64_t upperBound) {
  SmallVector<Value> matrixValue;
  auto memrefType = llvm::cast<MemRefType>(matrix.getType());
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> loadOffsets(memrefType.getShape().size(), c0);

  for (int i = 0; i < upperBound; i = i + 16) {

    Value offset_i = arith::ConstantIndexOp::create(rewriter, loc, i);
    loadOffsets[loadOffsets.size() - 2] = offset_i;

    Value v0 = vector::LoadOp::create(
        rewriter, loc, VectorType::get({32}, ipType), matrix, loadOffsets);

    matrixValue.push_back(v0);
  }

  return matrixValue;
}

static Value createTileMulOutProdOp(OpBuilder &rewriter, Location loc,
                                    Type ipType, Value lhs, Value rhs,
                                    Value acc) {
  if (ipType.isBF16())
    return ace::TileMulFOutProdOp::create(rewriter, loc, lhs, rhs, acc);

  if (ipType.isSignlessInteger(8))
    return ace::TileMulIOutProdOp::create(rewriter, loc, lhs, rhs, acc);

  llvm_unreachable("Unsupported input type");
}

static scf::ForOp
createLoops(OpBuilder &rewriter, Location loc, Value lowerBound,
            Value upperBound, Value step, SmallVector<Value> loopItrArgs,
            Type ipType, Type opType, Operation *vectorOpLhs,
            Operation *vectorOpRhs, vector::ContractionOp contractOp,
            scf::ForOp outerLoop, scf::ForOp innerLoop, Value ivOuterLoop,
            Value mask1, Value mask2, Value mask3, int group,
            unsigned int blockingFactor) {

  auto newLoop = scf::ForOp::create(
      rewriter, loc, lowerBound, upperBound, step, loopItrArgs,
      [&](OpBuilder &rewriterNewInnerLoop, Location locNewInnerLoop,
          Value ivNewInnerLoop, ValueRange iterArgsNewInnerLoop) {
        IRMapping mapping;
        if (outerLoop)
          mapping.map(vectorOpLhs->getOperand(
                          getIndexPosition(contractOp.getLhs(), outerLoop) + 1),
                      ivOuterLoop);

        mapping.map(vectorOpLhs->getOperand(
                        getIndexPosition(contractOp.getLhs(), innerLoop) + 1),
                    ivNewInnerLoop);
        auto lhsClone = rewriterNewInnerLoop.clone(*vectorOpLhs, mapping);

        IRMapping rhsMapping;

        Value matB;
        Operation *rhsOp = vectorOpRhs;

        // Clone for the subview type operations
        if (rhsOp->getNumOperands() > 0) {

          if (outerLoop) {
            int64_t outerPos = getIndexPosition(contractOp.getRhs(), outerLoop);

            if (outerPos >= 0) {
              unsigned operandIdx = static_cast<unsigned>(outerPos + 1);

              if (operandIdx < rhsOp->getNumOperands())
                rhsMapping.map(rhsOp->getOperand(operandIdx), ivOuterLoop);
            }
          }

          int64_t innerPos = getIndexPosition(contractOp.getRhs(), innerLoop);

          if (innerPos >= 0) {
            unsigned operandIdx = static_cast<unsigned>(innerPos + 1);

            if (operandIdx < rhsOp->getNumOperands())
              rhsMapping.map(rhsOp->getOperand(operandIdx), ivNewInnerLoop);
          }

          auto rhsClone = rewriterNewInnerLoop.clone(*rhsOp, rhsMapping);
          matB = rhsClone->getResult(0);

        } else {
          matB = rhsOp->getResult(0);
        }

        SmallVector<Value> ops;

        // Case 1: A(VNNI^T) x B(VNNI).
        if (group == 1) {
          SmallVector<Value> matAValue = loadData(
              rewriter, loc, lhsClone->getResult(0), ipType, opType, 32);

          SmallVector<Value> matBValue =
              loadData(rewriter, loc, matB, ipType, opType, 64);

          for (int i = 0, k = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
              Value op =
                  createTileMulOutProdOp(rewriter, loc, ipType, matAValue[i],
                                         matBValue[j], iterArgsNewInnerLoop[k]);

              k++;
              ops.push_back(op);
            }
          }
        }

        // Case 3: A(flat) x B(flat).
        if (group == 3) {

          SmallVector<Value> ops0;

          SmallVector<Value> matAValue =
              transposeShuffle(rewriter, loc, lhsClone->getResult(0), ipType,
                               opType, mask1, mask2, mask3, 32, blockingFactor);
          SmallVector<Value> matBValue =
              vnniShuffle(rewriter, loc, matB, ipType, opType, 64, 0, group);

          for (int i = 0, k = 0; i < 8; i += 4) {
            for (int j = 0; j < 4; j++) {
              Value op =
                  createTileMulOutProdOp(rewriter, loc, ipType, matAValue[i],
                                         matBValue[j], iterArgsNewInnerLoop[k]);

              k++;
              ops0.push_back(op);
            }
          }

          SmallVector<Value> ops1;
          matBValue =
              vnniShuffle(rewriter, loc, matB, ipType, opType, 64, 2, group);

          for (int i = 1, k = 0; i < 8; i += 4) {
            for (int j = 0; j < 4; j++) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops0[k]);

              k++;
              ops1.push_back(op);
            }
          }

          SmallVector<Value> ops2;
          matBValue =
              vnniShuffle(rewriter, loc, matB, ipType, opType, 64, 4, group);

          for (int i = 2, k = 0; i < 8; i += 4) {
            for (int j = 0; j < 4; j++) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops1[k]);

              k++;
              ops2.push_back(op);
            }
          }

          matBValue =
              vnniShuffle(rewriter, loc, matB, ipType, opType, 64, 6, group);

          for (int i = 3, k = 0; i < 8; i += 4) {
            for (int j = 0; j < 4; j++) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops2[k]);

              k++;
              ops.push_back(op);
            }
          }
        }

        // Case 4: A(flat^T) x B(flat).
        if (group == 4) {

          SmallVector<Value> matAValue =
              vnniShuffle(rewriter, loc, lhsClone->getResult(0), ipType, opType,
                          32, 0, group);

          SmallVector<Value> matBValue =
              vnniShuffle(rewriter, loc, matB, ipType, opType, 64, 0, group);

          for (int i = 0, k = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
              Value op =
                  createTileMulOutProdOp(rewriter, loc, ipType, matAValue[i],
                                         matBValue[j], iterArgsNewInnerLoop[k]);

              k++;
              ops.push_back(op);
            }
          }
        }

        // Case 5: A(flat) x B(flat^T)
        if (group == 5) {

          SmallVector<Value> ops0;

          SmallVector<Value> matAValue =
              transposeShuffle(rewriter, loc, lhsClone->getResult(0), ipType,
                               opType, mask1, mask2, mask3, 32, 2);

          SmallVector<Value> matBValue = transposeShuffle(
              rewriter, loc, matB, ipType, opType, mask1, mask2, mask3, 64, 2);

          for (int i = 0, k = 0; i < 8; i += 4) {
            for (int j = 0; j < 16; j += 4) {
              Value op =
                  createTileMulOutProdOp(rewriter, loc, ipType, matAValue[i],
                                         matBValue[j], iterArgsNewInnerLoop[k]);

              k++;
              ops0.push_back(op);
            }
          }

          SmallVector<Value> ops1;
          for (int i = 1, k = 0; i < 8; i += 4) {
            for (int j = 1; j < 16; j += 4) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops0[k]);

              k++;
              ops1.push_back(op);
            }
          }

          SmallVector<Value> ops2;
          for (int i = 2, k = 0; i < 8; i += 4) {
            for (int j = 2; j < 16; j += 4) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops1[k]);

              k++;
              ops2.push_back(op);
            }
          }

          for (int i = 3, k = 0; i < 8; i += 4) {
            for (int j = 3; j < 16; j += 4) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops2[k]);

              k++;
              ops.push_back(op);
            }
          }
        }

        // Case 6: A(flat^T) x B(flat^T)
        if (group == 6) {

          SmallVector<Value> ops0;

          SmallVector<Value> matAValue =
              vnniShuffle(rewriter, loc, lhsClone->getResult(0), ipType, opType,
                          64, 0, group);

          SmallVector<Value> matBValue = transposeShuffle(
              rewriter, loc, matB, ipType, opType, mask1, mask2, mask3, 32, 2);

          for (int i = 0, k = 0; i < 4; i++) {
            for (int j = 0; j < 8; j += 4) {
              Value op =
                  createTileMulOutProdOp(rewriter, loc, ipType, matAValue[i],
                                         matBValue[j], iterArgsNewInnerLoop[k]);

              k++;
              ops0.push_back(op);
            }
          }

          SmallVector<Value> ops1;
          matAValue = vnniShuffle(rewriter, loc, lhsClone->getResult(0), ipType,
                                  opType, 64, 2, group);

          for (int i = 0, k = 0; i < 4; i++) {
            for (int j = 1; j < 8; j += 4) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops0[k]);

              k++;
              ops1.push_back(op);
            }
          }

          SmallVector<Value> ops2;
          matAValue = vnniShuffle(rewriter, loc, lhsClone->getResult(0), ipType,
                                  opType, 64, 4, group);

          for (int i = 0, k = 0; i < 4; i++) {
            for (int j = 2; j < 8; j += 4) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops1[k]);

              k++;
              ops2.push_back(op);
            }
          }

          matAValue = vnniShuffle(rewriter, loc, lhsClone->getResult(0), ipType,
                                  opType, 64, 6, group);

          for (int i = 0, k = 0; i < 4; i++) {
            for (int j = 3; j < 8; j += 4) {
              Value op = createTileMulOutProdOp(
                  rewriter, loc, ipType, matAValue[i], matBValue[j], ops2[k]);

              k++;
              ops.push_back(op);
            }
          }
        }

        scf::YieldOp::create(rewriterNewInnerLoop, locNewInnerLoop, ops);
      });

  return newLoop;
}

static int caseValue(vector::ContractionOp contractOp) {

  bool isLhsVnni = false;
  bool isRhsVnni = false;

  bool isLhsTransposed = false;
  bool isRhsTransposed = false;

  auto shapeTypeLhs = dyn_cast<ShapedType>(contractOp.getOperand(0).getType());
  auto shapeTypeRhs = dyn_cast<ShapedType>(contractOp.getOperand(1).getType());

  auto shapeLhs = shapeTypeLhs.getShape();
  auto shapeRhs = shapeTypeRhs.getShape();

  if (shapeLhs[shapeLhs.size() - 1] == 2 || shapeLhs[shapeLhs.size() - 1] == 4)
    isLhsVnni = true;

  if (shapeRhs[shapeRhs.size() - 1] == 2 || shapeRhs[shapeRhs.size() - 1] == 4)
    isRhsVnni = true;

  AffineMap mapA =
      contractOp.getMatchingIndexingMap(&contractOp->getOpOperand(0));
  AffineMap mapB =
      contractOp.getMatchingIndexingMap(&contractOp->getOpOperand(1));
  AffineMap mapAcc =
      contractOp.getMatchingIndexingMap(&contractOp->getOpOperand(2));

  size_t mapAccSizeRes = mapAcc.getResults().size();
  size_t mapASizeRes = mapA.getResults().size();
  size_t mapBSizeRes = mapB.getResults().size();

  if (isLhsVnni) {
    if (mapAcc.getResult(mapAccSizeRes - 2) != mapA.getResult(mapASizeRes - 3))
      isLhsTransposed = true;
  } else {
    if (mapAcc.getResult(mapAccSizeRes - 2) != mapA.getResult(mapASizeRes - 2))
      isLhsTransposed = true;
  }

  if (isRhsVnni) {
    if (mapAcc.getResult(mapAccSizeRes - 1) != mapB.getResult(mapBSizeRes - 2))
      isRhsTransposed = true;
  } else {
    if (mapAcc.getResult(mapAccSizeRes - 1) != mapB.getResult(mapBSizeRes - 1))
      isRhsTransposed = true;
  }

  if (isLhsVnni && isLhsTransposed && isRhsVnni && !isRhsTransposed)
    return 1;

  if (!isLhsVnni && !isLhsTransposed && !isRhsVnni && !isRhsTransposed)
    return 3;

  if (!isLhsVnni && isLhsTransposed && !isRhsVnni && !isRhsTransposed)
    return 4;

  if (!isLhsVnni && !isLhsTransposed && !isRhsVnni && isRhsTransposed)
    return 5;

  if (!isLhsVnni && isLhsTransposed && !isRhsVnni && isRhsTransposed)
    return 6;

  return 0;
}

struct VectorContractToACEOuterProduct
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {

    if (contractOp.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(contractOp,
                                         "Expects add combining kind.");

    unsigned int blockingFactor =
        contractOp.getLhsType().getElementType().isBF16() ? 2 : 4;
    bool isVnni =
        isInVnniLayout(contractOp.getOperation(),
                       contractOp.getIndexingMapsArray(), blockingFactor);

    int group = caseValue(contractOp);

    if (group != 1 && group != 3 && group != 4 && group != 5 && group != 6)
      return rewriter.notifyMatchFailure(contractOp, "Invalid group.");

    VectorType lhsTy = contractOp.getLhsType();
    if (!lhsTy.getElementType().isBF16() &&
        !lhsTy.getElementType().isSignlessInteger(8) &&
        !lhsTy.getElementType().isF8E4M3FN() &&
        !lhsTy.getElementType().isF8E5M2())
      return rewriter.notifyMatchFailure(
          contractOp, "Only BF16/Int8/F8 lowering is supported.");

    if (lhsTy.getElementType() != contractOp.getRhsType().getElementType())
      return rewriter.notifyMatchFailure(
          contractOp, "Contraction should have same lhs and rhs type.");

    VectorType accTy = dyn_cast<VectorType>(contractOp.getAccType());
    if (!accTy)
      return rewriter.notifyMatchFailure(contractOp, "Wrong accmulator type.");

    if (((lhsTy.getElementType().isBF16() ||
          lhsTy.getElementType().isF8E4M3FN() ||
          lhsTy.getElementType().isF8E5M2()) &&
         !accTy.getElementType().isF32()) ||
        (lhsTy.getElementType().isSignlessInteger(8) &&
         !accTy.getElementType().isSignlessInteger(32)))
      return rewriter.notifyMatchFailure(contractOp,
                                         "Only F32 for BF16 or Int32 for Int8 "
                                         "accumulation type is supported.");

    Operation *accReadOp =
        traceToVectorReadLikeParentOperation(contractOp.getAcc());

    Operation *resultWriteOp =
        traceToVectorWriteLikeUserOperation(contractOp.getResult());

    if (!accReadOp || !resultWriteOp)
      return rewriter.notifyMatchFailure(
          contractOp, "The ACC operand of the vector.contract should be a "
                      "transfer_read or a load. And, the result should be "
                      "stored using transfer_write or store.");

    Type ipType = rewriter.getBF16Type();
    Type opType = rewriter.getF32Type();

    if (lhsTy.getElementType().isSignlessInteger(8)) {
      ipType = rewriter.getIntegerType(8);
      opType = rewriter.getIntegerType(32);
    }

    if (lhsTy.getElementType().isF8E4M3FN())
      ipType = rewriter.getF8E4M3FNType();

    if (lhsTy.getElementType().isF8E5M2())
      ipType = rewriter.getF8E5M2Type();

    if (accReadOp->getBlock() == contractOp->getBlock() &&
        resultWriteOp->getBlock() != contractOp->getBlock())
      return rewriter.notifyMatchFailure(
          contractOp, "The accumulator store is in different block.");

    if (accReadOp->getBlock() != contractOp->getBlock() &&
        resultWriteOp->getBlock() == contractOp->getBlock())
      return rewriter.notifyMatchFailure(
          contractOp, "The accumulator read is in different block.");

    Value srcBuffAcc;
    SmallVector<Value> indicesAcc;

    llvm::TypeSwitch<Operation *>(accReadOp).Case<TransferReadOp, LoadOp>(
        [&](auto readOp) {
          srcBuffAcc = readOp.getOperand(0);

          auto indices = readOp.getIndices();
          indicesAcc.reserve(indices.size());

          llvm::transform(indices, std::back_inserter(indicesAcc),
                          [&](OpFoldResult ofr) {
                            return mlir::getValueOrCreateConstantIndexOp(
                                rewriter, contractOp.getLoc(), ofr);
                          });
        });

    auto outputShapes =
        mlir::cast<mlir::MemRefType>(srcBuffAcc.getType()).getShape();
    unsigned int M = outputShapes[outputShapes.size() - 2];
    unsigned int N = outputShapes[outputShapes.size() - 1];

    if ((M != 32 || N != 64) && group != 6)
      return rewriter.notifyMatchFailure(
          contractOp,
          "The M tile size should be 32 and N tile size should be 64.");

    if ((M != 64 || N != 32) && group == 6)
      return rewriter.notifyMatchFailure(
          contractOp,
          "The M tile size should be 64 and N tile size should be 32.");

    unsigned int dimValue = blockingFactor;
    if (group == 3 || group == 2 || group == 5 || group == 6)
      dimValue = 4 * blockingFactor;

    // Case 2: The acc are passed as iter args through the reduction loop.
    // We support, reduction loop depth until 2. TODO: Support for n-depth
    // reduction loop.
    // TODOs: Re-factor 2a and 2b.
    SmallVector<scf::ForOp> loopLists;
    Operation *current = contractOp;
    while (true) {
      Operation *parent = current->getParentOfType<scf::ForOp>();

      if (!parent)
        return rewriter.notifyMatchFailure(
            contractOp,
            "Accumulator read and contract op not within scf.for op");

      loopLists.push_back(dyn_cast<scf::ForOp>(parent));

      if (accReadOp->getBlock() == parent->getBlock()) {
        break;
      }

      current = parent;
    }

    if (loopLists.size() > 2 || loopLists.size() == 0)
      return rewriter.notifyMatchFailure(
          contractOp, "Rewrite is supported until reduction loop depth of 2.");

    auto srcIndxLhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                      contractOp.getLhs(), false);
    if (failed(srcIndxLhs))
      return rewriter.notifyMatchFailure(contractOp,
                                         "The LHS src is not a MemRef type.");
    auto [srcBuffLhs, indicesLhs] = *srcIndxLhs;

    auto srcIndxRhs = getSrcIndxValue(rewriter, contractOp.getLoc(),
                                      contractOp.getRhs(), false);
    if (failed(srcIndxRhs))
      return rewriter.notifyMatchFailure(contractOp,
                                         "The RHS src is not a MemRef type.");

    auto [srcBuffRhs, indicesRhs] = *srcIndxRhs;
    Operation *vectorOpLhs;
    llvm::TypeSwitch<Operation *>(contractOp.getLhs().getDefiningOp())
        .Case<TransferReadOp, LoadOp>([&](auto readOp) {
          vectorOpLhs = readOp.getBase().getDefiningOp();
        });

    Operation *vectorOpRhs;
    llvm::TypeSwitch<Operation *>(contractOp.getRhs().getDefiningOp())
        .Case<TransferReadOp, LoadOp>([&](auto readOp) {
          vectorOpRhs = readOp.getBase().getDefiningOp();
        });

    // Retrive all the contaction operation within the loop.
    SmallVector<vector::ContractionOp> ops;
    for (mlir::Operation &op : loopLists[0].getBody()->getOperations()) {

      if (auto contract = llvm::dyn_cast<mlir::vector::ContractionOp>(op)) {

        LogicalResult validate = validateContractOps(
            rewriter, contract, dimValue, srcBuffLhs, srcBuffRhs, true, ipType);

        if (failed(validate)) {
          return rewriter.notifyMatchFailure(
              contractOp,
              "The associated contract operations doesn't satisfy "
              "the re-write conditions either the dimensions are "
              "wrong or MemRef source are different or many users.");
        }

        ops.push_back(contract);
      }
    }

    SmallVector<vector::ContractionOp> checkOps(ops.begin(), ops.end());
    if (!isVnni) {
      unsigned int pairCount = 0;
      for (size_t j = 0; j < checkOps.size(); j++) {
        if (!checkOps[j])
          continue;
        for (size_t i = j; i < checkOps.size(); i++) {
          if (i != j &&
              validatePairVectorContract(checkOps[j], checkOps[i], true, 16)) {
            checkOps[i] = nullptr;
            pairCount = pairCount + 2;
            continue;
          }
        }
      }

      if (pairCount != ops.size())
        return rewriter.notifyMatchFailure(
            contractOp, "Coudn't find the pair vector contract ");
    }

    scf::ForOp innerLoop;
    scf::ForOp outerLoop;

    scf::ForOp newLoop;

    ModuleOp module = contractOp->getParentOfType<ModuleOp>();
    OpBuilder builder(module.getBodyRegion());

    auto i32Ty = builder.getI32Type();
    auto memrefTy = MemRefType::get({16}, i32Ty);
    auto tensorTy = RankedTensorType::get({16}, i32Ty);

    auto initAttr = DenseIntElementsAttr::get(
        tensorTy,
        ArrayRef<int32_t>{0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0});
    memref::GlobalOp::create(builder, contractOp.getLoc(), "mask_1",
                             builder.getStringAttr("private"), memrefTy,
                             initAttr, true, IntegerAttr());

    auto initAttr2 = DenseIntElementsAttr::get(
        tensorTy,
        ArrayRef<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0});
    memref::GlobalOp::create(builder, contractOp.getLoc(), "mask_2",
                             builder.getStringAttr("private"), memrefTy,
                             initAttr2, true, IntegerAttr());

    auto initAttr3 = DenseIntElementsAttr::get(
        tensorTy,
        ArrayRef<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1});
    memref::GlobalOp::create(builder, contractOp.getLoc(), "mask_3",
                             builder.getStringAttr("private"), memrefTy,
                             initAttr3, true, IntegerAttr());

    // case 2a: Reduction loop depth is 2.
    if (loopLists.size() == 2) {
      outerLoop = loopLists[1];
      innerLoop = loopLists[0];

      SmallVector<Value> loopItrArgs = createTileZeros(
          rewriter, outerLoop.getLoc(), opType, outerLoop, ops.size());

      SmallVector<Value> mask =
          loadMasks(rewriter, innerLoop.getLoc(), memrefTy);

      newLoop = scf::ForOp::create(
          rewriter, outerLoop.getLoc(), outerLoop.getLowerBound(),
          outerLoop.getUpperBound(), outerLoop.getStep(), loopItrArgs,
          [&](OpBuilder &rewriterOuterLoop, Location locOuterLoop,
              Value ivOuterLoop, ValueRange iterArgsOuterLoop) {
            auto newInnerLoop = createLoops(
                rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
                innerLoop.getUpperBound(), innerLoop.getStep(),
                iterArgsOuterLoop, ipType, opType, vectorOpLhs, vectorOpRhs,
                contractOp, outerLoop, innerLoop, ivOuterLoop, mask[0], mask[1],
                mask[2], group, blockingFactor);

            scf::YieldOp::create(rewriterOuterLoop, locOuterLoop,
                                 newInnerLoop.getResults());
          });
    }

    // Case 2b: Reduction loop depth is 1.
    if (loopLists.size() == 1) {

      innerLoop = loopLists[0];

      SmallVector<Value> loopItrArgs = createTileZeros(
          rewriter, innerLoop.getLoc(), opType, innerLoop, ops.size());

      SmallVector<Value> mask =
          loadMasks(rewriter, innerLoop.getLoc(), memrefTy);
      newLoop = createLoops(
          rewriter, innerLoop.getLoc(), innerLoop.getLowerBound(),
          innerLoop.getUpperBound(), innerLoop.getStep(), loopItrArgs, ipType,
          opType, vectorOpLhs, vectorOpRhs, contractOp, nullptr, innerLoop,
          nullptr, mask[0], mask[1], mask[2], group, blockingFactor);

      // This helps the final store back to the acc uses the same code for
      // the both reduction loop depth 1 or 2.
      outerLoop = innerLoop;
    }

    // Write back to the C Matrix

    // Copy the amx tile accumulation results to a MemRef buffer, add the
    // initial accumulation value, and store back to the C-Matrix
    Location loc = outerLoop.getLoc();

    SmallVector<Value> dps = newLoop.getResults();
    auto bufferType = MemRefType::get({16, 16}, opType);
    auto resultBuffer0 = memref::AllocaOp::create(rewriter, loc, bufferType);

    Value c16_i16 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerType(16),
        rewriter.getIntegerAttr(rewriter.getIntegerType(16), 16));
    Value c64_i16 = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerType(16),
        rewriter.getIntegerAttr(rewriter.getIntegerType(16), 64));

    auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto c16 = arith::ConstantIndexOp::create(rewriter, loc, 16);
    auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto one_indx = rewriter.getIndexAttr(1);
    SmallVector<OpFoldResult> strides(indicesAcc.size(), one_indx);
    SmallVector<OpFoldResult> sizes(indicesAcc.size(), one_indx);

    llvm::SmallVector<memref::SubViewOp> subviews;
    sizes[sizes.size() - 1] = rewriter.getIndexAttr(16);
    sizes[sizes.size() - 2] = rewriter.getIndexAttr(16);

    llvm::SmallVector<memref::AllocaOp> allocas;

    for (unsigned int i = 0; i < M; i = i + 16) {
      for (unsigned int j = 0; j < N; j = j + 32) {

        auto offset_i = arith::ConstantIndexOp::create(rewriter, loc, i);
        auto offset_j = arith::ConstantIndexOp::create(rewriter, loc, j);
        auto offset_j16 =
            arith::ConstantIndexOp::create(rewriter, loc, (j + 16));

        indicesAcc[indicesAcc.size() - 1] = offset_j;
        indicesAcc[indicesAcc.size() - 2] = offset_i;

        SmallVector<OpFoldResult> offsets_0 = getAsOpFoldResult(indicesAcc);
        auto subview = memref::SubViewOp::create(rewriter, loc, srcBuffAcc,
                                                 offsets_0, sizes, strides);

        subviews.push_back(subview);

        indicesAcc[indicesAcc.size() - 1] = offset_j16;
        SmallVector<OpFoldResult> offsets_1 = getAsOpFoldResult(indicesAcc);
        auto subview1 = memref::SubViewOp::create(rewriter, loc, srcBuffAcc,
                                                  offsets_1, sizes, strides);

        subviews.push_back(subview1);

        allocas.push_back(memref::AllocaOp::create(rewriter, loc, bufferType));
        allocas.push_back(memref::AllocaOp::create(rewriter, loc, bufferType));
      }
    }

    // Create a loop that iterates over the MxN memerf, retrives two rows +
    // shuffle them, add up the C element values and stores them to temp buffer.
    for (size_t i = 0; i < dps.size(); i = i + 4) {
      scf::ForOp::create(
          rewriter, loc, c0, c16, one, ValueRange{},
          [&](OpBuilder &nestedBuilder, Location loc, Value iv,
              ValueRange iterArgs) {
            Value cast_indx = arith::IndexCastOp::create(
                rewriter, loc, rewriter.getI32Type(), iv);

            auto mov0 = amx::TileMovRowOp::create(
                rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
                c16_i16, c64_i16, dps[i], cast_indx);
            auto row0 = vector::BitCastOp::create(
                rewriter, loc, VectorType::get({16}, opType), mov0);

            auto mov1 = amx::TileMovRowOp::create(
                rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
                c16_i16, c64_i16, dps[i + 1], cast_indx);
            auto row1 = vector::BitCastOp::create(
                rewriter, loc, VectorType::get({16}, opType), mov1);

            auto mov2 = amx::TileMovRowOp::create(
                rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
                c16_i16, c64_i16, dps[i + 2], cast_indx);
            auto row2 = vector::BitCastOp::create(
                rewriter, loc, VectorType::get({16}, opType), mov2);

            auto mov3 = amx::TileMovRowOp::create(
                rewriter, loc, VectorType::get({16}, rewriter.getI32Type()),
                c16_i16, c64_i16, dps[i + 3], cast_indx);
            auto row3 = vector::BitCastOp::create(
                rewriter, loc, VectorType::get({16}, opType), mov3);

            Value shuffle0 = row0;
            Value shuffle1 = row1;

            Value shuffle2 = row2;
            Value shuffle3 = row3;

            if (group == 3 || group == 4) {
              if (ipType.isBF16()) {
                /*shuffle0 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row0, row1,
                    ArrayRef<int64_t>{0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7,
                                      20, 21, 22, 23});

                shuffle1 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row0, row1,
                    ArrayRef<int64_t>{8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14,
                                      15, 28, 29, 30, 31});

                shuffle2 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row2, row3,
                    ArrayRef<int64_t>{0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7,
                                      20, 21, 22, 23});

                shuffle3 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row2, row3,
                    ArrayRef<int64_t>{8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14,
                                      15, 28, 29, 30, 31}); */

                auto a0 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row0, row1,
                    ArrayRef<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19,
                                      20, 21, 22, 23});

                auto a1 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row0, row1,
                    ArrayRef<int64_t>{8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26,
                                      27, 28, 29, 30, 31});

                shuffle0 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), a0, a0,
                    ArrayRef<int64_t>{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12,
                                      13, 14, 15});

                shuffle1 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), a1, a1,
                    ArrayRef<int64_t>{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12,
                                      13, 14, 15});

                auto a2 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row2, row3,
                    ArrayRef<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19,
                                      20, 21, 22, 23});

                auto a3 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row2, row3,
                    ArrayRef<int64_t>{8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26,
                                      27, 28, 29, 30, 31});

                shuffle2 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), a2, a2,
                    ArrayRef<int64_t>{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12,
                                      13, 14, 15});

                shuffle3 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), a3, a3,
                    ArrayRef<int64_t>{0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12,
                                      13, 14, 15});

              } else {

                auto a0 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row0, row1,
                    ArrayRef<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19,
                                      20, 21, 22, 23});

                auto a1 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row0, row1,
                    ArrayRef<int64_t>{8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26,
                                      27, 28, 29, 30, 31});

                auto a2 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row2, row3,
                    ArrayRef<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19,
                                      20, 21, 22, 23});

                auto a3 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), row2, row3,
                    ArrayRef<int64_t>{8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26,
                                      27, 28, 29, 30, 31});

                shuffle0 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), a0, a2,
                    ArrayRef<int64_t>{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19,
                                      24, 25, 26, 27});

                shuffle1 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), a0, a2,
                    ArrayRef<int64_t>{4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22,
                                      23, 28, 29, 30, 31});

                shuffle2 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), a1, a3,
                    ArrayRef<int64_t>{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19,
                                      24, 25, 26, 27});

                shuffle3 = vector::ShuffleOp::create(
                    rewriter, loc, VectorType::get(16, opType), a1, a3,
                    ArrayRef<int64_t>{4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22,
                                      23, 28, 29, 30, 31});
              }
            }

            indicesAcc[indicesAcc.size() - 2] = iv;
            indicesAcc[indicesAcc.size() - 1] = c0;

            Value valueCRow0 = vector::LoadOp::create(
                rewriter, loc, VectorType::get(16, opType), subviews[i],
                indicesAcc);

            Value valueCRow1 = vector::LoadOp::create(
                rewriter, loc, VectorType::get(16, opType), subviews[i + 1],
                indicesAcc);

            Value valueCRow2 = vector::LoadOp::create(
                rewriter, loc, VectorType::get(16, opType), subviews[i + 2],
                indicesAcc);

            Value valueCRow3 = vector::LoadOp::create(
                rewriter, loc, VectorType::get(16, opType), subviews[i + 3],
                indicesAcc);

            Value addOp0;
            Value addOp1;
            Value addOp2;
            Value addOp3;

            if (ipType.isBF16() || ipType.isF8E5M2() || ipType.isF8E4M3FN()) {
              addOp0 =
                  arith::AddFOp::create(rewriter, loc, shuffle0, valueCRow0);

              addOp1 =
                  arith::AddFOp::create(rewriter, loc, shuffle1, valueCRow1);

              addOp2 =
                  arith::AddFOp::create(rewriter, loc, shuffle2, valueCRow2);

              addOp3 =
                  arith::AddFOp::create(rewriter, loc, shuffle3, valueCRow3);
            }

            if (ipType.isSignlessInteger(8)) {
              addOp0 =
                  arith::AddIOp::create(rewriter, loc, shuffle0, valueCRow0);

              addOp1 =
                  arith::AddIOp::create(rewriter, loc, shuffle1, valueCRow1);

              addOp2 =
                  arith::AddIOp::create(rewriter, loc, shuffle2, valueCRow2);

              addOp3 =
                  arith::AddIOp::create(rewriter, loc, shuffle3, valueCRow3);
            }

            vector::StoreOp::create(rewriter, loc, addOp0, allocas[i],
                                    ValueRange{iv, c0});
            vector::StoreOp::create(rewriter, loc, addOp1, allocas[i + 1],
                                    ValueRange{iv, c0});

            vector::StoreOp::create(rewriter, loc, addOp2, allocas[i + 2],
                                    ValueRange{iv, c0});
            vector::StoreOp::create(rewriter, loc, addOp3, allocas[i + 3],
                                    ValueRange{iv, c0});

            scf::YieldOp::create(nestedBuilder, loc);
          });
      auto vectorType = mlir::VectorType::get({16, 16}, opType);

      int64_t srcRank =
          (dyn_cast<ShapedType>(resultBuffer0.getType())).getRank();
      Value padding = ub::PoisonOp::create(rewriter, loc, opType);
      auto map = AffineMap::getMinorIdentityMap(srcRank, vectorType.getRank(),
                                                rewriter.getContext());
      SmallVector<bool> inBounds(vectorType.getRank(), true);

      auto vec0 = vector::TransferReadOp::create(rewriter, loc, vectorType,
                                                 allocas[i], ValueRange{c0, c0},
                                                 padding, map, inBounds);

      auto vec1 = vector::TransferReadOp::create(
          rewriter, loc, vectorType, allocas[i + 1], ValueRange{c0, c0},
          padding, map, inBounds);

      auto vec2 = vector::TransferReadOp::create(
          rewriter, loc, vectorType, allocas[i + 2], ValueRange{c0, c0},
          padding, map, inBounds);

      auto vec3 = vector::TransferReadOp::create(
          rewriter, loc, vectorType, allocas[i + 3], ValueRange{c0, c0},
          padding, map, inBounds);

      Value resultWriteOp = contractionUsersAfterYield(ops[i].getResult());
      if (auto vecType = llvm::dyn_cast<VectorType>(resultWriteOp.getType())) {
        auto vecRow =
            mlir::vector::ShapeCastOp::create(rewriter, loc, vecType, vec0);
        rewriter.replaceAllUsesWith(resultWriteOp, vecRow);
      }

      Value resultWriteOp1 = contractionUsersAfterYield(ops[i + 1].getResult());
      if (auto vecType = llvm::dyn_cast<VectorType>(resultWriteOp1.getType())) {
        auto vecRow =
            mlir::vector::ShapeCastOp::create(rewriter, loc, vecType, vec1);
        rewriter.replaceAllUsesWith(resultWriteOp1, vecRow);
      }

      Value resultWriteOp2 = contractionUsersAfterYield(ops[i + 2].getResult());
      if (auto vecType = llvm::dyn_cast<VectorType>(resultWriteOp2.getType())) {
        auto vecRow =
            mlir::vector::ShapeCastOp::create(rewriter, loc, vecType, vec2);
        rewriter.replaceAllUsesWith(resultWriteOp2, vecRow);
      }

      Value resultWriteOp3 = contractionUsersAfterYield(ops[i + 3].getResult());
      if (auto vecType = llvm::dyn_cast<VectorType>(resultWriteOp3.getType())) {
        auto vecRow =
            mlir::vector::ShapeCastOp::create(rewriter, loc, vecType, vec3);
        rewriter.replaceAllUsesWith(resultWriteOp3, vecRow);
      }
    }
    return success();
  }
};

} // namespace

void x86::populateVectorContractToACEOuterProductPatterns(
    RewritePatternSet &patterns) {
  patterns.add<VectorContractToACEOuterProduct>(patterns.getContext());
}
