//===- XsmmOps.td - Xsmm dialect ops ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_XSMM_OPS
#define TPP_XSMM_OPS

include "TPP/Dialect/Xsmm/XsmmDialect.td"
include "TPP/Dialect/Xsmm/XsmmEnum.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class StaticMemRefRankOf<list<Type> allowedTypes, list<int> ranks> :
    Type<And<[MemRefOf<allowedTypes>.predicate,
              HasAnyRankOfPred<ranks>, HasStaticShapePred]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " static " #
         MemRefOf<allowedTypes>.summary,
         "::mlir::MemRefType">;

def XsmmMemRef : AnyTypeOf<[StaticMemRefRankOf<[F32, BF16, F8E5M2, F8E4M3FN], [1, 2, 3, 4]>,
                            F32, BF16, F8E5M2, F8E4M3FN, I64]>;

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

def Xsmm_BinaryOp : Xsmm_Op<"binary", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "binary operation.";
  let description = [{
    Binary operation. See description for Xsmm_TernaryCallOp. The only
    difference is the number of operands for the computation is restricted to
    two.
  }];

  let arguments = (ins Xsmm_DataType:$data_type, Xsmm_BinaryKind:$callee,
                       Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `data_type` `=` $data_type `,` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    Value getDispatch() { return getInputs()[0]; }

    bool hasScalarLhs() {
      Type operand = getInputs()[1].getType();
      return !isa<ShapedType>(operand);
    }

    bool hasScalarRhs() {
      Type operand = getInputs()[2].getType();
      return !isa<ShapedType>(operand);
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

def Xsmm_UnaryOp : Xsmm_Op<"unary", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "unary call operation.";
  let description = [{
    Binary operation. See description for Xsmm_TernaryCallOp. The only
    difference is the number of operands for the computation is restricted to one.
  }];

  let arguments = (ins Xsmm_DataType:$data_type, Xsmm_UnaryKind:$callee,
                       Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    $callee `(` `data_type` `=` $data_type `,` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    Value getDispatch() { return getInputs()[0]; }

    bool hasScalarInput() {
      Type operand = getInputs()[1].getType();
      return !isa<ShapedType>(operand);
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===//

def GemmMemRef : AnyTypeOf<[StaticMemRefRankOf<[F32, BF16, F8E5M2, F8E4M3FN], [2, 3]>, I64]>;

def Xsmm_GemmOp : Xsmm_Op<"gemm", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "matmul call operation.";
  let arguments = (ins Xsmm_DataType:$data_type,
                       OptionalAttr<Xsmm_DataType>:$out_type,
                       Variadic<GemmMemRef>:$inputs);

  let assemblyFormat = [{
    `(` `data_type` `=` $data_type `,` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    Value getDispatch() { return getInputs()[0]; }

    Value getOperandA() { return getInputs()[1]; }

    Value getOperandB() { return getInputs()[2]; }

    Value getOutput() { return getInputs()[3]; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BrgemmOp
//===----------------------------------------------------------------------===//

def BrgemmMemRef : AnyTypeOf<[StaticMemRefRankOf<[F32, BF16, F8E5M2, F8E4M3FN], [2, 3, 4]>, I64]>;

def Xsmm_BrgemmOp : Xsmm_Op<"brgemm", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "brgemm call operation.";
  let arguments = (ins Xsmm_DataType:$data_type,
                       OptionalAttr<Xsmm_DataType>:$out_type,
                       Variadic<BrgemmMemRef>:$inputs);

  let assemblyFormat = [{
    `(` `data_type` `=` $data_type `,` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    Value getDispatch() { return getInputs()[0]; }

    Value getOperandA() { return getInputs()[1]; }

    Value getOperandB() { return getInputs()[2]; }

    Value getOutput() { return getInputs()[3]; }

    Value getBatch() { return getInputs()[4]; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FusedBrgemmOp
//===----------------------------------------------------------------------===//

def Xsmm_FusedBrgemmOp : Xsmm_Op<"fused_brgemm", 
                         [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "fused brgemm call operation.";
  let arguments = (ins Xsmm_DataType:$data_type, Variadic<XsmmMemRef>:$inputs);

  let assemblyFormat = [{
    `(` `data_type` `=` $data_type `,` $inputs `)`
    attr-dict `:` functional-type($inputs, results)
  }];

  let extraClassDeclaration = [{
    Value getDispatch() { return getInputs()[0]; }

    Value getOperandA() { return getInputs()[1]; }

    Value getOperandB() { return getInputs()[2]; }

    Value getOutput() { return getInputs()[3]; }

    Value getOperandD() { return getInputs()[4]; };

    Value getBatch() { return getInputs()[5]; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BinaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_BinaryDispatchOp : Xsmm_Op<"binary.dispatch", [Pure]> {
  let summary = "dispatch binary operation.";
  let description = [{
    See 'ternary.dispatch'.
  }];

  let arguments = (ins
    Xsmm_BinaryKind:$kind,
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    TypedArrayAttrBase<Xsmm_BinaryFlags, "binary flags">:$flags,
    Xsmm_DataType:$data_type);

  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// UnaryDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_UnaryDispatchOp : Xsmm_Op<"unary.dispatch", [Pure]> {
  let summary = "dispatch unary operation.";
  let description = [{
    See 'ternary.dispatch'.
  }];

  let arguments = (ins
    Xsmm_UnaryKind:$kind,
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    TypedArrayAttrBase<Xsmm_UnaryFlags, "unary flags">:$flags,
    Xsmm_DataType:$data_type);

  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GemmDispatchOp
//===----------------------------------------------------------------------===//

class Xsmm_GemmLikeOp<string mnemonic, list<Trait> traits = []> :
  Xsmm_Op<mnemonic, !listconcat(traits, [Pure])> {
  let description = [{
    Base class for 'gemm.dispatch' and 'brgemm.dispatch'. The operation has
    the following arguments: 1) inputs carry information on leading dimensions and
    sizes; for example,  in 'matmul.dispatch' the inputs are m, n, k, lda, ldb and
    ldc. Inputs is a dense attribute of I64 elements. 2) flags carry information on
    the different flags that can be used for matmul and brgemm (i.e., VNNI_B). For
    more details, see: `Xsmm_GemmFlags`.
  }];

  let arguments = (ins
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    TypedArrayAttrBase<Xsmm_GemmFlags, "gemm flags">:$flags,
    Xsmm_DataType:$data_type,
    OptionalAttr<Xsmm_DataType>:$out_type);

  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;
}

def Xsmm_GemmDispatchOp : Xsmm_GemmLikeOp<"gemm.dispatch"> {
  let summary = "dispatch for matmul operation.";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BrgemmDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_BrgemmDispatchOp : Xsmm_GemmLikeOp<"brgemm.dispatch"> {
  let summary = "dispatch for brgemm operation.";
  let hasVerifier = 1;
  
}

//===----------------------------------------------------------------------===//
// IntelAMXTileConfigDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_IntelAMXTileConfigDispatchOp : Xsmm_GemmLikeOp<"IntelAMXtileConfig.dispatch"> {
  let summary = "dispatch for Intel amx tileConfig operation.";
}

//===----------------------------------------------------------------------===//
// FusedBrgemmDispatchOp
//===----------------------------------------------------------------------===//

def Xsmm_FusedBrgemmDispatchOp : Xsmm_Op<"fused_brgemm.dispatch", [Pure]> {
  let summary = "dispatch fused brgemm operation.";
  let description = [{
    Implements C = unary(binary(BRGEMM(A, B), D)). The operation has the
    following arguments: 1) inputs carry information on leading dimensions and
    sizes; Inputs is a dense attribute of I64 elements. 2) `binary_kind` and
    `unary_kind` to represent the kind of unary and binary to invoke, respectively.
    3) `flags` carry the flags associated with the brgemm operation (i.e., beta 0
    or 1). `unary_flags` and `binary_flags` are the flags associated with the unary
    and binary, respectively.
  }];


  let arguments = (ins
    ConfinedAttr<DenseI64ArrayAttr,
                [DenseArrayNonNegative<DenseI64ArrayAttr>]>:$inputs,
    Xsmm_BinaryKind:$binary_kind,
    Xsmm_UnaryKind:$unary_kind,
    TypedArrayAttrBase<Xsmm_GemmFlags, "gemm flags">:$flags,
    TypedArrayAttrBase<Xsmm_UnaryFlags, "unary flags">:$unary_flags,
    TypedArrayAttrBase<Xsmm_BinaryFlags, "binary flags">:$binary_flags,
    Xsmm_DataType:$data_type);

  let results = (outs I64:$results);
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// IntelAMXTileConfigOp
//===----------------------------------------------------------------------===//

def Xsmm_IntelAMXTileConfigOp : Xsmm_Op<"IntelAMXtileConfig",  [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "invoke for Intel AMX tileConfig operation.";
  let arguments = (ins I64:$dispatch, Variadic<AnyMemRef>:$inputs);
}

#endif // TPP_XSMM_OPS
