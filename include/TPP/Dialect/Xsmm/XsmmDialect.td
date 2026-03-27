//===- XsmmDialect.td - Xsmm dialect ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_XSMM_DIALECT
#define TPP_XSMM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Xsmm dialect definition.
//===----------------------------------------------------------------------===//

def Xsmm_Dialect : Dialect {
    let name = "xsmm";
    let summary = "Xsmm dialect.";
    let description = [{
        see: https://github.com/libxsmm/libxsmm
    }];
    let cppNamespace = "::mlir::xsmm";
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Xsmm_Op<string mnemonic, list<Trait> traits = []> :
        Op<Xsmm_Dialect, mnemonic, traits>;

#endif // TPP_XSMM_DIALECT
