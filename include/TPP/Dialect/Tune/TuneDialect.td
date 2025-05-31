//===- TuneDialect.td - Tune dialect ----------------------*- tablegen -*--===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_TUNE_DIALECT
#define TPP_TUNE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tune dialect definition.
//===----------------------------------------------------------------------===//

def Tune_Dialect : Dialect {
    let name = "tune";
    let summary = "Tune dialect.";
    let description = [{
        TODO
    }];
    let cppNamespace = "::mlir::tune";
    let usePropertiesForAttributes = 1;
}

//===----------------------------------------------------------------------===//
// Base operation definition.
//===----------------------------------------------------------------------===//

class Tune_Op<string mnemonic, list<Trait> traits = []> :
        Op<Tune_Dialect, mnemonic, traits>;

#endif // TPP_TUNE_DIALECT
