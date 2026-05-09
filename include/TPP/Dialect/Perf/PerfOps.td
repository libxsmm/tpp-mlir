//===- PerfOps.td - Perf dialect ops -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TPP_PERF_OPS
#define TPP_PERF_OPS

include "TPP/Dialect/Perf/PerfDialect.td"
include "TPP/Dialect/Perf/PerfTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// StartTimerOp
//===----------------------------------------------------------------------===//

def Perf_StartTimerOp : Perf_Op<"start_timer", []> {
  let summary = "Start a timer.";
  let description = [{
    The `perf.start_timer` operation creates a new unique timer
    which begins time measurement.

    See `perf.stop_timer` for timer termination.

    Example:

    ```mlir

    %timer = perf.start_timer : !perf.timer
    ... // ops under measurement

    ```
  }];

  let arguments = (ins);
  let results = (outs Perf_TimerType:$timer);

  let assemblyFormat = [{
    attr-dict `:` type($timer)
  }];

  let extraClassDeclaration = [{
    static std::string getLibraryCallName() {
      return "perf_start_timer";
    }
  }];
}

//===----------------------------------------------------------------------===//
// StopTimerOp
//===----------------------------------------------------------------------===//

def Perf_StopTimerOp : Perf_Op<"stop_timer", []> {
  let summary = "Stops a timer.";
  let description = [{
    The `perf.stop_timer` operation stops the specified
    timer and returns elapsed time delta.
    Once a timer is stopped, it cannot be used again.

    See `perf.start_timer` for timer creation.

    Example:

    ```mlir

    %timer = perf.start_timer : !perf.timer
    ... // ops under measurement
    %delta = perf.stop_timer(%timer : !perf.timer) : f64

    ```
  }];

  let arguments = (ins Perf_TimerType:$timer);
  let results = (outs F64:$delta);

  let assemblyFormat = [{
    `(` $timer `:` type($timer) `)` attr-dict
    `:` type($delta)
  }];

  let extraClassDeclaration = [{
    static std::string getLibraryCallName() {
      return "perf_stop_timer";
    }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// BenchOp
//===----------------------------------------------------------------------===//

def Perf_BenchOp : Perf_Op<"bench",
    [AutomaticAllocationScope, SingleBlockImplicitTerminator<"perf::YieldOp">,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface,
     ["getEntrySuccessorOperands", "getSuccessorInputs"]>,
     RangedTypesMatchWith<"iter_args types match types of yield",
                          "$_self",
                          "iterArgs",
                          "getRegion().front().getTerminator()->getOperandTypes()">]> {
  let summary = "Benchmark the enclosed code.";
  let description = [{
    The `perf.bench` operation generates benchmarking code
    around the enclosed region.

    The performance results are gathered over the specified
    number of iterations and returned as an f64.

    If no extra return values are defined, then the yield terminator may be
    left out in the custom syntax and the builders will insert one implicitly.

    For example - a simple benchmark:

    ```mlir
    %delta = perf.bench (%n : i64) -> f64 {
      ... // body - ops under measurement
    }
    ```

    `perf.bench` also accepts optional arguments. The initial argument values
    are bound to the benchmarked region and the their values are carried over
    between benchmarking iterations. On the last iteration, their final values
    are passed to the `perf.bench` results.
    The types of the arguments and the results must match, and the benchmarking
    region must be terminated with a matching `perf.yield` operation.

    When arguments are present, the region must terminate with `perf.yield`.
    Note, that in this case, calling BenchOp::build will not insert the terminator
    implicitly. The caller must insert `perf.yield` separately with intended
    return values.

    An example of a benchmark with an output result:

    ```mlir
    %delta, %sum = perf.bench (%n : i64) iter_args(%arg0 = %val) -> (f64, i32) {
      %sum_next = arith.addi %arg0, %cst : i32
      perf.yield %sum_next : i32
    } -> i32
    ```

    `perf.bench` is essentially a utility operation that generates
    a benchmarking loop.
    For example, the following input:
    ```mlir
    %delta, %res, ... = perf.bench (%n : i64) iter_args(...) -> ... {
      ... // body - ops under measurement

      // Yield current iteration values to the next iteration iter_args (%x, ...)
      // or to the bench op results (%res, ...) if it is the final iteration.
      perf.yield %x, ...
    }
    ```
    is materialized as:
    ```mlir
    %timer = perf.timer_start() : i64
    %res, ... = loop %iv from 0 to %n step 1
        iter_args(%x, ...) -> type(%res), ... {
      ... // body - ops under measurement
      yield %x, ...
    }
    %delta = perf.stop_timer(%timer) : f64
    ```
  }];

  let arguments = (ins I64:$numIters, Variadic<AnyType>:$iterArgs);
  let results = (outs Variadic<AnyType>:$bodyResults);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$numIters,
      CArg<"ValueRange", "{}">:$iterArgs)>
  ];

  let extraClassDeclaration = [{
    YieldOp getYieldOp();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def Perf_YieldOp : Perf_Op<"yield", [ HasParent<"BenchOp">,
              Pure, Terminator, ReturnLike]> {
  let summary = "Yield values to parent operation.";
  let description = [{
    The `perf.yield` operation yields an SSA value from the perf dialect op
    region and terminates the regions. The semantics of how the values are
    yielded is defined by the parent operation.
    If `perf.yield` has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the `perf.yield` may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// SinkOp
//===----------------------------------------------------------------------===//

def Perf_SinkOp : Perf_Op<"sink", [ConditionallySpeculatable]> {
  let summary = "Prevent removal of unused values.";
  let description = [{
    The `perf.sink` operation acts as a dummy, not speculatable
    operation that prevents removal of unused values and their defining ops
    by optimization passes.

    Example:

    ```mlir

    perf.bench (%n, %deltas: i64, memref<?xf64>) {
      %a = arith.addi %b, %c : i64
      perf.sink(%a) : i64 // make sure that 'addi' is not optimized away
    }

    ```
  }];

  let arguments = (ins AnyType:$input);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input)
  }];

  let extraClassDeclaration = [{
    static std::string applyTypeMangling(std::string name, Type type);

    std::string getLibraryCallName() {
      return applyTypeMangling("perf_sink", getInput().getType());
    }

    ::mlir::Speculation::Speculatability getSpeculatability() {
      return ::mlir::Speculation::NotSpeculatable;
    }
  }];
}

#endif // TPP_PERF_OPS
