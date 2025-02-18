# TPP Runner

This is basically a copy of `mlir-runner`, using `JitRunnerMain` and its call backs.

The main difference is that we add a wrapper function to call the kernel (entry) function to allow for benchmarking.

## Functionality

The main function of this modified TPP runner is to:
 * Find the kernel function and:
   * Discover its arguments (input and output)
   * Allocate memrefs for all and random-initialise the inputs
 * Compile the kernel MLIR module with `main`, `return` and kernel functions
 * Run the main function, gathering the outputs (tensors)
 * Run the kernel function, passing those tensors as arguments
   * Run multiple times and output statistics in benchmark mode
 * Run the `return` function to print/cleanup the results

Note: We're using a new benchmark harness (in `benchmarks/harness`) to discover paths, args, etc.

## Implementation

First approach is to use the existing callbacks to wrap our needed functionality:
 * `mlirTransformer` for parsing the kernel's arguments, preparing the input and calling the kernel, printing the results
 * `llvmModuleBuilder` for finding the libraries and making sure the IR is correct

We'll probably need a way to measure execution of the kernel, not the tensor preparation and print. 
This could be done in IR (via `mlirTransformer`) or via some other callback.

If we add new callbacks, we must upstream this.

## Entry Point

Just like `mlir-runner`, `tpp-run` is supposed to work with `tpp-opt`, `mlir-opt`, etc.
However, it also introduces MLIR functions, so it has some internal passes to convert those to LLVM, and it requires the original functions *not* to be in the LLVM Dialect.

For these reasons, the entry point of `tpp-run` is _"after all code-gen passes of the optimizer"_ and _"just before the first LLVM lowering"_.

So, if in `mlir-opt` you'd pass LLVM lowering flags to run on `mlir-runner`, with `tpp-opt`, you cannot.
All other passes, however, even including partial conversions (ex. `scf-to-cf`) need to be passed, as we can't assume what the original IR had used.

This may change in the future when the program gets more complex, but for now, it's a safe point.
