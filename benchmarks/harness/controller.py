#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    TPP-MLIR Benchmark Controller

    Runs an MLIR kernel multiple times and takes the statistics, comparing to
    a known-good result, checking if any difference is statistically significant.

    Arguments:
     * filename: kernel to run on
     * -n N: Number of times to run (default 1000)
     * -e ENTRY: Entry point name (default "entry")

    Like LLVM's FileCheck, we try to get information from comments on the MLIR
    file (//).

    We can look at RUN lines, for details on how to process the file, like entry
    point, tpp-opt args.

    We also support new ones:
     * BENCH_TOTAL_FLOPS: Number of FP ops in the kernel

    Much of the code was shamelessly stolen from my previous harness:
    https://github.com/Linaro/benchmark_harness/
"""

import os
import sys
import re
import shlex
import argparse
import io

from Logger import Logger
from Execute import Execute
from FileCheckParser import FileCheckParser
from TPPHelper import TPPHelper


class BenchmarkController(object):
    """Entry point of the benchmark harness"""

    def __init__(self, args, loglevel):
        self.args = args
        self.logger = Logger("controller.bench.controller", loglevel)
        self.helper = TPPHelper(loglevel)
        self.loglevel = loglevel
        # If we're in a git repo, find the base dir, otherwise, this is the base dir
        self.baseDir = self.helper.findGitRoot(os.path.dirname(__file__))
        self.logger.debug(f"Base dir: {self.baseDir}")
        self.build_dir = args.build
        if not self.build_dir:
            self.build_dir = self.baseDir
        self.programs = self.helper.findTPPProgs(self.build_dir)
        self.output = ""
        self.mean = 0.0
        # Output is always in seconds, we need to convert anyway
        self.unit = "ms"  # or 'gflops'
        self.benchmark = self._read_input(args.benchmark)

    def _read_input(self, input):
        """Reads an input and returns its contents"""

        if isinstance(input, io.IOBase):
            data = input.read()
            input.close()
            return data

        try:
            with open(input) as file:
                return file.read()
        except IOError as err:
            self.logger.error(f"Cannot open file '{input}': {err.strerror}")
            return {}
        except Exception as err:
            self.logger.error(
                f"Uncaught error while parsing file: {err.strerror}"
            )
            return {}

    def verifyArgs(self):
        """Verify cmd-line and IR file arguments, update defaults, etc"""

        # Parse the IR file for user arguments
        self.logger.info("Parsing FileCheck lines, updating arguments")
        fileArgs = FileCheckParser(self.loglevel).parse(self.benchmark)

        # Command line arguments have preference
        if not self.args.n and "iters" in fileArgs:
            self.args.n = fileArgs["iters"]
        if not self.args.entry and "entry" in fileArgs:
            self.args.entry = fileArgs["entry"]
        if not self.args.flops and "flops" in fileArgs:
            self.args.flops = fileArgs["flops"]
            self.unit = "gflops"
        if not self.args.opt_args and "opt-args" in fileArgs:
            self.args.opt_args = fileArgs["opt-args"]
        if not self.args.run_args and "run-args" in fileArgs:
            self.args.run_args = fileArgs["run-args"]

        # Make sure we get all we need
        self.logger.info("Validating arguments")
        if not self.args.n:
            logger.warning("Number of iterations not found, default to 1000")
            self.args.n = 1000

        if not self.args.entry:
            logger.error("No entry point defined, bailing")
            return False

        return True

    def run(self):
        """Run tpp-opt and tpp-run to get the timings"""

        irContents = ""
        executor = Execute(self.loglevel)

        # Only run tpp-opt if we have the arguments
        if self.args.opt_args:
            self.logger.info("Running optimiser, to prepare the IR file")
            optCmd = [self.programs["tpp-opt"]]
            optCmd.extend(shlex.split(self.args.opt_args))

            # Run tpp-opt and capture the output IR
            optResult = executor.run(optCmd, input=self.benchmark)
            if 0 != optResult.returncode:
                self.logger.error(
                    f"Error executing tpp-opt: {optResult.stderr}"
                )
                return False
            irContents = optResult.stdout
        else:
            # Bypass tpp-opt and just dump the file
            irContents = self.benchmark

        # Actually run the file in benchmark mode, no output
        self.logger.info("Running the kernel with the arguments provided")
        runCmd = [
            self.programs["tpp-run"],
            "-n",
            str(self.args.n),
            "-e",
            self.args.entry,
            "--entry-point-result=void",
            "--print=0",
        ]
        if self.args.seed is not None:
            runCmd.extend(["--seed", str(self.args.seed)])
        if self.args.splat_to_random != 0:
            runCmd.append("--splat-to-random")
        if self.args.init_type:
            runCmd.extend(["--init-type", self.args.init_type])
        if self.args.run_args:
            runCmd.extend(shlex.split(self.args.run_args.replace("'", "")))
        runResult = executor.run(runCmd, irContents)
        if 0 != runResult.returncode:
            self.logger.error(f"Error executing tpp-run: {runResult.stderr}")
            return False
        self.output = runResult.stdout

        return True

    def verifyStats(self):
        """Verify the results, should be in format 'mean'"""

        if not self.output:
            self.logger.error(
                "Benchmark produced no output, can't verify results"
            )
            return False

        # Parse results (always in seconds, as per timer)
        m = re.search(r"([\d\.\-e]+)", self.output)
        if m:
            self.mean = float(m.group(1))
            self.logger.info(
                f"Mean time: {self.mean*1000} ms"
            )
        else:
            self.logger.error("Cannot find mean in output")
            return False

        # If we asked for flops, we need to convert
        if self.args.flops:
            mean = self.args.flops / self.mean
            self.mean = mean
            # We annotate in flops (easier to calculate / compare) but we display in Gflops
            self.mean /= 1e9
        else:
            # Output is in seconds, but we display in milliseconds
            self.mean *= 1000

        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPP-MLIR Benchmark Harness")

    # Required argument: benchmark name (can be a file, a directory, or stdin)
    parser.add_argument(
        "benchmark",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="MLIR file or directory containing MLIR files",
    )

    # Required, but auto-detected if omitted
    parser.add_argument(
        "-n",
        type=int,
        help="Number of times to execute the kernel (checks RUN line)",
    )
    parser.add_argument(
        "-flops",
        type=float,
        help="Known number of FP OPs (checks BENCH_TOTAL_FLOPS line)",
    )
    parser.add_argument(
        "-entry", type=str, help="Name of the entry point (checks RUN line)"
    )
    parser.add_argument(
        "-opt-args", type=str, help="tpp-opt arguments (checks RUN line)"
    )
    parser.add_argument("-run-args", type=str, help="tpp-run arguments")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="The verbosity of logging output",
    )
    parser.add_argument(
        "-q", "--quiet", action="count", default=0, help="Suppress warnings"
    )
    parser.add_argument(
        "--disable-lsan", action="count", default=0, help="Disable LSAN"
    )
    parser.add_argument(
        "--build", type=str, default="", help="Path to the build dir"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--splat-to-random",
        type=int,
        default=1,
        help="Replace splat dense tensors with random value (default: enabled)",
    )
    parser.add_argument(
        "--linalg-to-xsmm",
        type=int,
        default=0,
        help="Directly lower linalg to xsmm dialect",
    )
    parser.add_argument(
        "--init-type",
        type=str,
        default="normal",
        help="Random initializer type (default: normal)",
    )
    args = parser.parse_args()

    # List of ASAN_OPTIONS
    asan_options = (
        [os.getenv("ASAN_OPTIONS")] if os.getenv("ASAN_OPTIONS") else []
    )

    # Some tensors may not be freed but we still want numbers
    if args.disable_lsan:
        asan_options.append("detect_leaks=0")

    # Apply ASAN_OPTIONS to environment
    if asan_options:
        os.environ["ASAN_OPTIONS"] = ":".join(asan_options)

    # Creates the logger object
    loglevel = args.verbose - (args.quiet > 0)
    logger = Logger("controller", loglevel)

    # Creates a controller from command line arguments
    controller = BenchmarkController(args, loglevel)

    # Checks all parameters are good
    if not controller.verifyArgs():
        logger.error("Argument verification error")
        print("\n\n")
        parser.print_help()
        sys.exit(1)

    # Runs the benchmark
    if not controller.run():
        logger.error("Error executing the benchmark")
        sys.exit(1)

    # Checks stats
    if not controller.verifyStats():
        logger.error("Error verifying the statistics")
        sys.exit(1)

    # Success prints basic stats
    if args.flops:
        print(f"{(controller.mean):9.3f} {controller.unit}")
    else:
        print(f"{(controller.mean):3.9f} {controller.unit}")
