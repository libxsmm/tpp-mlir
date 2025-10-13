# -*- Python -*-

import os
import lit.formats
import lit.util

from lit.llvm import llvm_config

# from lit.llvm.subst import ToolSubst
# from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "TPP_OPT"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tpp_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))
config.substitutions.append(("%llvmlibdir", config.llvm_lib_dir))
config.substitutions.append(("%tpplibdir", config.tpp_obj_root + "/lib/"))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = []

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.tpp_obj_root, "test")
config.tpp_tools_dir = os.path.join(config.tpp_obj_root, "bin")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.tpp_tools_dir, config.llvm_tools_dir]
tools = ["mlir-gen", "tpp-opt", "tpp-run", "fpcmp", "tpp-sched"]

# Define '*-registered-target' feature for each target for 'REQUIRES' directive
# to work as expected.
config.targets = frozenset(config.targets_to_build.split())

for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower() + "-registered-target")

llvm_config.add_tool_substitutions(tools, tool_dirs)
