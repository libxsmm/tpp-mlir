
#!/usr/bin/env python

from pathlib import Path
import re
import sys


python_packages_path = Path(__file__).parent.parent / "python_packages"
if python_packages_path.exists():
    sys.path.append(str(python_packages_path))

cmake_cache_path = Path(__file__).parent.parent / "CMakeCache.txt"
if cmake_cache_path.exists():
    cmake_cache = cmake_cache_path.open("r").read()
    match = re.search(r"MLIR_DIR:UNINITIALIZED=(.*)/lib/cmake/mlir", cmake_cache)
    if match and (mlir_build_dir := match.group(1)):
        mlir_python_lib_path = (
            Path(mlir_build_dir) / "tools/mlir/python_packages/mlir_core"
        )
        if mlir_python_lib_path.exists():
            sys.path.append(str(mlir_python_lib_path))


from mlir_tpp.gen.__main__ import main

print(main(sys.argv[1:]))
