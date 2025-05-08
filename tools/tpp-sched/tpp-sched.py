#!/usr/bin/env python3

import sys
from pathlib import Path


# Enable automagically finding TPP-MLIR's python modules (which include
# and extend MLIR's Python bindings).
python_packages_path = Path(__file__).parent.parent / "python_packages"
if python_packages_path.exists():
    sys.path = [str(python_packages_path)] + sys.path


from mlir.tpp.sched import main


main.main(sys.argv[1:])
