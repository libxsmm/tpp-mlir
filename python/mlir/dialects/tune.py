from .._mlir_libs._tppDialects.tune import *

from .._mlir_libs import get_dialect_registry
register_dialect(get_dialect_registry())
