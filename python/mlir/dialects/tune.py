from .._mlir_libs import get_dialect_registry
from .._mlir_libs._tppDialects.tune import register_dialect

register_dialect(get_dialect_registry())
