from ._xsmm_ops_gen import *
from .._mlir_libs._tppDialects.xsmm import *

from .._mlir_libs import get_dialect_registry
register_dialect(get_dialect_registry())
