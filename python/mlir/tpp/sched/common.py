from mlir.dialects import transform
from mlir.dialects.transform import structured, tune


# Wrapper to addresss verbosity.
def apply_registered_pass(*args, **kwargs):
    return transform.apply_registered_pass(transform.AnyOpType.get(), *args, **kwargs)


# Wrapper to addresss verbosity.
def select(*args, **kwargs):
    return tune.select(transform.AnyParamType.get(), *args, **kwargs)


# Wrapper to addresss verbosity.
def match(*args, **kwargs):
    return structured.MatchOp(transform.AnyOpType.get(), *args, **kwargs)
