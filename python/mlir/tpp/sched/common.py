from mlir.dialects import transform
from mlir.dialects.transform import structured


# Wrapper to addresss verbosity.
def apply_registered_pass(*args, **kwargs):
    return transform.ApplyRegisteredPassOp(transform.AnyOpType.get(), *args, **kwargs)


# Wrapper to addresss verbosity.
def match(*args, **kwargs):
    return structured.MatchOp(transform.AnyOpType.get(), *args, **kwargs)
