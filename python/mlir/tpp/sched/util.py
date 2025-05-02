from enum import Enum

GpuBackend = Enum("GpuBackend", [("intel", "intel"), ("cuda", "cuda")])


# Raise this exception during transform ops generation to early stop schedule generation.
class PipelineInterrupt(Exception):
    pass
