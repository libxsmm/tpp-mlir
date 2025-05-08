import ctypes

from pathlib import Path


utils_lib = None
# Automagically find TPP-MLIR's "runtime" library.
utils_so = (
    Path(__file__).parent.parent.parent.parent.parent
    / "lib/libtpp_xsmm_runner_utils.so"
)
if utils_so.exists():
    utils_lib  = ctypes.cdll.LoadLibrary(utils_so)


def has_amx() -> bool:
    if utils_lib is not None:
        return utils_lib.xsmm_has_amx() != 0
    raise NotImplementedError()
