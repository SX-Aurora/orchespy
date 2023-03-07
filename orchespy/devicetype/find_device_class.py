# device types definitions
from .host import Host
import numpy
try:
    from .cuda import CUDAGPU
except ImportError:
    CUDAGPU = None
try:
    from .ve import VE
except ImportError:
    VE = None


def find_device_class(a):
    if isinstance(a, numpy.ndarray):
        return Host
    if VE is not None:
        if isinstance(a, VE().numpy_class.ndarray):
            return VE
    if CUDAGPU is not None:
        if isinstance(a, CUDAGPU().numpy_class.ndarray):
            return CUDAGPU
    return None
