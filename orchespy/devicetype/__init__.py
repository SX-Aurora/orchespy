# device types definitions
from .host import Host
try:
    from .cuda import CUDAGPU
except ImportError:
    CUDAGPU = None
try:
    from .ve import VE
except ImportError:
    VE = None

__all__ = ['Host', 'CUDAGPU', 'VE']
