from .base import Base
try:
    from .cuda import CUDAGPU
except ImportError:
    CUDAGPU = None

from .. import transfer

try:
    from ._transfer import cunlc
except ImportError:
    cunlc = None

import nlcpy
import numpy

if cunlc is not None:
    _NDARRAY_CONVERTER_SRC = {
        CUDAGPU: cunlc.convert_from_cupy_to_nlcpy,
    }

    _NDARRAY_CONVERTER_DST = {
        CUDAGPU: cunlc.convert_from_nlcpy_to_cupy,
    }
else:
    _NDARRAY_CONVERTER_SRC = {
    }

    _NDARRAY_CONVERTER_DST = {
    }


class VE(Base):
    """Device type class for VE

    Specify as the target of the orchespy's decorator or function.

    Examples
    --------
    Specify VE as the target of decorator.
    The function exec_on_ve() will be execute on VE.

    >>> from orchespy.devicetype import VE
    >>> from orchespy import device
    >>> import numpy
    >>>
    >>> @device(VE)
    ... def exec_on_ve(x, y):
    ...     return x * y
    >>>
    >>> x = numpy.ones((2,2))
    >>> y = numpy.ones((2,2))
    >>> z = exec_on_ve(x, y)


    """
    def __init__(self):
        pass

    def get_ndarray_on_host(self, ndarray):
        assert(isinstance(ndarray, nlcpy.ndarray))
        return ndarray.get()

    def get_ndarray_on_device(self, ndarray):
        assert(isinstance(ndarray, numpy.ndarray))
        return nlcpy.asarray(ndarray)

    @classmethod
    def find_device(self, ndarray):
        assert(isinstance(ndarray, nlcpy.ndarray))
        return VE()

    @property
    def numpy_class(self):
        return nlcpy

    # for direct conversion
    def func_to_transfer_ndarray_from(self, srcdev):
        return _NDARRAY_CONVERTER_SRC.get(type(srcdev))

    def func_to_transfer_ndarray_to(self, dstdev):
        return _NDARRAY_CONVERTER_DST.get(type(dstdev))


transfer.register_devicetype(VE, nlcpy.ndarray)
