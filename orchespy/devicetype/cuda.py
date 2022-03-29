from .base import Base
from .. import transfer
import numpy
import cupy


class CUDAGPU(Base):
    """Device type class for CUDAGPU.

    Specify as the target of the orchespy's decorator or function.

    Examples
    --------
    Specify CUDAGPU as the target of decorator.
    The function exec_on_gpu() will be execute on CUDAGPU.

    >>> from orchespy.devicetype import CUDAGPU
    >>> from orchespy import device
    >>> import numpy
    >>>
    >>> @device(CUDAGPU)
    ... def exec_on_gpu(x, y):
    ...     return x * y
    >>>
    >>> x = numpy.ones((2,2))
    >>> y = numpy.ones((2,2))
    >>> z = exec_on_gpu(x, y)


    """
    def __init__(self):
        pass

    def get_ndarray_on_host(self, ndarray):
        assert(isinstance(ndarray, cupy.ndarray))
        return ndarray.get()

    def get_ndarray_on_device(self, ndarray):
        assert(isinstance(ndarray, numpy.ndarray))
        return cupy.asarray(ndarray)

    @classmethod
    def find_device(self, ndarray):
        assert(isinstance(ndarray, cupy.ndarray))
        # TODO: create CUDAGPU instance from ndarray.device
        #       for multi-GPU support
        return CUDAGPU()

    @property
    def numpy_class(self):
        return cupy


transfer.register_devicetype(CUDAGPU, cupy.ndarray)
