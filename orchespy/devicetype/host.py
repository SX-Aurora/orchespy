from .base import Base
from .. import transfer
import numpy


class Host(Base):
    """Device type class for Host.

    Specify as the target of the orchespy's decorator or function.

    Examples
    --------
    Specify Host as the target of decorator.
    The function exec_on_host() will be execute on Host.

    >>> from orchespy.devicetype import Host
    >>> from orchespy import device
    >>> import nlcpy
    >>>
    >>> @device(Host)
    ... def exec_on_host(x, y):
    ...     return x * y
    >>>
    >>> x = nlcpy.ones((2,2))
    >>> y = nlcpy.ones((2,2))
    >>> z = exec_on_host(x, y)


    """

    def get_ndarray_on_host(self, ary):
        assert(isinstance(ary, numpy.ndarray))
        # ary is numpy.ndarray, which is on the host.
        return ary

    def get_ndarray_on_device(self, ary):
        assert(isinstance(ary, numpy.ndarray))
        # ary is numpy.ndarray, which is on the host.
        return ary

    @classmethod
    def find_device(self, ndarray):
        assert(isinstance(ndarray, numpy.ndarray))
        return Host()

    @property
    def numpy_class(self):
        return numpy


transfer.register_devicetype(Host, numpy.ndarray)
