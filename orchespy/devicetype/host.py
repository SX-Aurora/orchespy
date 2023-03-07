from .base import Base
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
    def can_transfer(self, obj):
        return isinstance(obj, numpy.ndarray)

    def can_transfer_to(self, obj, target):
        return isinstance(obj, numpy.ndarray) and type(target) == Host

    def create_ndarray_on_device(self, obj):
        _order = "F" if not obj.flags.c_contiguous and obj.flags.f_contiguous else "C"
        return numpy.empty(obj.shape, dtype=obj.dtype, order=_order)

    def transfer_array_content(self, dst, src):
        numpy.copyto(dst, src)

    def transfer_array_content_to(self, dst, src):
        numpy.copyto(dst, src)

    @classmethod
    def get_device(self, ndarray):
        assert isinstance(ndarray, numpy.ndarray)
        return Host()

    @property
    def numpy_class(self):
        return numpy
