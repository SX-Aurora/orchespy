from .base import Base
from .host import Host
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

    If you have multiple identical devices,
    you can specify any device for the decorator.
    Also, when operating within the device decorator function,
    It is necessary to switch the output device in advance.

    >>> from orchespy.devicetype import CUDAGPU
    >>> from orchespy import device
    >>> import numpy
    >>> import cupy as cp
    >>>
    >>> @device(CUDAGPU(1))
    ... def exec_on_gpu(x, y):
    ...     return x * y
    ...
    >>> x = numpy.ones((2,2)) * 3
    >>> y = numpy.ones((2,2)) * 2
    >>> cp.cuda.runtime.setDevice(1)
    >>> z = exec_on_gpu(x, y)
    >>> z.device.id
    1
    >>> type(z)
    <class 'cupy.ndarray'>
    >>> z
    array([[6., 6.],
           [6., 6.]])
    """
    def __init__(self, device_id=0):
        if not isinstance(device_id, int):
            raise TypeError('an integer is required')
        if device_id >= cupy.cuda.runtime.getDeviceCount():
            raise ValueError('This ID exceeds the number of'
                             ' cupy.cuda.runtime.getDeviceCount')
        self._device_id = device_id

    def can_transfer(self, obj):
        return isinstance(obj, (cupy.ndarray, numpy.ndarray))

    def can_transfer_to(self, obj, target):
        return (isinstance(obj, cupy.ndarray) and
                type(target) in (CUDAGPU, Host))

    def create_ndarray_on_device(self, obj):
        prev_cuda = cupy.cuda.runtime.getDevice()
        try:
            cupy.cuda.runtime.setDevice(self._device_id)
            if isinstance(obj, numpy.ndarray):
                _order = "F" if not obj.flags.c_contiguous and\
                    obj.flags.f_contiguous else "C"
            else:
                _order = "F" if not obj._c_contiguous and obj._f_contiguous else "C"
            return cupy.empty(obj.shape, dtype=obj.dtype, order=_order)
        finally:
            cupy.cuda.runtime.setDevice(prev_cuda)

    def transfer_array_content(self, dst, src):
        prev_cuda = cupy.cuda.runtime.getDevice()
        try:
            cupy.cuda.runtime.setDevice(self._device_id)
            if isinstance(src, cupy.ndarray):
                cupy.copyto(dst, src)
            else:
                dst.set(src)
        finally:
            cupy.cuda.runtime.setDevice(prev_cuda)

    def transfer_array_content_to(self, dst, src):
        prev_cuda = cupy.cuda.runtime.getDevice()
        try:
            cupy.cuda.runtime.setDevice(self._device_id)
            if isinstance(dst, cupy.ndarray):
                cupy.copyto(dst, src)
            else:
                cupy.asnumpy(src, out=dst)
        finally:
            cupy.cuda.runtime.setDevice(prev_cuda)

    @classmethod
    def get_device(self, ndarray):
        assert isinstance(ndarray, cupy.ndarray)
        return CUDAGPU(ndarray.device.id)

    @property
    def numpy_class(self):
        return cupy
