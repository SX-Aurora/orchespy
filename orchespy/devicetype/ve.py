from .base import Base
from .host import Host
import numpy
import nlcpy
try:
    from .cuda import CUDAGPU
except ImportError:
    CUDAGPU = None
try:
    from ._transfer import cunlc
except ImportError:
    cunlc = None


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

    If you have multiple identical devices,
    you can specify any device for the decorator.
    Also, when operating within the device decorator function,
    It is necessary to switch the output device in advance.

    >>> from orchespy.devicetype import VE
    >>> from orchespy import device
    >>> import numpy
    >>> import nlcpy as vp
    >>>
    >>> @device(VE(1))
    ... def exec_on_ve(x, y):
    ...     return x * y
    ...
    >>> x = numpy.ones((2,2)) * 3
    >>> y = numpy.ones((2,2)) * 2
    >>> vp.venode.VE(1).use()
    <VE node logical_id=1, physical_id=1>
    >>> z = exec_on_ve(x, y)
    >>> z.venode.id
    1
    >>> type(z)
    <class 'nlcpy.core.core.ndarray'>
    >>> z
    array([[6., 6.],
           [6., 6.]])
    """
    def __init__(self, device_id=0):
        if not isinstance(device_id, int):
            raise TypeError('an integer is required')
        if device_id >= nlcpy.venode.get_num_available_venodes():
            raise ValueError('This ID exceeds the number of'
                             ' nlcpy.venode.get_num_available_venodes')
        self._device_id = device_id

    def _get_order(self, obj):
        if isinstance(obj, numpy.ndarray):
            return "F" if not obj.flags.c_contiguous and\
                obj.flags.f_contiguous else "C"
        else:
            return "F" if not obj._c_contiguous and obj._f_contiguous else "C"

    def can_transfer(self, obj):
        if CUDAGPU is None:
            return isinstance(obj, (nlcpy.ndarray, numpy.ndarray))
        return isinstance(obj, (nlcpy.ndarray,
                                CUDAGPU().numpy_class.ndarray,
                                numpy.ndarray))

    def can_transfer_to(self, obj, target):
        if CUDAGPU is None:
            return isinstance(obj, nlcpy.ndarray) and\
                type(target) in (VE, Host)
        return isinstance(obj, nlcpy.ndarray) and\
            type(target) in (VE, CUDAGPU, Host)

    def create_ndarray_on_device(self, obj):
        prev_ve = nlcpy.venode.VE()
        try:
            nlcpy.venode.VE(self._device_id).use()
            _order = self._get_order(obj)
            return nlcpy.empty(obj.shape, dtype=obj.dtype, order=_order)
        finally:
            prev_ve.use()

    def transfer_array_content(self, dst, src):
        prev_ve = nlcpy.venode.VE()
        try:
            nlcpy.venode.VE(self._device_id).use()

            if isinstance(src, nlcpy.ndarray):
                try:
                    nlcpy.copyto(dst, src)
                except ValueError:
                    raise NotImplementedError('This communication is not supported.')
            elif isinstance(src, numpy.ndarray):
                dst.set(src)
            elif cunlc is not None:
                if CUDAGPU is None:
                    raise NotImplementedError('This communication is not supported.')
                if isinstance(src, CUDAGPU().numpy_class.ndarray):
                    if src.nbytes != dst.nbytes:
                        if src.dtype == 'bool':
                            _order = self._get_order(src)
                            tmp_buf = CUDAGPU().numpy_class.empty(src.shape,
                                                                  dtype='i4',
                                                                  order=_order)
                            CUDAGPU().numpy_class.copyto(tmp_buf, src.astype('i4'))
                            cunlc.convert_from_cupy_to_nlcpy(dst, tmp_buf)
                        else:
                            dst.set(src.get())
                    else:
                        cunlc.convert_from_cupy_to_nlcpy(dst, src)
                else:
                    raise ValueError('src is an unsupported device')
            else:
                raise ValueError('src is an unsupported device')
        finally:
            prev_ve.use()

    def transfer_array_content_to(self, dst, src):
        prev_ve = nlcpy.venode.VE()
        try:
            nlcpy.venode.VE(self._device_id).use()

            if isinstance(dst, nlcpy.ndarray):
                try:
                    nlcpy.copyto(dst, src)
                except ValueError:
                    raise NotImplementedError('This communication is not supported.')
            elif isinstance(dst, numpy.ndarray):
                src.get(out=dst)
            elif cunlc is not None:
                if CUDAGPU is None:
                    raise NotImplementedError('This communication is not supported.')
                if isinstance(dst, CUDAGPU().numpy_class.ndarray):
                    if dst.nbytes != src.nbytes:
                        if dst.dtype == 'bool':
                            _order = self._get_order(src)
                            tmp_buf = CUDAGPU().numpy_class.empty(src.shape,
                                                                  dtype='i4',
                                                                  order=_order)
                            cunlc.convert_from_nlcpy_to_cupy(tmp_buf, src)
                            CUDAGPU().numpy_class.copyto(dst, tmp_buf.astype('?'))
                        else:
                            dst.set(src.get())
                    else:
                        cunlc.convert_from_nlcpy_to_cupy(dst, src)
                else:
                    raise ValueError('dst is an unsupported device')
            else:
                raise ValueError('dst is an unsupported device')
        finally:
            prev_ve.use()

    @classmethod
    def get_device(self, ndarray):
        assert isinstance(ndarray, nlcpy.ndarray)
        return VE(ndarray.venode.id)

    @property
    def numpy_class(self):
        return nlcpy
