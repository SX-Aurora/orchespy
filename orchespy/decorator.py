import functools
from .transfer import transfer_array


def device(target, numpy_module_arg=None):
    """ Execute a decorated function on a specified device.

    Parameters
    ----------
    target : devicetype
        A device where the function is executed.
        Specify the devicetype class that corresponds to the device.
        See :class:`orchespy.devicetype` for what you can specify.
    numpy_module_arg : string, optional
        The argument name to pass the
        namespace of NumPy-compatible package for the target device.

    See Also
    --------
    orchespy.devicetype: specify to target parameter.

    Examples
    --------
    Decorate function to execute on CUDAGPU. the function of exec_on_gpu()
    is executed on CUDAGPU.
    The ndarray specified args of function are converted to cupy.ndarray
    and processed.

    >>> from orchespy import device
    >>> from orchespy.devicetype import CUDAGPU
    >>> import numpy as np
    >>>
    >>> @device(CUDAGPU)
    ... def exec_on_gpu(x, y):
    ...     return x * y
    >>>
    >>> x = np.linspace(1, 5, 5)
    >>> y = np.linspace(1, 5, 5)
    >>> z = exec_on_gpu(x, y)

    Decorate function to execute on VE. the function of create_array_on_VE()
    is executed on VE. and return nlcpy.ndarray.

    >>> from orchespy import device
    >>> from orchespy.devicetype import VE
    >>>
    >>> @device(VE, numpy_module_arg='xp')
    ... def create_array_on_VE(xp):
    ...     return xp.random.rand(2, 2)
    >>>
    >>> x = create_array_on_VE()

    If you have multiple identical devices, you can specify any device for the decorator.
    Also, when operating within the device decorator function,
    It is necessary to switch the output device in advance.

    >>> from orchespy import device
    >>> import numpy as np
    >>> import nlcpy as vp
    >>> @device(VE(1))
    ... def exec_on_ve(x, y):
    ...     return x * y
    ...
    >>> x = np.ones((2,2))
    >>> y = np.ones((2,2))
    >>> vp.venode.VE(1).use()
    <VE node logical_id=1, physical_id=1>
    >>> mul = exec_on_ve(x, y)
    >>> mul.venode.id
    1
    """
    if isinstance(target, type):
        target = target()

    def _device(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with target as xp:
                args_converted = tuple((transfer_array(a, target)
                                        for a in args))
                kwargs_converted = dict([(k, transfer_array(kwargs[k], target))
                                         for k in kwargs])
                if numpy_module_arg is not None:
                    kwargs_converted[numpy_module_arg] = xp
                return func(*args_converted, **kwargs_converted)
        return wrapper
    return _device
